from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch


@dataclass(frozen=True)
class ReconstructionInputs:
    num_faces: torch.Tensor
    num_edges: torch.Tensor
    face_range_mask: torch.Tensor
    adjacency: torch.Tensor


@dataclass(frozen=True)
class ReconstructionResult:
    samples: list[dict[str, object]]
    stats: dict[str, int]
    validity_in_range: float | None
    timing_seconds: dict[str, float | None] = field(default_factory=dict)


@dataclass(frozen=True)
class SampleOutcome:
    index: int
    valid: bool
    failure_reason: str | None
    step_written: bool
    compilable: bool | None
    step_write_error: bool


EXPECTED_PREDICTION_KEYS = (
    "num_face_logits",
    "num_edge_logits",
    "face_z_hat",
    "face_pos_hat",
    "edge_z_hat",
    "edge_bbox_hat",
    "edge_corners_hat",
    "adj_face_logits",
)


def order_worker_results(
    results: list[object],
    *,
    expected_indices: list[int],
) -> list[object]:
    """Restore deterministic sample order and reject incomplete worker output."""
    by_index: dict[int, object] = {}
    for result in results:
        index = int(getattr(result, "index"))
        if index in by_index:
            raise RuntimeError("Worker result indices are missing or duplicated.")
        by_index[index] = result
    if set(by_index) != set(expected_indices):
        raise RuntimeError("Worker result indices are missing or duplicated.")
    return [by_index[index] for index in expected_indices]


def build_reconstruction_result(
    inputs: ReconstructionInputs,
    *,
    outcomes: list[SampleOutcome],
    output_dir: str | Path,
    timing_seconds: dict[str, float | None] | None = None,
) -> ReconstructionResult:
    batch_size = int(inputs.num_faces.shape[0])
    ordered = order_worker_results(
        list(outcomes),
        expected_indices=list(range(batch_size)),
    )
    output_dir = Path(output_dir)
    samples: list[dict[str, object]] = []
    failure_counts = {
        "empty_prediction": 0,
        "reconstruction_exception": 0,
        "not_closed_or_zero_volume": 0,
    }
    step_write_errors = 0
    valid_flags = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=inputs.num_faces.device,
    )
    for outcome in ordered:
        index = outcome.index
        face_count = int(inputs.num_faces[index].item())
        edge_count = int(inputs.num_edges[index].item())
        if outcome.valid:
            if outcome.failure_reason is not None:
                raise RuntimeError("A valid sample cannot have a failure reason.")
            valid_flags[index] = True
        else:
            if outcome.failure_reason not in failure_counts:
                raise RuntimeError("An invalid sample must have a stable failure reason.")
            failure_counts[outcome.failure_reason] += 1
        step_write_errors += int(outcome.step_write_error)
        expected_step = output_dir / (
            f"sample_idx_{index}_{face_count}f_{edge_count}e.step"
        )
        samples.append(
            {
                "index": index,
                "num_faces": face_count,
                "num_edges": edge_count,
                "in_face_range": bool(inputs.face_range_mask[index].item()),
                "valid": outcome.valid,
                "failure_reason": outcome.failure_reason,
                "compilable": outcome.compilable,
                "step_file": expected_step.name if outcome.step_written else None,
            }
        )

    in_range = inputs.face_range_mask
    validity_in_range = None
    if bool(in_range.any().item()):
        validity_in_range = float(valid_flags[in_range].float().mean().item())
    return ReconstructionResult(
        samples=samples,
        stats={
            "num_samples": batch_size,
            "fails_empty_after_filter": failure_counts["empty_prediction"],
            "fails_recon_exception": failure_counts["reconstruction_exception"],
            "not_occ_valid": 0,
            "fails_not_closed_or_zero_volume": failure_counts[
                "not_closed_or_zero_volume"
            ],
            "fails_step_write": step_write_errors,
        },
        validity_in_range=validity_in_range,
        timing_seconds=dict(timing_seconds or {}),
    )


def validate_decoder_predictions(
    predictions: Mapping[str, torch.Tensor],
    *,
    max_faces: int,
) -> dict[str, list[int]]:
    max_edges = max_faces * 3
    expected_shapes = {
        "num_face_logits": (max_faces + 1,),
        "num_edge_logits": (max_edges + 1,),
        "face_z_hat": (max_faces, 108),
        "face_pos_hat": (max_faces, 6),
        "edge_z_hat": (max_edges, 18),
        "edge_bbox_hat": (max_edges, 6),
        "edge_corners_hat": (max_edges, 6),
        "adj_face_logits": (max_edges, max_faces),
    }
    missing = [key for key in EXPECTED_PREDICTION_KEYS if key not in predictions]
    if missing:
        raise KeyError(f"Decoder output is missing keys: {', '.join(missing)}")

    batch_size: int | None = None
    schema: dict[str, list[int]] = {}
    for key, tail_shape in expected_shapes.items():
        value = predictions[key]
        if not torch.is_tensor(value):
            raise TypeError(f"Decoder output '{key}' is not a tensor.")
        if tuple(value.shape[1:]) != tail_shape:
            raise ValueError(
                f"Decoder output '{key}' has shape {tuple(value.shape)}; "
                f"expected [batch, {', '.join(map(str, tail_shape))}]."
            )
        if batch_size is None:
            batch_size = int(value.shape[0])
        elif value.shape[0] != batch_size:
            raise ValueError("Decoder outputs do not share one batch dimension.")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Decoder output '{key}' contains non-finite values.")
        schema[key] = list(value.shape)
    return schema


def prepare_reconstruction_inputs(
    predictions: Mapping[str, torch.Tensor],
    *,
    face_min: int,
    face_max: int,
) -> ReconstructionInputs:
    num_faces = predictions["num_face_logits"].argmax(dim=-1)
    num_edges = predictions["num_edge_logits"].argmax(dim=-1)
    batch_size = int(num_faces.shape[0])
    max_faces = int(predictions["face_z_hat"].shape[1])
    max_edges = int(predictions["edge_z_hat"].shape[1])
    device = num_faces.device

    face_valid = torch.arange(max_faces, device=device)[None, :] < num_faces[:, None]
    edge_valid = torch.arange(max_edges, device=device)[None, :] < num_edges[:, None]
    logits = predictions["adj_face_logits"].masked_fill(
        ~face_valid[:, None, :], float("-inf")
    )
    rows_with_faces = (edge_valid[:, :, None] & face_valid[:, None, :]).any(
        dim=-1, keepdim=True
    )
    logits = torch.where(
        rows_with_faces.expand_as(logits),
        logits,
        torch.zeros_like(logits),
    )
    top_two_faces = torch.topk(logits, k=2, dim=-1).indices
    adjacency = torch.zeros_like(logits, dtype=torch.bool)
    batch_indices = torch.arange(batch_size, device=device).view(-1, 1, 1)
    edge_indices = torch.arange(max_edges, device=device).view(1, -1, 1)
    adjacency[batch_indices, edge_indices, top_two_faces] = True
    adjacency &= edge_valid[:, :, None]
    adjacency &= face_valid[:, None, :]

    return ReconstructionInputs(
        num_faces=num_faces,
        num_edges=num_edges,
        face_range_mask=(num_faces >= face_min) & (num_faces <= face_max),
        adjacency=adjacency,
    )


def optimize_prepared_batch(
    geometries: Sequence[Any],
    *,
    device: torch.device,
) -> dict[int, Any]:
    """Apply the legacy surface-offset objective to all prepared samples on CUDA."""
    if not geometries:
        return {}
    if device.type != "cuda":
        raise ValueError("Batched reconstruction optimization requires CUDA.")

    import numpy as np
    from chamferdist import ChamferDistance

    surface_arrays = []
    sample_slices: dict[int, slice] = {}
    grouped_targets: dict[int, list[tuple[int, Any]]] = {}
    face_weights = []
    for geometry in geometries:
        sample_start = len(surface_arrays)
        num_faces = int(len(geometry.surface_initial))
        if num_faces <= 0:
            raise RuntimeError("Prepared geometry must contain at least one face.")
        for local_index, adjacency in enumerate(geometry.face_edge_adjacency):
            surface_arrays.append(geometry.surface_initial[local_index])
            edge_points = np.asarray(
                geometry.edge_wcs[adjacency],
                dtype=np.float32,
            ).reshape(-1, 3)
            if len(edge_points) == 0:
                raise RuntimeError("Prepared geometry contains a face without edges.")
            global_index = len(surface_arrays) - 1
            grouped_targets.setdefault(len(edge_points), []).append(
                (global_index, edge_points)
            )
            face_weights.append(1.0 / num_faces)
        sample_slices[int(geometry.index)] = slice(
            sample_start,
            sample_start + num_faces,
        )

    surface = torch.as_tensor(
        np.stack(surface_arrays),
        dtype=torch.float32,
        device=device,
    )
    state = torch.nn.Parameter(
        torch.tensor(
            [1.0, 0.0, 0.0, 0.0],
            dtype=torch.float32,
            device=device,
        ).repeat(len(surface_arrays), 1)
    )
    optimizer = torch.optim.AdamW(
        [state],
        lr=1e-3,
        betas=(0.95, 0.999),
        weight_decay=1e-6,
        eps=1e-8,
    )
    loss_function = ChamferDistance().to(device)
    weights = torch.tensor(face_weights, dtype=torch.float32, device=device)
    target_groups = []
    for target_count in sorted(grouped_targets):
        entries = grouped_targets[target_count]
        indices = torch.tensor(
            [entry[0] for entry in entries],
            dtype=torch.long,
            device=device,
        )
        targets = torch.as_tensor(
            np.stack([entry[1] for entry in entries]),
            dtype=torch.float32,
            device=device,
        )
        target_groups.append((indices, targets))

    with torch.enable_grad():
        for _ in range(200):
            surface_updated = surface + state[:, 1:].reshape(-1, 1, 1, 3)
            loss = torch.zeros((), dtype=torch.float32, device=device)
            for indices, targets in target_groups:
                distances = loss_function(
                    surface_updated[indices].reshape(len(indices), -1, 3),
                    targets,
                    bidirectional=False,
                    reverse=True,
                    batch_reduction=None,
                    point_reduction="sum",
                )
                loss = loss + (distances * weights[indices]).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    optimized = surface_updated.detach().cpu().numpy()
    return {
        index: optimized[sample_slice]
        for index, sample_slice in sample_slices.items()
    }


class ReconstructionRunner:
    """Persistent execution boundary for legacy or staged batch reconstruction."""

    def __init__(
        self,
        *,
        mode: str,
        workers: int,
        device: torch.device,
    ) -> None:
        if mode not in {"legacy", "batched"}:
            raise ValueError("Reconstruction mode must be legacy or batched.")
        if workers <= 0:
            raise ValueError("Reconstruction workers must be positive.")
        if mode == "legacy" and workers != 1:
            raise ValueError("Legacy reconstruction requires exactly one worker.")
        if device.type != "cuda":
            raise ValueError("Reconstruction requires a CUDA device.")
        self.mode = mode
        self.workers = workers
        self.device = device
        self._executor: ProcessPoolExecutor | None = None

    def __enter__(self) -> "ReconstructionRunner":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def _ensure_executor(self) -> ProcessPoolExecutor:
        if self._executor is None:
            from src.inference.reconstruction_worker import initialize_occ_worker

            self._executor = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=get_context("spawn"),
                initializer=initialize_occ_worker,
            )
        return self._executor

    def _run_worker_stage(
        self,
        function: Callable[[Any], Any],
        tasks: Sequence[Any],
    ) -> list[Any]:
        if not tasks:
            return []
        executor = self._ensure_executor()
        results = list(executor.map(function, tasks, chunksize=1))
        return order_worker_results(
            results,
            expected_indices=[int(task.index) for task in tasks],
        )

    def reconstruct(
        self,
        predictions: Mapping[str, torch.Tensor],
        *,
        face_min: int,
        face_max: int,
        output_dir: str | Path,
        save_step_policy: str,
    ) -> ReconstructionResult:
        if self.mode == "legacy":
            started = time.perf_counter()
            result = reconstruct_predictions(
                predictions,
                face_min=face_min,
                face_max=face_max,
                output_dir=output_dir,
                save_step_policy=save_step_policy,
            )
            return replace(
                result,
                timing_seconds={
                    "preparation": None,
                    "gpu_optimization": None,
                    "finalization": None,
                    "legacy_total": time.perf_counter() - started,
                },
            )
        return self._reconstruct_batched(
            predictions,
            face_min=face_min,
            face_max=face_max,
            output_dir=output_dir,
            save_step_policy=save_step_policy,
        )

    def _reconstruct_batched(
        self,
        predictions: Mapping[str, torch.Tensor],
        *,
        face_min: int,
        face_max: int,
        output_dir: str | Path,
        save_step_policy: str,
    ) -> ReconstructionResult:
        import numpy as np

        from src.inference.reconstruction_worker import (
            FinalizationTask,
            SampleGeometryInput,
            finalize_geometry_sample,
            prepare_geometry_sample,
        )

        if save_step_policy not in {"none", "valid", "all"}:
            raise ValueError("save_step_policy must be one of: none, valid, all.")
        output_dir = Path(output_dir)
        if save_step_policy != "none":
            output_dir.mkdir(parents=True, exist_ok=True)

        preparation_started = time.perf_counter()
        inputs = prepare_reconstruction_inputs(
            predictions,
            face_min=face_min,
            face_max=face_max,
        )
        cpu_values = {
            key: value.detach().cpu().numpy()
            for key, value in predictions.items()
            if key
            in {
                "face_z_hat",
                "face_pos_hat",
                "edge_z_hat",
                "edge_bbox_hat",
                "edge_corners_hat",
            }
        }
        num_faces = inputs.num_faces.detach().cpu().numpy()
        num_edges = inputs.num_edges.detach().cpu().numpy()
        adjacency = inputs.adjacency.detach().cpu().numpy()
        preparation_tasks = []
        outcomes: list[SampleOutcome] = []
        for index, (face_count_raw, edge_count_raw) in enumerate(
            zip(num_faces, num_edges)
        ):
            face_count = int(face_count_raw)
            edge_count = int(edge_count_raw)
            if face_count <= 0 or edge_count <= 0:
                outcomes.append(
                    SampleOutcome(
                        index=index,
                        valid=False,
                        failure_reason="empty_prediction",
                        step_written=False,
                        compilable=False if save_step_policy == "all" else None,
                        step_write_error=False,
                    )
                )
                continue
            preparation_tasks.append(
                SampleGeometryInput(
                    index=index,
                    face_z=np.ascontiguousarray(
                        cpu_values["face_z_hat"][index, :face_count]
                    ),
                    face_pos=np.ascontiguousarray(
                        cpu_values["face_pos_hat"][index, :face_count]
                    ),
                    edge_z=np.ascontiguousarray(
                        cpu_values["edge_z_hat"][index, :edge_count]
                    ),
                    edge_bbox=np.ascontiguousarray(
                        cpu_values["edge_bbox_hat"][index, :edge_count]
                    ),
                    edge_corners=np.ascontiguousarray(
                        cpu_values["edge_corners_hat"][index, :edge_count]
                    ),
                    adjacency=np.ascontiguousarray(
                        adjacency[index, :edge_count, :face_count]
                    ),
                )
            )
        preparation_results = self._run_worker_stage(
            prepare_geometry_sample,
            preparation_tasks,
        )
        geometries = []
        for result in preparation_results:
            if result.geometry is None:
                outcomes.append(
                    SampleOutcome(
                        index=result.index,
                        valid=False,
                        failure_reason=result.failure_reason,
                        step_written=False,
                        compilable=False if save_step_policy == "all" else None,
                        step_write_error=False,
                    )
                )
            else:
                geometries.append(result.geometry)
        preparation_seconds = time.perf_counter() - preparation_started

        torch.cuda.synchronize(self.device)
        optimization_started = time.perf_counter()
        optimized_surfaces = optimize_prepared_batch(
            geometries,
            device=self.device,
        )
        torch.cuda.synchronize(self.device)
        optimization_seconds = time.perf_counter() - optimization_started

        finalization_started = time.perf_counter()
        finalization_tasks = []
        for geometry in geometries:
            index = int(geometry.index)
            face_count = int(num_faces[index])
            edge_count = int(num_edges[index])
            step_path = None
            if save_step_policy != "none":
                step_path = str(
                    output_dir
                    / f"sample_idx_{index}_{face_count}f_{edge_count}e.step"
                )
            finalization_tasks.append(
                FinalizationTask(
                    index=index,
                    surface_wcs=optimized_surfaces[index],
                    edge_wcs=geometry.edge_wcs,
                    face_edge_adjacency=geometry.face_edge_adjacency,
                    edge_vertex_adjacency=geometry.edge_vertex_adjacency,
                    step_path=step_path,
                    save_step_policy=save_step_policy,
                )
            )
        finalization_results = self._run_worker_stage(
            finalize_geometry_sample,
            finalization_tasks,
        )
        outcomes.extend(
            SampleOutcome(
                index=result.index,
                valid=result.valid,
                failure_reason=result.failure_reason,
                step_written=result.step_written,
                compilable=result.compilable,
                step_write_error=result.step_write_error,
            )
            for result in finalization_results
        )
        finalization_seconds = time.perf_counter() - finalization_started
        return build_reconstruction_result(
            inputs,
            outcomes=outcomes,
            output_dir=output_dir,
            timing_seconds={
                "preparation": preparation_seconds,
                "gpu_optimization": optimization_seconds,
                "finalization": finalization_seconds,
                "legacy_total": None,
            },
        )


def reconstruct_predictions(
    predictions: Mapping[str, torch.Tensor],
    *,
    face_min: int,
    face_max: int,
    output_dir: str | Path,
    save_step_policy: str = "all",
) -> ReconstructionResult:
    from src.eval.validity import compute_validity_batch_latentsortbrep

    if save_step_policy not in {"none", "valid", "all"}:
        raise ValueError("save_step_policy must be one of: none, valid, all.")
    output_dir = Path(output_dir)
    if save_step_policy != "none":
        output_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_reconstruction_inputs(
        predictions,
        face_min=face_min,
        face_max=face_max,
    )
    step_prefix = output_dir / "sample"
    flags, raw_stats, failure_reasons, step_written, compilable = (
        compute_validity_batch_latentsortbrep(
            face_z=predictions["face_z_hat"],
            face_pos=predictions["face_pos_hat"],
            edge_z=predictions["edge_z_hat"],
            edge_bbox=predictions["edge_bbox_hat"],
            edge_corners=predictions["edge_corners_hat"],
            adj_face_bin=prepared.adjacency,
            num_face=prepared.num_faces,
            num_edge=prepared.num_edges,
            save_step_path=(
                str(step_prefix) if save_step_policy != "none" else None
            ),
            save_step_policy=save_step_policy,
        )
    )

    samples: list[dict[str, object]] = []
    for index in range(int(flags.shape[0])):
        face_count = int(prepared.num_faces[index].item())
        edge_count = int(prepared.num_edges[index].item())
        expected_step = output_dir / (
            f"sample_idx_{index}_{face_count}f_{edge_count}e.step"
        )
        samples.append(
            {
                "index": index,
                "num_faces": face_count,
                "num_edges": edge_count,
                "in_face_range": bool(prepared.face_range_mask[index].item()),
                "valid": bool(flags[index].item()),
                "failure_reason": failure_reasons[index],
                "compilable": compilable[index],
                "step_file": expected_step.name if step_written[index] else None,
            }
        )

    in_range = prepared.face_range_mask
    validity_in_range = None
    if bool(in_range.any().item()):
        validity_in_range = float(flags[in_range].float().mean().item())
    stats = {
        key: int(value.reshape(-1)[0].item())
        for key, value in raw_stats.items()
    }
    return ReconstructionResult(
        samples=samples,
        stats=stats,
        validity_in_range=validity_in_range,
    )
