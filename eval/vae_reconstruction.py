from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_SCHEMA_VERSION = 3
MODEL_INPUT_FIELDS = (
    "surf_z",
    "surf_pos",
    "edge_z",
    "edge_pos",
    "adj_face",
    "num_face",
    "num_edge",
)
FAILURE_REASONS = (
    "empty_prediction",
    "reconstruction_exception",
    "not_closed_or_zero_volume",
)


def source_revision() -> dict[str, object]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "working_tree_dirty": None}
    return {
        "git_commit": revision,
        "working_tree_dirty": bool(status.strip()),
    }


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_name, path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def write_samples_jsonl(
    path: Path,
    records: Sequence[Mapping[str, object]],
) -> None:
    ordered = sorted((dict(record) for record in records), key=lambda item: int(item["index"]))
    indices = [int(record["index"]) for record in ordered]
    if indices != list(range(len(ordered))):
        raise RuntimeError("VAE sample indices must be complete, unique, and contiguous.")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for record in ordered:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temporary_name, path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def select_model_inputs(batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    missing = [name for name in MODEL_INPUT_FIELDS if name not in batch]
    if missing:
        raise ValueError(f"VAE evaluation batch is missing: {', '.join(missing)}.")
    selected = {name: batch[name] for name in MODEL_INPUT_FIELDS}
    if any(not torch.is_tensor(value) for value in selected.values()):
        raise TypeError("VAE model inputs must all be tensors.")
    return selected


def validate_resume_batch(
    payload: Mapping[str, object],
    *,
    fingerprint: str,
    batch_index: int,
    sample_start: int,
    uids: Sequence[str],
) -> None:
    if payload.get("format_version") != EVALUATION_SCHEMA_VERSION:
        raise ValueError("Resume batch uses an unsupported format version.")
    if payload.get("fingerprint") != fingerprint:
        raise ValueError("Resume batch fingerprint does not match this evaluation run.")
    samples = payload.get("samples")
    identity = (
        payload.get("batch_index") == batch_index
        and payload.get("sample_start") == sample_start
        and payload.get("uids") == list(uids)
        and payload.get("sample_count") == len(uids)
        and isinstance(samples, list)
        and [sample.get("uid") for sample in samples if isinstance(sample, Mapping)]
        == list(uids)
    )
    if not identity:
        raise ValueError("Resume batch identity does not match the manifest order.")


def use_ground_truth_counts(
    predictions: Mapping[str, torch.Tensor],
    *,
    num_faces: torch.Tensor,
    num_edges: torch.Tensor,
) -> dict[str, torch.Tensor]:
    replaced = dict(predictions)
    face_counts = num_faces.reshape(-1).to(device=predictions["num_face_logits"].device)
    edge_counts = num_edges.reshape(-1).to(device=predictions["num_edge_logits"].device)
    if face_counts.min() < 0 or face_counts.max() >= predictions["num_face_logits"].shape[-1]:
        raise ValueError("Ground-truth face count is outside the decoder class range.")
    if edge_counts.min() < 0 or edge_counts.max() >= predictions["num_edge_logits"].shape[-1]:
        raise ValueError("Ground-truth edge count is outside the decoder class range.")
    face_logits = torch.zeros_like(predictions["num_face_logits"])
    edge_logits = torch.zeros_like(predictions["num_edge_logits"])
    face_logits.scatter_(1, face_counts.long().unsqueeze(1), 1.0)
    edge_logits.scatter_(1, edge_counts.long().unsqueeze(1), 1.0)
    replaced["num_face_logits"] = face_logits
    replaced["num_edge_logits"] = edge_logits
    return replaced


def validate_vae_outputs(
    encoded: Mapping[str, object],
    predictions: Mapping[str, torch.Tensor],
    *,
    batch_size: int,
    latent_len: int,
    latent_dim: int,
    max_face: int,
    device: torch.device,
) -> dict[str, object]:
    from src.inference.reconstruction import validate_decoder_predictions

    expected = (batch_size, latent_len, latent_dim)
    for name in ("mu", "logvar"):
        value = encoded.get(name)
        if not torch.is_tensor(value):
            raise TypeError(f"VAE encoded output '{name}' must be a tensor.")
        if tuple(value.shape) != expected:
            raise ValueError(
                f"VAE encoded output '{name}' has shape {tuple(value.shape)}; "
                f"expected {expected}."
            )
        if value.dtype != torch.float32:
            raise ValueError(f"VAE encoded output '{name}' must remain float32.")
        if value.device != device:
            raise ValueError(f"VAE encoded output '{name}' is on the wrong device.")
        if not torch.isfinite(value).all():
            raise ValueError(f"VAE encoded output '{name}' contains non-finite values.")
    decoder_schema = validate_decoder_predictions(predictions, max_faces=max_face)
    for name, value in predictions.items():
        if value.dtype != torch.float32:
            raise ValueError(f"VAE decoder output '{name}' must remain float32.")
        if value.device != device:
            raise ValueError(f"VAE decoder output '{name}' is on the wrong device.")
    return {
        "latent": list(encoded["mu"].shape),
        "decoder": decoder_schema,
    }


def adjacency_counts_per_sample(
    predictions: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
) -> tuple[list[int], list[int]]:
    logits = predictions["adj_face_logits"]
    batch_size, max_edges, max_faces = logits.shape
    face_counts = batch["num_face"].reshape(batch_size)
    edge_counts = batch["num_edge"].reshape(batch_size)
    face_valid = torch.arange(max_faces, device=logits.device)[None] < face_counts[:, None]
    edge_valid = torch.arange(max_edges, device=logits.device)[None] < edge_counts[:, None]
    masked = logits.masked_fill(~face_valid[:, None, :], float("-inf"))
    indices = torch.topk(masked, k=2, dim=-1).indices
    predicted = torch.zeros_like(logits, dtype=torch.bool)
    batch_indices = torch.arange(batch_size, device=logits.device).view(-1, 1, 1)
    edge_indices = torch.arange(max_edges, device=logits.device).view(1, -1, 1)
    predicted[batch_indices, edge_indices, indices] = True
    predicted &= edge_valid[:, :, None] & face_valid[:, None, :]
    target = batch["adj_face"].bool()
    valid = edge_valid[:, :, None] & face_valid[:, None, :]
    hits = (((predicted == target) & valid).sum(dim=(1, 2))).tolist()
    totals = valid.sum(dim=(1, 2)).tolist()
    return [int(value) for value in hits], [int(value) for value in totals]


def make_sample_records(
    *,
    uids: Sequence[str],
    sample_start: int,
    reconstruction_samples: Sequence[Mapping[str, object]],
    adjacency_hits: Sequence[int],
    adjacency_totals: Sequence[int],
) -> list[dict[str, object]]:
    count = len(uids)
    if not (
        len(reconstruction_samples) == count
        and len(adjacency_hits) == count
        and len(adjacency_totals) == count
    ):
        raise RuntimeError("VAE per-sample outputs do not share one batch size.")
    records: list[dict[str, object]] = []
    for local_index, (uid, reconstruction) in enumerate(
        zip(uids, reconstruction_samples, strict=True)
    ):
        if int(reconstruction["index"]) != local_index:
            raise RuntimeError("VAE reconstruction returned unexpected sample order.")
        total = int(adjacency_totals[local_index])
        hits = int(adjacency_hits[local_index])
        records.append(
            {
                "index": sample_start + local_index,
                "uid": str(uid),
                "ground_truth_num_faces": int(reconstruction["num_faces"]),
                "ground_truth_num_edges": int(reconstruction["num_edges"]),
                "valid": bool(reconstruction["valid"]),
                "failure_reason": reconstruction["failure_reason"],
                "adjacency_hits": hits,
                "adjacency_total": total,
                "adjacency_accuracy": hits / max(total, 1),
            }
        )
    return records


def _environment(device: torch.device) -> dict[str, object]:
    packages: dict[str, str | None] = {}
    for name in ("numpy", "torch", "occwl", "chamferdist"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    properties = torch.cuda.get_device_properties(device)
    return {
        "python": platform.python_version(),
        "packages": packages,
        "torch_cuda": torch.version.cuda,
        "gpu": properties.name,
        "gpu_total_memory_bytes": int(properties.total_memory),
    }


def _move_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _summary_from_batches(
    *,
    variant: str,
    state: str,
    checkpoint: Mapping[str, object],
    batch_payloads: Sequence[Mapping[str, object]],
    dataset: str,
    tensor_schema: Mapping[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    samples = [
        dict(sample)
        for payload in batch_payloads
        for sample in payload["samples"]
        if isinstance(sample, Mapping)
    ]
    samples.sort(key=lambda sample: int(sample["index"]))
    sample_count = sum(int(payload["sample_count"]) for payload in batch_payloads)
    if len(samples) != sample_count:
        raise RuntimeError("VAE batch payloads have incomplete per-sample records.")
    if [int(sample["index"]) for sample in samples] != list(range(sample_count)):
        raise RuntimeError("VAE evaluation batches contain missing or duplicate indices.")
    uids = [str(sample["uid"]) for sample in samples]
    if len(set(uids)) != sample_count:
        raise RuntimeError("VAE evaluation batches contain duplicate sample IDs.")
    valid_count = sum(bool(sample["valid"]) for sample in samples)
    failures = {name: 0 for name in FAILURE_REASONS}
    for sample in samples:
        failure = sample["failure_reason"]
        if bool(sample["valid"]):
            if failure is not None:
                raise RuntimeError("A valid VAE reconstruction cannot have a failure reason.")
        elif failure not in failures:
            raise RuntimeError("An invalid VAE reconstruction needs a stable failure reason.")
        else:
            failures[str(failure)] += 1
    if valid_count + sum(failures.values()) != sample_count:
        raise RuntimeError("VAE validity and failure counts do not cover the evaluated split.")
    adjacency_hits = sum(int(sample["adjacency_hits"]) for sample in samples)
    adjacency_total = sum(int(sample["adjacency_total"]) for sample in samples)
    loss_sum = sum(float(payload["loss_sum"]) for payload in batch_payloads)
    return (
        {
            "format_version": EVALUATION_SCHEMA_VERSION,
            "dataset": dataset,
            "variant": variant,
            "state": state,
            "checkpoint": dict(checkpoint),
            "sample_count": sample_count,
            "posterior": "mean",
            "dtype": "float32",
            "reconstruction_count_source": "ground_truth",
            "reconstruction": "closed_shell_and_nonzero_volume",
            "valid_count": valid_count,
            "valid_rate": valid_count / sample_count,
            "adjacency_hits": adjacency_hits,
            "adjacency_total": adjacency_total,
            "adjacency_accuracy": adjacency_hits / max(adjacency_total, 1),
            "validation_loss": loss_sum / sample_count,
            "failure_counts": failures,
            "tensor_schema": dict(tensor_schema),
        },
        samples,
    )


def _load_contract(config_path: Path) -> dict[str, object]:
    from src.training.config import load_training_recipe

    recipe = load_training_recipe(
        config_path,
        expected_task="vae",
        repository_root=REPOSITORY_ROOT,
    )
    data = recipe.section("data")
    split = data.get("val_split")
    if not isinstance(split, str) or not split:
        raise ValueError("VAE recipe data.val_split must be a non-empty string.")
    return {
        "config": {
            "filename": recipe.path.name,
            "sha256": recipe.sha256,
        },
        "manifest_path": recipe.manifest_path,
        "manifest_relative": str(data["manifest"]),
        "split": split,
        "bbox_scale": float(data.get("bbox_scale", 3.0)),
        "model": recipe.section("model"),
        "variant": recipe.variant,
    }


def validate_recipe_variant(*, configured: object, requested: str) -> str:
    if not isinstance(configured, str) or configured != requested:
        raise ValueError(
            "The requested variant must match the VAE recipe root variant."
        )
    return requested


def prepare_portable_target(
    *,
    checkpoint: str | Path,
    dataset: str,
    variant: str,
    model: torch.nn.Module,
) -> dict[str, object]:
    """Strictly load one self-describing portable VAE without hashing it."""
    from src.training.checkpoint import load_portable_vae_weights

    path = Path(checkpoint).expanduser().resolve()
    identity = load_portable_vae_weights(
        model,
        path,
        expected_dataset=dataset,
        expected_variant=variant,
    )
    metadata = identity["release_metadata"]
    if not isinstance(metadata, Mapping):
        raise RuntimeError("Portable VAE identity has no release metadata.")
    return {
        "variant": variant,
        "path": path,
        "state": metadata["state"],
        "checkpoint": identity,
    }


def _run_target(
    *,
    target: Mapping[str, object],
    target_dir: Path,
    model: torch.nn.Module,
    dataloader: Any,
    protocol: Mapping[str, object],
    execution: Mapping[str, object],
    implementation: Mapping[str, object],
    model_config: Mapping[str, object],
    device: torch.device,
    resume: bool,
    show_progress: bool,
) -> dict[str, object]:
    from tqdm.auto import tqdm

    from src.inference.reconstruction import ReconstructionRunner
    variant = str(target["variant"])
    state = str(target["state"])
    checkpoint = dict(target["checkpoint"])
    fingerprint = _canonical_sha256(
        {
            "protocol": protocol,
            "execution": execution,
            "implementation": implementation,
            "checkpoint": checkpoint,
            "variant": variant,
            "state": state,
            "model": dict(model_config),
        }
    )
    batch_dir = target_dir / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    expected_batch_count = (
        int(protocol["sample_count"]) + int(protocol["batch_size"]) - 1
    ) // int(protocol["batch_size"])
    expected_batch_paths = {
        batch_dir / f"batch_{index:05d}.json"
        for index in range(expected_batch_count)
    }
    unexpected_batches = sorted(set(batch_dir.glob("batch_*.json")) - expected_batch_paths)
    if unexpected_batches:
        names = ", ".join(path.name for path in unexpected_batches)
        raise ValueError(f"VAE run directory contains unexpected batch files: {names}")
    model.eval()
    payloads: list[dict[str, object]] = []
    tensor_schema: dict[str, object] | None = None
    max_face = int(model_config["max_face"])
    latent_len = int(model_config["latent_len"])
    latent_dim = int(model_config["latent_dim"])
    with ReconstructionRunner(
        mode="batched",
        workers=int(execution["reconstruction_workers"]),
        device=device,
    ) as runner:
        progress = tqdm(
            dataloader,
            desc=f"{variant}/{state}",
            disable=not show_progress,
        )
        for batch_index, cpu_batch in enumerate(progress):
            uids = [str(uid) for uid in cpu_batch["uid"]]
            sample_start = batch_index * int(protocol["batch_size"])
            batch_path = batch_dir / f"batch_{batch_index:05d}.json"
            if resume and batch_path.is_file():
                existing = json.loads(batch_path.read_text(encoding="utf-8"))
                validate_resume_batch(
                    existing,
                    fingerprint=fingerprint,
                    batch_index=batch_index,
                    sample_start=sample_start,
                    uids=uids,
                )
                payloads.append(existing)
                tensor_schema = dict(existing["tensor_schema"])
                continue
            batch = _move_to_device(cpu_batch, device)
            model_inputs = select_model_inputs(batch)
            started = time.perf_counter()
            with torch.inference_mode():
                encoded, predictions = model(
                    **model_inputs,
                    sample_posterior=False,
                    return_pred=True,
                )
                current_schema = validate_vae_outputs(
                    encoded,
                    predictions,
                    batch_size=len(uids),
                    latent_len=latent_len,
                    latent_dim=latent_dim,
                    max_face=max_face,
                    device=device,
                )
                loss, _ = model.loss_from_predictions(
                    encoded=encoded,
                    decoded=predictions,
                    **model_inputs,
                )
                if not torch.isfinite(loss):
                    raise ValueError("VAE validation loss is non-finite.")
                adjacency_hits, adjacency_totals = adjacency_counts_per_sample(
                    predictions,
                    batch,
                )
            reconstruction_predictions = use_ground_truth_counts(
                predictions,
                num_faces=batch["num_face"],
                num_edges=batch["num_edge"],
            )
            reconstruction = runner.reconstruct(
                reconstruction_predictions,
                face_min=1,
                face_max=max_face,
                output_dir=target_dir / "steps-unused",
                save_step_policy="none",
            )
            records = make_sample_records(
                uids=uids,
                sample_start=sample_start,
                reconstruction_samples=reconstruction.samples,
                adjacency_hits=adjacency_hits,
                adjacency_totals=adjacency_totals,
            )
            failure_counts = {
                name: sum(record["failure_reason"] == name for record in records)
                for name in FAILURE_REASONS
            }
            if sum(bool(record["valid"]) for record in records) + sum(
                failure_counts.values()
            ) != len(records):
                raise RuntimeError("VAE reconstruction outcomes do not cover the batch.")
            if tensor_schema is None:
                tensor_schema = current_schema
            else:
                expected_tails = {
                    "latent": list(tensor_schema["latent"])[1:],
                    "decoder": {
                        name: list(shape)[1:]
                        for name, shape in tensor_schema["decoder"].items()
                    },
                }
                current_tails = {
                    "latent": list(current_schema["latent"])[1:],
                    "decoder": {
                        name: list(shape)[1:]
                        for name, shape in current_schema["decoder"].items()
                    },
                }
                if current_tails != expected_tails:
                    raise RuntimeError("VAE tensor schema changed between evaluation batches.")
            payload = {
                "format_version": EVALUATION_SCHEMA_VERSION,
                "fingerprint": fingerprint,
                "batch_index": batch_index,
                "sample_start": sample_start,
                "sample_count": len(records),
                "uids": uids,
                "samples": records,
                "valid_count": sum(bool(record["valid"]) for record in records),
                "adjacency_hits": sum(adjacency_hits),
                "adjacency_total": sum(adjacency_totals),
                "loss_sum": float(loss.item()) * len(records),
                "failure_counts": failure_counts,
                "tensor_schema": current_schema,
                "elapsed_seconds": time.perf_counter() - started,
            }
            _atomic_json(batch_path, payload)
            payloads.append(payload)
    if tensor_schema is None:
        raise RuntimeError("VAE evaluation produced no batches.")
    summary, samples = _summary_from_batches(
        variant=variant,
        state=state,
        checkpoint=checkpoint,
        batch_payloads=payloads,
        dataset=str(protocol["dataset"]),
        tensor_schema=tensor_schema,
    )
    if summary["sample_count"] != int(protocol["sample_count"]):
        raise RuntimeError("VAE evaluation did not cover the selected validation samples.")
    write_samples_jsonl(target_dir / "samples.jsonl", samples)
    _atomic_json(target_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a VAE checkpoint with FP32 posterior-mean reconstruction."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--variant", required=True, help="Released VAE variant.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--reconstruction-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Evaluate only the first N ordered validation samples.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0 or args.reconstruction_workers <= 0:
        raise ValueError(
            "Batch and worker counts must be positive (DataLoader workers may be zero)."
        )
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive.")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    from torch.utils.data import DataLoader, Subset

    from src.data.brep_dataset import PortableBrepDataset
    from src.data.manifest import load_dataset_manifest
    from src.vae.latentbrep.latentbrep_vae import LatentBiSPBrepVAE

    contract = _load_contract(args.config)
    variant = validate_recipe_variant(
        configured=contract["variant"],
        requested=args.variant,
    )
    manifest = load_dataset_manifest(contract["manifest_path"])
    split = str(contract["split"])
    split_paths = manifest.split(split)
    dataset = PortableBrepDataset(
        data_root=args.data_root,
        manifest_path=contract["manifest_path"],
        split=split,
        max_face=int(contract["model"]["max_face"]),
        bbox_scale=float(contract["bbox_scale"]),
        augment=False,
    )
    selected_count = len(dataset)
    if args.max_samples is not None:
        if args.max_samples > len(dataset):
            raise ValueError("--max-samples exceeds the evaluation split size.")
        selected_count = args.max_samples
    selected_dataset = Subset(dataset, range(selected_count))
    dataloader = DataLoader(
        selected_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("VAE reconstruction evaluation requires CUDA.")
    model_config = dict(contract["model"])
    model = LatentBiSPBrepVAE(**model_config)
    target = prepare_portable_target(
        checkpoint=args.checkpoint,
        dataset=manifest.dataset,
        variant=variant,
        model=model,
    )
    protocol = {
        "config": contract["config"],
        "manifest": {
            "path": contract["manifest_relative"],
            "sha256": manifest.sha256,
            "split": split,
            "split_sha256": manifest.split_sha256[split],
            "split_count": len(split_paths),
        },
        "dataset": manifest.dataset,
        "sample_count": selected_count,
        "full_split": selected_count == len(split_paths),
        "batch_size": args.batch_size,
        "posterior": "mean",
        "dtype": "float32",
        "reconstruction_count_source": "ground_truth",
        "adjacency_denominator": "ground_truth_edge_by_face_extent",
        "reconstruction": "closed_shell_and_nonzero_volume",
    }
    execution = {
        "device_type": device.type,
        "reconstruction_mode": "batched",
        "reconstruction_workers": args.reconstruction_workers,
        "num_workers": args.num_workers,
        "mode": "portable_checkpoint",
        "variant": target["variant"],
        "state": target["state"],
    }
    implementation = {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        **source_revision(),
    }
    run_config = {
        "format_version": EVALUATION_SCHEMA_VERSION,
        "implementation": implementation,
        "protocol": protocol,
        "execution": execution,
        "checkpoint": target["checkpoint"],
        "environment": _environment(device),
    }
    run_config_path = output_dir / "run_config.json"
    if args.resume:
        if not run_config_path.is_file():
            raise FileNotFoundError("--resume requires an existing run_config.json.")
        existing_run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        if existing_run_config != run_config:
            raise ValueError("Resume run configuration does not match this evaluation.")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(run_config_path, run_config)

    model.to(device=device, dtype=torch.float32)
    summary = _run_target(
        target=target,
        target_dir=output_dir,
        model=model,
        dataloader=dataloader,
        protocol=protocol,
        execution=execution,
        implementation=implementation,
        model_config=model_config,
        device=device,
        resume=args.resume,
        show_progress=not args.no_progress,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
