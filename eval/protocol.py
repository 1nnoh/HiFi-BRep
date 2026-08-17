from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.utils.config import load_config


FAILURE_REASONS = (
    "empty_prediction",
    "reconstruction_exception",
    "not_closed_or_zero_volume",
)
SAVE_STEP_POLICIES = ("none", "valid", "all")
RECONSTRUCTION_MODES = ("legacy", "batched")


@dataclass(frozen=True)
class EvaluationProtocol:
    format_version: int
    num_samples: int
    batch_size: int
    base_seed: int
    num_inference_steps: int
    eta: float
    dtype: str
    reconstruction_criterion: str
    save_steps: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationExecution:
    reconstruction_mode: str
    reconstruction_workers: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BatchSpec:
    index: int
    sample_start: int
    sample_count: int
    seed: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _require_mapping(values: Mapping[str, Any], key: str) -> dict[str, Any]:
    selected = values.get(key)
    if not isinstance(selected, dict):
        raise ValueError(f"Evaluation config field '{key}' must be a mapping.")
    return dict(selected)


def load_evaluation_protocol(
    config_path: str | Path,
    *,
    num_samples: int | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
    num_inference_steps: int | None = None,
    eta: float | None = None,
    save_steps: str | None = None,
) -> EvaluationProtocol:
    raw = load_config(str(Path(config_path).resolve()))
    values = _require_mapping(raw, "protocol")
    protocol = EvaluationProtocol(
        format_version=int(raw.get("format_version", 0)),
        num_samples=int(values["num_samples"] if num_samples is None else num_samples),
        batch_size=int(values["batch_size"] if batch_size is None else batch_size),
        base_seed=int(values["base_seed"] if seed is None else seed),
        num_inference_steps=int(
            values["num_inference_steps"]
            if num_inference_steps is None
            else num_inference_steps
        ),
        eta=float(values["eta"] if eta is None else eta),
        dtype=str(values["dtype"]),
        reconstruction_criterion=str(values["reconstruction_criterion"]),
        save_steps=str(values["save_steps"] if save_steps is None else save_steps),
    )
    if protocol.format_version != 1:
        raise ValueError("Evaluation format_version must be 1.")
    if protocol.num_samples <= 0:
        raise ValueError("Evaluation num_samples must be positive.")
    if protocol.batch_size <= 0:
        raise ValueError("Evaluation batch_size must be positive.")
    if protocol.num_inference_steps <= 0:
        raise ValueError("Evaluation num_inference_steps must be positive.")
    if protocol.eta < 0:
        raise ValueError("Evaluation eta must be non-negative.")
    if protocol.dtype != "float32":
        raise ValueError("Evaluation dtype must remain float32.")
    if protocol.reconstruction_criterion != "closed_shell_and_nonzero_volume":
        raise ValueError(
            "Evaluation reconstruction_criterion must remain "
            "closed_shell_and_nonzero_volume."
        )
    if protocol.save_steps not in SAVE_STEP_POLICIES:
        available = ", ".join(SAVE_STEP_POLICIES)
        raise ValueError(f"Evaluation save_steps must be one of: {available}.")
    return protocol


def load_evaluation_execution(
    config_path: str | Path,
    *,
    reconstruction_mode: str | None = None,
    reconstruction_workers: int | None = None,
) -> EvaluationExecution:
    raw = load_config(str(Path(config_path).resolve()))
    values = _require_mapping(raw, "execution")
    mode = str(
        values["reconstruction_mode"]
        if reconstruction_mode is None
        else reconstruction_mode
    )
    workers = int(
        values["reconstruction_workers"]
        if reconstruction_workers is None
        else reconstruction_workers
    )
    execution = EvaluationExecution(
        reconstruction_mode=mode,
        reconstruction_workers=workers,
    )
    if execution.reconstruction_mode not in RECONSTRUCTION_MODES:
        available = ", ".join(RECONSTRUCTION_MODES)
        raise ValueError(
            f"Evaluation reconstruction_mode must be one of: {available}."
        )
    if execution.reconstruction_workers <= 0:
        raise ValueError("Evaluation reconstruction_workers must be positive.")
    if (
        execution.reconstruction_mode == "legacy"
        and execution.reconstruction_workers != 1
    ):
        raise ValueError("Evaluation legacy reconstruction requires exactly one worker.")
    return execution


def build_batch_plan(protocol: EvaluationProtocol) -> list[BatchSpec]:
    batches: list[BatchSpec] = []
    for index, sample_start in enumerate(
        range(0, protocol.num_samples, protocol.batch_size)
    ):
        batches.append(
            BatchSpec(
                index=index,
                sample_start=sample_start,
                sample_count=min(
                    protocol.batch_size,
                    protocol.num_samples - sample_start,
                ),
                seed=protocol.base_seed + index,
            )
        )
    return batches


def compute_protocol_fingerprint(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _histogram(values: Sequence[int]) -> dict[str, int]:
    counts = Counter(values)
    return {str(key): counts[key] for key in sorted(counts)}


def aggregate_samples(
    samples: Sequence[Mapping[str, object]],
    *,
    requested_count: int,
    save_step_policy: str | None = None,
) -> dict[str, object]:
    if requested_count <= 0:
        raise ValueError("requested_count must be positive.")
    if save_step_policy is not None and save_step_policy not in SAVE_STEP_POLICIES:
        available = ", ".join(SAVE_STEP_POLICIES)
        raise ValueError(f"save_step_policy must be one of: {available}.")
    indices = [int(sample["index"]) for sample in samples]
    if sorted(indices) != list(range(requested_count)):
        raise ValueError(
            "Sample indices must contain each requested index exactly once."
        )

    failure_counts = {reason: 0 for reason in FAILURE_REASONS}
    face_counts: list[int] = []
    edge_counts: list[int] = []
    in_face_range_count = 0
    closed_solid_count = 0
    closed_solid_in_range_count = 0
    qualified_count = 0
    compilability_values: list[bool | None] = []
    for sample in samples:
        face_counts.append(int(sample["num_faces"]))
        edge_counts.append(int(sample["num_edges"]))
        in_face_range = bool(sample["in_face_range"])
        closed_solid = bool(sample["closed_solid"])
        failure_reason = sample.get("failure_reason")
        compilable = sample.get("compilable")
        if compilable is not None and not isinstance(compilable, bool):
            raise ValueError("Sample compilable must be true, false, or null.")
        compilability_values.append(compilable)
        if closed_solid and failure_reason is not None:
            raise ValueError("A closed-solid sample cannot have a failure_reason.")
        if not closed_solid:
            if failure_reason not in FAILURE_REASONS:
                raise ValueError(
                    "A non-closed sample must have a supported failure_reason."
                )
            failure_counts[str(failure_reason)] += 1
        in_face_range_count += int(in_face_range)
        closed_solid_count += int(closed_solid)
        closed_solid_in_range_count += int(closed_solid and in_face_range)
        qualified_count += int(closed_solid and in_face_range)

    evaluated_count = len(samples)
    if closed_solid_count + sum(failure_counts.values()) != evaluated_count:
        raise ValueError("Closed-solid and failure counts do not cover every sample.")
    if save_step_policy == "all":
        if not all(isinstance(value, bool) for value in compilability_values):
            raise ValueError(
                "save_steps=all requires a boolean compilable value for every sample."
            )
        compilability_measured = True
    elif save_step_policy in {"none", "valid"}:
        if any(value is not None for value in compilability_values):
            raise ValueError(
                f"save_steps={save_step_policy} must not report Compilability."
            )
        compilability_measured = False
    else:
        compilability_measured = all(
            isinstance(value, bool) for value in compilability_values
        )
    compilable_count = None
    compilability_failure_count = None
    compilability_rate = None
    if compilability_measured:
        compilable_count = sum(bool(value) for value in compilability_values)
        compilability_failure_count = evaluated_count - compilable_count
        compilability_rate = _rate(compilable_count, evaluated_count)
    out_of_face_range_count = evaluated_count - in_face_range_count
    return {
        "requested_count": requested_count,
        "evaluated_count": evaluated_count,
        "in_face_range_count": in_face_range_count,
        "in_face_range_rate": _rate(in_face_range_count, evaluated_count),
        "out_of_face_range_count": out_of_face_range_count,
        "out_of_face_range_rate": _rate(out_of_face_range_count, evaluated_count),
        "closed_solid_count": closed_solid_count,
        "closed_solid_rate_all": _rate(closed_solid_count, evaluated_count),
        "closed_solid_in_range_count": closed_solid_in_range_count,
        "closed_solid_rate_in_range": _rate(
            closed_solid_in_range_count,
            in_face_range_count,
        ),
        "qualified_count": qualified_count,
        "qualified_rate": _rate(qualified_count, evaluated_count),
        "compilability_measured": compilability_measured,
        "compilable_count": compilable_count,
        "compilability_failure_count": compilability_failure_count,
        "compilability_rate": compilability_rate,
        "failure_counts": failure_counts,
        "face_count_histogram": _histogram(face_counts),
        "edge_count_histogram": _histogram(edge_counts),
    }


def atomic_write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def prepare_run_directory(
    output_dir: str | Path,
    run_config: Mapping[str, object],
    *,
    resume: bool,
) -> Path:
    output_dir = Path(output_dir).expanduser().resolve()
    config_path = output_dir / "run_config.json"
    if resume:
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Cannot resume without an existing run_config.json in {output_dir}."
            )
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("protocol_fingerprint") != run_config.get(
            "protocol_fingerprint"
        ):
            raise ValueError(
                "Existing run protocol fingerprint does not match this invocation."
            )
        return output_dir

    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(f"Output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Use --resume only for the same protocol."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "batches").mkdir()
    atomic_write_json(config_path, run_config)
    return output_dir


def validate_batch_payload(
    payload: Mapping[str, object],
    batch: BatchSpec,
) -> list[Mapping[str, object]]:
    if payload.get("batch") != batch.to_dict():
        raise ValueError(
            f"Completed batch {batch.index} does not match its batch specification."
        )
    samples = payload.get("samples")
    if not isinstance(samples, list) or not all(
        isinstance(sample, dict) for sample in samples
    ):
        raise ValueError(f"Completed batch {batch.index} has invalid samples.")
    expected_indices = list(
        range(batch.sample_start, batch.sample_start + batch.sample_count)
    )
    actual_indices = [int(sample["index"]) for sample in samples]
    if actual_indices != expected_indices:
        raise ValueError(
            f"Completed batch {batch.index} has unexpected sample indices."
        )
    return samples
