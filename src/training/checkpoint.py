from __future__ import annotations

import os
import pickle
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from src.training.runtime import restore_rng_state, validate_rng_state


FORMAT_VERSION = 1
TRAINING_CHECKPOINT_FORMAT_VERSION = 2
TRAINING_CHECKPOINT_FIELDS = frozenset(
    {
        "format_version",
        "step",
        "epoch",
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "ema",
        "metadata",
        "rng_state",
    }
)
PORTABLE_VAE_KIND = "hifi-brep-portable-full-vae"
PORTABLE_VAE_VERSION = 1
PORTABLE_VAE_STATES = ("online", "ema")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
PORTABLE_VAE_METADATA_FIELDS = frozenset(
    {"kind", "version", "dataset", "variant", "state", "tensor_summary"}
)


@dataclass(frozen=True)
class TrainingArtifactSpec:
    task: str
    dataset: str
    variant: str
    state: str
    face_range: tuple[int, int]
    vae_path: Path | None = None
    decoder_path: Path | None = None
    diffusion_path: Path | None = None
    diffusion_weight_state: str | None = None

    def __post_init__(self) -> None:
        if self.task not in ("vae", "diffusion"):
            raise ValueError("Training artifact task must be 'vae' or 'diffusion'.")
        if not self.dataset or not self.variant:
            raise ValueError("Training artifact dataset and variant must be non-empty.")
        if self.state not in PORTABLE_VAE_STATES:
            raise ValueError("Training artifact state must be 'online' or 'ema'.")
        if (
            len(self.face_range) != 2
            or any(type(bound) is not int for bound in self.face_range)
            or not 0 < self.face_range[0] <= self.face_range[1]
        ):
            raise ValueError("Training artifact face range is invalid.")
        if self.task == "vae":
            if self.vae_path is None:
                raise ValueError("VAE artifact path is required.")
            if any(
                value is not None
                for value in (
                    self.decoder_path,
                    self.diffusion_path,
                    self.diffusion_weight_state,
                )
            ):
                raise ValueError("VAE artifact spec cannot contain diffusion outputs.")
            return
        expected_weight_state = f"{self.state}/model"
        if self.decoder_path is None or self.diffusion_path is None:
            raise ValueError("Diffusion artifact paths are required.")
        if self.vae_path is not None:
            raise ValueError("Diffusion artifact spec cannot contain a full-VAE output.")
        if self.diffusion_weight_state != expected_weight_state:
            raise ValueError(
                "Diffusion artifact weight state must match the selected training state."
            )
        if self.decoder_path == self.diffusion_path:
            raise ValueError("Decoder and diffusion artifact paths must be different.")


def _strict_tensor_state(
    value: object,
    *,
    description: str,
) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise TypeError(f"{description} must be a non-empty state dictionary.")
    if not all(isinstance(key, str) and torch.is_tensor(tensor) for key, tensor in value.items()):
        raise TypeError(f"{description} must contain only string tensor entries.")
    return dict(value)


def summarize_tensor_state(value: object) -> dict[str, object]:
    """Return the exact tensor count, storage bytes, and dtype counts."""
    state = _strict_tensor_state(value, description="Model state")
    dtype_counts: dict[str, int] = {}
    tensor_bytes = 0
    for tensor in state.values():
        dtype = str(tensor.dtype)
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        tensor_bytes += tensor.numel() * tensor.element_size()
    return {
        "tensor_count": len(state),
        "tensor_bytes": tensor_bytes,
        "dtype_counts": {
            dtype: dtype_counts[dtype]
            for dtype in sorted(dtype_counts)
        },
    }


def _validate_sha256(value: object, *, description: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{description} must be a lowercase SHA256 digest.")
    return value


def _validate_portable_source(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("Portable VAE source must be a mapping.")
    required = {"filename", "bytes", "sha256", "step"}
    if set(value) != required:
        raise ValueError("Portable VAE source schema is invalid.")
    filename = value.get("filename")
    if (
        not isinstance(filename, str)
        or not filename
        or "/" in filename
        or "\\" in filename
        or "\x00" in filename
    ):
        raise ValueError("Portable VAE source filename must be a basename.")
    source_bytes = value.get("bytes")
    if isinstance(source_bytes, bool) or not isinstance(source_bytes, int) or source_bytes <= 0:
        raise ValueError("Portable VAE source bytes must be positive.")
    step = value.get("step")
    if step is not None and (
        isinstance(step, bool) or not isinstance(step, int) or step < 0
    ):
        raise ValueError("Portable VAE source step must be a non-negative integer or null.")
    return {
        "filename": filename,
        "bytes": source_bytes,
        "sha256": _validate_sha256(
            value.get("sha256"),
            description="Portable VAE source SHA256",
        ),
        "step": step,
    }


def _validate_tensor_summary(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "tensor_count",
        "tensor_bytes",
        "dtype_counts",
    }:
        raise ValueError("Portable VAE tensor summary schema is invalid.")
    tensor_count = value.get("tensor_count")
    tensor_bytes = value.get("tensor_bytes")
    dtype_counts = value.get("dtype_counts")
    if (
        isinstance(tensor_count, bool)
        or not isinstance(tensor_count, int)
        or tensor_count <= 0
        or isinstance(tensor_bytes, bool)
        or not isinstance(tensor_bytes, int)
        or tensor_bytes <= 0
        or not isinstance(dtype_counts, Mapping)
        or not dtype_counts
        or not all(
            isinstance(dtype, str)
            and dtype
            and not isinstance(count, bool)
            and isinstance(count, int)
            and count > 0
            for dtype, count in dtype_counts.items()
        )
        or sum(dtype_counts.values()) != tensor_count
    ):
        raise ValueError("Portable VAE tensor summary values are invalid.")
    return {
        "tensor_count": tensor_count,
        "tensor_bytes": tensor_bytes,
        "dtype_counts": {
            str(dtype): int(dtype_counts[dtype])
            for dtype in sorted(dtype_counts)
        },
    }


def _validate_portable_metadata(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) not in (
        PORTABLE_VAE_METADATA_FIELDS,
        PORTABLE_VAE_METADATA_FIELDS | {"source"},
    ):
        raise ValueError("Portable VAE metadata schema is invalid.")
    if value.get("kind") != PORTABLE_VAE_KIND:
        raise ValueError("Portable VAE kind is invalid.")
    version = value.get("version")
    if type(version) is not int or version != PORTABLE_VAE_VERSION:
        raise ValueError("Portable VAE version is invalid.")
    dataset = value.get("dataset")
    variant = value.get("variant")
    state = value.get("state")
    if not isinstance(dataset, str) or not dataset:
        raise ValueError("Portable VAE dataset must be a non-empty string.")
    if not isinstance(variant, str) or not variant:
        raise ValueError("Portable VAE variant must be a non-empty string.")
    if state not in PORTABLE_VAE_STATES:
        raise ValueError("Portable VAE state must be 'online' or 'ema'.")
    metadata = {
        "kind": PORTABLE_VAE_KIND,
        "version": PORTABLE_VAE_VERSION,
        "dataset": dataset,
        "variant": variant,
        "state": state,
        "tensor_summary": _validate_tensor_summary(value.get("tensor_summary")),
    }
    if "source" in value:
        metadata["source"] = _validate_portable_source(value.get("source"))
    return metadata


def build_portable_vae_payload(
    model_state: object,
    *,
    dataset: str,
    variant: str,
    state: str,
    source: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the model-only, safe-loadable full-VAE artifact schema."""
    selected = _strict_tensor_state(model_state, description="Portable VAE model")
    metadata_values: dict[str, object] = {
        "kind": PORTABLE_VAE_KIND,
        "version": PORTABLE_VAE_VERSION,
        "dataset": dataset,
        "variant": variant,
        "state": state,
        "tensor_summary": summarize_tensor_state(selected),
    }
    if source is not None:
        metadata_values["source"] = dict(source)
    metadata = _validate_portable_metadata(metadata_values)
    return {
        "metadata": metadata,
        "model": selected,
    }


def validate_portable_vae_payload(
    payload: object,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    """Validate the complete portable VAE schema and tensor summary."""
    if not isinstance(payload, Mapping) or set(payload) != {"metadata", "model"}:
        raise ValueError("Portable VAE root schema is invalid.")
    metadata = _validate_portable_metadata(payload.get("metadata"))
    state = _strict_tensor_state(
        payload.get("model"),
        description="Portable VAE model",
    )
    if metadata["tensor_summary"] != summarize_tensor_state(state):
        raise ValueError("Portable VAE tensor summary does not match model tensors.")
    return metadata, state


def load_portable_vae_weights(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    expected_dataset: str,
    expected_variant: str,
    expected_state: str | None = None,
) -> dict[str, object]:
    """Safe-load, validate, and strictly load a self-describing full VAE."""
    if not isinstance(expected_dataset, str) or not expected_dataset:
        raise ValueError("Expected portable VAE dataset must be a non-empty string.")
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Portable VAE checkpoint does not exist: {path}")
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    metadata, state = validate_portable_vae_payload(payload)
    expected = {"variant": expected_variant}
    if expected_state is not None:
        expected["state"] = expected_state
    mismatches = {
        field: (metadata[field], value)
        for field, value in expected.items()
        if metadata[field] != value
    }
    if str(metadata["dataset"]).casefold() != expected_dataset.casefold():
        mismatches["dataset"] = (metadata["dataset"], expected_dataset)
    if mismatches:
        fields = ", ".join(sorted(mismatches))
        raise ValueError(f"Portable VAE metadata mismatch: {fields}.")
    model.load_state_dict(state, strict=True)
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "payload_key": "model",
        "release_metadata": {
            key: metadata[key]
            for key in (
                "kind",
                "version",
                "dataset",
                "variant",
                "state",
                "tensor_summary",
            )
        },
        "sha256": None,
        "sha256_provenance": "not_computed",
        "runtime_sha256_verified": False,
    }


def _tensor_state(value: object, *, description: str) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise TypeError(f"{description} must be a non-empty state dictionary.")
    state = {
        str(key): tensor
        for key, tensor in value.items()
        if isinstance(key, str) and torch.is_tensor(tensor)
    }
    if not state:
        raise TypeError(f"{description} contains no tensors.")
    return state


def extract_legacy_model_state(
    payload: object,
    state: str,
) -> tuple[dict[str, torch.Tensor], str]:
    """Select online or EMA model tensors from historical training packages."""
    if state not in ("online", "ema"):
        raise ValueError("Legacy checkpoint state must be 'online' or 'ema'.")
    if not isinstance(payload, Mapping):
        raise TypeError("Legacy checkpoint root must be a mapping.")
    if state == "online":
        if "model" in payload:
            return _tensor_state(payload["model"], description="Legacy online model"), "model"
        return _tensor_state(payload, description="Legacy bare model"), "root"

    ema = payload.get("ema")
    if not isinstance(ema, Mapping):
        raise KeyError("Legacy checkpoint has no EMA state.")
    if isinstance(ema.get("model"), Mapping):
        return _tensor_state(ema["model"], description="EMA shadow model"), "ema:model"
    for prefix in ("ema_model.", "module.ema_model."):
        selected = {
            key.removeprefix(prefix): value
            for key, value in ema.items()
            if isinstance(key, str) and key.startswith(prefix) and torch.is_tensor(value)
        }
        if selected:
            return selected, f"ema:{prefix}*"
    raise KeyError("Legacy EMA state has no 'ema_model.*' tensors.")


def load_legacy_weights(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    state: str,
) -> str:
    path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    selected, source = extract_legacy_model_state(payload, state)
    model.load_state_dict(selected, strict=True)
    return source


def _atomic_torch_save_many(payload: object, paths: list[Path]) -> None:
    if not paths:
        raise ValueError("At least one checkpoint target is required.")
    resolved = [path.expanduser().resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Checkpoint targets must be unique.")
    parent = resolved[0].parent
    if any(path.parent != parent for path in resolved):
        raise ValueError("Checkpoint targets must share one output directory.")
    parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".checkpoint-payload.",
        suffix=".tmp",
        dir=parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    staged_targets: list[Path] = []
    try:
        torch.save(payload, temporary_path)
        for path in resolved:
            target_descriptor, target_name = tempfile.mkstemp(
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=parent,
            )
            os.close(target_descriptor)
            staged = Path(target_name)
            staged.unlink()
            try:
                os.link(temporary_path, staged)
            except OSError:
                shutil.copyfile(temporary_path, staged)
            staged_targets.append(staged)
        for staged, path in zip(staged_targets, resolved, strict=True):
            os.replace(staged, path)
    finally:
        for staged in staged_targets:
            if staged.exists():
                staged.unlink()
        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_torch_save_payloads(payloads: Mapping[Path, object]) -> None:
    if not payloads:
        raise ValueError("At least one artifact target is required.")
    resolved = [(path.expanduser().resolve(), payload) for path, payload in payloads.items()]
    paths = [path for path, _ in resolved]
    if len(set(paths)) != len(paths):
        raise ValueError("Artifact targets must be unique.")
    staged: list[tuple[Path, Path]] = []
    backups: dict[Path, Path | None] = {}
    installed: list[Path] = []
    try:
        for path, payload in resolved:
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
            )
            os.close(descriptor)
            temporary_path = Path(temporary_name)
            staged.append((temporary_path, path))
            torch.save(payload, temporary_path)

        for _, path in staged:
            if not path.exists():
                backups[path] = None
                continue
            if not path.is_file():
                raise ValueError(f"Artifact target is not a file: {path}")
            descriptor, backup_name = tempfile.mkstemp(
                prefix=f".{path.name}.",
                suffix=".bak",
                dir=path.parent,
            )
            os.close(descriptor)
            backup_path = Path(backup_name)
            backup_path.unlink()
            try:
                os.link(path, backup_path)
            except OSError:
                shutil.copyfile(path, backup_path)
            backups[path] = backup_path

        for temporary_path, path in staged:
            os.replace(temporary_path, path)
            installed.append(path)
    except Exception as error:
        rollback_errors: list[str] = []
        for path in reversed(installed):
            backup_path = backups[path]
            try:
                if backup_path is None:
                    path.unlink(missing_ok=True)
                else:
                    os.replace(backup_path, path)
            except OSError as rollback_error:
                rollback_errors.append(f"{path}: {rollback_error}")
        if rollback_errors:
            raise RuntimeError(
                "Artifact installation failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from error
        raise
    finally:
        for temporary_path, _ in staged:
            if temporary_path.exists():
                temporary_path.unlink()
        for backup_path in backups.values():
            if backup_path is not None and backup_path.exists():
                backup_path.unlink()


def _model_artifact_identity(
    path: Path,
    *,
    payload_key: str,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "payload_key": payload_key,
        "release_metadata": dict(metadata),
        "sha256": None,
        "sha256_provenance": "not_computed",
        "runtime_sha256_verified": False,
    }


def save_training_artifacts(
    spec: TrainingArtifactSpec,
    *,
    model_state: object,
    vae_state: object | None = None,
) -> dict[str, dict[str, object]]:
    """Atomically save the model-only artifacts selected by validation."""
    selected_model = _strict_tensor_state(
        model_state,
        description="Selected training model",
    )
    if spec.task == "vae":
        if vae_state is not None:
            raise ValueError("VAE artifact export does not accept a separate VAE state.")
        assert spec.vae_path is not None
        payload = build_portable_vae_payload(
            selected_model,
            dataset=spec.dataset,
            variant=spec.variant,
            state=spec.state,
        )
        _atomic_torch_save_payloads({spec.vae_path: payload})
        metadata, _ = validate_portable_vae_payload(payload)
        identity = _model_artifact_identity(
            spec.vae_path.expanduser().resolve(),
            payload_key="model",
            metadata={key: metadata[key] for key in PORTABLE_VAE_METADATA_FIELDS},
        )
        return {"vae": identity}

    if vae_state is None:
        raise ValueError("Diffusion artifact export requires the selected full-VAE state.")
    selected_vae = _strict_tensor_state(
        vae_state,
        description="Selected full-VAE model",
    )
    decoder_state = {
        key: tensor
        for key, tensor in selected_vae.items()
        if key.startswith("decoder.")
    }
    if not decoder_state:
        raise ValueError("Selected full-VAE model contains no decoder.* tensors.")
    face_range = list(spec.face_range)
    decoder_metadata: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "variant": spec.variant,
        "face_range": face_range,
        "component": "vae_decoder",
        "weight_state": "decoder.*",
    }
    diffusion_metadata: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "variant": spec.variant,
        "face_range": face_range,
        "component": "diffusion",
        "weight_state": spec.diffusion_weight_state,
    }
    decoder_payload: dict[str, object] = dict(decoder_state)
    decoder_payload["_release_metadata"] = decoder_metadata
    diffusion_payload: dict[str, object] = {
        "model": selected_model,
        "_release_metadata": diffusion_metadata,
    }
    assert spec.decoder_path is not None
    assert spec.diffusion_path is not None
    _atomic_torch_save_payloads(
        {
            spec.decoder_path: decoder_payload,
            spec.diffusion_path: diffusion_payload,
        }
    )
    return {
        "decoder": _model_artifact_identity(
            spec.decoder_path.expanduser().resolve(),
            payload_key="root:decoder.*",
            metadata=decoder_metadata,
        ),
        "diffusion": _model_artifact_identity(
            spec.diffusion_path.expanduser().resolve(),
            payload_key="model",
            metadata=diffusion_metadata,
        ),
    }


def _validate_checkpoint_rng_state(rng_state: Mapping[str, Any]) -> None:
    if "per_rank" not in rng_state:
        validate_rng_state(rng_state)
        return
    if set(rng_state) != {"per_rank"}:
        raise ValueError("Training checkpoint per-rank RNG schema is invalid.")
    per_rank = rng_state.get("per_rank")
    if not isinstance(per_rank, list) or not per_rank:
        raise ValueError("Training checkpoint per-rank RNG state must be non-empty.")
    for rank, state in enumerate(per_rank):
        if not isinstance(state, Mapping):
            raise ValueError(
                f"Training checkpoint RNG state for rank {rank} must be a mapping."
            )
        validate_rng_state(state)


class TrainingCheckpoint:
    @staticmethod
    def _payload(
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: object | None,
        scaler: object | None,
        ema_state: Mapping[str, Any] | None,
        step: int,
        epoch: int,
        metadata: Mapping[str, Any],
        rng_state: Mapping[str, Any],
    ) -> dict[str, object]:
        _validate_checkpoint_rng_state(rng_state)
        return {
            "format_version": TRAINING_CHECKPOINT_FORMAT_VERSION,
            "step": int(step),
            "epoch": int(epoch),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "ema": dict(ema_state) if ema_state is not None else None,
            "metadata": dict(metadata),
            "rng_state": dict(rng_state),
        }

    @staticmethod
    def save(
        path: str | Path,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: object | None,
        scaler: object | None,
        ema_state: Mapping[str, Any] | None,
        step: int,
        epoch: int,
        metadata: Mapping[str, Any],
        rng_state: Mapping[str, Any],
    ) -> None:
        TrainingCheckpoint.save_many(
            [path],
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema_state=ema_state,
            step=step,
            epoch=epoch,
            metadata=metadata,
            rng_state=rng_state,
        )

    @staticmethod
    def save_many(
        paths: list[str | Path],
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: object | None,
        scaler: object | None,
        ema_state: Mapping[str, Any] | None,
        step: int,
        epoch: int,
        metadata: Mapping[str, Any],
        rng_state: Mapping[str, Any],
    ) -> None:
        payload = TrainingCheckpoint._payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema_state=ema_state,
            step=step,
            epoch=epoch,
            metadata=metadata,
            rng_state=rng_state,
        )
        _atomic_torch_save_many(payload, [Path(path) for path in paths])

    @staticmethod
    def load_for_resume(
        path: str | Path,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: object | None,
        scaler: object | None,
        expected_metadata: Mapping[str, Any],
        rng_rank: int = 0,
        rng_generators: Mapping[str, torch.Generator] | None = None,
    ) -> dict[str, Any]:
        checkpoint_path = Path(path).expanduser().resolve()
        try:
            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        except pickle.UnpicklingError as error:
            raise ValueError(
                "Unsupported training checkpoint format; expected safe "
                f"format_version {TRAINING_CHECKPOINT_FORMAT_VERSION}. Legacy "
                "training resume checkpoints are not supported."
            ) from error
        if not isinstance(payload, Mapping):
            raise TypeError("Training checkpoint root must be a mapping.")
        format_version = payload.get("format_version")
        if format_version != TRAINING_CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                "Unsupported training checkpoint format_version "
                f"{format_version!r}; expected {TRAINING_CHECKPOINT_FORMAT_VERSION}. "
                "Legacy training resume checkpoints are not supported."
            )
        missing = sorted(TRAINING_CHECKPOINT_FIELDS - set(payload))
        unexpected = sorted(set(payload) - TRAINING_CHECKPOINT_FIELDS)
        if missing or unexpected:
            raise ValueError(
                "Training checkpoint root schema is invalid: "
                f"missing={missing}, unexpected={unexpected}."
            )
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("Training checkpoint metadata must be a mapping.")
        mismatches = {
            key: (metadata.get(key), expected)
            for key, expected in expected_metadata.items()
            if metadata.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"Resume metadata mismatch: {mismatches!r}.")
        if metadata.get("resume_boundary") is not True:
            raise ValueError("Training checkpoint was not saved at a resumable epoch boundary.")
        for name in ("step", "epoch"):
            value = payload.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"Training checkpoint {name} must be a non-negative integer."
                )
        model_state = _strict_tensor_state(
            payload.get("model"),
            description="Training checkpoint model",
        )
        optimizer_state = payload.get("optimizer")
        if (
            not isinstance(optimizer_state, Mapping)
            or set(optimizer_state) != {"state", "param_groups"}
            or not isinstance(optimizer_state.get("state"), Mapping)
            or not isinstance(optimizer_state.get("param_groups"), list)
        ):
            raise ValueError("Training checkpoint optimizer state is invalid.")
        scheduler_state = payload.get("scheduler")
        if scheduler is None:
            if scheduler_state is not None:
                raise ValueError(
                    "Training checkpoint has scheduler state but this run has no scheduler."
                )
        elif not isinstance(scheduler_state, Mapping):
            raise ValueError("Training checkpoint has no valid scheduler state.")
        scaler_state = payload.get("scaler")
        if scaler is None:
            if scaler_state is not None:
                raise ValueError(
                    "Training checkpoint has scaler state but this run has no scaler."
                )
        elif not isinstance(scaler_state, Mapping):
            raise ValueError("Training checkpoint has no valid scaler state.")
        ema_state = payload.get("ema")
        if ema_state is not None and not isinstance(ema_state, Mapping):
            raise ValueError("Training checkpoint EMA state is invalid.")
        rng_state = payload.get("rng_state")
        if not isinstance(rng_state, Mapping):
            raise ValueError("Training checkpoint RNG state must be a mapping.")
        if "per_rank" in rng_state:
            if set(rng_state) != {"per_rank"}:
                raise ValueError("Training checkpoint per-rank RNG schema is invalid.")
            per_rank = rng_state["per_rank"]
            if not isinstance(per_rank, list) or not 0 <= rng_rank < len(per_rank):
                raise ValueError("Training checkpoint has no RNG state for this rank.")
            rng_state = per_rank[rng_rank]
        if not isinstance(rng_state, Mapping):
            raise ValueError("Training checkpoint RNG rank state must be a mapping.")
        validate_rng_state(rng_state, generators=rng_generators)

        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(dict(optimizer_state))
        if scheduler is not None:
            assert isinstance(scheduler_state, Mapping)
            scheduler.load_state_dict(dict(scheduler_state))
        if scaler is not None:
            assert isinstance(scaler_state, Mapping)
            scaler.load_state_dict(dict(scaler_state))
        restore_rng_state(rng_state, generators=rng_generators)
        return dict(payload)
