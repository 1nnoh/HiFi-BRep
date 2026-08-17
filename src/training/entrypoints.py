from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from src.data.brep_dataset import PortableBrepDataset
from src.data.manifest import load_dataset_manifest
from src.data.validation import validate_dataset_provenance, validate_split_access
from src.inference.config import load_demo_config
from src.training.checkpoint import (
    TrainingArtifactSpec,
    load_legacy_weights,
    load_portable_vae_weights,
)
from src.training.config import load_training_recipe
from src.training.engine import apply_cli_overrides, run_training_loop
from src.training.runtime import seed_everything
from src.training.tasks import DiffusionTask, VaeTask


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DECODER_CONFIG_FIELDS = {
    "max_face": "max_face",
    "latent_dim": "latent_dim",
    "d_model": "dec_d_model",
    "nhead": "dec_nhead",
    "dropout": "dec_dropout",
    "mem_self_depth": "mem_self_depth",
    "face_cross_depth": "face_cross_depth",
    "edge_cross_depth": "edge_cross_depth",
    "latent_len": "latent_len",
    "use_mem_pos": "use_mem_pos",
    "use_mem_sa": "use_mem_sa",
    "use_softplus": "use_softplus",
    "use_sin": "use_sin",
}
DIFFUSION_SAMPLING_FIELDS = (
    "num_train_timesteps",
    "beta_start",
    "beta_end",
    "beta_schedule",
    "prediction_type",
    "variance_type",
)


def seed_model_initialization(values: dict[str, Any]) -> int:
    """Make model parameters depend on the recipe seed, not ambient RNG state."""
    training = values.get("training")
    if not isinstance(training, dict):
        raise ValueError("Training configuration must contain a training mapping.")
    seed = int(training.get("seed", 42))
    seed_everything(seed)
    return seed


def _resolve_cli_checkpoint(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    root = os.environ.get("HIFI_BREP_CHECKPOINT_ROOT")
    return ((Path(root) if root else REPOSITORY_ROOT) / expanded).resolve()


def configured_vae_source(values: Mapping[str, Any]) -> dict[str, str]:
    """Return the exact self-describing VAE contract from a diffusion recipe."""
    configured = values.get("vae_source")
    if not isinstance(configured, Mapping) or set(configured) != {
        "variant",
        "state",
    }:
        raise ValueError(
            "Diffusion recipe vae_source must contain exactly variant and state."
        )
    variant = configured.get("variant")
    state = configured.get("state")
    if not isinstance(variant, str) or not variant:
        raise ValueError("Diffusion recipe VAE variant must be a non-empty string.")
    if state not in ("online", "ema"):
        raise ValueError("Diffusion recipe VAE state must be 'online' or 'ema'.")
    return {"variant": variant, "state": state}


def _decoder_config(values: Mapping[str, Any]) -> dict[str, object]:
    missing = sorted(
        source for source in DECODER_CONFIG_FIELDS.values() if source not in values
    )
    if missing:
        raise ValueError(
            "Training VAE model cannot construct the inference decoder; missing: "
            + ", ".join(missing)
            + "."
        )
    return {
        target: values[source]
        for target, source in DECODER_CONFIG_FIELDS.items()
    }


def build_training_artifact_spec(
    values: Mapping[str, Any],
    *,
    dataset: str,
    output_dir: Path,
    repository_root: Path = REPOSITORY_ROOT,
) -> TrainingArtifactSpec:
    task = values.get("task")
    if task not in ("vae", "diffusion"):
        raise ValueError("Training artifact task must be 'vae' or 'diffusion'.")
    variant = values.get("variant")
    if not isinstance(variant, str) or not variant:
        raise ValueError("Training recipe variant must be a non-empty string.")
    validation = values.get("validation")
    if not isinstance(validation, Mapping):
        raise ValueError("Training recipe validation must be a mapping.")
    state = validation.get("state")
    if state not in ("online", "ema"):
        raise ValueError("Training validation state must be 'online' or 'ema'.")

    repository_root = repository_root.expanduser().resolve()
    demo = load_demo_config(
        repository_root / "configs" / "demo.yaml",
        variant,
    )
    expected_state = demo.diffusion_state.removesuffix("/model")
    if state != expected_state:
        raise ValueError(
            "Training validation state must match the selected Demo variant state."
        )
    model_values = values.get("model")
    if not isinstance(model_values, Mapping):
        raise ValueError("Training recipe model must be a mapping.")
    vae_values = model_values if task == "vae" else values.get("vae_model")
    if not isinstance(vae_values, Mapping):
        raise ValueError("Diffusion recipe vae_model must be a mapping.")
    if _decoder_config(vae_values) != demo.decoder:
        raise ValueError(
            "Training VAE decoder architecture does not match configs/demo.yaml."
        )
    if task == "diffusion":
        if dict(model_values) != demo.model:
            raise ValueError(
                "Training diffusion architecture does not match configs/demo.yaml."
            )
        sampling = values.get("diffusion")
        if not isinstance(sampling, Mapping):
            raise ValueError("Diffusion recipe diffusion must be a mapping.")
        mismatches = [
            field
            for field in DIFFUSION_SAMPLING_FIELDS
            if sampling.get(field) != demo.sampling.get(field)
        ]
        if mismatches:
            raise ValueError(
                "Training diffusion schedule does not match configs/demo.yaml: "
                + ", ".join(mismatches)
                + "."
            )
        if configured_vae_source(values) != {"variant": variant, "state": state}:
            raise ValueError(
                "Diffusion recipe vae_source must match its variant and validation state."
            )

    checkpoint_root = (repository_root / "checkpoints").resolve()
    try:
        decoder_relative = demo.vae_checkpoint.relative_to(checkpoint_root)
        diffusion_relative = demo.diffusion_checkpoint.relative_to(checkpoint_root)
    except ValueError as error:
        raise ValueError(
            "Demo checkpoint paths must remain under the repository checkpoint root."
        ) from error
    group = decoder_relative.parent.as_posix()
    if group.casefold() != dataset.casefold():
        raise ValueError("Training dataset does not match the selected Demo variant.")

    artifact_root = output_dir.expanduser().resolve() / "artifacts"
    common = {
        "task": task,
        "dataset": dataset,
        "variant": variant,
        "state": state,
        "face_range": (demo.face_min, demo.face_max),
    }
    if task == "vae":
        filename = (
            f"hifi-brep-vae-{demo.face_min}-{demo.face_max}-{state}.pt"
        )
        return TrainingArtifactSpec(
            **common,
            vae_path=artifact_root / "training" / decoder_relative.parent / filename,
        )
    return TrainingArtifactSpec(
        **common,
        decoder_path=artifact_root / decoder_relative,
        diffusion_path=artifact_root / diffusion_relative,
        diffusion_weight_state=demo.diffusion_state,
    )


def _legacy_checkpoint_identity(
    checkpoint: Path,
    *,
    state: str,
    payload_key: str,
) -> dict[str, object]:
    return {
        "filename": checkpoint.name,
        "bytes": checkpoint.stat().st_size,
        "payload_key": payload_key,
        "state": state,
        "sha256": None,
        "sha256_provenance": "not_computed",
        "runtime_sha256_verified": False,
    }


def _datasets(
    *,
    values: dict[str, Any],
    data_root: Path,
    manifest_path: Path,
) -> tuple[PortableBrepDataset, PortableBrepDataset]:
    data = dict(values["data"])
    model = dict(values["model"])
    common = {
        "data_root": data_root,
        "manifest_path": manifest_path,
        "max_face": int(model.get("max_face", 50)),
        "bbox_scale": float(data.get("bbox_scale", 3.0)),
    }
    train_dataset = PortableBrepDataset(
        split=str(data.get("train_split", "train")),
        augment=bool(data.get("augment", False)),
        augment_probability=float(data.get("augment_probability", 0.5)),
        **common,
    )
    val_dataset = PortableBrepDataset(
        split=str(data.get("val_split", "val")),
        augment=False,
        **common,
    )
    return train_dataset, val_dataset


def _prepare(args: object, *, task: str) -> tuple[object, dict[str, Any], object, Path]:
    recipe = load_training_recipe(
        args.config,
        expected_task=task,
        repository_root=REPOSITORY_ROOT,
    )
    effective = apply_cli_overrides(recipe.values, args)
    manifest = load_dataset_manifest(recipe.manifest_path)
    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise NotADirectoryError(f"Processed data root does not exist: {data_root}")
    validate_dataset_provenance(data_root, manifest)
    data_config = dict(effective["data"])
    validate_split_access(
        data_root,
        manifest,
        split_names=(
            str(data_config.get("train_split", "train")),
            str(data_config.get("val_split", "val")),
        ),
    )
    return recipe, effective, manifest, data_root


def run_vae_training(args: object) -> dict[str, Any]:
    from src.vae.latentbrep.latentbrep_vae import LatentBiSPBrepVAE

    recipe, effective, manifest, data_root = _prepare(args, task="vae")
    train_dataset, val_dataset = _datasets(
        values=effective,
        data_root=data_root,
        manifest_path=recipe.manifest_path,
    )
    artifact_spec = build_training_artifact_spec(
        effective,
        dataset=manifest.dataset,
        output_dir=Path(args.output_dir),
    )
    seed_model_initialization(effective)
    model = LatentBiSPBrepVAE(**dict(effective["model"]))
    provenance: dict[str, Any] = {"initialization": "random"}
    if args.init_from is not None:
        checkpoint = _resolve_cli_checkpoint(args.init_from)
        source = load_legacy_weights(model, checkpoint, state=args.init_state)
        provenance = {
            "initialization": "init_from",
            **_legacy_checkpoint_identity(
                checkpoint,
                state=args.init_state,
                payload_key=source,
            ),
        }
    return run_training_loop(
        repository_root=REPOSITORY_ROOT,
        effective_config=effective,
        recipe_sha256=recipe.sha256,
        manifest_sha256=manifest.sha256,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        task=VaeTask(sample_posterior=True),
        output_dir=Path(args.output_dir),
        resume_path=_resolve_cli_checkpoint(args.resume) if args.resume else None,
        max_train_steps=args.max_train_steps,
        checkpoint_provenance=provenance,
        artifact_spec=artifact_spec,
    )


def run_diffusion_training(args: object) -> dict[str, Any]:
    from src.flow.flow import DiT
    from src.vae.latentbrep.latentbrep_vae import LatentBiSPBrepVAE

    recipe, effective, manifest, data_root = _prepare(args, task="diffusion")
    train_dataset, val_dataset = _datasets(
        values=effective,
        data_root=data_root,
        manifest_path=recipe.manifest_path,
    )
    artifact_spec = build_training_artifact_spec(
        effective,
        dataset=manifest.dataset,
        output_dir=Path(args.output_dir),
    )
    vae_checkpoint = _resolve_cli_checkpoint(args.vae_checkpoint)
    vae_source = configured_vae_source(effective)
    vae = LatentBiSPBrepVAE(**dict(effective["vae_model"]))
    vae_identity = load_portable_vae_weights(
        vae,
        vae_checkpoint,
        expected_dataset=manifest.dataset,
        expected_variant=vae_source["variant"],
        expected_state=vae_source["state"],
    )
    seed_model_initialization(effective)
    model = DiT(**dict(effective["model"]))
    provenance: dict[str, Any] = {
        "vae": vae_identity,
        "diffusion_initialization": "random",
    }
    if args.init_from is not None:
        checkpoint = _resolve_cli_checkpoint(args.init_from)
        source = load_legacy_weights(model, checkpoint, state=args.init_state)
        provenance["diffusion_initialization"] = {
            **_legacy_checkpoint_identity(
                checkpoint,
                state=args.init_state,
                payload_key=source,
            ),
        }
    latent_shape = (
        int(effective["model"]["latent_num"]),
        int(effective["model"]["in_channels"]),
    )
    task = DiffusionTask(
        vae=vae,
        latent_shape=latent_shape,
        scheduler_config=dict(effective["diffusion"]),
        sample_posterior=bool(effective["diffusion"].get("sample_posterior", True)),
    )
    return run_training_loop(
        repository_root=REPOSITORY_ROOT,
        effective_config=effective,
        recipe_sha256=recipe.sha256,
        manifest_sha256=manifest.sha256,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        task=task,
        output_dir=Path(args.output_dir),
        resume_path=_resolve_cli_checkpoint(args.resume) if args.resume else None,
        max_train_steps=args.max_train_steps,
        checkpoint_provenance=provenance,
        max_additional_train_steps=args.max_additional_train_steps,
        stop_after_epoch=getattr(args, "stop_after_epoch", None),
        artifact_spec=artifact_spec,
    )
