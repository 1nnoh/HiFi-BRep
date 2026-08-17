from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.flow.config import validate_unconditional_dit_config
from src.utils.config import load_config


DIFFUSION_WEIGHT_STATES = ("online/model", "ema/model")
DEMO_CONFIG_FIELDS = frozenset({"inherit_from", "model", "decoder", "sampling", "variants"})
DECODER_CONFIG_FIELDS = frozenset(
    {
        "max_face",
        "latent_dim",
        "d_model",
        "nhead",
        "dropout",
        "mem_self_depth",
        "edge_cross_depth",
        "face_cross_depth",
        "dec_depth",
        "mlp_hidden",
        "use_mem_sa",
        "use_mem_pos",
        "latent_len",
        "use_softplus",
        "use_sin",
    }
)
SAMPLING_CONFIG_FIELDS = frozenset(
    {
        "num_train_timesteps",
        "beta_start",
        "beta_end",
        "beta_schedule",
        "prediction_type",
        "variance_type",
        "num_inference_steps",
        "eta",
    }
)
VARIANT_CONFIG_FIELDS = frozenset(
    {
        "face_min",
        "face_max",
        "vae_checkpoint",
        "diffusion_checkpoint",
        "diffusion_state",
    }
)


@dataclass(frozen=True)
class DemoConfig:
    variant: str
    face_min: int
    face_max: int
    vae_checkpoint: Path
    diffusion_checkpoint: Path
    diffusion_state: str
    model: dict[str, Any]
    decoder: dict[str, Any]
    sampling: dict[str, Any]

    @property
    def latent_shape(self) -> tuple[int, int]:
        return int(self.model["latent_num"]), int(self.model["in_channels"])


def _require_mapping(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Demo config field '{key}' must be a mapping.")
    return dict(value)


def _reject_unknown_fields(
    values: Mapping[str, Any],
    allowed: frozenset[str],
    *,
    description: str,
) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(
            f"{description} contains unsupported fields: {', '.join(unknown)}."
        )


def _resolve_checkpoint(path: str, repository_root: Path) -> Path:
    checkpoint = Path(path)
    if not checkpoint.is_absolute():
        checkpoint = repository_root / checkpoint
    return checkpoint.resolve()


def _checkpoint_override_path(path: str, checkpoint_dir: Path) -> Path:
    configured = Path(path)
    if configured.is_absolute():
        relative = Path(configured.name)
    elif configured.parts and configured.parts[0] == "checkpoints":
        relative = Path(*configured.parts[1:])
    else:
        relative = configured
    return checkpoint_dir / relative


def load_demo_config(
    config_path: str | Path,
    variant: str,
    *,
    checkpoint_dir: str | Path | None = None,
) -> DemoConfig:
    """Load one named, internally consistent Demo checkpoint pair."""
    config_path = Path(config_path).resolve()
    raw = load_config(str(config_path))
    if not isinstance(raw, dict):
        raise ValueError("Demo config root must be a mapping.")
    _reject_unknown_fields(raw, DEMO_CONFIG_FIELDS, description="Demo config")
    variants = _require_mapping(raw, "variants")
    if variant not in variants:
        available = ", ".join(sorted(variants))
        raise ValueError(f"Unknown variant '{variant}'. Available variants: {available}.")

    selected = variants[variant]
    if not isinstance(selected, dict):
        raise ValueError(f"Variant '{variant}' must be a mapping.")
    _reject_unknown_fields(
        selected,
        VARIANT_CONFIG_FIELDS,
        description=f"Variant '{variant}'",
    )

    repository_root = config_path.parent.parent
    vae_checkpoint_value = str(selected["vae_checkpoint"])
    diffusion_checkpoint_value = str(selected["diffusion_checkpoint"])
    vae_checkpoint = _resolve_checkpoint(vae_checkpoint_value, repository_root)
    diffusion_checkpoint = _resolve_checkpoint(
        diffusion_checkpoint_value, repository_root
    )
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        vae_checkpoint = _checkpoint_override_path(vae_checkpoint_value, checkpoint_dir)
        diffusion_checkpoint = _checkpoint_override_path(
            diffusion_checkpoint_value,
            checkpoint_dir,
        )

    face_min = int(selected["face_min"])
    face_max = int(selected["face_max"])
    if not 0 < face_min <= face_max:
        raise ValueError(
            f"Invalid face range for variant '{variant}': {face_min}-{face_max}."
        )
    diffusion_state = str(selected.get("diffusion_state", "online/model"))
    if diffusion_state not in DIFFUSION_WEIGHT_STATES:
        available = ", ".join(DIFFUSION_WEIGHT_STATES)
        raise ValueError(
            f"Variant '{variant}' diffusion_state must be one of: {available}."
        )

    model = validate_unconditional_dit_config(
        _require_mapping(raw, "model"),
        description="Demo model config",
    )
    decoder = _require_mapping(raw, "decoder")
    _reject_unknown_fields(
        decoder,
        DECODER_CONFIG_FIELDS,
        description="Demo decoder config",
    )
    sampling = _require_mapping(raw, "sampling")
    _reject_unknown_fields(
        sampling,
        SAMPLING_CONFIG_FIELDS,
        description="Demo sampling config",
    )

    return DemoConfig(
        variant=variant,
        face_min=face_min,
        face_max=face_max,
        vae_checkpoint=vae_checkpoint,
        diffusion_checkpoint=diffusion_checkpoint,
        diffusion_state=diffusion_state,
        model=model,
        decoder=decoder,
        sampling=sampling,
    )
