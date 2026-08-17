from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from diffusers import DDIMScheduler, DDPMScheduler
from torch import nn
from tqdm.auto import tqdm

from src.inference.config import DemoConfig


RELEASE_FORMAT_VERSION = 1
RELEASE_COMPONENT_STATES = {
    "vae_decoder": ("decoder.*", "root:decoder.*"),
    "diffusion": (None, "model"),
}


@dataclass(frozen=True)
class SamplingConfig:
    num_train_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "sample"
    variance_type: str = "fixed_small"
    num_inference_steps: int = 400
    eta: float = 0.2

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        num_inference_steps: int | None = None,
        eta: float | None = None,
    ) -> "SamplingConfig":
        fields = {
            "num_train_timesteps": int(values["num_train_timesteps"]),
            "beta_start": float(values["beta_start"]),
            "beta_end": float(values["beta_end"]),
            "beta_schedule": str(values["beta_schedule"]),
            "prediction_type": str(values["prediction_type"]),
            "variance_type": str(values.get("variance_type", "fixed_small")),
            "num_inference_steps": int(
                values["num_inference_steps"]
                if num_inference_steps is None
                else num_inference_steps
            ),
            "eta": float(values["eta"] if eta is None else eta),
        }
        sampling = cls(**fields)
        if sampling.num_train_timesteps <= 0 or sampling.num_inference_steps <= 0:
            raise ValueError("Diffusion timestep counts must be positive.")
        if sampling.eta < 0:
            raise ValueError("DDIM eta must be non-negative.")
        return sampling


def _strict_tensor_state(
    value: object,
    *,
    description: str,
) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise TypeError(f"{description} must be a non-empty state dictionary.")
    if not all(
        isinstance(key, str) and torch.is_tensor(tensor)
        for key, tensor in value.items()
    ):
        raise TypeError(f"{description} must contain only string tensor entries.")
    return dict(value)


def _release_metadata(
    payload: Mapping[str, object],
    *,
    checkpoint_path: Path,
    expected_variant: str,
    expected_face_range: tuple[int, int],
    expected_component: str,
    expected_weight_state: str,
) -> dict[str, object]:
    metadata = payload.get("_release_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(
            f"Checkpoint '{checkpoint_path.name}' has no release metadata."
        )
    schema_errors = []
    if type(metadata.get("format_version")) is not int:
        schema_errors.append("format_version")
    face_range = metadata.get("face_range")
    if (
        not isinstance(face_range, list)
        or len(face_range) != 2
        or any(type(bound) is not int for bound in face_range)
    ):
        schema_errors.append("face_range")
    for field in ("variant", "component", "weight_state"):
        if not isinstance(metadata.get(field), str) or not metadata[field]:
            schema_errors.append(field)
    if schema_errors:
        fields = ", ".join(sorted(schema_errors))
        raise ValueError(
            f"Checkpoint '{checkpoint_path.name}' release metadata schema is invalid: "
            f"{fields}."
        )
    expected = {
        "format_version": RELEASE_FORMAT_VERSION,
        "variant": expected_variant,
        "face_range": list(expected_face_range),
        "component": expected_component,
        "weight_state": expected_weight_state,
    }
    mismatches = {
        field: (metadata.get(field), value)
        for field, value in expected.items()
        if metadata.get(field) != value
    }
    if mismatches:
        fields = ", ".join(sorted(mismatches))
        raise ValueError(
            f"Checkpoint '{checkpoint_path.name}' release metadata mismatch: {fields}."
        )
    return expected


def _read_release_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_variant: str,
    expected_face_range: tuple[int, int],
    expected_component: str,
    expected_weight_state: str,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if expected_component not in RELEASE_COMPONENT_STATES:
        available = ", ".join(sorted(RELEASE_COMPONENT_STATES))
        raise ValueError(f"Release component must be one of: {available}.")
    required_state, payload_key = RELEASE_COMPONENT_STATES[expected_component]
    if required_state is not None and expected_weight_state != required_state:
        raise ValueError(
            f"Component '{expected_component}' requires weight state '{required_state}'."
        )
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"Checkpoint '{checkpoint_path.name}' root must be a mapping."
        )
    metadata = _release_metadata(
        payload,
        checkpoint_path=checkpoint_path,
        expected_variant=expected_variant,
        expected_face_range=expected_face_range,
        expected_component=expected_component,
        expected_weight_state=expected_weight_state,
    )
    if expected_component == "vae_decoder":
        root_state = {
            key: value
            for key, value in payload.items()
            if key != "_release_metadata"
        }
        prefixed = _strict_tensor_state(
            root_state,
            description=f"VAE decoder checkpoint '{checkpoint_path.name}'",
        )
        if not all(key.startswith("decoder.") for key in prefixed):
            raise ValueError(
                f"VAE decoder checkpoint '{checkpoint_path.name}' contains "
                "tensors outside decoder.*."
            )
        state = {
            key.removeprefix("decoder."): value
            for key, value in prefixed.items()
        }
    else:
        if set(payload) != {"model", "_release_metadata"}:
            raise ValueError(
                f"Diffusion checkpoint '{checkpoint_path.name}' root schema is invalid."
            )
        state = _strict_tensor_state(
            payload.get("model"),
            description=f"Diffusion checkpoint '{checkpoint_path.name}' model",
        )
    identity = {
        "filename": checkpoint_path.name,
        "bytes": checkpoint_path.stat().st_size,
        "payload_key": payload_key,
        "release_metadata": metadata,
        "sha256": None,
        "sha256_provenance": "not_computed",
        "runtime_sha256_verified": False,
    }
    return state, identity


def inspect_release_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_variant: str,
    expected_face_range: tuple[int, int],
    expected_component: str,
    expected_weight_state: str,
) -> dict[str, object]:
    """Safely inspect one self-describing release artifact without hashing it."""
    _, identity = _read_release_checkpoint(
        checkpoint_path,
        expected_variant=expected_variant,
        expected_face_range=expected_face_range,
        expected_component=expected_component,
        expected_weight_state=expected_weight_state,
    )
    return identity


def load_module_checkpoint(
    module: nn.Module,
    checkpoint_path: str | Path,
    *,
    expected_variant: str,
    expected_face_range: tuple[int, int],
    expected_component: str,
    expected_weight_state: str,
) -> dict[str, object]:
    """Validate metadata and tensor schema, then strictly load a release artifact."""
    state, identity = _read_release_checkpoint(
        checkpoint_path,
        expected_variant=expected_variant,
        expected_face_range=expected_face_range,
        expected_component=expected_component,
        expected_weight_state=expected_weight_state,
    )
    module.load_state_dict(state, strict=True)
    return identity


@torch.inference_mode()
def sample_latents(
    model: nn.Module,
    *,
    batch_size: int,
    latent_shape: Sequence[int],
    sampling: SamplingConfig,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    show_progress: bool = True,
) -> torch.Tensor:
    """Run the existing online-model DDIM sampling rule without trainer state."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if len(latent_shape) != 2 or any(int(size) <= 0 for size in latent_shape):
        raise ValueError("latent_shape must contain two positive dimensions.")

    scheduler_config = DDPMScheduler(
        num_train_timesteps=sampling.num_train_timesteps,
        beta_start=sampling.beta_start,
        beta_end=sampling.beta_end,
        beta_schedule=sampling.beta_schedule,
        prediction_type=sampling.prediction_type,
        clip_sample=True,
        clip_sample_range=10.0,
        variance_type=sampling.variance_type,
    ).config
    scheduler = DDIMScheduler.from_config(scheduler_config)
    scheduler.set_timesteps(sampling.num_inference_steps, device=device)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    latent = torch.randn(
        (batch_size, *(int(size) for size in latent_shape)),
        device=device,
        dtype=dtype,
        generator=generator,
    )

    model.eval()
    timesteps = tqdm(
        scheduler.timesteps,
        desc="DDIM sampling",
        disable=not show_progress,
    )
    for timestep in timesteps:
        continuous_timestep = torch.full(
            (batch_size,),
            (timestep.float() + 0.5) / float(sampling.num_train_timesteps),
            device=device,
        )
        prediction = model(latent, continuous_timestep)
        latent = scheduler.step(
            prediction,
            timestep,
            latent,
            eta=sampling.eta,
            generator=generator,
        ).prev_sample
    return latent


class DemoGenerator:
    """Minimal VAE decoder + selected diffusion runtime for generation."""

    def __init__(
        self,
        config: DemoConfig,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        from src.flow.flow import DiT
        from src.vae.latentbrep.latentbrep_decoder import DecoderBiSP

        self.config = config
        self.device = device
        self.dtype = dtype
        self.decoder = DecoderBiSP(**config.decoder)
        self.model = DiT(**config.model)

        face_range = (config.face_min, config.face_max)
        self.vae_checkpoint_identity = load_module_checkpoint(
            self.decoder,
            config.vae_checkpoint,
            expected_variant=config.variant,
            expected_face_range=face_range,
            expected_component="vae_decoder",
            expected_weight_state="decoder.*",
        )
        self.diffusion_checkpoint_identity = load_module_checkpoint(
            self.model,
            config.diffusion_checkpoint,
            expected_variant=config.variant,
            expected_face_range=face_range,
            expected_component="diffusion",
            expected_weight_state=config.diffusion_state,
        )
        self.checkpoint_identities = {
            "vae": self.vae_checkpoint_identity,
            "diffusion": self.diffusion_checkpoint_identity,
        }

        self.decoder.to(device=device, dtype=dtype).eval()
        self.model.to(device=device, dtype=dtype).eval()

    @torch.inference_mode()
    def generate(
        self,
        *,
        num_samples: int,
        seed: int,
        sampling: SamplingConfig,
        show_progress: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        latent = sample_latents(
            self.model,
            batch_size=num_samples,
            latent_shape=self.config.latent_shape,
            sampling=sampling,
            device=self.device,
            dtype=self.dtype,
            seed=seed,
            show_progress=show_progress,
        )
        predictions = self.decoder(z=latent, use_pred=True)
        return latent, predictions
