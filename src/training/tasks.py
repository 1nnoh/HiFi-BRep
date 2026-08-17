from __future__ import annotations

from typing import Any, Mapping

import torch
from diffusers import DDPMScheduler
from torch import nn

from src.training.diffusion import freeze_vae, validate_latents


class VaeTask:
    def __init__(self, *, sample_posterior: bool = True) -> None:
        self.sample_posterior = bool(sample_posterior)

    def loss(
        self,
        model: nn.Module,
        batch: Mapping[str, Any],
        *,
        training: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del training
        loss, logs = model(
            **batch,
            sample_posterior=self.sample_posterior,
        )
        return loss, dict(logs)


class DiffusionTask:
    def __init__(
        self,
        *,
        vae: nn.Module,
        latent_shape: tuple[int, int],
        scheduler_config: Mapping[str, Any],
        sample_posterior: bool,
    ) -> None:
        self.vae = vae
        freeze_vae(self.vae)
        self.latent_shape = tuple(int(value) for value in latent_shape)
        self.scheduler = DDPMScheduler(
            num_train_timesteps=int(scheduler_config["num_train_timesteps"]),
            beta_start=float(scheduler_config["beta_start"]),
            beta_end=float(scheduler_config["beta_end"]),
            beta_schedule=str(scheduler_config["beta_schedule"]),
            prediction_type=str(scheduler_config["prediction_type"]),
            clip_sample=True,
            clip_sample_range=10.0,
            variance_type=str(scheduler_config.get("variance_type", "fixed_small")),
        )
        self.sample_posterior = bool(sample_posterior)

    def loss(
        self,
        model: nn.Module,
        batch: Mapping[str, Any],
        *,
        training: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del training
        self.vae.eval()
        with torch.no_grad():
            clean = self.vae(
                **batch,
                sample_posterior=self.sample_posterior,
                encode_only=True,
            )
        validate_latents(clean, latent_shape=self.latent_shape)
        noise = torch.randn_like(clean)
        batch_size = clean.shape[0]
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=clean.device,
            dtype=torch.long,
        )
        noisy = self.scheduler.add_noise(clean, noise, timesteps)
        continuous_timesteps = (
            timesteps.to(dtype=torch.float32) + 0.5
        ) / float(self.scheduler.config.num_train_timesteps)
        prediction = model(noisy, continuous_timesteps)
        prediction_type = self.scheduler.config.prediction_type
        if prediction_type == "sample":
            target = clean
        elif prediction_type == "epsilon":
            target = noise
        elif prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(clean, noise, timesteps)
        else:
            raise ValueError(f"Unsupported diffusion prediction_type: {prediction_type}.")
        per_sample = (prediction - target).square().mean(dim=(1, 2))
        loss = per_sample.mean()
        return loss, {"loss": loss.detach(), "loss_raw": per_sample.mean().detach()}
