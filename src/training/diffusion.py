from __future__ import annotations

import torch
from torch import nn


def freeze_vae(vae: nn.Module) -> None:
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)


def validate_latents(
    latents: torch.Tensor,
    *,
    latent_shape: tuple[int, int] = (48, 32),
) -> None:
    expected = (latents.shape[0], *latent_shape) if latents.ndim >= 1 else None
    if latents.ndim != 3 or tuple(latents.shape[1:]) != tuple(latent_shape):
        raise ValueError(
            f"VAE latent shape must be [B,{latent_shape[0]},{latent_shape[1]}]; "
            f"received {tuple(latents.shape)}."
        )
    if not torch.is_floating_point(latents):
        raise ValueError("VAE latents must use a floating-point dtype.")
    if not torch.isfinite(latents).all():
        raise ValueError("VAE latents must contain only finite values.")
