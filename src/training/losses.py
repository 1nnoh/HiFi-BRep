from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def masked_reduce(
    loss: torch.Tensor,
    mask: torch.Tensor | None,
    reduction: str,
) -> torch.Tensor:
    if reduction not in ("none", "sum", "mean", "batch"):
        raise ValueError(f"Unsupported reduction: {reduction}.")
    if mask is None:
        if reduction == "none":
            return loss
        if reduction == "sum":
            return loss.sum()
        per_sample = loss.flatten(1).mean(-1)
        return per_sample if reduction == "batch" else per_sample.mean()
    if mask.ndim != loss.ndim:
        raise ValueError(f"Mask rank {mask.ndim} does not match loss rank {loss.ndim}.")
    weights = mask.to(device=loss.device, dtype=loss.dtype).expand_as(loss)
    weighted = loss * weights
    if reduction == "none":
        return weighted
    if reduction == "sum":
        return weighted.sum()
    per_sample = weighted.flatten(1).sum(-1) / weights.flatten(1).sum(-1).clamp_min(1e-6)
    return per_sample if reduction == "batch" else per_sample.mean()


class MaskedMSELoss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return masked_reduce((prediction - target).square(), mask, self.reduction)


class RowSoftmaxTwoPeakLoss(nn.Module):
    def __init__(self, temperature: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        *,
        edge_mask: torch.Tensor | None = None,
        face_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if logits.ndim != 3 or target.shape != logits.shape:
            raise ValueError("Adjacency logits and targets must have shape [B,E,F].")
        batch, edges, faces = logits.shape
        if edge_mask is None:
            edge_mask = torch.ones((batch, edges), dtype=torch.bool, device=logits.device)
        if face_mask is None:
            face_mask = torch.ones((batch, faces), dtype=torch.bool, device=logits.device)
        combined = edge_mask.unsqueeze(-1) & face_mask.unsqueeze(1)
        face_columns = face_mask.unsqueeze(1).expand_as(logits)
        masked_logits = logits.masked_fill(~face_columns, float("-inf"))
        row_has_valid = combined.any(dim=-1, keepdim=True)
        effective_logits = torch.where(row_has_valid, masked_logits, torch.zeros_like(logits))
        log_probability = F.log_softmax(effective_logits / self.temperature, dim=-1)
        log_probability = torch.where(combined, log_probability, torch.zeros_like(log_probability))
        return masked_reduce(-(target * log_probability), combined, self.reduction)


def build_padding_masks(
    *,
    num_face: torch.Tensor,
    num_edge: torch.Tensor,
    max_face: int,
    max_edge: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = num_face.shape[0]
    face_count = num_face.reshape(batch)
    edge_count = num_edge.reshape(batch)
    face_padding = torch.arange(max_face, device=num_face.device).unsqueeze(0) >= face_count.unsqueeze(1)
    edge_padding = torch.arange(max_edge, device=num_edge.device).unsqueeze(0) >= edge_count.unsqueeze(1)
    return face_padding, edge_padding
