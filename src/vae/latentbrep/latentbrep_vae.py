from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.training.losses import MaskedMSELoss, RowSoftmaxTwoPeakLoss, build_padding_masks
from src.vae.latentbrep.latentbrep_decoder import DecoderBiSP
from src.vae.latentbrep.latentbrep_encoder import EncoderATPBiLatent


class LatentBiSPBrepVAE(nn.Module):
    """HiFi-BRep variational autoencoder with the historical checkpoint layout."""

    def __init__(
        self,
        *,
        max_face: int = 50,
        enc_d_model: int = 512,
        enc_nhead: int = 8,
        enc_dropout: float = 0.1,
        enc_self_depth: int = 2,
        enc_cross_depth: int = 2,
        latent_len: int = 16,
        learnable_latent: bool = True,
        dec_d_model: int = 512,
        dec_nhead: int = 8,
        dec_dropout: float = 0.1,
        mem_self_depth: int = 2,
        face_cross_depth: int = 2,
        edge_cross_depth: int = 2,
        use_mem_pos: bool = False,
        use_mem_sa: bool = True,
        use_softplus: bool = False,
        use_sin: bool = False,
        latent_dim: int = 64,
        w_kl: float = 1e-6,
        w_num_face: float = 1.0,
        w_num_edge: float = 0.7,
        w_face_z: float = 1.0,
        w_face_pos: float = 1.0,
        w_edge_z: float = 1.0,
        w_edge_bbox: float = 1.0,
        w_edge_corners: float = 1.0,
        w_adj_face: float = 1.0,
        **_: object,
    ) -> None:
        super().__init__()
        self.max_face = max_face
        self.max_edge = max_face * 3
        self.latent_len = latent_len
        self.latent_dim = latent_dim
        self.encoder = EncoderATPBiLatent(
            max_face=max_face,
            d_model=enc_d_model,
            nhead=enc_nhead,
            dropout=enc_dropout,
            enc_self_depth=enc_self_depth,
            enc_cross_depth=enc_cross_depth,
            latent_len=latent_len,
            mlp_hidden=None,
            use_pos_emb=True,
            learnable_latent=learnable_latent,
            use_sin=use_sin,
        )
        self.quant_proj = nn.Linear(enc_d_model, 2 * latent_dim)
        self.decoder = DecoderBiSP(
            max_face=max_face,
            latent_dim=latent_dim,
            d_model=dec_d_model,
            nhead=dec_nhead,
            dropout=dec_dropout,
            mem_self_depth=mem_self_depth,
            face_cross_depth=face_cross_depth,
            edge_cross_depth=edge_cross_depth,
            latent_len=latent_len,
            use_mem_pos=use_mem_pos,
            use_mem_sa=use_mem_sa,
            use_softplus=use_softplus,
            use_sin=use_sin,
        )
        self.mse = MaskedMSELoss(reduction="batch")
        self.adj_loss = RowSoftmaxTwoPeakLoss(reduction="batch")
        self.loss_weights = {
            "kl": float(w_kl),
            "num_face": float(w_num_face),
            "num_edge": float(w_num_edge),
            "face_z": float(w_face_z),
            "face_pos": float(w_face_pos),
            "edge_z": float(w_edge_z),
            "edge_bbox": float(w_edge_bbox),
            "edge_corners": float(w_edge_corners),
            "adj_face": float(w_adj_face),
        }

    def reparameterize(
        self,
        mean: torch.Tensor,
        log_variance: torch.Tensor,
        *,
        sample: bool,
    ) -> torch.Tensor:
        if not sample:
            return mean
        standard_deviation = torch.exp(0.5 * log_variance).clamp_max(3.0)
        return mean + torch.randn_like(standard_deviation) * standard_deviation

    def encode(
        self,
        *,
        surf_z: torch.Tensor,
        surf_pos: torch.Tensor,
        edge_z: torch.Tensor,
        edge_pos: torch.Tensor,
        adj_face: torch.Tensor,
        num_face: torch.Tensor | None = None,
        num_edge: torch.Tensor | None = None,
    ) -> dict[str, object]:
        encoded = self.encoder(
            surf_z=surf_z,
            surf_pos=surf_pos,
            edge_z=edge_z,
            edge_pos=edge_pos,
            adj_face=adj_face,
            num_face=num_face,
            num_edge=num_edge,
        )
        mean, log_variance = self.quant_proj(encoded["latent_tokens"]).chunk(2, dim=-1)
        return {"mu": mean, "logvar": log_variance, "enc_cache": encoded}

    def _loss(
        self,
        *,
        decoded: dict[str, torch.Tensor],
        mean: torch.Tensor,
        log_variance: torch.Tensor,
        surf_z: torch.Tensor,
        surf_pos: torch.Tensor,
        edge_z: torch.Tensor,
        edge_pos: torch.Tensor,
        adj_face: torch.Tensor,
        num_face: torch.Tensor,
        num_edge: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, max_faces = surf_pos.shape[:2]
        max_edges = edge_pos.shape[1]
        face_padding, edge_padding = build_padding_masks(
            num_face=num_face,
            num_edge=num_edge,
            max_face=max_faces,
            max_edge=max_edges,
        )
        face_mask = (~face_padding).to(surf_pos.dtype).unsqueeze(-1)
        edge_mask = (~edge_padding).to(edge_pos.dtype).unsqueeze(-1)
        kl = 0.5 * (
            torch.exp(log_variance) + mean.square() - 1.0 - log_variance
        ).sum(dim=(1, 2))
        face_target = num_face.reshape(batch).clamp(0, decoded["num_face_logits"].shape[-1] - 1)
        edge_target = num_edge.reshape(batch).clamp(0, decoded["num_edge_logits"].shape[-1] - 1)
        components = {
            "kl": kl,
            "num_face": F.cross_entropy(
                decoded["num_face_logits"],
                face_target,
                label_smoothing=0.005,
                reduction="none",
            ),
            "num_edge": F.cross_entropy(
                decoded["num_edge_logits"],
                edge_target,
                label_smoothing=0.005,
                reduction="none",
            ),
            "face_z": self.mse(decoded["face_z_hat"], surf_z.reshape(batch, max_faces, 108), face_mask),
            "face_pos": self.mse(decoded["face_pos_hat"], surf_pos, face_mask),
            "edge_z": self.mse(decoded["edge_z_hat"], edge_z.reshape(batch, max_edges, 18), edge_mask),
            "edge_bbox": self.mse(decoded["edge_bbox_hat"], edge_pos[..., :6], edge_mask),
            "edge_corners": self.mse(decoded["edge_corners_hat"], edge_pos[..., 6:12], edge_mask),
            "adj_face": self.adj_loss(
                decoded["adj_face_logits"],
                adj_face.to(decoded["adj_face_logits"].dtype) * 0.5,
                face_mask=~face_padding,
                edge_mask=~edge_padding,
            ),
        }
        per_sample = sum(self.loss_weights[name] * value for name, value in components.items())
        total = per_sample.mean()
        logs = {f"loss_{name}": value.mean().detach() for name, value in components.items()}
        logs.update(
            {
                "loss_total": total.detach(),
                "post_mean_abs": mean.abs().mean().detach(),
                "post_std_mean": (0.5 * log_variance).exp().mean().detach(),
            }
        )
        return total, logs

    def loss_from_predictions(
        self,
        *,
        encoded: dict[str, object],
        decoded: dict[str, torch.Tensor],
        surf_z: torch.Tensor,
        surf_pos: torch.Tensor,
        edge_z: torch.Tensor,
        edge_pos: torch.Tensor,
        adj_face: torch.Tensor,
        num_face: torch.Tensor,
        num_edge: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mean = encoded["mu"]
        log_variance = encoded["logvar"].clamp(min=-10.0, max=10.0)
        return self._loss(
            decoded=decoded,
            mean=mean,
            log_variance=log_variance,
            surf_z=surf_z,
            surf_pos=surf_pos,
            edge_z=edge_z,
            edge_pos=edge_pos,
            adj_face=adj_face,
            num_face=num_face,
            num_edge=num_edge,
        )

    def forward(
        self,
        *,
        surf_z: torch.Tensor | None = None,
        surf_pos: torch.Tensor | None = None,
        edge_z: torch.Tensor | None = None,
        edge_pos: torch.Tensor | None = None,
        adj_face: torch.Tensor | None = None,
        num_face: torch.Tensor | None = None,
        num_edge: torch.Tensor | None = None,
        sample_posterior: bool = True,
        return_pred: bool = False,
        encode_only: bool = False,
        return_mu_logvar: bool = False,
        decode_z: torch.Tensor | None = None,
        **_: object,
    ) -> object:
        if decode_z is not None:
            return self.decoder(decode_z, use_pred=True)
        required = {
            "surf_z": surf_z,
            "surf_pos": surf_pos,
            "edge_z": edge_z,
            "edge_pos": edge_pos,
            "adj_face": adj_face,
            "num_face": num_face,
            "num_edge": num_edge,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"VAE forward is missing tensors: {', '.join(missing)}.")
        encoded = self.encode(
            surf_z=surf_z,
            surf_pos=surf_pos,
            edge_z=edge_z,
            edge_pos=edge_pos,
            adj_face=adj_face,
            num_face=num_face,
            num_edge=num_edge,
        )
        mean = encoded["mu"]
        log_variance = encoded["logvar"].clamp(min=-10.0, max=10.0)
        if return_mu_logvar:
            return mean, log_variance
        latent = self.reparameterize(mean, log_variance, sample=sample_posterior)
        if encode_only:
            return latent
        decoded = self.decoder(latent, num_face=num_face, num_edge=num_edge)
        if return_pred:
            return encoded, decoded
        return self._loss(
            decoded=decoded,
            mean=mean,
            log_variance=log_variance,
            surf_z=surf_z,
            surf_pos=surf_pos,
            edge_z=edge_z,
            edge_pos=edge_pos,
            adj_face=adj_face,
            num_face=num_face,
            num_edge=num_edge,
        )
