from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from src.vae.modules import BiModalBlock


def _flatten_last_dims(value: torch.Tensor, dimensions: int) -> torch.Tensor:
    shape = value.shape
    return value.reshape(*shape[:-dimensions], -1).contiguous()


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Optional[int] = None,
        activation: type[nn.Module] = nn.SiLU,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = hidden or max(in_dim, out_dim, 128)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class EncoderATPBiLatent(nn.Module):
    """Encode padded face/edge geometry into a fixed latent token sequence."""

    def __init__(
        self,
        *,
        max_face: int = 50,
        d_model: int = 512,
        nhead: int = 8,
        dropout: float = 0.1,
        enc_self_depth: int = 2,
        enc_cross_depth: int = 2,
        latent_len: int = 16,
        mlp_hidden: int | None = None,
        use_pos_emb: bool = True,
        use_sin: bool = False,
        use_soft_nonadj: bool = False,
        learnable_latent: bool = True,
        nonadj_bias: float = -6.0,
    ) -> None:
        super().__init__()
        self.max_face = max_face
        self.max_edge = max_face * 3
        self.d_model = d_model
        self.latent_len = latent_len
        self.use_pos_emb = use_pos_emb
        self.use_soft_nonadj = use_soft_nonadj
        self.nonadj_bias = nonadj_bias
        self.use_sin = use_sin

        self.emb_surf_z = MLP(36 * 3, d_model, hidden=mlp_hidden, dropout=dropout)
        self.emb_surf_p = MLP(6, d_model, hidden=mlp_hidden, dropout=dropout)
        self.emb_edge_z = MLP(6 * 3, d_model, hidden=mlp_hidden, dropout=dropout)
        self.emb_edge_p = MLP(6, d_model, hidden=mlp_hidden, dropout=dropout)
        self.emb_edge_v = MLP(6, d_model, hidden=mlp_hidden, dropout=dropout)

        if use_pos_emb and not use_sin:
            self.face_pos_emb = nn.Parameter(torch.randn(self.max_face, d_model) * 0.02)
            self.edge_pos_emb = nn.Parameter(torch.randn(self.max_edge, d_model) * 0.02)
        if use_pos_emb and use_sin:
            position = torch.arange(0, 1000).unsqueeze(1)
            divisor = torch.exp(
                torch.arange(0, 256, 2) * (-math.log(10000.0) / 256)
            )
            encoding = torch.zeros(1000, 256)
            encoding[:, 0::2] = torch.sin(position * divisor)
            encoding[:, 1::2] = torch.cos(position * divisor)
            self.register_buffer("pe", encoding)
            self.face_token_mlp = MLP(256, d_model, hidden=mlp_hidden, dropout=dropout)
            self.edge_token_mlp = MLP(256, d_model, hidden=mlp_hidden, dropout=dropout)

        self.bimodal_blocks = nn.ModuleList(
            [
                BiModalBlock(d_model=d_model, nhead=nhead, dropout=dropout)
                for _ in range(enc_self_depth)
            ]
        )
        self.learnable_latent = learnable_latent
        if learnable_latent:
            self.latent_queries = nn.Parameter(torch.randn(latent_len, d_model) * 0.02)
        else:
            if not hasattr(self, "pe"):
                position = torch.arange(0, 1000).unsqueeze(1)
                divisor = torch.exp(
                    torch.arange(0, 256, 2) * (-math.log(10000.0) / 256)
                )
                encoding = torch.zeros(1000, 256)
                encoding[:, 0::2] = torch.sin(position * divisor)
                encoding[:, 1::2] = torch.cos(position * divisor)
                self.register_buffer("pe", encoding)
            self.mlp_latent = MLP(256, d_model, hidden=mlp_hidden, dropout=dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            norm_first=True,
            batch_first=True,
            dim_feedforward=min(d_model * 4, 1024),
            dropout=dropout,
        )
        self.cross_attn = nn.TransformerDecoder(
            decoder_layer,
            enc_cross_depth,
            nn.LayerNorm(d_model),
        )
        self.cross_norm_q = MLP(d_model, d_model, hidden=mlp_hidden, dropout=dropout)
        self.cross_norm_o = nn.LayerNorm(d_model)
        self.cross_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.cross_attn.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.ones_(self.cross_norm_o.weight)
        nn.init.zeros_(self.cross_norm_o.bias)

    @staticmethod
    def _build_masks_from_counts(
        device: torch.device,
        batch_size: int,
        max_face: int,
        num_face: torch.Tensor | None,
        num_edge: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if num_face is None:
            return None, None
        num_face = num_face.reshape(batch_size).to(device=device, dtype=torch.long)
        face_index = torch.arange(max_face, device=device).unsqueeze(0)
        face_padding = face_index >= num_face.unsqueeze(1)
        if num_edge is None:
            num_edge = num_face * 3
        else:
            num_edge = num_edge.reshape(batch_size).to(device=device, dtype=torch.long)
        edge_index = torch.arange(max_face * 3, device=device).unsqueeze(0)
        edge_padding = edge_index >= num_edge.unsqueeze(1)
        return face_padding, edge_padding

    @staticmethod
    def _fallback_masks_from_zero_rows(
        surface_position: torch.Tensor,
        edge_position: torch.Tensor,
        epsilon: float = 1e-12,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            surface_position.abs().sum(dim=-1) < epsilon,
            edge_position.abs().sum(dim=-1) < epsilon,
        )

    def forward(
        self,
        *,
        surf_z: torch.Tensor,
        surf_pos: torch.Tensor,
        edge_z: torch.Tensor,
        edge_pos: torch.Tensor,
        adj_face: torch.Tensor,
        num_face: torch.Tensor | None = None,
        num_edge: torch.Tensor | None = None,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        batch_size, max_faces = surf_pos.shape[:2]
        max_edges = edge_pos.shape[1]
        face_padding, edge_padding = self._build_masks_from_counts(
            surf_pos.device,
            batch_size,
            self.max_face,
            num_face,
            num_edge,
        )
        if face_padding is None or edge_padding is None:
            face_padding, edge_padding = self._fallback_masks_from_zero_rows(
                surf_pos,
                edge_pos,
            )

        surface_tokens = self.emb_surf_z(_flatten_last_dims(surf_z, 2))
        surface_tokens = surface_tokens + self.emb_surf_p(surf_pos)
        edge_tokens = self.emb_edge_z(_flatten_last_dims(edge_z, 2))
        edge_tokens = edge_tokens + self.emb_edge_p(edge_pos[..., :6])
        edge_tokens = edge_tokens + self.emb_edge_v(edge_pos[..., 6:12])

        if self.use_pos_emb and not self.use_sin:
            surface_tokens = surface_tokens + self.face_pos_emb.unsqueeze(0)
            edge_tokens = edge_tokens + self.edge_pos_emb.unsqueeze(0)
        elif self.use_pos_emb and self.use_sin:
            surface_tokens = surface_tokens + self.face_token_mlp(self.pe[:max_faces]).unsqueeze(0)
            edge_tokens = edge_tokens + self.edge_token_mlp(self.pe[:max_edges]).unsqueeze(0)

        face_valid = ~face_padding
        edge_valid = ~edge_padding
        for block in self.bimodal_blocks:
            surface_tokens, edge_tokens = block(
                surface_tokens,
                edge_tokens,
                face_valid_mask=face_valid,
                edge_valid_mask=edge_valid,
                adj_face=adj_face,
                use_soft_nonadj=self.use_soft_nonadj,
                nonadj_bias=self.nonadj_bias,
            )

        context = torch.cat((surface_tokens, edge_tokens), dim=1)
        context_valid = torch.cat((face_valid, edge_valid), dim=1)
        if self.learnable_latent:
            queries = self.latent_queries.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            queries = self.mlp_latent(self.pe[: self.latent_len]).unsqueeze(0)
            queries = queries.expand(batch_size, -1, -1)
        normalized_queries = self.cross_norm_q(queries)
        latent = self.cross_attn(
            tgt=normalized_queries,
            memory=context,
            memory_key_padding_mask=~context_valid,
        )
        latent = self.cross_norm_o(latent + queries)
        latent = latent + self.cross_ffn(latent)
        return {
            "latent_tokens": latent,
            "surf_tokens": surface_tokens,
            "edge_tokens": edge_tokens,
            "face_pad_mask": face_padding,
            "edge_pad_mask": edge_padding,
        }
