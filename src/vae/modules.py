from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SDPA(nn.Module):
    """Projected scaled dot-product attention with boolean or additive masks."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.o_proj.bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(attn_mask, torch.Tensor) and attn_mask.dtype == torch.bool:
            zero = torch.tensor(0.0, dtype=query.dtype, device=query.device)
            negative = torch.tensor(-1e9, dtype=query.dtype, device=query.device)
            attn_mask = torch.where(attn_mask, zero, negative)

        if isinstance(attn_mask, torch.Tensor) and attn_mask.dtype.is_floating_point:
            all_blocked = (
                torch.isneginf(attn_mask).all(dim=-1, keepdim=True)
                if attn_mask.isinf().any()
                else (attn_mask < -1e8).all(dim=-1, keepdim=True)
            )
            attn_mask = torch.where(
                all_blocked,
                torch.zeros_like(attn_mask),
                attn_mask,
            )

        batch_size, query_length, dimension = query.shape
        key_length = key.shape[1]
        query = self.q_proj(query).view(
            batch_size,
            query_length,
            self.nhead,
            self.d_head,
        ).transpose(1, 2)
        key = self.k_proj(key).view(
            batch_size,
            key_length,
            self.nhead,
            self.d_head,
        ).transpose(1, 2)
        value = self.v_proj(value).view(
            batch_size,
            key_length,
            self.nhead,
            self.d_head,
        ).transpose(1, 2)

        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        output = output.transpose(1, 2).contiguous().view(
            batch_size,
            query_length,
            dimension,
        )
        return self.o_proj(output)


class BiModalBlock(nn.Module):
    """Apply masked self- and cross-attention to face and edge tokens."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm_s1 = nn.LayerNorm(d_model)
        self.norm_e1 = nn.LayerNorm(d_model)
        self.sa_s = SDPA(d_model, nhead, dropout)
        self.sa_e = SDPA(d_model, nhead, dropout)

        self.norm_s2q = nn.LayerNorm(d_model)
        self.norm_s2k = nn.LayerNorm(d_model)
        self.norm_e2q = nn.LayerNorm(d_model)
        self.norm_e2k = nn.LayerNorm(d_model)
        self.x_s_from_e = SDPA(d_model, nhead, dropout)
        self.x_e_from_s = SDPA(d_model, nhead, dropout)

        self.norm_s3 = nn.LayerNorm(d_model)
        self.norm_e3 = nn.LayerNorm(d_model)
        self.ffn_s = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.ffn_e = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(
        self,
        surface_tokens: torch.Tensor,
        edge_tokens: torch.Tensor,
        face_valid_mask: torch.Tensor,
        edge_valid_mask: torch.Tensor,
        adj_face: torch.Tensor,
        use_soft_nonadj: bool = False,
        nonadj_bias: float = -6.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, face_count, _ = surface_tokens.shape
        edge_count = edge_tokens.shape[1]

        surface_mask = face_valid_mask[:, None, None, :].expand(
            batch_size,
            1,
            face_count,
            face_count,
        )
        edge_mask = edge_valid_mask[:, None, None, :].expand(
            batch_size,
            1,
            edge_count,
            edge_count,
        )
        surface_tokens = surface_tokens + self.sa_s(
            self.norm_s1(surface_tokens),
            self.norm_s1(surface_tokens),
            self.norm_s1(surface_tokens),
            attn_mask=surface_mask,
        )
        edge_tokens = edge_tokens + self.sa_e(
            self.norm_e1(edge_tokens),
            self.norm_e1(edge_tokens),
            self.norm_e1(edge_tokens),
            attn_mask=edge_mask,
        )

        adjacency = adj_face.bool()
        allow_edge_to_surface = adjacency.transpose(1, 2)
        valid_edges = edge_valid_mask[:, None, :].expand(
            batch_size,
            face_count,
            edge_count,
        )
        allow_edge_to_surface = allow_edge_to_surface & valid_edges

        allow_surface_to_edge = adjacency
        valid_faces = face_valid_mask[:, None, :].expand(
            batch_size,
            edge_count,
            face_count,
        )
        allow_surface_to_edge = allow_surface_to_edge & valid_faces

        blocked_value = -1e9 if not use_soft_nonadj else float(nonadj_bias)
        edge_to_surface_mask = torch.where(
            allow_edge_to_surface,
            torch.tensor(0.0, dtype=surface_tokens.dtype, device=surface_tokens.device),
            torch.tensor(
                blocked_value,
                dtype=surface_tokens.dtype,
                device=surface_tokens.device,
            ),
        ).unsqueeze(1)
        surface_to_edge_mask = torch.where(
            allow_surface_to_edge,
            torch.tensor(0.0, dtype=surface_tokens.dtype, device=surface_tokens.device),
            torch.tensor(
                blocked_value,
                dtype=surface_tokens.dtype,
                device=surface_tokens.device,
            ),
        ).unsqueeze(1)

        surface_tokens = surface_tokens + self.x_s_from_e(
            self.norm_s2q(surface_tokens),
            self.norm_s2k(edge_tokens),
            self.norm_s2k(edge_tokens),
            attn_mask=edge_to_surface_mask,
        )
        edge_tokens = edge_tokens + self.x_e_from_s(
            self.norm_e2q(edge_tokens),
            self.norm_e2k(surface_tokens),
            self.norm_e2k(surface_tokens),
            attn_mask=surface_to_edge_mask,
        )
        surface_tokens = surface_tokens + self.ffn_s(self.norm_s3(surface_tokens))
        edge_tokens = edge_tokens + self.ffn_e(self.norm_e3(edge_tokens))
        return surface_tokens, edge_tokens


class DecBiBlock(nn.Module):
    """Decode coupled face and edge tokens against the latent memory."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_fm_q = nn.LayerNorm(d_model)
        self.n_fm_k = nn.LayerNorm(d_model)
        self.n_em_q = nn.LayerNorm(d_model)
        self.n_em_k = nn.LayerNorm(d_model)
        self.ca_f_mem = SDPA(d_model, nhead, dropout)
        self.ca_e_mem = SDPA(d_model, nhead, dropout)

        self.n_fs = nn.LayerNorm(d_model)
        self.n_es = nn.LayerNorm(d_model)
        self.sa_f = SDPA(d_model, nhead, dropout)
        self.sa_e = SDPA(d_model, nhead, dropout)

        self.n_fq = nn.LayerNorm(d_model)
        self.n_fk = nn.LayerNorm(d_model)
        self.n_eq = nn.LayerNorm(d_model)
        self.n_ek = nn.LayerNorm(d_model)
        self.ca_f_from_e = SDPA(d_model, nhead, dropout)
        self.ca_e_from_f = SDPA(d_model, nhead, dropout)

        self.n_ff = nn.LayerNorm(d_model)
        self.n_ef = nn.LayerNorm(d_model)
        self.ffn_f = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.ffn_e = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        for feed_forward in (self.ffn_f, self.ffn_e):
            for module in feed_forward:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        face_tokens: torch.Tensor,
        edge_tokens: torch.Tensor,
        memory: torch.Tensor,
        face_pad_mask: torch.Tensor,
        edge_pad_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, face_count, _ = face_tokens.shape
        edge_count = edge_tokens.shape[1]
        face_valid_mask = ~face_pad_mask
        edge_valid_mask = ~edge_pad_mask

        face_tokens = face_tokens + self.ca_f_mem(
            self.n_fm_q(face_tokens),
            self.n_fm_k(memory),
            self.n_fm_k(memory),
            attn_mask=None,
        )
        edge_tokens = edge_tokens + self.ca_e_mem(
            self.n_em_q(edge_tokens),
            self.n_em_k(memory),
            self.n_em_k(memory),
            attn_mask=None,
        )

        allow_faces = face_valid_mask[:, None, None, :].expand(
            batch_size,
            1,
            face_count,
            face_count,
        )
        allow_edges = edge_valid_mask[:, None, None, :].expand(
            batch_size,
            1,
            edge_count,
            edge_count,
        )
        face_tokens = face_tokens + self.sa_f(
            self.n_fs(face_tokens),
            self.n_fs(face_tokens),
            self.n_fs(face_tokens),
            attn_mask=allow_faces,
        )
        edge_tokens = edge_tokens + self.sa_e(
            self.n_es(edge_tokens),
            self.n_es(edge_tokens),
            self.n_es(edge_tokens),
            attn_mask=allow_edges,
        )

        allow_edge_keys = edge_valid_mask[:, None, None, :].expand(
            batch_size,
            1,
            face_count,
            edge_count,
        )
        allow_face_keys = face_valid_mask[:, None, None, :].expand(
            batch_size,
            1,
            edge_count,
            face_count,
        )
        face_tokens = face_tokens + self.ca_f_from_e(
            self.n_fq(face_tokens),
            self.n_ek(edge_tokens),
            self.n_ek(edge_tokens),
            attn_mask=allow_edge_keys,
        )
        edge_tokens = edge_tokens + self.ca_e_from_f(
            self.n_eq(edge_tokens),
            self.n_fk(face_tokens),
            self.n_fk(face_tokens),
            attn_mask=allow_face_keys,
        )
        face_tokens = face_tokens + self.ffn_f(self.n_ff(face_tokens))
        edge_tokens = edge_tokens + self.ffn_e(self.n_ef(edge_tokens))
        return face_tokens, edge_tokens
