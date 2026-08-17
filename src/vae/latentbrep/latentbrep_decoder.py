from __future__ import annotations

import math

import torch
from torch import nn

from src.vae.modules import DecBiBlock


class MLP(nn.Module):
    """Two-layer projection used by the released decoder."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: int | None = None,
        act: type[nn.Module] = nn.SiLU,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = hidden or max(in_dim, out_dim, 128)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act(),
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


class DecoderBiSP(nn.Module):
    """Decode latent tokens into face, edge, count, and adjacency predictions."""

    def __init__(
        self,
        *,
        max_face: int = 50,
        latent_dim: int = 32,
        d_model: int = 512,
        nhead: int = 8,
        dropout: float = 0.1,
        mem_self_depth: int = 2,
        edge_cross_depth: int = 2,
        face_cross_depth: int = 2,
        dec_depth: int = 2,
        mlp_hidden: int | None = None,
        use_mem_sa: bool = True,
        use_mem_pos: bool = False,
        latent_len: int = 48,
        use_softplus: bool = False,
        use_sin: bool = False,
    ) -> None:
        super().__init__()
        self.max_face = max_face
        self.max_edge = max_face * 3
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.nhead = nhead
        self.dropout = dropout
        self.mem_self_depth = mem_self_depth
        self.edge_cross_depth = edge_cross_depth
        self.face_cross_depth = face_cross_depth
        self.dec_depth = dec_depth
        self.mlp_hidden = mlp_hidden
        self.use_mem_sa = use_mem_sa
        self.use_mem_pos = use_mem_pos
        self.latent_len = latent_len
        self.use_softplus = use_softplus
        self.use_sin = use_sin
        if edge_cross_depth != face_cross_depth:
            raise ValueError("edge_cross_depth and face_cross_depth must match.")
        dec_depth = face_cross_depth if face_cross_depth is not None else dec_depth

        self.mem_proj_in = (
            nn.Identity()
            if latent_dim == d_model
            else MLP(
                latent_dim,
                d_model,
                hidden=mlp_hidden,
                dropout=dropout,
            )
        )
        if use_mem_sa:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=min(4 * d_model, 1024),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.mem_sa = nn.TransformerEncoder(
                encoder_layer,
                num_layers=mem_self_depth,
                norm=nn.LayerNorm(d_model),
            )
        else:
            self.mem_sa = None
        if use_mem_pos:
            self.mem_pos_emb = nn.Parameter(
                torch.randn(latent_len, d_model) * 0.02
            )

        if not use_sin:
            self.face_queries = nn.Parameter(
                torch.randn(self.max_face, d_model) * 0.02
            )
            self.face_pos_emb = nn.Parameter(
                torch.randn(self.max_face, d_model) * 0.02
            )
            self.edge_queries = nn.Parameter(
                torch.randn(self.max_edge, d_model) * 0.02
            )
            self.edge_pos_emb = nn.Parameter(
                torch.randn(self.max_edge, d_model) * 0.02
            )
        else:
            position = torch.arange(0, 1000).unsqueeze(1)
            divisor = torch.exp(
                torch.arange(0, 256, 2) * (-math.log(10000.0) / 256)
            )
            positional_encoding = torch.zeros(1000, 256)
            positional_encoding[:, 0::2] = torch.sin(position * divisor)
            positional_encoding[:, 1::2] = torch.cos(position * divisor)
            self.register_buffer("pe", positional_encoding)
            self.face_token_mlp = MLP(
                256,
                d_model,
                hidden=mlp_hidden,
                dropout=dropout,
            )
            self.edge_token_mlp = MLP(
                256,
                d_model,
                hidden=mlp_hidden,
                dropout=dropout,
            )

        self.num_queries = nn.Parameter(torch.randn(2, d_model))
        count_decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            norm_first=True,
            batch_first=True,
            dim_feedforward=min(4 * d_model, 1024),
            dropout=dropout,
        )
        self.num_cross = nn.TransformerDecoder(
            count_decoder_layer,
            num_layers=2,
            norm=nn.LayerNorm(d_model),
        )
        self.head_face_len = MLP(
            d_model,
            max_face + 1,
            hidden=mlp_hidden,
            dropout=dropout,
        )
        self.head_edge_len = MLP(
            d_model,
            self.max_edge + 1,
            hidden=mlp_hidden,
            dropout=dropout,
        )
        self.dec_blocks = nn.ModuleList(
            [
                DecBiBlock(d_model=d_model, nhead=nhead, dropout=dropout)
                for _ in range(dec_depth)
            ]
        )
        self.head_face_z = MLP(
            d_model,
            108,
            hidden=mlp_hidden,
            dropout=dropout,
        )
        self.head_face_cC = MLP(
            d_model,
            6,
            hidden=mlp_hidden,
            dropout=dropout,
        )
        self.head_edge_z = MLP(
            d_model,
            18,
            hidden=mlp_hidden,
            dropout=dropout,
        )
        self.head_edge_cC = MLP(
            d_model,
            6,
            hidden=mlp_hidden,
            dropout=dropout,
        )
        self.head_edge_v0 = MLP(
            d_model,
            6,
            hidden=mlp_hidden,
            dropout=dropout,
        )
        adjacency_dim = d_model // 2
        self.emb_edge_for_face = MLP(
            d_model,
            adjacency_dim,
            hidden=mlp_hidden,
            dropout=dropout,
        )
        self.emb_face_for_face = MLP(
            d_model,
            adjacency_dim,
            hidden=mlp_hidden,
            dropout=dropout,
        )

    @staticmethod
    def _decode_center_size_with_sqrt(
        prediction: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        x_center, y_center, z_center = (
            prediction[..., 0],
            prediction[..., 1],
            prediction[..., 2],
        )
        size = torch.sqrt(prediction[..., 3:6].pow(2) + eps)
        x_min = x_center - size[..., 0] / 2
        x_max = x_center + size[..., 0] / 2
        y_min = y_center - size[..., 1] / 2
        y_max = y_center + size[..., 1] / 2
        z_min = z_center - size[..., 2] / 2
        z_max = z_center + size[..., 2] / 2
        return torch.stack(
            [x_min, y_min, z_min, x_max, y_max, z_max],
            dim=-1,
        )

    @staticmethod
    def _decode_center_size_to_corners(
        prediction: torch.Tensor,
        min_size: float = 1e-6,
        beta: float = 1.0,
        do_sort: bool = False,
    ) -> torch.Tensor:
        x_center, y_center, z_center = (
            prediction[..., 0],
            prediction[..., 1],
            prediction[..., 2],
        )
        size = nn.functional.softplus(
            prediction[..., 3:6],
            beta=beta,
        ) + min_size
        x_min = x_center - 0.5 * size[..., 0]
        x_max = x_center + 0.5 * size[..., 0]
        y_min = y_center - 0.5 * size[..., 1]
        y_max = y_center + 0.5 * size[..., 1]
        z_min = z_center - 0.5 * size[..., 2]
        z_max = z_center + 0.5 * size[..., 2]
        if do_sort:
            x_min, x_max = torch.minimum(x_min, x_max), torch.maximum(x_min, x_max)
            y_min, y_max = torch.minimum(y_min, y_max), torch.maximum(y_min, y_max)
            z_min, z_max = torch.minimum(z_min, z_max), torch.maximum(z_min, z_max)
        return torch.stack(
            [x_min, y_min, z_min, x_max, y_max, z_max],
            dim=-1,
        )

    @staticmethod
    def _build_masks_from_num(
        device: torch.device,
        batch_size: int,
        max_face: int,
        num_face: torch.Tensor | None,
        num_edge: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if num_face is None:
            return None, None
        if num_face.dim() == 2 and num_face.size(1) == 1:
            num_face = num_face.squeeze(1)
        num_face = num_face.to(device).long()
        face_indices = torch.arange(max_face, device=device).unsqueeze(0).expand(
            batch_size,
            -1,
        )
        face_pad = face_indices >= num_face.unsqueeze(1)

        max_edge = max_face * 3
        if num_edge is None:
            num_edge = num_face * 3
        else:
            if num_edge.dim() == 2 and num_edge.size(1) == 1:
                num_edge = num_edge.squeeze(1)
            num_edge = num_edge.to(device).long()
        edge_indices = torch.arange(max_edge, device=device).unsqueeze(0).expand(
            batch_size,
            -1,
        )
        edge_pad = edge_indices >= num_edge.unsqueeze(1)
        return face_pad, edge_pad

    @staticmethod
    def _build_masks_from_logit(
        device: torch.device,
        batch_size: int,
        max_face: int,
        num_face_logit: torch.Tensor | None,
        num_edge_logit: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if num_face_logit is not None and num_face_logit.dim() == 3:
            num_face_logit = num_face_logit.squeeze(1)
        if num_edge_logit is not None and num_edge_logit.dim() == 3:
            num_edge_logit = num_edge_logit.squeeze(1)
        predicted_faces = (
            num_face_logit.argmax(-1) if num_face_logit is not None else None
        )
        predicted_edges = (
            num_edge_logit.argmax(-1) if num_edge_logit is not None else None
        )
        return DecoderBiSP._build_masks_from_num(
            device=device,
            batch_size=batch_size,
            max_face=max_face,
            num_face=predicted_faces,
            num_edge=predicted_edges,
        )

    def forward(
        self,
        z: torch.Tensor,
        use_pred: bool = False,
        num_face: torch.Tensor | None = None,
        num_edge: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size, _, _ = z.shape
        memory = self.mem_proj_in(z)
        if self.use_mem_pos:
            memory = memory + self.mem_pos_emb.unsqueeze(0)
        if self.mem_sa is not None:
            memory = self.mem_sa(memory)

        count_queries = self.num_queries.unsqueeze(0).expand(batch_size, -1, -1)
        count_tokens = self.num_cross(tgt=count_queries, memory=memory)
        num_face_logits = self.head_face_len(count_tokens[:, 0])
        num_edge_logits = self.head_edge_len(count_tokens[:, 1])

        if use_pred:
            face_pad_mask, edge_pad_mask = self._build_masks_from_logit(
                device=z.device,
                batch_size=batch_size,
                max_face=self.max_face,
                num_face_logit=num_face_logits,
                num_edge_logit=num_edge_logits,
            )
        else:
            face_pad_mask, edge_pad_mask = self._build_masks_from_num(
                device=z.device,
                batch_size=batch_size,
                max_face=self.max_face,
                num_face=num_face,
                num_edge=num_edge,
            )
        if face_pad_mask is None or edge_pad_mask is None:
            face_pad_mask = torch.zeros(
                batch_size,
                self.max_face,
                dtype=torch.bool,
                device=z.device,
            )
            edge_pad_mask = torch.zeros(
                batch_size,
                self.max_edge,
                dtype=torch.bool,
                device=z.device,
            )

        memory = torch.cat([memory, count_tokens], dim=1)
        if not self.use_sin:
            face_tokens = (self.face_queries + self.face_pos_emb).unsqueeze(0).expand(
                batch_size,
                -1,
                -1,
            )
            edge_tokens = (self.edge_queries + self.edge_pos_emb).unsqueeze(0).expand(
                batch_size,
                -1,
                -1,
            )
        else:
            face_tokens = self.face_token_mlp(self.pe[: self.max_face]).unsqueeze(0).expand(
                batch_size,
                -1,
                -1,
            )
            edge_tokens = self.edge_token_mlp(self.pe[: self.max_edge]).unsqueeze(0).expand(
                batch_size,
                -1,
                -1,
            )

        for block in self.dec_blocks:
            face_tokens, edge_tokens = block(
                face_tokens,
                edge_tokens,
                memory,
                face_pad_mask,
                edge_pad_mask,
            )

        face_tokens = face_tokens.masked_fill(face_pad_mask.unsqueeze(-1), 0.0)
        edge_tokens = edge_tokens.masked_fill(edge_pad_mask.unsqueeze(-1), 0.0)

        face_z_hat = self.head_face_z(face_tokens)
        face_center_size = self.head_face_cC(face_tokens)
        if not self.use_softplus:
            face_pos_hat = self._decode_center_size_with_sqrt(face_center_size)
        else:
            face_pos_hat = self._decode_center_size_to_corners(face_center_size)

        edge_z_hat = self.head_edge_z(edge_tokens)
        edge_center_size = self.head_edge_cC(edge_tokens)
        if not self.use_softplus:
            edge_bbox_hat = self._decode_center_size_with_sqrt(edge_center_size)
        else:
            edge_bbox_hat = self._decode_center_size_to_corners(edge_center_size)
        edge_corners_hat = self.head_edge_v0(edge_tokens)

        edge_adjacency = self.emb_edge_for_face(edge_tokens)
        face_adjacency = self.emb_face_for_face(face_tokens)
        adjacency_dimension = max(1, edge_adjacency.shape[-1])
        adj_face_logits = torch.bmm(
            edge_adjacency,
            face_adjacency.transpose(1, 2),
        ) / math.sqrt(adjacency_dimension)

        return {
            "num_face_logits": num_face_logits,
            "num_edge_logits": num_edge_logits,
            "face_z_hat": face_z_hat,
            "face_pos_hat": face_pos_hat,
            "edge_z_hat": edge_z_hat,
            "edge_bbox_hat": edge_bbox_hat,
            "edge_corners_hat": edge_corners_hat,
            "adj_face_logits": adj_face_logits,
        }
