from __future__ import annotations

import math

import torch
from timm.models.vision_transformer import Attention, Mlp
from torch import nn


def modulate(
    value: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return value * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Embed continuous diffusion timesteps into the DiT hidden dimension."""

    def __init__(
        self,
        hidden_size: int,
        output_size: int | None = None,
        frequency_embedding_size: int = 256,
        time_scale: float = 1000.0,
        max_period: float = 10000.0,
    ) -> None:
        super().__init__()
        if output_size is None:
            output_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, output_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.time_scale = float(time_scale)
        self.max_period = float(max_period)

    @staticmethod
    def timestep_embedding(
        timestep: torch.Tensor,
        dimension: int,
        max_period: float = 10000.0,
    ) -> torch.Tensor:
        half = dimension // 2
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(
                0,
                half,
                dtype=torch.float32,
                device=timestep.device,
            )
            / half
        )
        arguments = timestep[:, None].float() * frequencies[None]
        embedding = torch.cat(
            [torch.cos(arguments), torch.sin(arguments)],
            dim=-1,
        )
        if dimension % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])],
                dim=-1,
            )
        return embedding

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        scaled = timestep * self.time_scale
        frequencies = self.timestep_embedding(
            scaled,
            self.frequency_embedding_size,
            self.max_period,
        )
        return self.mlp(frequencies)


class DiTBlock(nn.Module):
    """Apply one adaptive-layer-normalized transformer block."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        **block_kwargs: object,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            **block_kwargs,
        )
        self.norm2 = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approximate_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approximate_gelu,
            drop=0.1,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(
        self,
        value: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        (
            shift_attention,
            scale_attention,
            gate_attention,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(conditioning).chunk(6, dim=1)
        value = value + gate_attention.unsqueeze(1) * self.attn(
            modulate(self.norm1(value), shift_attention, scale_attention)
        )
        value = value + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(value), shift_mlp, scale_mlp)
        )
        return value


class FinalLayer(nn.Module):
    """Project hidden DiT tokens back to latent channels."""

    def __init__(self, hidden_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(
        self,
        value: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(conditioning).chunk(2, dim=1)
        value = modulate(self.norm_final(value), shift, scale)
        return self.linear(value)


class DiT(nn.Module):
    """Unconditional diffusion transformer for HiFi-BRep latent tokens."""

    def __init__(
        self,
        in_channels: int = 16,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = False,
        latent_num: int = 128,
        use_pos: bool = True,
    ) -> None:
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.latent_num = latent_num
        self.cond_feature_dim = hidden_size
        self.x_embedder = nn.Linear(in_channels, hidden_size)
        self.use_pos = use_pos
        self.t_embedder = TimestepEmbedder(
            hidden_size,
            output_size=hidden_size,
        )

        if use_pos:
            self.pos_embed = nn.Parameter(torch.randn(latent_num, hidden_size))
            self.pos_drop = nn.Dropout(p=0.1)
            self.pos_jitter_std = 0.03
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=0.1,
                    proj_drop=0.1,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        weight = self.x_embedder.weight.data
        nn.init.xavier_uniform_(weight.view([weight.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_pos:
            if self.training and self.pos_jitter_std > 0:
                position = self.pos_embed + (
                    torch.randn_like(self.pos_embed) * self.pos_jitter_std
                )
            else:
                position = self.pos_embed
            x = self.x_embedder(x) + self.pos_drop(position)
        else:
            x = self.x_embedder(x)
        conditioning = self.t_embedder(t)

        for block in self.blocks:
            x = block(x, conditioning)

        x = self.final_layer(x, conditioning)
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=-1)
        return x
