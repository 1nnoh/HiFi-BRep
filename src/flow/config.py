from __future__ import annotations

from typing import Any, Mapping


DIT_CONFIG_FIELDS = frozenset(
    {
        "in_channels",
        "hidden_size",
        "depth",
        "num_heads",
        "mlp_ratio",
        "learn_sigma",
        "latent_num",
        "use_pos",
    }
)


def validate_unconditional_dit_config(
    values: Mapping[str, Any],
    *,
    description: str,
) -> dict[str, Any]:
    """Reject unsupported or conditioned DiT configuration fields."""
    unknown = sorted(set(values) - DIT_CONFIG_FIELDS)
    if unknown:
        raise ValueError(
            f"{description} contains unsupported fields: {', '.join(unknown)}."
        )
    return dict(values)
