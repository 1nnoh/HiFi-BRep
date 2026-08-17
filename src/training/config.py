from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.flow.config import validate_unconditional_dit_config


FORMAT_VERSION = 1
VAE_STATES = ("online", "ema")
WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


def _require_mapping(values: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = values.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Training recipe field '{key}' must be a mapping.")
    return dict(value)


def _find_absolute_strings(value: object, *, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            found.extend(_find_absolute_strings(child, prefix=child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_find_absolute_strings(child, prefix=f"{prefix}[{index}]"))
    elif isinstance(value, str):
        if Path(value).is_absolute() or WINDOWS_ABSOLUTE.match(value):
            found.append(prefix)
    return found


@dataclass(frozen=True)
class TrainingRecipe:
    path: Path
    repository_root: Path
    values: Mapping[str, Any]
    sha256: str
    task: str
    variant: str
    manifest_path: Path

    def section(self, name: str) -> dict[str, Any]:
        return _require_mapping(self.values, name)


def load_training_recipe(
    path: str | Path,
    *,
    expected_task: str,
    repository_root: str | Path,
) -> TrainingRecipe:
    recipe_path = Path(path).expanduser().resolve()
    raw_bytes = recipe_path.read_bytes()
    values = yaml.safe_load(raw_bytes)
    if not isinstance(values, dict):
        raise ValueError("Training recipe root must be a mapping.")
    if values.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported training recipe format_version: {values.get('format_version')!r}."
        )
    task = values.get("task")
    if task != expected_task:
        raise ValueError(f"Expected a '{expected_task}' recipe, received {task!r}.")
    recipe_variant = values.get("variant")
    if not isinstance(recipe_variant, str) or not recipe_variant:
        raise ValueError("Training recipe variant must be a non-empty string.")
    absolute_fields = _find_absolute_strings(values)
    if absolute_fields:
        raise ValueError(
            "Training recipes cannot contain machine absolute paths: "
            + ", ".join(absolute_fields)
            + "."
        )
    for section in ("data", "training", "model", "optimizer"):
        _require_mapping(values, section)
    if task == "diffusion":
        validate_unconditional_dit_config(
            _require_mapping(values, "model"),
            description="Diffusion model config",
        )
    data = _require_mapping(values, "data")
    manifest = data.get("manifest")
    if not isinstance(manifest, str) or not manifest:
        raise ValueError("Training recipe data.manifest must be a relative path.")
    root = Path(repository_root).expanduser().resolve()
    manifest_path = (root / manifest).resolve()
    if not manifest_path.is_relative_to(root):
        raise ValueError("Training recipe manifest resolves outside the repository.")
    if task == "diffusion":
        vae_source = values.get("vae_source")
        if not isinstance(vae_source, dict) or set(vae_source) != {
            "variant",
            "state",
        }:
            raise ValueError(
                "Diffusion recipe vae_source must contain exactly variant and state."
            )
        vae_variant = vae_source.get("variant")
        state = vae_source.get("state")
        if not isinstance(vae_variant, str) or not vae_variant:
            raise ValueError("Diffusion recipe VAE variant must be a non-empty string.")
        if state not in VAE_STATES:
            raise ValueError("Diffusion recipe VAE state must be 'online' or 'ema'.")
        if vae_variant != recipe_variant:
            raise ValueError(
                "Diffusion recipe vae_source.variant must match the root variant."
            )
        release_artifact = values.get("release_artifact")
        if release_artifact is not None:
            if not isinstance(release_artifact, dict) or set(release_artifact) != {
                "epoch",
                "state",
            }:
                raise ValueError(
                    "Diffusion recipe release_artifact must contain exactly epoch and state."
                )
            release_epoch = release_artifact.get("epoch")
            training = _require_mapping(values, "training")
            if (
                isinstance(release_epoch, bool)
                or not isinstance(release_epoch, int)
                or release_epoch <= 0
                or release_epoch > int(training.get("epochs", 0))
            ):
                raise ValueError(
                    "Diffusion recipe release_artifact.epoch must be within training epochs."
                )
            validation = _require_mapping(values, "validation")
            if release_artifact.get("state") != validation.get("state"):
                raise ValueError(
                    "Diffusion recipe release_artifact.state must match validation.state."
                )
    return TrainingRecipe(
        path=recipe_path,
        repository_root=root,
        values=values,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        task=task,
        variant=recipe_variant,
        manifest_path=manifest_path,
    )
