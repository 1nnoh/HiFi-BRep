from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml


def _read_yaml(path: Path) -> dict[str, Any]:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if values is None:
        return {}
    if not isinstance(values, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return dict(values)


def load_config(
    path: str | Path,
    default_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load a YAML config with optional inheritance and environment expansion."""
    config_path = Path(path).expanduser().resolve()
    selected = _read_yaml(config_path)
    inherited = selected.get("inherit_from")
    if inherited is not None:
        if not isinstance(inherited, str) or not inherited:
            raise ValueError("inherit_from must be a non-empty path string.")
        inherited_path = Path(inherited).expanduser()
        if not inherited_path.is_absolute():
            inherited_path = config_path.parent / inherited_path
        config = load_config(inherited_path, default_path)
    elif default_path is not None:
        config = _read_yaml(Path(default_path).expanduser().resolve())
    else:
        config = {}
    update_recursive(config, selected)
    return expand_env_in_config(config)


def expand_env_in_config(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: expand_env_in_config(child) for key, child in value.items()}
    if isinstance(value, list):
        return [expand_env_in_config(child) for child in value]
    if isinstance(value, tuple):
        return tuple(expand_env_in_config(child) for child in value)
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    return value


def update_recursive(
    target: dict[str, Any],
    updates: Mapping[str, Any],
) -> None:
    """Update a nested mapping in place."""
    for key, value in updates.items():
        if isinstance(value, Mapping):
            child = target.get(key)
            if not isinstance(child, dict):
                child = {}
                target[key] = child
            update_recursive(child, value)
        else:
            target[key] = value
