from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from src.data.manifest import DatasetManifest, resolve_sample_path


def validate_dataset_provenance(
    data_root: str | Path,
    manifest: DatasetManifest,
) -> bool:
    root = Path(data_root).expanduser().resolve()
    path = root / "DATASET_PROVENANCE.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Dataset provenance is not valid JSON.") from exc
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("Dataset provenance uses an unsupported format version.")
    if payload.get("dataset") != manifest.dataset:
        raise ValueError("Dataset provenance dataset does not match the manifest.")
    manifest_record = payload.get("manifest")
    if not isinstance(manifest_record, dict) or manifest_record.get("sha256") != manifest.sha256:
        raise ValueError("Dataset provenance manifest SHA256 does not match.")
    return True


def validate_split_access(
    data_root: str | Path,
    manifest: DatasetManifest,
    *,
    split_names: Sequence[str],
) -> None:
    root = Path(data_root).expanduser().resolve()
    for split_name in split_names:
        paths = manifest.split(split_name)
        if not paths:
            raise ValueError(f"Dataset split '{split_name}' is empty.")
        representatives = (paths[0], paths[-1]) if len(paths) > 1 else paths
        for relative_path in representatives:
            if not resolve_sample_path(root, relative_path).is_file():
                raise FileNotFoundError(
                    f"Processed dataset cannot resolve manifest sample: {relative_path}"
                )


def validate_manifest_files(
    data_root: str | Path,
    manifest: DatasetManifest,
) -> dict[str, int]:
    root = Path(data_root).expanduser().resolve()
    counts: dict[str, int] = {}
    for split_name, paths in manifest.splits.items():
        for relative_path in paths:
            if not resolve_sample_path(root, relative_path).is_file():
                raise FileNotFoundError(
                    f"Processed dataset is missing manifest sample: {relative_path}"
                )
        counts[split_name] = len(paths)
    return counts
