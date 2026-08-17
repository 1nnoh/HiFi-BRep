from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence


SCHEMA_VERSION = 1
REQUIRED_SPLITS = ("train", "val", "test")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def relative_path_hash(paths: Sequence[str]) -> str:
    """Hash an ordered path list independently of JSON formatting."""
    return _sha256_bytes("".join(f"{path}\n" for path in paths).encode("utf-8"))


def _validate_relative_path(value: object, *, split: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Manifest split '{split}' contains a non-string path.")
    if "\\" in value:
        raise ValueError(f"Manifest path must use POSIX separators: {value!r}.")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"Manifest path must be relative and traversal-free: {value!r}.")
    if path.suffix != ".pkl":
        raise ValueError(f"Manifest path must identify a .pkl sample: {value!r}.")
    normalized = path.as_posix()
    if normalized != value:
        raise ValueError(f"Manifest path is not normalized: {value!r}.")
    return normalized


@dataclass(frozen=True)
class DatasetManifest:
    path: Path
    schema_version: int
    dataset: str
    processed_format: str
    splits: Mapping[str, tuple[str, ...]]
    sha256: str
    split_sha256: Mapping[str, str]

    @property
    def counts(self) -> dict[str, int]:
        return {name: len(paths) for name, paths in self.splits.items()}

    def split(self, name: str) -> tuple[str, ...]:
        try:
            return self.splits[name]
        except KeyError as exc:
            available = ", ".join(sorted(self.splits))
            raise ValueError(f"Unknown split '{name}'. Available splits: {available}.") from exc


def load_dataset_manifest(path: str | Path) -> DatasetManifest:
    manifest_path = Path(path).expanduser().resolve()
    raw_bytes = manifest_path.read_bytes()
    try:
        payload = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid dataset manifest JSON: {manifest_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Dataset manifest root must be a JSON object.")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported dataset manifest schema_version: {payload.get('schema_version')!r}."
        )
    dataset = payload.get("dataset")
    processed_format = payload.get("processed_format")
    if not isinstance(dataset, str) or not dataset:
        raise ValueError("Dataset manifest requires a non-empty 'dataset' string.")
    if not isinstance(processed_format, str) or not processed_format:
        raise ValueError("Dataset manifest requires a non-empty 'processed_format' string.")

    raw_splits = payload.get("splits")
    if not isinstance(raw_splits, dict):
        raise ValueError("Dataset manifest requires a 'splits' mapping.")
    missing = [name for name in REQUIRED_SPLITS if name not in raw_splits]
    if missing:
        raise ValueError(f"Dataset manifest is missing splits: {', '.join(missing)}.")

    splits: dict[str, tuple[str, ...]] = {}
    owners: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for split in REQUIRED_SPLITS:
        values = raw_splits[split]
        if not isinstance(values, list):
            raise ValueError(f"Manifest split '{split}' must be a JSON array.")
        paths = tuple(_validate_relative_path(value, split=split) for value in values)
        if len(set(paths)) != len(paths):
            raise ValueError(f"Manifest split '{split}' contains duplicate paths.")
        for relative_path in paths:
            previous = owners.setdefault(relative_path, split)
            if previous != split:
                raise ValueError(
                    f"Manifest splits overlap at {relative_path!r}: {previous} and {split}."
                )
        splits[split] = paths
        hashes[split] = relative_path_hash(paths)

    declared_counts = payload.get("split_counts")
    if declared_counts is not None:
        expected_counts = {name: len(splits[name]) for name in REQUIRED_SPLITS}
        if declared_counts != expected_counts:
            raise ValueError(
                f"Manifest split_counts do not match entries: {declared_counts!r} != {expected_counts!r}."
            )
    declared_hashes = payload.get("split_sha256")
    if declared_hashes is not None and declared_hashes != hashes:
        raise ValueError("Manifest split_sha256 values do not match the ordered entries.")

    return DatasetManifest(
        path=manifest_path,
        schema_version=SCHEMA_VERSION,
        dataset=dataset,
        processed_format=processed_format,
        splits=splits,
        sha256=_sha256_bytes(raw_bytes),
        split_sha256=hashes,
    )


def resolve_sample_path(data_root: str | Path, relative_path: str) -> Path:
    root = Path(data_root).expanduser().resolve()
    normalized = _validate_relative_path(relative_path, split="runtime")
    candidate = (root / Path(*PurePosixPath(normalized).parts)).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Resolved sample escapes data root: {relative_path!r}.")
    return candidate
