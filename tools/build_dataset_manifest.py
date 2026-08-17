from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Sequence

from src.data.manifest import relative_path_hash


def _load_split(path: Path, *, source_prefix: PurePosixPath) -> list[str]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list):
        raise ValueError(f"Source split must be a JSON array: {path}")
    relative_paths: list[str] = []
    prefix_parts = source_prefix.parts
    for value in values:
        if not isinstance(value, str):
            raise ValueError(f"Source split contains a non-string entry: {path}")
        source = PurePosixPath(value)
        if source.parts[: len(prefix_parts)] != prefix_parts:
            raise ValueError(f"Source path does not start with {source_prefix}: {value}")
        relative = PurePosixPath(*source.parts[len(prefix_parts) :])
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".pkl":
            raise ValueError(f"Unsafe or unsupported source path: {value}")
        if len(relative.parts) != 2:
            raise ValueError(f"Expected chunk/sample.pkl layout, received: {relative}")
        relative_paths.append(relative.as_posix())
    if len(set(relative_paths)) != len(relative_paths):
        raise ValueError(f"Source split contains duplicate entries: {path}")
    return relative_paths


def build_manifest(
    *,
    dataset: str,
    source_prefix: str,
    split_paths: dict[str, Path | None],
) -> dict[str, object]:
    prefix = PurePosixPath(source_prefix)
    splits = {
        name: _load_split(path, source_prefix=prefix) if path is not None else []
        for name, path in split_paths.items()
    }
    owners: dict[str, str] = {}
    for split, paths in splits.items():
        for relative_path in paths:
            previous = owners.setdefault(relative_path, split)
            if previous != split:
                raise ValueError(
                    f"Source splits overlap at {relative_path}: {previous} and {split}."
                )
    payload: dict[str, object] = {
        "schema_version": 1,
        "dataset": dataset,
        "processed_format": "hifi_brep_bspline_v1",
        "split_counts": {name: len(paths) for name, paths in splits.items()},
        "split_sha256": {
            name: relative_path_hash(paths) for name, paths in splits.items()
        },
        "splits": splits,
    }
    return payload


def _atomic_json_write(payload: object, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.chmod(temporary_name, 0o644)
        os.replace(temporary_name, output_path)
    finally:
        temporary_path = Path(temporary_name)
        if temporary_path.exists():
            temporary_path.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert source path lists to a portable manifest.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-prefix", required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Output manifest already exists: {args.output}")
    payload = build_manifest(
        dataset=args.dataset,
        source_prefix=args.source_prefix,
        split_paths={"train": args.train, "val": args.val, "test": args.test},
    )
    _atomic_json_write(payload, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "counts": payload["split_counts"],
                "split_sha256": payload["split_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
