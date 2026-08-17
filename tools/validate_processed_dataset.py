from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify every file referenced by a portable dataset manifest."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from src.data.manifest import load_dataset_manifest
    from src.data.validation import validate_dataset_provenance, validate_manifest_files

    manifest = load_dataset_manifest(args.manifest)
    provenance = validate_dataset_provenance(args.data_root, manifest)
    counts = validate_manifest_files(args.data_root, manifest)
    print(
        json.dumps(
            {
                "dataset": manifest.dataset,
                "manifest_sha256": manifest.sha256,
                "provenance_verified": provenance,
                "split_counts": counts,
                "split_sha256": dict(manifest.split_sha256),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
