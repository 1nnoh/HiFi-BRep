from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert STEP solids to the HiFi-BRep processed PKL representation."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--layout", choices=("abc", "relative"), default="abc")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-face", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N sorted files.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from src.preprocessing.pipeline import run_preprocessing

    summary = run_preprocessing(
        input_root=args.input_root,
        output_root=args.output_root,
        layout=args.layout,
        workers=args.workers,
        max_face=args.max_face,
        resume=args.resume,
        limit=args.limit,
    )
    print(summary["counts"])
    return 1 if summary["counts"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
