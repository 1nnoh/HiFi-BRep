from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the HiFi-BRep VAE.")
    parser.add_argument("--config", type=Path, required=True, help="Portable YAML recipe.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=os.environ.get("HIFI_BREP_DATA_ROOT"),
        help="Processed dataset root (or HIFI_BREP_DATA_ROOT).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=os.environ.get("HIFI_BREP_CHECKPOINT_ROOT"),
        help="New run directory (or HIFI_BREP_CHECKPOINT_ROOT).",
    )
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--resume", type=Path, help="Strictly resume a full training checkpoint.")
    start.add_argument("--init-from", type=Path, help="Initialize only compatible model weights.")
    parser.add_argument("--init-state", choices=("online", "ema"), default="online")
    parser.add_argument(
        "--max-train-steps",
        type=int,
        help="Optional update limit without changing the configured recipe.",
    )
    parser.add_argument("--per-device-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--precision", choices=("no", "fp16", "bf16"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.data_root is None:
        raise SystemExit("--data-root or HIFI_BREP_DATA_ROOT is required.")
    if args.output_dir is None:
        raise SystemExit("--output-dir or HIFI_BREP_CHECKPOINT_ROOT is required.")
    from src.training.entrypoints import run_vae_training

    run_vae_training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
