from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train HiFi-BRep latent diffusion.")
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
    parser.add_argument(
        "--vae-checkpoint",
        type=Path,
        required=True,
        help="Portable full-VAE checkpoint used to encode posterior samples.",
    )
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--resume", type=Path, help="Strictly resume a full training checkpoint.")
    start.add_argument("--init-from", type=Path, help="Initialize only compatible DiT weights.")
    parser.add_argument("--init-state", choices=("online", "ema"), default="online")
    parser.add_argument(
        "--max-train-steps",
        type=int,
        help="Optional update limit without changing the configured recipe.",
    )
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        help="Stop at this completed epoch without shortening the configured scheduler.",
    )
    parser.add_argument(
        "--max-additional-train-steps",
        type=int,
        help="Stop after N additional updates without changing the resumed scheduler.",
    )
    parser.add_argument("--per-device-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--precision", choices=("no", "fp16", "bf16"))
    parser.add_argument("--optimizer-type", choices=("adam", "adamw"))
    parser.add_argument("--weight-decay", type=float)
    return parser


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.stop_after_epoch is not None:
        if args.stop_after_epoch <= 0:
            raise ValueError("--stop-after-epoch must be positive.")
        if args.max_train_steps is not None or args.max_additional_train_steps is not None:
            raise ValueError(
                "--stop-after-epoch is mutually exclusive with update-step limits."
            )
    if args.weight_decay is not None and args.weight_decay < 0:
        raise ValueError("--weight-decay must be non-negative.")
    if args.max_additional_train_steps is not None:
        if args.resume is None:
            raise ValueError("--max-additional-train-steps requires --resume.")
        if args.max_additional_train_steps <= 0:
            raise ValueError("--max-additional-train-steps must be positive.")
        if args.max_train_steps is not None:
            raise ValueError(
                "--max-train-steps and --max-additional-train-steps are mutually exclusive."
            )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = validate_args(build_parser().parse_args(argv))
    if args.data_root is None:
        raise SystemExit("--data-root or HIFI_BREP_DATA_ROOT is required.")
    if args.output_dir is None:
        raise SystemExit("--output-dir or HIFI_BREP_CHECKPOINT_ROOT is required.")
    from src.training.entrypoints import run_diffusion_training

    run_diffusion_training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
