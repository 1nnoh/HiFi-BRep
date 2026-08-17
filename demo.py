from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "demo.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate unconditional B-rep samples with a released HiFi-BRep model.",
    )
    parser.add_argument(
        "--variant",
        required=True,
        help="Variant name defined by --config.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Demo model and sampling configuration.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Directory containing the configured VAE and diffusion checkpoint pair.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for STEP files and manifest.json.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of shapes to sample.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for DDIM sampling.")
    parser.add_argument("--steps", type=int, help="Override the configured DDIM step count.")
    parser.add_argument("--eta", type=float, help="Override the configured DDIM eta.")
    parser.add_argument(
        "--device",
        default="cuda",
        help="CUDA device, for example cuda or cuda:1.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the DDIM progress bar.",
    )
    return parser


def _default_output_dir(variant: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return REPOSITORY_ROOT / "outputs" / "demo" / f"{variant}-{timestamp}"


def _prepare_output_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(f"Output path is not a directory: {path}")
        if any(path.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {path}. Choose a new directory."
            )
    path.mkdir(parents=True, exist_ok=True)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be positive")
    if args.eta is not None and args.eta < 0:
        parser.error("--eta must be non-negative")

    from src.inference.config import load_demo_config

    try:
        config = load_demo_config(
            args.config,
            args.variant,
            checkpoint_dir=args.checkpoint_dir,
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    import torch

    from src.inference.generator import DemoGenerator, SamplingConfig
    from src.inference.reconstruction import (
        reconstruct_predictions,
        validate_decoder_predictions,
    )

    try:
        device = torch.device(args.device)
    except RuntimeError as exc:
        parser.error(str(exc))
    if device.type != "cuda":
        parser.error("The Demo requires a CUDA device for STEP reconstruction.")
    if not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available.")
    sampling = SamplingConfig.from_mapping(
        config.sampling,
        num_inference_steps=args.steps,
        eta=args.eta,
    )

    try:
        generator = DemoGenerator(config, device=device, dtype=torch.float32)
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    output_dir = _prepare_output_dir(
        args.output_dir or _default_output_dir(args.variant)
    )

    print(f"Variant: {config.variant} faces={config.face_min}-{config.face_max}")
    print(f"VAE checkpoint: {config.vae_checkpoint}")
    print(
        f"Diffusion checkpoint: {config.diffusion_checkpoint} "
        f"[{config.diffusion_state}]"
    )
    print(f"Output directory: {output_dir}")

    latent, predictions = generator.generate(
        num_samples=args.num_samples,
        seed=args.seed,
        sampling=sampling,
        show_progress=not args.no_progress,
    )
    if tuple(latent.shape[1:]) != config.latent_shape:
        raise RuntimeError(
            f"Unexpected latent shape {tuple(latent.shape)}; "
            f"expected [batch, {config.latent_shape[0]}, {config.latent_shape[1]}]."
        )
    if not torch.isfinite(latent).all():
        raise RuntimeError("Diffusion produced non-finite latent values.")
    decoder_schema = validate_decoder_predictions(
        predictions,
        max_faces=int(config.decoder["max_face"]),
    )
    reconstruction = reconstruct_predictions(
        predictions,
        face_min=config.face_min,
        face_max=config.face_max,
        output_dir=output_dir,
    )

    manifest = {
        "variant": config.variant,
        "face_range": [config.face_min, config.face_max],
        "checkpoints": generator.checkpoint_identities,
        "sampling": {
            "seed": args.seed,
            "num_samples": args.num_samples,
            "num_train_timesteps": sampling.num_train_timesteps,
            "num_inference_steps": sampling.num_inference_steps,
            "eta": sampling.eta,
            "scheduler": "DDIM",
            "prediction_type": sampling.prediction_type,
        },
        "runtime": {
            "device": str(device),
            "dtype": "float32",
            "torch_version": torch.__version__,
        },
        "latent_shape": list(latent.shape),
        "decoder_schema": decoder_schema,
        "validity_in_face_range": reconstruction.validity_in_range,
        "reconstruction_stats": reconstruction.stats,
        "samples": reconstruction.samples,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
