from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from eval.protocol import (
    aggregate_samples,
    atomic_write_json,
    build_batch_plan,
    compute_protocol_fingerprint,
    load_evaluation_execution,
    load_evaluation_protocol,
    prepare_run_directory,
    validate_batch_payload,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVALUATION_CONFIG = REPOSITORY_ROOT / "configs" / "evaluation.yaml"
DEFAULT_MODEL_CONFIG = REPOSITORY_ROOT / "configs" / "demo.yaml"
EVALUATION_SCHEMA_VERSION = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate unconditional HiFi-BRep generation.",
    )
    parser.add_argument(
        "--variant",
        required=True,
        help="Variant name defined by --model-config.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--evaluation-config",
        type=Path,
        default=DEFAULT_EVALUATION_CONFIG,
    )
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--eta", type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--save-steps",
        choices=("none", "valid", "all"),
    )
    parser.add_argument(
        "--reconstruction-mode",
        choices=("legacy", "batched"),
    )
    parser.add_argument("--reconstruction-workers", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def _runtime_descriptor(torch: Any, device: Any) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(device)
    try:
        diffusers_version = importlib.metadata.version("diffusers")
    except importlib.metadata.PackageNotFoundError:
        diffusers_version = "unknown"
    try:
        import OCC

        occ_version = str(getattr(OCC, "VERSION", "unknown"))
    except ImportError:
        occ_version = "unknown"
    return {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "diffusers_version": diffusers_version,
        "pythonocc_version": occ_version,
        "gpu_name": properties.name,
        "gpu_total_memory_bytes": int(properties.total_memory),
        "gpu_compute_capability": list(torch.cuda.get_device_capability(device)),
    }


def _sampling_payload(sampling: Any) -> dict[str, object]:
    return {
        "scheduler": "DDIM",
        "num_train_timesteps": sampling.num_train_timesteps,
        "num_inference_steps": sampling.num_inference_steps,
        "beta_start": sampling.beta_start,
        "beta_end": sampling.beta_end,
        "beta_schedule": sampling.beta_schedule,
        "prediction_type": sampling.prediction_type,
        "variance_type": sampling.variance_type,
        "eta": sampling.eta,
    }


def _source_revision() -> dict[str, object]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "working_tree_dirty": None}
    return {
        "git_commit": revision,
        "working_tree_dirty": bool(status.strip()),
    }


def _atomic_write_jsonl(
    path: Path,
    records: Sequence[Mapping[str, object]],
) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary_path.replace(path)


def _batch_path(output_dir: Path, batch_index: int) -> Path:
    return output_dir / "batches" / f"batch_{batch_index:05d}.json"


def _load_existing_batches(
    output_dir: Path,
    *,
    batches: Sequence[Any],
    protocol_fingerprint: str,
) -> tuple[list[dict[str, object]], list[Any]]:
    expected_paths = {_batch_path(output_dir, batch.index) for batch in batches}
    actual_paths = set((output_dir / "batches").glob("batch_*.json"))
    unexpected = sorted(actual_paths - expected_paths)
    if unexpected:
        names = ", ".join(path.name for path in unexpected)
        raise ValueError(f"Run directory contains unexpected batch files: {names}")

    completed: list[dict[str, object]] = []
    pending: list[Any] = []
    for batch in batches:
        path = _batch_path(output_dir, batch.index)
        if not path.is_file():
            pending.append(batch)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol_fingerprint") != protocol_fingerprint:
            raise ValueError(
                f"Completed batch {batch.index} has a different protocol fingerprint."
            )
        validate_batch_payload(payload, batch)
        completed.append(payload)
    return completed, pending


def _make_sample_records(
    reconstruction: Any,
    *,
    batch: Any,
    output_dir: Path,
    step_output_dir: Path,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for local_index, sample in enumerate(reconstruction.samples):
        if int(sample["index"]) != local_index:
            raise RuntimeError("Reconstruction returned unexpected local sample indices.")
        step_file = sample["step_file"]
        relative_step = None
        if step_file is not None:
            relative_step = (
                step_output_dir.relative_to(output_dir) / str(step_file)
            ).as_posix()
        records.append(
            {
                "index": batch.sample_start + local_index,
                "batch_index": batch.index,
                "batch_sample_index": local_index,
                "batch_seed": batch.seed,
                "num_faces": int(sample["num_faces"]),
                "num_edges": int(sample["num_edges"]),
                "in_face_range": bool(sample["in_face_range"]),
                "closed_solid": bool(sample["valid"]),
                "failure_reason": sample["failure_reason"],
                "compilable": sample.get("compilable"),
                "step_file": relative_step,
            }
        )
    if len(records) != batch.sample_count:
        raise RuntimeError(
            f"Batch {batch.index} reconstructed {len(records)} samples; "
            f"expected {batch.sample_count}."
        )
    return records


def _validate_batch_reconstruction(
    records: Sequence[Mapping[str, object]],
    stats: Mapping[str, int],
    *,
    save_step_policy: str,
) -> None:
    closed_count = sum(bool(record["closed_solid"]) for record in records)
    failure_count = sum(
        int(stats[key])
        for key in (
            "fails_empty_after_filter",
            "fails_recon_exception",
            "fails_not_closed_or_zero_volume",
        )
    )
    if closed_count + failure_count != len(records):
        raise RuntimeError(
            "Closed-solid and reconstruction failure counts do not cover the batch."
        )
    compilability = [record.get("compilable") for record in records]
    if save_step_policy == "all":
        if not all(isinstance(value, bool) for value in compilability):
            raise RuntimeError(
                "save_steps=all must classify Compilability for every sample."
            )
    elif any(value is not None for value in compilability):
        raise RuntimeError(
            f"save_steps={save_step_policy} cannot form a Compilability measurement."
        )


def _build_summary(
    run_config: Mapping[str, object],
    samples: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    metrics = aggregate_samples(
        samples,
        requested_count=int(run_config["protocol"]["num_samples"]),  # type: ignore[index]
        save_step_policy=str(run_config["protocol"]["save_steps"]),  # type: ignore[index]
    )
    checkpoints = run_config.get("checkpoints")
    protocol = run_config.get("protocol")
    if not isinstance(checkpoints, Mapping) or not isinstance(protocol, Mapping):
        raise ValueError("Generation run configuration schema is invalid.")
    decoder = checkpoints.get("vae")
    diffusion = checkpoints.get("diffusion")
    if not isinstance(decoder, Mapping) or not isinstance(diffusion, Mapping):
        raise ValueError("Generation checkpoint identities are invalid.")
    release_metadata = diffusion.get("release_metadata")
    if not isinstance(release_metadata, Mapping):
        raise ValueError("Diffusion checkpoint identity has no release metadata.")
    weight_state = release_metadata.get("weight_state")
    state = str(weight_state).removesuffix("/model")
    if state not in ("online", "ema"):
        raise ValueError("Diffusion checkpoint identity has an invalid weight state.")
    decoder_filename = decoder.get("filename")
    diffusion_filename = diffusion.get("filename")
    if not all(
        isinstance(filename, str)
        and filename
        and Path(filename).name == filename
        for filename in (decoder_filename, diffusion_filename)
    ):
        raise ValueError("Generation checkpoint filenames must be basenames.")
    protocol_fields = (
        "num_samples",
        "batch_size",
        "base_seed",
        "num_inference_steps",
        "eta",
        "dtype",
        "reconstruction_criterion",
        "save_steps",
    )
    missing_protocol = [field for field in protocol_fields if field not in protocol]
    if missing_protocol:
        raise ValueError(
            "Generation protocol is missing: " + ", ".join(missing_protocol) + "."
        )
    return {
        "format_version": 1,
        "variant": run_config["variant"],
        "face_range": run_config["face_range"],
        "checkpoints": {
            "decoder": {
                "filename": decoder_filename,
                "state": state,
            },
            "diffusion": {
                "filename": diffusion_filename,
                "state": state,
            },
        },
        "protocol": {
            field: protocol[field]
            for field in protocol_fields
        },
        "metrics": {
            "in_face_range_count": metrics["in_face_range_count"],
            "closed_solid_count": metrics["closed_solid_count"],
            "qualified_count": metrics["qualified_count"],
            "qualified_rate": metrics["qualified_rate"],
            "failure_counts": metrics["failure_counts"],
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        official_protocol = load_evaluation_protocol(args.evaluation_config)
        protocol = load_evaluation_protocol(
            args.evaluation_config,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            seed=args.seed,
            num_inference_steps=args.steps,
            eta=args.eta,
            save_steps=args.save_steps,
        )
        execution = load_evaluation_execution(
            args.evaluation_config,
            reconstruction_mode=args.reconstruction_mode,
            reconstruction_workers=args.reconstruction_workers,
        )
        from src.inference.config import load_demo_config

        model_config = load_demo_config(
            args.model_config,
            args.variant,
            checkpoint_dir=args.checkpoint_dir,
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    import torch

    try:
        device = torch.device(args.device)
    except RuntimeError as exc:
        parser.error(str(exc))
    if device.type != "cuda":
        parser.error("Generation evaluation requires a CUDA device.")
    if not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available.")
    if device.index is not None:
        torch.cuda.set_device(device)

    from src.inference.generator import (
        DemoGenerator,
        SamplingConfig,
        inspect_release_checkpoint,
    )

    face_range = (model_config.face_min, model_config.face_max)
    try:
        checkpoint_identities = {
            "vae": inspect_release_checkpoint(
                model_config.vae_checkpoint,
                expected_variant=model_config.variant,
                expected_face_range=face_range,
                expected_component="vae_decoder",
                expected_weight_state="decoder.*",
            ),
            "diffusion": inspect_release_checkpoint(
                model_config.diffusion_checkpoint,
                expected_variant=model_config.variant,
                expected_face_range=face_range,
                expected_component="diffusion",
                expected_weight_state=model_config.diffusion_state,
            ),
        }
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    sampling = SamplingConfig.from_mapping(
        model_config.sampling,
        num_inference_steps=protocol.num_inference_steps,
        eta=protocol.eta,
    )
    batches = build_batch_plan(protocol)
    runtime = _runtime_descriptor(torch, device)
    fingerprint_payload = {
        "format_version": 1,
        "implementation": {
            "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
            **_source_revision(),
        },
        "variant": model_config.variant,
        "face_range": [model_config.face_min, model_config.face_max],
        "checkpoints": checkpoint_identities,
        "protocol": protocol.to_dict(),
        "execution": execution.to_dict(),
        "sampling": _sampling_payload(sampling),
        "architecture": {
            "latent_shape": list(model_config.latent_shape),
            "model": model_config.model,
            "decoder": model_config.decoder,
        },
        "runtime": runtime,
    }
    protocol_fingerprint = compute_protocol_fingerprint(fingerprint_payload)
    run_config = {
        **fingerprint_payload,
        "protocol_fingerprint": protocol_fingerprint,
        "official_protocol": protocol == official_protocol,
        "batching": {
            "num_batches": len(batches),
            "seed_rule": "base_seed + batch_index",
        },
    }
    try:
        output_dir = prepare_run_directory(
            args.output_dir,
            run_config,
            resume=args.resume,
        )
        (output_dir / "batches").mkdir(exist_ok=True)
        completed_payloads, pending_batches = _load_existing_batches(
            output_dir,
            batches=batches,
            protocol_fingerprint=protocol_fingerprint,
        )
    except (FileNotFoundError, NotADirectoryError, FileExistsError, ValueError) as exc:
        parser.error(str(exc))

    print(
        f"Variant {model_config.variant}: {protocol.num_samples} samples, "
        f"batch size {protocol.batch_size}, {protocol.num_inference_steps}-step DDIM"
    )
    print(
        "Checkpoints: "
        f"{checkpoint_identities['vae']['filename']} + "
        f"{checkpoint_identities['diffusion']['filename']}"
    )
    print(
        "Reconstruction: "
        f"{execution.reconstruction_mode}, "
        f"workers={execution.reconstruction_workers}"
    )
    print(f"Output: {output_dir}")

    generator = None
    reconstruction_runner = None
    if pending_batches:
        generator = DemoGenerator(model_config, device=device, dtype=torch.float32)
        if generator.checkpoint_identities != checkpoint_identities:
            raise RuntimeError(
                "Checkpoint identity changed between run configuration and strict load."
            )
        from src.inference.reconstruction import ReconstructionRunner

        reconstruction_runner = ReconstructionRunner(
            mode=execution.reconstruction_mode,
            workers=execution.reconstruction_workers,
            device=device,
        )

    batch_payloads = list(completed_payloads)
    try:
        for batch in pending_batches:
            assert generator is not None
            assert reconstruction_runner is not None
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            generation_started = time.perf_counter()
            latent, predictions = generator.generate(
                num_samples=batch.sample_count,
                seed=batch.seed,
                sampling=sampling,
                show_progress=not args.no_progress,
            )
            torch.cuda.synchronize(device)
            generation_seconds = time.perf_counter() - generation_started
            expected_latent_shape = (batch.sample_count, *model_config.latent_shape)
            if tuple(latent.shape) != expected_latent_shape:
                raise RuntimeError(
                    f"Batch {batch.index} latent shape is {tuple(latent.shape)}; "
                    f"expected {expected_latent_shape}."
                )
            if latent.dtype != torch.float32 or latent.device.type != "cuda":
                raise RuntimeError(
                    f"Batch {batch.index} latent must remain CUDA float32."
                )
            if not torch.isfinite(latent).all():
                raise RuntimeError(f"Batch {batch.index} contains non-finite latents.")

            from src.inference.reconstruction import validate_decoder_predictions

            decoder_schema = validate_decoder_predictions(
                predictions,
                max_faces=int(model_config.decoder["max_face"]),
            )
            step_output_dir = output_dir / "steps" / f"batch_{batch.index:05d}"
            torch.cuda.synchronize(device)
            reconstruction_started = time.perf_counter()
            reconstruction = reconstruction_runner.reconstruct(
                predictions,
                face_min=model_config.face_min,
                face_max=model_config.face_max,
                output_dir=step_output_dir,
                save_step_policy=protocol.save_steps,
            )
            torch.cuda.synchronize(device)
            reconstruction_seconds = time.perf_counter() - reconstruction_started
            records = _make_sample_records(
                reconstruction,
                batch=batch,
                output_dir=output_dir,
                step_output_dir=step_output_dir,
            )
            _validate_batch_reconstruction(
                records,
                reconstruction.stats,
                save_step_policy=protocol.save_steps,
            )
            batch_payload = {
                "format_version": 1,
                "protocol_fingerprint": protocol_fingerprint,
                "batch": batch.to_dict(),
                "latent_shape": list(latent.shape),
                "decoder_schema": decoder_schema,
                "reconstruction_stats": reconstruction.stats,
                "timing_seconds": {
                    "generation": generation_seconds,
                    "reconstruction": reconstruction_seconds,
                    "reconstruction_preparation": reconstruction.timing_seconds.get(
                        "preparation"
                    ),
                    "reconstruction_gpu_optimization": (
                        reconstruction.timing_seconds.get("gpu_optimization")
                    ),
                    "reconstruction_finalization": reconstruction.timing_seconds.get(
                        "finalization"
                    ),
                },
                "cuda_memory_bytes": {
                    "peak_allocated": int(torch.cuda.max_memory_allocated(device)),
                    "peak_reserved": int(torch.cuda.max_memory_reserved(device)),
                },
                "samples": records,
            }
            atomic_write_json(_batch_path(output_dir, batch.index), batch_payload)
            batch_payloads.append(batch_payload)
            print(
                f"Batch {batch.index + 1}/{len(batches)} complete: "
                f"samples {batch.sample_start}-{batch.sample_start + batch.sample_count - 1}"
            )
            del latent, predictions, reconstruction
            gc.collect()
    finally:
        if reconstruction_runner is not None:
            reconstruction_runner.close()

    batch_payloads.sort(key=lambda payload: int(payload["batch"]["index"]))  # type: ignore[index]
    samples = [
        sample
        for payload in batch_payloads
        for sample in validate_batch_payload(
            payload,
            batches[int(payload["batch"]["index"])],  # type: ignore[index]
        )
    ]
    summary = _build_summary(run_config, samples)
    _atomic_write_jsonl(output_dir / "samples.jsonl", samples)
    atomic_write_json(output_dir / "summary.json", summary)
    metrics = summary["metrics"]
    print(
        f"Complete: qualified_rate={metrics['qualified_rate']:.6f}, "
        f"closed_solid_count={metrics['closed_solid_count']}"
    )
    print(f"Summary: {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
