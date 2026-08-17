from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import torch.distributed as distributed
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, DistributedDataParallelKwargs
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.training.checkpoint import (
    TrainingArtifactSpec,
    TrainingCheckpoint,
    save_training_artifacts,
)
from src.training.early_stopping import EarlyStoppingController
from src.training.ema import ModelEMA
from src.training.runtime import (
    capture_rng_state,
    compute_training_schedule,
    restore_rng_state,
    seed_everything,
    seed_worker,
)


def global_loader_batch_size(*, per_device_batch_size: int, world_size: int) -> int:
    if per_device_batch_size <= 0 or world_size <= 0:
        raise ValueError("Per-device batch size and world size must be positive.")
    return int(per_device_batch_size) * int(world_size)


def optimizer_update_succeeded(accelerator: object) -> bool:
    return bool(
        getattr(accelerator, "sync_gradients")
        and not getattr(accelerator, "optimizer_step_was_skipped")
    )


def validation_selection(
    config: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    metric = config.get("selection_metric")
    mode = config.get("selection_mode")
    if metric is None and mode is None:
        return None, None
    if not isinstance(metric, str) or not metric:
        raise ValueError("validation.selection_metric must be a non-empty string.")
    if mode not in ("min", "max"):
        raise ValueError("validation.selection_mode must be 'min' or 'max'.")
    return metric, str(mode)


def validation_improved(*, value: float, best: float | None, mode: str) -> bool:
    if not math.isfinite(value):
        raise FloatingPointError("Validation selection metric is non-finite.")
    if mode not in ("min", "max"):
        raise ValueError("Validation selection mode must be 'min' or 'max'.")
    return best is None or (value < best if mode == "min" else value > best)


def scheduled_checkpoint_names(
    epoch: int,
    config: Mapping[str, Any],
) -> tuple[str, ...]:
    if epoch <= 0:
        raise ValueError("Checkpoint epoch must be positive and one-based.")
    if "latest_every_epochs" not in config and "milestone_every_epochs" not in config:
        legacy_every = int(config.get("every_epochs", 0))
        if legacy_every <= 0:
            raise ValueError("Checkpoint cadence must be positive.")
        if epoch % legacy_every == 0:
            return (f"checkpoint-epoch-{epoch:04d}.pt",)
        return ()
    latest_every = int(config.get("latest_every_epochs", 0))
    milestone_every = int(config.get("milestone_every_epochs", 0))
    if latest_every <= 0 or milestone_every <= 0:
        raise ValueError("Checkpoint latest and milestone cadences must be positive.")
    names: list[str] = []
    if epoch % latest_every == 0:
        names.append("checkpoint-latest.pt")
    if epoch % milestone_every == 0:
        names.append(f"checkpoint-epoch-{epoch:04d}.pt")
    return tuple(names)


def create_tensorboard_writer(
    output_dir: Path,
    *,
    is_main_process: bool,
    writer_factory: Callable[..., object] | None = None,
) -> object | None:
    if not is_main_process:
        return None
    if writer_factory is None:
        from torch.utils.tensorboard import SummaryWriter

        writer_factory = SummaryWriter
    return writer_factory(log_dir=str(output_dir / "tensorboard"))


def write_train_tensorboard(writer: object, record: Mapping[str, object]) -> None:
    step = int(record["update_step"])
    loss_field = "loss" if "loss" in record else "loss_total"
    tags = {
        "train/loss": loss_field,
        "train/learning_rate": "learning_rate",
        "train/gradient_norm": "gradient_norm",
        "execution/samples_per_second": "samples_per_second",
        "execution/allocated_memory_gib": "allocated_memory_gib",
        "execution/reserved_memory_gib": "reserved_memory_gib",
    }
    for tag, field in tags.items():
        writer.add_scalar(tag, float(record[field]), step)


def write_validation_tensorboard(
    writer: object,
    record: Mapping[str, object],
    *,
    best_loss: float,
) -> None:
    step = int(record["update_step"])
    state = str(record["state"])
    loss_field = "loss" if "loss" in record else "loss_total"
    writer.add_scalar(
        f"validation/loss_{state}",
        float(record[loss_field]),
        step,
    )
    writer.add_scalar("validation/best_loss", float(best_loss), step)


def synchronize_stop_decision(accelerator: Accelerator, *, should_stop: bool) -> bool:
    flag = torch.tensor(
        int(should_stop),
        device=accelerator.device,
        dtype=torch.uint8,
    )
    if accelerator.num_processes > 1:
        distributed.broadcast(flag, src=0)
    return bool(flag.item())


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_source_compatibility(
    *,
    recipe_sha256: str,
    effective_config_sha256: str,
    manifest_sha256: str,
    git_state: Mapping[str, object],
) -> dict[str, object]:
    commit = git_state.get("commit")
    dirty = git_state.get("dirty")
    available = isinstance(commit, str) and bool(commit) and isinstance(dirty, bool)
    unavailable = commit is None and dirty is None
    if not available and not unavailable:
        raise ValueError(
            "Git state must be a commit/dirty pair or an explicit null pair."
        )
    return {
        "recipe_sha256": recipe_sha256,
        "effective_config_sha256": effective_config_sha256,
        "manifest_sha256": manifest_sha256,
        "git_commit": commit,
        "git_dirty": dirty,
    }


def apply_cli_overrides(values: Mapping[str, Any], args: object) -> dict[str, Any]:
    effective = deepcopy(dict(values))
    training = effective["training"]
    overrides = {
        "per_device_batch_size": getattr(args, "per_device_batch_size", None),
        "gradient_accumulation_steps": getattr(args, "gradient_accumulation_steps", None),
        "num_workers": getattr(args, "num_workers", None),
        "precision": getattr(args, "precision", None),
    }
    for key, value in overrides.items():
        if value is not None:
            training[key] = value
    optimizer = effective["optimizer"]
    optimizer_overrides = {
        "type": getattr(args, "optimizer_type", None),
        "weight_decay": getattr(args, "weight_decay", None),
    }
    for key, value in optimizer_overrides.items():
        if value is not None:
            optimizer[key] = value
    return effective


def _normalize_precision(value: object) -> str:
    if value is False:
        return "no"
    precision = str(value)
    if precision not in ("no", "fp16", "bf16"):
        raise ValueError("training.precision must be one of no, fp16, or bf16.")
    return precision


def _cosine_multiplier(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    initial_factor: float,
    final_factor: float,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        progress = step / max(warmup_steps, 1)
        return initial_factor + (1.0 - initial_factor) * progress
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return final_factor + (1.0 - final_factor) * cosine


def build_optimizer(
    parameters: object,
    optimizer_config: Mapping[str, Any],
) -> torch.optim.Optimizer:
    optimizer_type = str(optimizer_config.get("type", "adamw")).lower()
    if optimizer_type not in ("adam", "adamw"):
        raise ValueError("optimizer.type must be 'adam' or 'adamw'.")
    weight_decay = float(
        optimizer_config.get("weight_decay", 0.0 if optimizer_type == "adam" else 0.01)
    )
    if weight_decay < 0:
        raise ValueError("optimizer.weight_decay must be non-negative.")
    betas = optimizer_config.get("betas", [0.9, 0.99])
    kwargs = {
        "lr": float(optimizer_config["learning_rate"]),
        "weight_decay": weight_decay,
        "betas": (float(betas[0]), float(betas[1])),
        "eps": float(optimizer_config.get("eps", 1e-8)),
    }
    optimizer_class = torch.optim.Adam if optimizer_type == "adam" else torch.optim.AdamW
    return optimizer_class(parameters, **kwargs)


def resolve_warmup_steps(
    scheduler_config: Mapping[str, Any],
    *,
    total_steps: int,
    update_steps_per_epoch: int,
) -> int:
    fields = tuple(
        field
        for field in ("warmup_epochs", "warmup_steps", "warmup_fraction")
        if field in scheduler_config and scheduler_config[field] is not None
    )
    if len(fields) > 1:
        raise ValueError(
            "scheduler warmup_epochs, warmup_steps, and warmup_fraction are mutually exclusive."
        )
    if total_steps <= 0 or update_steps_per_epoch <= 0:
        raise ValueError("Scheduler step counts must be positive.")
    if not fields:
        warmup_fraction = 0.1
        return int(total_steps * warmup_fraction)
    field = fields[0]
    if field == "warmup_epochs":
        value = scheduler_config[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("scheduler.warmup_epochs must be a non-negative integer.")
        return int(value) * int(update_steps_per_epoch)
    if field == "warmup_steps":
        value = scheduler_config[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("scheduler.warmup_steps must be a non-negative integer.")
        return int(value)
    warmup_fraction = float(scheduler_config[field])
    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError("scheduler.warmup_fraction must be in [0, 1).")
    return int(total_steps * warmup_fraction)


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    scheduler_config: Mapping[str, Any],
) -> LambdaLR:
    initial_factor = float(scheduler_config.get("initial_factor", 0.01))
    final_factor = float(scheduler_config.get("final_factor", 0.01))
    return LambdaLR(
        optimizer,
        lr_lambda=lambda step: _cosine_multiplier(
            step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            initial_factor=initial_factor,
            final_factor=final_factor,
        ),
    )


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, path)
    finally:
        temporary_path = Path(temporary_name)
        if temporary_path.exists():
            temporary_path.unlink()


def _git_state(repository_root: Path) -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository_root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def _environment_summary() -> dict[str, object]:
    packages = {}
    for name in ("accelerate", "diffusers", "numpy", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "packages": packages,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpu_models": sorted(
            {torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())}
        ),
    }


def _dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> tuple[DataLoader, torch.Generator]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs: dict[str, Any] = {}
    if num_workers > 0:
        kwargs.update(prefetch_factor=2, persistent_workers=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
        **kwargs,
    )
    return loader, generator


def _gather_rng_states(
    accelerator: Accelerator,
    *,
    train_generator: torch.Generator,
) -> list[dict[str, Any]]:
    local_state = capture_rng_state(
        generators={"train_dataloader": train_generator}
    )
    if accelerator.num_processes == 1:
        return [local_state]
    states: list[dict[str, Any] | None] = [None] * accelerator.num_processes
    distributed.all_gather_object(states, local_state)
    if any(state is None for state in states):
        raise RuntimeError("Failed to gather RNG state from every training rank.")
    return [state for state in states if state is not None]


def _batch_size(batch: Mapping[str, Any]) -> int:
    for value in batch.values():
        if torch.is_tensor(value) and value.ndim > 0:
            return int(value.shape[0])
        if isinstance(value, (list, tuple)):
            return len(value)
    raise ValueError("Cannot infer batch size from the training batch.")


def accumulate_weighted_logs(
    sums: dict[str, torch.Tensor],
    logs: Mapping[str, torch.Tensor],
    *,
    batch_size: int,
    device: torch.device,
) -> None:
    if batch_size <= 0:
        raise ValueError("Training log batch size must be positive.")
    for name, value in logs.items():
        if not torch.is_tensor(value) or value.numel() != 1:
            raise ValueError(f"Training log '{name}' must be a scalar tensor.")
        scalar = value.detach().to(device=device, dtype=torch.float64)
        sums[name] = sums.get(
            name, torch.zeros((), device=device, dtype=torch.float64)
        ) + scalar * batch_size


def normalize_weighted_logs(
    sums: Mapping[str, torch.Tensor],
    *,
    sample_count: int,
) -> dict[str, float]:
    if sample_count <= 0:
        raise ValueError("Training log sample count must be positive.")
    return {
        name: float((value / sample_count).item())
        for name, value in sums.items()
    }


@torch.no_grad()
def _validate(
    *,
    accelerator: Accelerator,
    model: nn.Module,
    task: object,
    dataloader: DataLoader,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    sums: dict[str, torch.Tensor] = {}
    count = torch.zeros((), device=accelerator.device, dtype=torch.long)
    for batch_index, batch in enumerate(dataloader):
        batch_count = _batch_size(batch)
        with accelerator.autocast():
            _, logs = task.loss(model, batch, training=False)
        for name, value in logs.items():
            scalar = value.detach().to(device=accelerator.device, dtype=torch.float64)
            sums[name] = sums.get(
                name, torch.zeros((), device=accelerator.device, dtype=torch.float64)
            ) + scalar * batch_count
        count += batch_count
        if max_batches > 0 and batch_index + 1 >= max_batches:
            break
    count = accelerator.reduce(count, reduction="sum")
    results: dict[str, float] = {}
    for name, value in sums.items():
        total = accelerator.reduce(value, reduction="sum")
        results[name] = float((total / count.clamp_min(1)).item())
    return results


def run_deterministic_validation(
    *,
    accelerator: Accelerator,
    model: nn.Module,
    task: object,
    dataloader: DataLoader,
    max_batches: int,
    deterministic_seed: int,
    generator: torch.Generator,
) -> dict[str, float]:
    validation_rng = capture_rng_state(
        generators={"validation_dataloader": generator}
    )
    try:
        rank_seed = int(deterministic_seed) + int(accelerator.process_index)
        seed_everything(rank_seed)
        generator.manual_seed(rank_seed)
        return _validate(
            accelerator=accelerator,
            model=model,
            task=task,
            dataloader=dataloader,
            max_batches=max_batches,
        )
    finally:
        restore_rng_state(
            validation_rng,
            generators={"validation_dataloader": generator},
        )


def run_training_loop(
    *,
    repository_root: Path,
    effective_config: Mapping[str, Any],
    recipe_sha256: str,
    manifest_sha256: str,
    train_dataset: Dataset,
    val_dataset: Dataset,
    model: nn.Module,
    task: object,
    output_dir: Path,
    resume_path: Path | None,
    max_train_steps: int | None,
    checkpoint_provenance: Mapping[str, Any],
    max_additional_train_steps: int | None = None,
    stop_after_epoch: int | None = None,
    artifact_spec: TrainingArtifactSpec | None = None,
) -> dict[str, Any]:
    execution_stop_epoch = stop_after_epoch
    if max_train_steps is not None and max_train_steps <= 0:
        raise ValueError("Maximum train steps must be positive.")
    if max_additional_train_steps is not None:
        if resume_path is None:
            raise ValueError("Additional train steps require a resume checkpoint.")
        if max_additional_train_steps <= 0:
            raise ValueError("Additional train steps must be positive.")
        if max_train_steps is not None:
            raise ValueError("Absolute and additional train-step limits are exclusive.")
    training = dict(effective_config["training"])
    optimizer_config = dict(effective_config["optimizer"])
    scheduler_config = dict(effective_config.get("scheduler", {}))
    ema_config = dict(effective_config.get("ema", {}))
    validation_config = dict(effective_config.get("validation", {}))
    early_stopping_config = dict(effective_config.get("early_stopping", {}))
    checkpoint_config = dict(effective_config.get("checkpoint", {}))
    selection_metric, selection_mode = validation_selection(validation_config)
    early_stopping: EarlyStoppingController | None = None
    if selection_metric is not None and selection_mode is not None:
        early_stopping = EarlyStoppingController.from_config(
            early_stopping_config,
            selection_metric=selection_metric,
            selection_mode=selection_mode,
        )
    elif bool(early_stopping_config.get("enabled", False)):
        raise ValueError(
            "Early stopping requires validation selection_metric and selection_mode."
        )
    precision = _normalize_precision(training.get("precision", "no"))
    gradient_accumulation = int(training.get("gradient_accumulation_steps", 1))
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation,
        mixed_precision=precision,
        step_scheduler_with_optimizer=False,
        dataloader_config=DataLoaderConfiguration(
            split_batches=True,
            even_batches=False,
        ),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
    )
    seed = int(training.get("seed", 42))
    seed_everything(seed + accelerator.process_index)
    per_device_batch_size = int(training["per_device_batch_size"])
    num_workers = int(training.get("num_workers", 4))
    pin_memory = bool(training.get("pin_memory", True))
    drop_last = bool(training.get("drop_last", True))
    epochs = int(training["epochs"])
    if execution_stop_epoch is not None:
        if execution_stop_epoch <= 0 or execution_stop_epoch > epochs:
            raise ValueError(
                "stop_after_epoch must be positive and no greater than training.epochs."
            )
        if max_train_steps is not None or max_additional_train_steps is not None:
            raise ValueError("Epoch and update-step execution limits are exclusive.")
    schedule = compute_training_schedule(
        num_samples=len(train_dataset),
        epochs=epochs,
        per_device_batch_size=per_device_batch_size,
        world_size=accelerator.num_processes,
        gradient_accumulation_steps=gradient_accumulation,
        drop_last=drop_last,
    )
    total_steps = schedule.total_update_steps
    if max_train_steps is not None:
        total_steps = min(total_steps, int(max_train_steps))

    train_loader, train_generator = _dataloader(
        train_dataset,
        batch_size=global_loader_batch_size(
            per_device_batch_size=per_device_batch_size,
            world_size=accelerator.num_processes,
        ),
        shuffle=True,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
    )
    val_loader, val_generator = _dataloader(
        val_dataset,
        batch_size=global_loader_batch_size(
            per_device_batch_size=per_device_batch_size,
            world_size=accelerator.num_processes,
        ),
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
    )
    optimizer = build_optimizer(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        optimizer_config,
    )
    warmup_steps = resolve_warmup_steps(
        scheduler_config,
        total_steps=total_steps,
        update_steps_per_epoch=schedule.update_steps_per_epoch,
    )
    lr_scheduler = _make_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        scheduler_config=scheduler_config,
    )
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        val_loader,
        lr_scheduler,
    )
    if len(train_loader) != schedule.micro_steps_per_epoch:
        raise RuntimeError(
            "Prepared DataLoader length disagrees with the recorded epoch schedule: "
            f"{len(train_loader)} != {schedule.micro_steps_per_epoch}."
        )
    if hasattr(task, "vae"):
        task.vae.to(accelerator.device)
    unwrapped_model = accelerator.unwrap_model(model)
    ema = ModelEMA(
        unwrapped_model,
        decay=float(ema_config.get("decay", 0.995)),
        update_every=int(ema_config.get("update_every", 10)),
        update_after_step=int(ema_config.get("update_after_step", 100)),
        inv_gamma=float(ema_config.get("inv_gamma", 1.0)),
        power=float(ema_config.get("power", 2.0 / 3.0)),
        min_value=float(ema_config.get("min_value", 0.0)),
    )
    ema.model.to(accelerator.device)

    effective_sha256 = canonical_sha256(effective_config)
    git_state = _git_state(repository_root)
    compatibility = {
        **build_source_compatibility(
            recipe_sha256=recipe_sha256,
            effective_config_sha256=effective_sha256,
            manifest_sha256=manifest_sha256,
            git_state=git_state,
        ),
        "task": effective_config["task"],
        "world_size": accelerator.num_processes,
        "global_batch_size": schedule.global_batch_size,
        "precision": precision,
        "total_update_steps": total_steps,
        "checkpoint_provenance": dict(checkpoint_provenance),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
    }
    output_dir = output_dir.expanduser().resolve()
    existing = output_dir / "run_config.json"
    output_error: str | None = None
    if accelerator.is_main_process:
        if resume_path is None and output_dir.exists() and any(output_dir.iterdir()):
            output_error = f"Output directory is not empty: {output_dir}"
        elif resume_path is not None and not existing.is_file():
            output_error = "Resume output directory has no run_config.json."
        elif resume_path is not None:
            previous = json.loads(existing.read_text(encoding="utf-8"))
            if previous.get("effective_config_sha256") != effective_sha256:
                output_error = "Existing run_config.json does not match the resume recipe."
    output_errors = [output_error]
    if accelerator.num_processes > 1:
        distributed.broadcast_object_list(output_errors, src=0)
    if output_errors[0] is not None:
        raise RuntimeError(output_errors[0])

    start_epoch = 0
    update_step = 0
    best_validation: dict[str, object] | None = None
    if resume_path is not None:
        resumed = TrainingCheckpoint.load_for_resume(
            resume_path,
            model=unwrapped_model,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            scaler=accelerator.scaler,
            expected_metadata=compatibility,
            rng_rank=accelerator.process_index,
            rng_generators={"train_dataloader": train_generator},
        )
        if resumed.get("ema") is None:
            raise ValueError("Training checkpoint has no EMA state.")
        ema.load_state_dict(resumed["ema"])
        start_epoch = int(resumed["epoch"])
        update_step = int(resumed["step"])
        restored_best = resumed["metadata"].get("best_validation")
        if restored_best is not None:
            if not isinstance(restored_best, Mapping):
                raise ValueError("Training checkpoint best_validation metadata is invalid.")
            if (
                restored_best.get("metric") != selection_metric
                or restored_best.get("mode") != selection_mode
                or not isinstance(restored_best.get("value"), (int, float))
            ):
                raise ValueError(
                    "Training checkpoint best_validation metadata does not match the recipe."
                )
            best_validation = dict(restored_best)
        if early_stopping is not None:
            restored_early_stopping = resumed["metadata"].get("early_stopping")
            if not isinstance(restored_early_stopping, Mapping):
                raise ValueError(
                    "Training checkpoint has no valid early-stopping state."
                )
            early_stopping.load_state_dict(restored_early_stopping)
            if best_validation is not None and (
                early_stopping.best_value != float(best_validation["value"])
                or early_stopping.best_epoch != int(best_validation["epoch"])
            ):
                raise ValueError(
                    "Checkpoint best validation and early-stopping state disagree."
                )

    run_config = {
        "format_version": 1,
        "recipe_sha256": recipe_sha256,
        "effective_config_sha256": effective_sha256,
        "manifest_sha256": manifest_sha256,
        "config": effective_config,
        "git": git_state,
        "environment": _environment_summary(),
        "execution": {
            "world_size": accelerator.num_processes,
            "per_device_batch_size": per_device_batch_size,
            "global_batch_size": schedule.global_batch_size,
            "gradient_accumulation_steps": gradient_accumulation,
            "micro_steps_per_epoch": schedule.micro_steps_per_epoch,
            "update_steps_per_epoch": schedule.update_steps_per_epoch,
            "total_update_steps": total_steps,
            "warmup_steps": warmup_steps,
            "precision": precision,
            "pin_memory": pin_memory,
            "seed": seed,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
        },
        "checkpoint_provenance": checkpoint_provenance,
    }
    if max_additional_train_steps is not None:
        run_config["execution"]["max_additional_train_steps"] = int(
            max_additional_train_steps
        )
    if execution_stop_epoch is not None:
        run_config["execution"]["stop_after_epoch"] = int(execution_stop_epoch)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        if resume_path is None:
            _atomic_json(existing, run_config)
    accelerator.wait_for_everyone()

    writer: object | None = None
    metrics_path = output_dir / "metrics.jsonl"
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    log_every = int(training.get("log_every_updates", 10))
    validate_every = int(validation_config.get("every_epochs", 1))
    validation_batches = int(validation_config.get("max_batches", -1))
    validation_seed = int(validation_config.get("deterministic_seed", seed))
    stopped_mid_epoch = False
    early_stopped = False
    completed_epochs = start_epoch
    pending_log_sums: dict[str, torch.Tensor] = {}
    pending_log_samples = 0
    performance_samples = 0
    performance_started_at = time.perf_counter()
    if accelerator.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(accelerator.device)

    def save_checkpoints(
        filenames: list[str],
        *,
        next_epoch: int,
        resumable: bool,
    ) -> None:
        filenames = list(dict.fromkeys(filenames))
        if not filenames:
            return
        accelerator.wait_for_everyone()
        rng_states = _gather_rng_states(
            accelerator,
            train_generator=train_generator,
        )
        metadata = {
            **compatibility,
            "resume_boundary": resumable,
            "best_validation": best_validation,
            "early_stopping": (
                early_stopping.state_dict() if early_stopping is not None else None
            ),
        }
        save_status: list[str | None] = [None]
        if accelerator.is_main_process:
            try:
                TrainingCheckpoint.save_many(
                    [output_dir / filename for filename in filenames],
                    model=unwrapped_model,
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    scaler=accelerator.scaler,
                    ema_state=ema.state_dict(),
                    step=update_step,
                    epoch=next_epoch,
                    metadata=metadata,
                    rng_state={"per_rank": rng_states},
                )
                if artifact_spec is not None and "checkpoint-best.pt" in filenames:
                    selected_model = (
                        ema.model
                        if artifact_spec.state == "ema"
                        else unwrapped_model
                    )
                    selected_vae_state = None
                    if artifact_spec.task == "diffusion":
                        vae = getattr(task, "vae", None)
                        if not isinstance(vae, nn.Module):
                            raise ValueError(
                                "Diffusion artifact export requires the training VAE."
                            )
                        selected_vae_state = vae.state_dict()
                    save_training_artifacts(
                        artifact_spec,
                        model_state=selected_model.state_dict(),
                        vae_state=selected_vae_state,
                    )
            except Exception as error:
                save_status[0] = f"{type(error).__name__}: {error}"
        if accelerator.num_processes > 1:
            distributed.broadcast_object_list(save_status, src=0)
        if save_status[0] is not None:
            raise RuntimeError(f"Checkpoint save failed: {save_status[0]}")
        accelerator.wait_for_everyone()

    writer = create_tensorboard_writer(
        output_dir,
        is_main_process=accelerator.is_main_process,
    )

    initial_update_step = update_step
    additional_stop_step = (
        update_step + int(max_additional_train_steps)
        if max_additional_train_steps is not None
        else None
    )

    def reached_step_limit() -> bool:
        return bool(
            (max_train_steps is not None and update_step >= max_train_steps)
            or (
                additional_stop_step is not None
                and update_step >= additional_stop_step
            )
        )

    try:
        if reached_step_limit():
            return {
                "update_step": update_step,
                "completed_epochs": start_epoch,
                "stopped_mid_epoch": False,
                "early_stopped": False,
                "output_dir": str(output_dir),
            }

        for epoch in range(start_epoch, epochs):
            if hasattr(train_loader, "set_epoch"):
                train_loader.set_epoch(epoch)
            model.train()
            progress = tqdm(
                total=len(train_loader),
                desc=f"Epoch {epoch + 1}/{epochs}",
                disable=not accelerator.is_local_main_process,
            )
            batches_seen = 0
            for batch in train_loader:
                batches_seen += 1
                with accelerator.accumulate(model):
                    with accelerator.autocast():
                        loss, logs = task.loss(model, batch, training=True)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Non-finite training loss at update {update_step}, epoch {epoch}."
                        )
                    batch_count = _batch_size(batch)
                    accumulate_weighted_logs(
                        pending_log_sums,
                        logs,
                        batch_size=batch_count,
                        device=accelerator.device,
                    )
                    pending_log_samples += batch_count
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        gradient_norm = accelerator.clip_grad_norm_(
                            model.parameters(), max_grad_norm
                        )
                        if (
                            not torch.isfinite(gradient_norm)
                            and accelerator.scaler is None
                        ):
                            raise FloatingPointError(
                                "Non-finite gradient norm at update "
                                f"{update_step}, epoch {epoch}."
                            )
                    optimizer.step()
                    update_succeeded = optimizer_update_succeeded(accelerator)
                    if update_succeeded:
                        lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                update_logs: dict[str, float] = {}
                global_sample_count = 0
                if accelerator.sync_gradients:
                    if update_succeeded:
                        global_sample_count = int(
                            accelerator.reduce(
                                torch.tensor(
                                    pending_log_samples,
                                    device=accelerator.device,
                                    dtype=torch.long,
                                ),
                                reduction="sum",
                            ).item()
                        )
                        global_log_sums = {
                            name: accelerator.reduce(value, reduction="sum")
                            for name, value in pending_log_sums.items()
                        }
                        update_logs = normalize_weighted_logs(
                            global_log_sums,
                            sample_count=global_sample_count,
                        )
                    pending_log_sums = {}
                    pending_log_samples = 0
                if update_succeeded:
                    update_step += 1
                    performance_samples += global_sample_count
                    ema.update(unwrapped_model)
                    should_log = (
                        update_step == 1
                        or update_step == initial_update_step + 1
                        or (
                            max_train_steps is not None
                            and update_step == max_train_steps
                        )
                        or (
                            additional_stop_step is not None
                            and update_step == additional_stop_step
                        )
                        or update_step % max(log_every, 1) == 0
                    )
                    if should_log:
                        if accelerator.device.type == "cuda":
                            torch.cuda.synchronize(accelerator.device)
                            allocated = torch.cuda.max_memory_allocated(
                                accelerator.device
                            )
                            reserved = torch.cuda.max_memory_reserved(
                                accelerator.device
                            )
                        else:
                            allocated = 0
                            reserved = 0
                        elapsed = time.perf_counter() - performance_started_at
                        performance = torch.tensor(
                            [elapsed, float(allocated), float(reserved)],
                            device=accelerator.device,
                            dtype=torch.float64,
                        )
                        performance = accelerator.reduce(performance, reduction="max")
                        elapsed = max(float(performance[0].item()), 1e-12)
                        record = {
                            "type": "train",
                            "epoch": epoch + 1,
                            "update_step": update_step,
                            "learning_rate": float(lr_scheduler.get_last_lr()[0]),
                            "gradient_norm": float(gradient_norm.detach().cpu()),
                            "samples_per_second": performance_samples / elapsed,
                            "allocated_memory_gib": float(performance[1].item()) / 2**30,
                            "reserved_memory_gib": float(performance[2].item()) / 2**30,
                            **update_logs,
                        }
                        if accelerator.is_main_process:
                            with metrics_path.open("a", encoding="utf-8") as stream:
                                stream.write(json.dumps(record, sort_keys=True) + "\n")
                            if writer is not None:
                                write_train_tensorboard(writer, record)
                                writer.flush()
                        performance_samples = 0
                        performance_started_at = time.perf_counter()
                        if accelerator.device.type == "cuda":
                            torch.cuda.reset_peak_memory_stats(accelerator.device)
                    if reached_step_limit():
                        stopped_mid_epoch = batches_seen < len(train_loader)
                        break
                progress.update(1)
            progress.close()

            completed_epoch = not stopped_mid_epoch
            if completed_epoch:
                completed_epochs = epoch + 1
            stop_after_epoch = False
            improved = False
            if completed_epoch and (epoch + 1) % max(validate_every, 1) == 0:
                validation_state = str(validation_config.get("state", "ema"))
                evaluation_model = (
                    ema.model if validation_state == "ema" else unwrapped_model
                )
                results = run_deterministic_validation(
                    accelerator=accelerator,
                    model=evaluation_model,
                    task=task,
                    dataloader=val_loader,
                    max_batches=validation_batches,
                    deterministic_seed=validation_seed,
                    generator=val_generator,
                )
                local_stop = False
                if selection_metric is not None:
                    if selection_metric not in results:
                        raise KeyError(
                            "Configured validation selection metric is absent: "
                            f"{selection_metric}."
                        )
                    if early_stopping is None:
                        raise RuntimeError("Validation selection has no state controller.")
                    selected_value = float(results[selection_metric])
                    improved, local_stop = early_stopping.update(
                        epoch=epoch + 1,
                        value=selected_value,
                    )
                    if improved:
                        best_validation = {
                            "metric": selection_metric,
                            "mode": selection_mode,
                            "value": selected_value,
                            "state": validation_state,
                            "epoch": epoch + 1,
                            "update_step": update_step,
                        }
                stop_after_epoch = synchronize_stop_decision(
                    accelerator,
                    should_stop=local_stop,
                )
                record = {
                    "type": "validation",
                    "state": validation_state,
                    "epoch": epoch + 1,
                    "update_step": update_step,
                    "improved": improved,
                    "early_stop": stop_after_epoch,
                    "early_stopping_non_improvement_count": (
                        early_stopping.non_improvement_count
                        if early_stopping is not None
                        else None
                    ),
                    **results,
                }
                if accelerator.is_main_process:
                    with metrics_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, sort_keys=True) + "\n")
                    if writer is not None and best_validation is not None:
                        write_validation_tensorboard(
                            writer,
                            record,
                            best_loss=float(best_validation["value"]),
                        )
                        writer.flush()
            checkpoint_targets: list[str] = []
            if improved:
                checkpoint_targets.append("checkpoint-best.pt")
            if completed_epoch:
                checkpoint_targets.extend(
                    scheduled_checkpoint_names(epoch + 1, checkpoint_config)
                )
                if (
                    (epoch + 1 == epochs or reached_step_limit())
                    and not checkpoint_targets
                    and "every_epochs" in checkpoint_config
                ):
                    checkpoint_targets.append(f"checkpoint-epoch-{epoch + 1:04d}.pt")
            if stop_after_epoch:
                early_stopped = True
                checkpoint_targets.append("checkpoint-early-stop.pt")
            reached_epoch_limit = bool(
                completed_epoch
                and execution_stop_epoch is not None
                and epoch + 1 >= execution_stop_epoch
            )
            if reached_epoch_limit and not checkpoint_targets:
                checkpoint_targets.append(f"checkpoint-epoch-{epoch + 1:04d}.pt")
            save_checkpoints(
                checkpoint_targets,
                next_epoch=epoch + 1,
                resumable=True,
            )
            if stop_after_epoch:
                break
            if reached_epoch_limit:
                break
            if reached_step_limit():
                if stopped_mid_epoch:
                    save_checkpoints(
                        ["checkpoint-step-limit-nonresumable.pt"],
                        next_epoch=epoch,
                        resumable=False,
                    )
                break

        return {
            "update_step": update_step,
            "completed_epochs": completed_epochs,
            "stopped_mid_epoch": stopped_mid_epoch,
            "early_stopped": early_stopped,
            "stopped_after_epoch": bool(
                execution_stop_epoch is not None
                and completed_epochs >= execution_stop_epoch
            ),
            "output_dir": str(output_dir),
        }
    finally:
        if writer is not None:
            writer.flush()
            writer.close()
        accelerator.end_training()
