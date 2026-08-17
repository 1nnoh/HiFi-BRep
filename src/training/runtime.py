from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch


NUMPY_RNG_STATE_FIELDS = frozenset(
    {"algorithm", "state", "position", "has_gauss", "cached_gaussian"}
)


@dataclass(frozen=True)
class TrainingSchedule:
    num_samples: int
    epochs: int
    per_device_batch_size: int
    world_size: int
    gradient_accumulation_steps: int
    global_batch_size: int
    micro_steps_per_epoch: int
    update_steps_per_epoch: int
    total_update_steps: int


def compute_training_schedule(
    *,
    num_samples: int,
    epochs: int,
    per_device_batch_size: int,
    world_size: int,
    gradient_accumulation_steps: int,
    drop_last: bool,
) -> TrainingSchedule:
    values = {
        "num_samples": num_samples,
        "epochs": epochs,
        "per_device_batch_size": per_device_batch_size,
        "world_size": world_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }
    invalid = [name for name, value in values.items() if int(value) <= 0]
    if invalid:
        raise ValueError(f"Training schedule values must be positive: {', '.join(invalid)}.")
    micro_batch = int(per_device_batch_size) * int(world_size)
    if drop_last:
        micro_steps = int(num_samples) // micro_batch
    else:
        micro_steps = math.ceil(int(num_samples) / micro_batch)
    if micro_steps <= 0:
        raise ValueError("No complete training batches remain with drop_last=True.")
    update_steps = math.ceil(micro_steps / int(gradient_accumulation_steps))
    return TrainingSchedule(
        num_samples=int(num_samples),
        epochs=int(epochs),
        per_device_batch_size=int(per_device_batch_size),
        world_size=int(world_size),
        gradient_accumulation_steps=int(gradient_accumulation_steps),
        global_batch_size=micro_batch * int(gradient_accumulation_steps),
        micro_steps_per_epoch=micro_steps,
        update_steps_per_epoch=update_steps,
        total_update_steps=update_steps * int(epochs),
    )


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def capture_rng_state(
    *,
    generators: Mapping[str, torch.Generator] | None = None,
) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "algorithm": str(numpy_state[0]),
            "state": torch.from_numpy(numpy_state[1].astype(np.int64)),
            "position": int(numpy_state[2]),
            "has_gauss": bool(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    if generators is not None:
        state["torch_generators"] = {
            str(name): generator.get_state()
            for name, generator in generators.items()
        }
    return state


def _numpy_rng_state(value: object) -> tuple[str, np.ndarray, int, int, float]:
    if not isinstance(value, Mapping) or set(value) != NUMPY_RNG_STATE_FIELDS:
        raise ValueError("NumPy RNG checkpoint schema is invalid.")
    algorithm = value.get("algorithm")
    tensor = value.get("state")
    position = value.get("position")
    has_gauss = value.get("has_gauss")
    cached_gaussian = value.get("cached_gaussian")
    if not isinstance(algorithm, str) or not algorithm:
        raise ValueError("NumPy RNG algorithm must be a non-empty string.")
    if (
        not torch.is_tensor(tensor)
        or tensor.device.type != "cpu"
        or tensor.dtype != torch.int64
        or tensor.ndim != 1
        or tensor.numel() == 0
    ):
        raise ValueError("NumPy RNG state must be a non-empty CPU int64 tensor.")
    if bool(torch.any(tensor < 0)) or bool(torch.any(tensor > 2**32 - 1)):
        raise ValueError("NumPy RNG state values must fit in uint32.")
    if (
        isinstance(position, bool)
        or not isinstance(position, int)
        or not 0 <= position <= tensor.numel()
    ):
        raise ValueError("NumPy RNG position is invalid.")
    if not isinstance(has_gauss, bool):
        raise ValueError("NumPy RNG Gaussian-cache flag must be Boolean.")
    if isinstance(cached_gaussian, bool) or not isinstance(
        cached_gaussian, (int, float)
    ):
        raise ValueError("NumPy RNG Gaussian-cache value must be numeric.")
    cached_value = float(cached_gaussian)
    if not math.isfinite(cached_value):
        raise ValueError("NumPy RNG Gaussian-cache value must be finite.")
    return (
        algorithm,
        tensor.detach().contiguous().numpy().astype(np.uint32),
        position,
        int(has_gauss),
        cached_value,
    )


def _validate_torch_rng_tensor(value: object, *, description: str) -> torch.Tensor:
    if (
        not torch.is_tensor(value)
        or value.device.type != "cpu"
        or value.dtype != torch.uint8
        or value.ndim != 1
        or value.numel() == 0
    ):
        raise ValueError(f"{description} must be a non-empty CPU uint8 tensor.")
    return value


def validate_rng_state(
    state: Mapping[str, Any],
    *,
    generators: Mapping[str, torch.Generator] | None = None,
) -> None:
    if not isinstance(state, Mapping):
        raise ValueError("RNG checkpoint must be a mapping.")
    required = {"python", "numpy", "torch_cpu"}
    allowed = required | {"torch_cuda", "torch_generators"}
    missing = sorted(required - set(state))
    unexpected = sorted(set(state) - allowed)
    if missing or unexpected:
        raise ValueError(
            f"RNG checkpoint schema is invalid: missing={missing}, "
            f"unexpected={unexpected}."
        )

    try:
        random.Random().setstate(state["python"])
    except (TypeError, ValueError) as error:
        raise ValueError("Python RNG checkpoint state is invalid.") from error
    numpy_state = _numpy_rng_state(state["numpy"])
    try:
        numpy_generator = np.random.RandomState()
        numpy_generator.set_state(numpy_state)
    except (TypeError, ValueError) as error:
        raise ValueError("NumPy RNG checkpoint state is invalid.") from error

    torch_cpu = _validate_torch_rng_tensor(
        state["torch_cpu"],
        description="Torch CPU RNG checkpoint state",
    )
    try:
        torch.Generator(device="cpu").set_state(torch_cpu)
    except RuntimeError as error:
        raise ValueError("Torch CPU RNG checkpoint state is invalid.") from error

    cuda_state = state.get("torch_cuda")
    if cuda_state is not None:
        if not isinstance(cuda_state, list) or not cuda_state:
            raise ValueError("Torch CUDA RNG checkpoint state must be a non-empty list.")
        for index, value in enumerate(cuda_state):
            _validate_torch_rng_tensor(
                value,
                description=f"Torch CUDA RNG checkpoint state {index}",
            )
        if torch.cuda.is_available() and len(cuda_state) != torch.cuda.device_count():
            raise ValueError(
                "CUDA RNG checkpoint device count does not match the current runtime."
            )

    saved_generators = state.get("torch_generators")
    if saved_generators is not None:
        if not isinstance(saved_generators, Mapping) or not all(
            isinstance(name, str) and name for name in saved_generators
        ):
            raise ValueError("Named RNG generator checkpoint state is invalid.")
        for name, value in saved_generators.items():
            _validate_torch_rng_tensor(
                value,
                description=f"Named RNG generator '{name}' checkpoint state",
            )
    if generators is not None:
        if not isinstance(saved_generators, Mapping):
            raise ValueError("RNG checkpoint has no named DataLoader generators.")
        if set(saved_generators) != set(generators):
            raise ValueError("RNG checkpoint DataLoader generator names do not match.")
        for name, generator in generators.items():
            try:
                torch.Generator(device=generator.device).set_state(
                    saved_generators[name]
                )
            except RuntimeError as error:
                raise ValueError(
                    f"Named RNG generator '{name}' checkpoint state is invalid."
                ) from error


def restore_rng_state(
    state: Mapping[str, Any],
    *,
    generators: Mapping[str, torch.Generator] | None = None,
) -> None:
    validate_rng_state(state, generators=generators)
    numpy_state = _numpy_rng_state(state["numpy"])
    random.setstate(state["python"])
    np.random.set_state(numpy_state)
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)
    if generators is not None:
        saved_generators = state.get("torch_generators")
        assert isinstance(saved_generators, Mapping)
        for name, generator in generators.items():
            generator.set_state(saved_generators[name])
