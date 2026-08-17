from __future__ import annotations

import copy
import math
from typing import Any, Mapping

import torch
from torch import nn


FORMAT_VERSION = 2


class ModelEMA:
    """EMA shadow model with the historical ema-pytorch update schedule."""

    def __init__(
        self,
        model: nn.Module,
        *,
        decay: float,
        update_every: int,
        update_after_step: int = 100,
        inv_gamma: float = 1.0,
        power: float = 2.0 / 3.0,
        min_value: float = 0.0,
    ) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1).")
        if update_every <= 0:
            raise ValueError("EMA update_every must be positive.")
        numeric_values = (decay, inv_gamma, power, min_value)
        if not all(math.isfinite(float(value)) for value in numeric_values):
            raise ValueError("EMA decay and warmup values must be finite.")
        if update_after_step < 0 or inv_gamma <= 0.0 or power <= 0.0:
            raise ValueError("EMA warmup values must be non-negative and finite.")
        if not 0.0 <= min_value <= decay:
            raise ValueError("EMA min_value must be between zero and decay.")
        self.decay = float(decay)
        self.update_every = int(update_every)
        self.update_after_step = int(update_after_step)
        self.inv_gamma = float(inv_gamma)
        self.power = float(power)
        self.min_value = float(min_value)
        self.updates = 0
        self.initialized = False
        self.model = copy.deepcopy(model).eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def _copy_from(self, online_model: nn.Module) -> None:
        online_state = online_model.state_dict()
        shadow_state = self.model.state_dict()
        if online_state.keys() != shadow_state.keys():
            raise RuntimeError("EMA and online model state dictionaries do not match.")
        for key, shadow in shadow_state.items():
            online = online_state[key].detach().to(
                device=shadow.device,
                dtype=shadow.dtype,
            )
            shadow.copy_(online)

    def current_decay(self) -> float:
        parameter = next(self.model.parameters(), None)
        if parameter is not None:
            device = parameter.device
        else:
            buffer = next(self.model.buffers(), None)
            device = buffer.device if buffer is not None else torch.device("cpu")
        step = torch.tensor(self.updates, device=device)
        epoch = (step - self.update_after_step - 1).clamp(min=0.0)
        if epoch.item() <= 0.0:
            return 0.0
        value = 1.0 - (1.0 + epoch / self.inv_gamma) ** (-self.power)
        return float(value.clamp(min=self.min_value, max=self.decay).item())

    @torch.no_grad()
    def update(self, online_model: nn.Module) -> None:
        step = self.updates
        self.updates += 1
        if not self.initialized:
            self._copy_from(online_model)
            self.initialized = True
            return
        if step % self.update_every != 0:
            return
        if step <= self.update_after_step:
            self._copy_from(online_model)
            return
        decay = self.current_decay()
        online_state = online_model.state_dict()
        shadow_state = self.model.state_dict()
        if online_state.keys() != shadow_state.keys():
            raise RuntimeError("EMA and online model state dictionaries do not match.")
        for key, shadow in shadow_state.items():
            online = online_state[key].detach().to(device=shadow.device, dtype=shadow.dtype)
            if torch.is_floating_point(shadow):
                shadow.lerp_(online, 1.0 - decay)
            else:
                shadow.copy_(online)

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "decay": self.decay,
            "update_every": self.update_every,
            "update_after_step": self.update_after_step,
            "inv_gamma": self.inv_gamma,
            "power": self.power,
            "min_value": self.min_value,
            "updates": self.updates,
            "initialized": self.initialized,
            "model": self.model.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("format_version") != FORMAT_VERSION:
            raise ValueError("Unsupported EMA checkpoint format.")
        if float(state.get("decay", -1.0)) != self.decay:
            raise ValueError("EMA decay does not match the current recipe.")
        if int(state.get("update_every", -1)) != self.update_every:
            raise ValueError("EMA update cadence does not match the current recipe.")
        settings = {
            "update_after_step": self.update_after_step,
            "inv_gamma": self.inv_gamma,
            "power": self.power,
            "min_value": self.min_value,
        }
        mismatches = {
            name: (state.get(name), expected)
            for name, expected in settings.items()
            if state.get(name) != expected
        }
        if mismatches:
            raise ValueError(f"EMA warmup settings do not match the current recipe: {mismatches!r}.")
        model_state = state.get("model")
        if not isinstance(model_state, Mapping):
            raise ValueError("EMA checkpoint has no shadow model state.")
        self.model.load_state_dict(model_state, strict=True)
        self.updates = int(state.get("updates", 0))
        self.initialized = bool(state.get("initialized", False))
