from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


@dataclass
class EarlyStoppingController:
    enabled: bool
    metric: str
    mode: str
    min_epochs: int
    patience: int
    min_delta: float
    best_value: float | None = None
    best_epoch: int | None = None
    non_improvement_count: int = 0

    def __post_init__(self) -> None:
        if not self.metric:
            raise ValueError("Early-stopping metric must be a non-empty string.")
        if self.mode not in ("min", "max"):
            raise ValueError("Early-stopping mode must be 'min' or 'max'.")
        if self.min_epochs <= 0:
            raise ValueError("Early-stopping min_epochs must be positive.")
        if self.patience <= 0:
            raise ValueError("Early-stopping patience must be positive.")
        if not math.isfinite(self.min_delta) or self.min_delta < 0.0:
            raise ValueError("Early-stopping min_delta must be finite and non-negative.")

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
        *,
        selection_metric: str,
        selection_mode: str,
    ) -> "EarlyStoppingController":
        metric = str(config.get("metric", selection_metric))
        mode = str(config.get("mode", selection_mode))
        if metric != selection_metric or mode != selection_mode:
            raise ValueError(
                "Early-stopping metric and mode must match validation selection."
            )
        return cls(
            enabled=bool(config.get("enabled", False)),
            metric=metric,
            mode=mode,
            min_epochs=int(config.get("min_epochs", 1)),
            patience=int(config.get("patience", 1)),
            min_delta=float(config.get("min_delta", 0.0)),
        )

    def _improved(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "min":
            return value < self.best_value - self.min_delta
        return value > self.best_value + self.min_delta

    def update(self, *, epoch: int, value: float) -> tuple[bool, bool]:
        if epoch <= 0:
            raise ValueError("Early-stopping epoch must be positive and one-based.")
        if not math.isfinite(value):
            raise FloatingPointError("Early-stopping metric is non-finite.")
        improved = self._improved(float(value))
        if improved:
            self.best_value = float(value)
            self.best_epoch = int(epoch)
            self.non_improvement_count = 0
        elif epoch >= self.min_epochs:
            self.non_improvement_count += 1
        should_stop = bool(
            self.enabled
            and epoch >= self.min_epochs
            and self.non_improvement_count >= self.patience
        )
        return improved, should_stop

    def state_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "metric": self.metric,
            "mode": self.mode,
            "min_epochs": self.min_epochs,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "non_improvement_count": self.non_improvement_count,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        expected = {
            "enabled": self.enabled,
            "metric": self.metric,
            "mode": self.mode,
            "min_epochs": self.min_epochs,
            "patience": self.patience,
            "min_delta": self.min_delta,
        }
        mismatches = {
            key: (state.get(key), value)
            for key, value in expected.items()
            if state.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"Early-stopping checkpoint configuration mismatch: {mismatches!r}."
            )
        best_value = state.get("best_value")
        best_epoch = state.get("best_epoch")
        count = state.get("non_improvement_count")
        if best_value is not None and (
            not isinstance(best_value, (int, float))
            or not math.isfinite(float(best_value))
        ):
            raise ValueError("Early-stopping best_value is invalid.")
        if best_epoch is not None and (
            not isinstance(best_epoch, int) or best_epoch <= 0
        ):
            raise ValueError("Early-stopping best_epoch is invalid.")
        if not isinstance(count, int) or count < 0:
            raise ValueError("Early-stopping non_improvement_count is invalid.")
        if (best_value is None) != (best_epoch is None):
            raise ValueError("Early-stopping best value and epoch must be restored together.")
        self.best_value = None if best_value is None else float(best_value)
        self.best_epoch = best_epoch
        self.non_improvement_count = count
