"""Experiment task registry."""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from odyssey.training.config import TrainConfig
    from odyssey.training.learner import Learner

_BUILDERS: dict[str, Callable[..., "Learner"]] = {}


def register_task(name: str):
    def decorator(fn: Callable[..., "Learner"]):
        _BUILDERS[name] = fn
        return fn

    return decorator


def build_learner(cfg: "TrainConfig", *, plot_progress: bool = True) -> "Learner":
    if cfg.task not in _BUILDERS:
        raise ValueError(f"Unknown task {cfg.task!r}; registered: {tuple(_BUILDERS)}")
    return _BUILDERS[cfg.task](cfg, plot_progress=plot_progress)
