"""Registered training experiments (classification, DETR, …)."""
from odyssey.experiments.registry import build_learner, register_task

import odyssey.experiments.classification  # noqa: F401
import odyssey.experiments.detr  # noqa: F401

__all__ = ["build_learner", "register_task"]
