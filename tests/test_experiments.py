"""Tests for experiment registry."""
from odyssey.experiments.registry import build_learner
from odyssey.training.config import compose_train_config
from odyssey.training.learner import Learner


def test_build_classification_learner():
    cfg = compose_train_config(["epochs=1", "compile=false", "loader.num_workers=0"])
    learn = build_learner(cfg, plot_progress=False)
    assert isinstance(learn, Learner)


def test_build_detr_learner():
    cfg = compose_train_config([
        "task=detr",
        "dataset=coco128",
        "batch_size=2",
        "compile=false",
        "loader.num_workers=0",
    ])
    learn = build_learner(cfg, plot_progress=False)
    assert isinstance(learn, Learner)
    assert hasattr(learn.model, "num_queries")
