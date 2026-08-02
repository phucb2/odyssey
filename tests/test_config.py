"""Tests for Hydra/OmegaConf training configuration."""
from odyssey.paths import DEFAULT_PROJECT, default_data_root, default_gpu_cache_path, run_path
from odyssey.training.config import TrainConfig, compose_train_config


def test_compose_train_config_defaults():
    cfg = compose_train_config()
    assert isinstance(cfg, TrainConfig)
    assert cfg.mode == "train"
    assert cfg.epochs == 10
    assert cfg.compile is True
    assert cfg.project_name == DEFAULT_PROJECT
    assert cfg.image_size == 28
    assert cfg.data_root == default_data_root("fashion_mnist")
    assert cfg.data_location == default_gpu_cache_path("fashion_mnist")
    assert cfg.analysis.lr_save == str(run_path(DEFAULT_PROJECT, "lr_find.png"))


def test_compose_train_config_overrides():
    cfg = compose_train_config(["epochs=3", "compile=false"])
    assert cfg.epochs == 3
    assert cfg.compile is False


def test_compose_train_config_analysis_mode():
    cfg = compose_train_config(["mode=analysis", "analysis.lr_finder=true"])
    assert cfg.mode == "analysis"
    assert cfg.analysis.lr_finder is True


def test_compose_train_config_nested_overrides():
    cfg = compose_train_config(["loader.num_workers=0", "analysis.lr_mult=1.5"])
    assert cfg.loader.num_workers == 0
    assert cfg.analysis.lr_mult == 1.5


def test_compose_train_config_cifar_image_size():
    cfg = compose_train_config(["dataset=cifar10"])
    assert cfg.image_size == 32
    assert cfg.data_root == default_data_root("cifar10")
