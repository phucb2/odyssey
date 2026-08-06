"""Training configuration (OmegaConf structured config + Hydra CLI)."""
from dataclasses import dataclass, field, replace
from typing import Sequence

from hydra import compose, initialize_config_module
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from odyssey.data.loaders import LoaderConfig
from odyssey.paths import (
    DEFAULT_PROJECT,
    default_data_root,
    default_gpu_cache_path,
    run_path,
)
from odyssey.training.weights import _INIT_PRESETS

_CONFIG_REGISTERED = False


@dataclass
class AnalysisConfig:
    lr_finder: bool = True
    lr: float = 1e-2
    lr_mult: float = 1.3
    lr_epochs: int = 1
    lr_save: str | None = None
    show_plot: bool = True
    act_hist: bool = True
    act_hist_epochs: int = 1
    act_hist_dir: str | None = None
    act_hist_batches: int = 2


@dataclass
class DetrConfig:
    num_queries: int = 50
    num_classes: int = 80
    d_model: int = 256
    freeze_backbone: bool = True
    enc_layers: int = 2
    dec_layers: int = 2
    score_threshold: float = 0.05


@dataclass
class TrainConfig:
    mode: str = "train"
    task: str = "classification"
    project_name: str = DEFAULT_PROJECT
    model: str = "cnn"
    epochs: int = 10
    lr: float = 1e-2
    init: str | None = None
    batch_size: int = 128
    image_size: int | None = None
    dataset: str = "fashion_mnist"
    data_root: str | None = None
    data_location: str | None = None
    fp16_data: bool = True
    one_hot_targets: bool = False
    width_mult: float = 2.0
    no_aug: bool = False
    pad_amount: int = 2
    flip_p: float = 0.5
    cutmix_size: int = 3
    act_hist: bool = False
    act_hist_dir: str | None = None
    compile: bool = True
    compile_mode: str = "default"
    compile_warmup_batches: int = 4
    grad_clip_norm: float | None = None
    grad_clip_value: float | None = None
    grad_accum: int = 1
    detr: DetrConfig = field(default_factory=DetrConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)


def _validate_train_config(cfg: TrainConfig) -> None:
    if cfg.mode not in {"train", "analysis"}:
        raise ValueError(f"mode must be 'train' or 'analysis', got {cfg.mode!r}")
    if cfg.task not in {"classification", "detr"}:
        raise ValueError(f"task must be 'classification' or 'detr', got {cfg.task!r}")
    if cfg.task == "classification" and cfg.model not in {"cnn", "resnet"}:
        raise ValueError(f"classification model must be 'cnn' or 'resnet', got {cfg.model!r}")
    if cfg.init is not None and cfg.init not in _INIT_PRESETS:
        raise ValueError(f"Unknown init preset {cfg.init!r}; choose from {tuple(_INIT_PRESETS)}")


def _default_image_size(dataset: str) -> int:
    if dataset == "fashion_mnist":
        return 28
    if dataset == "coco128":
        return 320
    return 32


def apply_path_defaults(cfg: TrainConfig) -> TrainConfig:
    "Fill dataset, image size, and run output paths from the workspace layout."
    project = cfg.project_name
    image_size = cfg.image_size if cfg.image_size is not None else _default_image_size(cfg.dataset)
    analysis = replace(
        cfg.analysis,
        lr_save=cfg.analysis.lr_save or str(run_path(project, "lr_find.png")),
        act_hist_dir=cfg.analysis.act_hist_dir or str(run_path(project, "activations")),
    )
    cfg = replace(
        cfg,
        image_size=image_size,
        data_root=cfg.data_root or default_data_root(cfg.dataset),
        data_location=cfg.data_location or default_gpu_cache_path(cfg.dataset),
        act_hist_dir=cfg.act_hist_dir or str(run_path(project, "activations")),
        analysis=analysis,
    )
    _validate_train_config(cfg)
    return cfg


def register_train_config() -> None:
    global _CONFIG_REGISTERED
    if _CONFIG_REGISTERED:
        return
    ConfigStore.instance().store(name="train", node=TrainConfig)
    _CONFIG_REGISTERED = True


register_train_config()


def compose_train_config(overrides: Sequence[str] | None = None) -> TrainConfig:
    "Compose TrainConfig from structured defaults plus Hydra-style overrides."
    with initialize_config_module(config_module="odyssey.conf", version_base="1.3"):
        cfg = compose(config_name="config", overrides=list(overrides or []))
    return apply_path_defaults(OmegaConf.to_object(cfg))


def train_config_from_hydra(cfg: DictConfig) -> TrainConfig:
    "Convert a Hydra-composed DictConfig into a validated TrainConfig."
    return apply_path_defaults(OmegaConf.to_object(cfg))
