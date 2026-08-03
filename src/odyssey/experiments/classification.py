"""Classification experiment builder."""
import torch
import torch.nn.functional as F
from torcheval.metrics import MulticlassAccuracy

from odyssey.data.augment import AugmentConfig, soft_cross_entropy
from odyssey.data.loaders import GpuPreloadConfig, create_gpu_dls, default_gpu_cache_path
from odyssey.experiments.registry import register_task
from odyssey.models import create_cnn_model, create_res_model
from odyssey.training.callbacks import (
    ChannelsLastCB,
    CompileCB,
    CompileWarmupCB,
    DeviceCB,
    GradClipCB,
    LRFind,
    MetricsCB,
    MixPrecisionCB,
    ProgressCB,
    TimingCB,
    TrainCB,
)
from odyssey.training.config import TrainConfig
from odyssey.training.learner import Learner


def _dataset_model_kwargs(cfg: TrainConfig) -> dict:
    if cfg.dataset == "cifar10":
        return {"sz": cfg.image_size, "n_out": 10, "in_ch": 3}
    return {"sz": cfg.image_size, "n_out": 10, "in_ch": 1}


def _resolve_one_hot_targets(cfg: TrainConfig, aug: AugmentConfig) -> bool:
    if cfg.one_hot_targets:
        return True
    return not cfg.no_aug and aug.enabled and aug.cutmix_size > 0


@register_task("classification")
def build_classification_learner(cfg: TrainConfig, *, plot_progress: bool = True) -> Learner:
    model_kw = _dataset_model_kwargs(cfg)
    aug = AugmentConfig(
        enabled=not cfg.no_aug,
        pad_amount=0 if cfg.no_aug else cfg.pad_amount,
        flip_p=cfg.flip_p,
        cutmix_size=0 if cfg.no_aug else cfg.cutmix_size,
    )
    use_one_hot = _resolve_one_hot_targets(cfg, aug)
    if cfg.model == "resnet":
        model = create_res_model(wide=True, width_mult=cfg.width_mult, **model_kw)
    else:
        model = create_cnn_model(**model_kw)
    cache_path = cfg.data_location or default_gpu_cache_path(cfg.dataset)
    preload = GpuPreloadConfig(
        dataset=cfg.dataset,
        root=cfg.data_root,
        data_location=cache_path,
        fp16=cfg.fp16_data,
        one_hot=use_one_hot,
        num_workers=cfg.loader.num_workers,
    )
    dls = create_gpu_dls(bs=cfg.batch_size, preload=preload, aug=aug)
    loss_func = soft_cross_entropy if use_one_hot else F.cross_entropy
    cbs = [
        CompileCB(mode=cfg.compile_mode, enabled=cfg.compile),
        TimingCB(),
        MetricsCB(accuracy=MulticlassAccuracy(num_classes=10)),
        DeviceCB(),
        ChannelsLastCB(),
        ProgressCB(plot=plot_progress),
        MixPrecisionCB(),
        CompileWarmupCB(n_batches=cfg.compile_warmup_batches if cfg.compile else 0),
        TrainCB(),
    ]
    if cfg.grad_clip_norm is not None or cfg.grad_clip_value is not None:
        cbs.append(GradClipCB(max_norm=cfg.grad_clip_norm, max_value=cfg.grad_clip_value))
    lr_find = LRFind(
        n_epochs=cfg.analysis.lr_epochs,
        lr_mult=cfg.analysis.lr_mult,
        start_lr=cfg.analysis.lr,
        show_plot=cfg.analysis.show_plot,
        save_path=cfg.analysis.lr_save,
    )
    learn = Learner(
        model, dls, loss_func, torch.optim.AdamW,
        lr=cfg.lr, cbs=cbs, init=cfg.init, lr_find=lr_find,
    )
    learn.pin_memory = cfg.loader.pin_memory
    learn.project_name = cfg.project_name
    return learn
