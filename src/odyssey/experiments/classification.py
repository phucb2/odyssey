"""Classification experiment builder."""
import torch
import torch.nn.functional as F
from torcheval.metrics import MulticlassAccuracy

from odyssey.data.augment import AugmentConfig, soft_cross_entropy
from odyssey.data.loaders import GpuPreloadConfig, create_gpu_dls, default_gpu_cache_path
from odyssey.experiments.registry import register_task
from odyssey.models import create_cnn_model, create_res_model
from odyssey.training.callbacks import TrainCB, default_cbs, make_lr_find
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
    cbs = default_cbs(
        train=TrainCB(),
        metrics=dict(accuracy=MulticlassAccuracy(num_classes=10)),
        compile=cfg.compile,
        compile_mode=cfg.compile_mode,
        compile_warmup_batches=cfg.compile_warmup_batches if cfg.compile else 0,
        channels_last=True,
        plot_progress=plot_progress,
        grad_clip_norm=cfg.grad_clip_norm,
        grad_clip_value=cfg.grad_clip_value,
        grad_accum=cfg.grad_accum,
    )
    learn = Learner(
        model, dls, loss_func, torch.optim.AdamW,
        lr=cfg.lr, cbs=cbs, init=cfg.init, lr_find=make_lr_find(cfg.analysis),
        pin_memory=cfg.loader.pin_memory,
        project_name=cfg.project_name,
    )
    return learn
