"""Build learner and run training CLI."""
import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torcheval.metrics import MulticlassAccuracy

from odyssey.data.augment import AugmentConfig, soft_cross_entropy
from odyssey.data.loaders import GpuPreloadConfig, create_gpu_dls, default_gpu_cache_path
from odyssey.models import create_cnn_model, create_res_model
from odyssey.tracking.diagnostics import describe_model
from odyssey.training.callbacks import (
    ActivationHistCB,
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
)
from odyssey.training.config import TrainConfig, train_config_from_hydra
from odyssey.training.cuda import enable_cuda_speed_settings
from odyssey.training.learner import Learner

enable_cuda_speed_settings()


def _set_progress_plot(cbs, plot: bool):
    return [ProgressCB(plot=plot) if isinstance(cb, ProgressCB) else cb for cb in cbs]


def _dataset_model_kwargs(cfg: TrainConfig) -> dict:
    if cfg.dataset == "cifar10":
        return {"sz": cfg.image_size, "n_out": 10, "in_ch": 3}
    return {"sz": cfg.image_size, "n_out": 10, "in_ch": 1}


def _resolve_one_hot_targets(cfg: TrainConfig, aug: AugmentConfig) -> bool:
    if cfg.one_hot_targets:
        return True
    return not cfg.no_aug and aug.enabled and aug.cutmix_size > 0


def _build_learner(cfg: TrainConfig, *, plot_progress: bool = True) -> Learner:
    model_kw = _dataset_model_kwargs(cfg)
    aug = AugmentConfig(
        enabled=not cfg.no_aug,
        pad_amount=0 if cfg.no_aug else cfg.pad_amount,
        flip_p=cfg.flip_p,
        cutmix_size=0 if cfg.no_aug else cfg.cutmix_size,
    )
    use_one_hot = _resolve_one_hot_targets(cfg, aug)
    if cfg.wide_resnet:
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


def run_training(cfg: TrainConfig) -> None:
    learn = _build_learner(cfg, plot_progress=cfg.mode == "train" or cfg.analysis.act_hist)
    print(describe_model(learn.model))

    try:
        if cfg.mode == "train":
            if cfg.act_hist:
                learn.cbs = learn.cbs + [
                    ActivationHistCB(save_dir=cfg.act_hist_dir),
                ]
            learn.fit(cfg.epochs)
            return

        ana = cfg.analysis
        if ana.lr_finder:
            learn.cbs = _set_progress_plot(learn.cbs, plot=False)
            learn.lr_find.run(learn)

        if ana.act_hist:
            learn.cbs = learn.cbs + [
                ActivationHistCB(
                    save_dir=ana.act_hist_dir,
                    n_batches=ana.act_hist_batches,
                ),
            ]
            learn.cbs = _set_progress_plot(learn.cbs, plot=True)
            learn.fit(ana.act_hist_epochs)
    finally:
        if getattr(learn, "dls", None) is not None:
            learn.dls.shutdown()


@hydra.main(version_base="1.3", config_path="pkg://odyssey.conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run_training(train_config_from_hydra(cfg))
