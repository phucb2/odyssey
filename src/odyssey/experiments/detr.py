"""DETR object-detection experiment builder."""
import torch

from odyssey.data.coco128 import create_coco128_dls
from odyssey.experiments.registry import register_task
from odyssey.models.detr import create_detr_model
from odyssey.training.callbacks import (
    DeviceCB,
    GradClipCB,
    LRFind,
    MetricsCB,
    MixPrecisionCB,
    ProgressCB,
    TimingCB,
)
from odyssey.training.config import TrainConfig
from odyssey.training.detection.callbacks import DetrLossMetricsCB, DetrMapMetricsCB, DetrTrainCB
from odyssey.training.detection.criterion import DetrCriterion, DetrCriterionConfig
from odyssey.training.learner import Learner


@register_task("detr")
def build_detr_learner(cfg: TrainConfig, *, plot_progress: bool = True) -> Learner:
    detr = cfg.detr
    dls = create_coco128_dls(
        bs=cfg.batch_size,
        root=cfg.data_root,
        image_size=cfg.image_size,
        loader=cfg.loader,
    )
    model = create_detr_model(
        num_classes=detr.num_classes,
        num_queries=detr.num_queries,
        d_model=detr.d_model,
        num_encoder_layers=detr.enc_layers,
        num_decoder_layers=detr.dec_layers,
        freeze_backbone=detr.freeze_backbone,
    )
    criterion = DetrCriterion(DetrCriterionConfig(num_classes=detr.num_classes))
    cbs = [
        TimingCB(),
        DetrLossMetricsCB(),
        DetrMapMetricsCB(num_classes=detr.num_classes, score_threshold=detr.score_threshold),
        MetricsCB(),
        DeviceCB(),
        ProgressCB(plot=plot_progress),
        MixPrecisionCB(),
        DetrTrainCB(),
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
        model,
        dls,
        criterion,
        torch.optim.AdamW,
        lr=cfg.lr,
        cbs=cbs,
        init=cfg.init,
        lr_find=lr_find,
    )
    learn.pin_memory = cfg.loader.pin_memory
    learn.project_name = cfg.project_name
    return learn
