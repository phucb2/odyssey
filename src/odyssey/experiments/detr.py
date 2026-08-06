"""DETR object-detection experiment builder."""
import torch

from odyssey.data.coco128 import create_coco128_dls
from odyssey.experiments.registry import register_task
from odyssey.models.detr import create_detr_model
from odyssey.training.callbacks import default_cbs, make_lr_find
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
    cbs = default_cbs(
        train=DetrTrainCB(),
        include_compile_cb=False,
        plot_progress=plot_progress,
        grad_clip_norm=cfg.grad_clip_norm,
        grad_clip_value=cfg.grad_clip_value,
        grad_accum=cfg.grad_accum,
        before_metrics=[
            DetrLossMetricsCB(),
            DetrMapMetricsCB(num_classes=detr.num_classes, score_threshold=detr.score_threshold),
        ],
    )
    learn = Learner(
        model,
        dls,
        criterion,
        torch.optim.AdamW,
        lr=cfg.lr,
        cbs=cbs,
        init=cfg.init,
        lr_find=make_lr_find(cfg.analysis),
        pin_memory=cfg.loader.pin_memory,
        project_name=cfg.project_name,
    )
    return learn
