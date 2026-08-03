"""Tests for DETR model, criterion, and coco128 integration."""
import torch

from odyssey.data.coco128 import detection_collate
from odyssey.models.detr import create_detr_model
from odyssey.paths import default_data_root
from odyssey.training.config import compose_train_config
from odyssey.training.detection.boxes import box_cxcywh_to_xyxy
from odyssey.training.detection.criterion import DetrCriterion, DetrCriterionConfig
from odyssey.training.detection.metrics import compute_map_metrics, decode_detr_outputs


def test_detr_criterion_forward():
    model = create_detr_model(num_classes=80, num_queries=10, d_model=64, freeze_backbone=True)
    criterion = DetrCriterion(DetrCriterionConfig(num_classes=80))
    images = torch.randn(2, 3, 64, 64)
    outputs = model(images)
    targets = [
        {"labels": torch.tensor([0, 15]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.4, 0.1, 0.1]])},
        {"labels": torch.tensor([1]), "boxes": torch.tensor([[0.6, 0.6, 0.3, 0.3]])},
    ]
    loss, loss_dict = criterion(outputs, targets)
    assert loss.ndim == 0
    assert set(loss_dict) == {"loss_ce", "loss_bbox", "loss_giou"}
    assert all(v.ndim == 0 for v in loss_dict.values())


def test_detection_collate_shapes():
    batch = [
        (torch.randn(3, 32, 32), {"labels": torch.tensor([0]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])}),
        (torch.randn(3, 32, 32), {"labels": torch.tensor([1, 2]), "boxes": torch.tensor([[0.1, 0.2, 0.1, 0.1], [0.3, 0.4, 0.2, 0.2]])}),
    ]
    images, targets = detection_collate(batch)
    assert images.shape == (2, 3, 32, 32)
    assert len(targets) == 2
    assert targets[0]["labels"].shape == (1,)
    assert targets[1]["boxes"].shape == (2, 4)


def test_compose_train_config_detr_coco128():
    cfg = compose_train_config(["task=detr", "dataset=coco128", "epochs=5"])
    assert cfg.task == "detr"
    assert cfg.dataset == "coco128"
    assert cfg.image_size == 320
    assert cfg.detr.num_queries == 50
    assert cfg.detr.num_classes == 80
    assert cfg.detr.score_threshold == 0.05
    assert cfg.data_root == default_data_root("coco128")


def test_map_metrics_perfect_match():
    gt_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    gt = [{"labels": torch.tensor([0]), "boxes": gt_boxes}]
    pred_boxes = box_cxcywh_to_xyxy(gt_boxes)
    pred = [{"boxes": pred_boxes, "scores": torch.tensor([0.99]), "labels": torch.tensor([0])}]
    metrics = compute_map_metrics(pred, gt, num_classes=1, iou_thresholds=(0.5,))
    assert metrics["map50"] == 1.0
    assert metrics["map50_95"] == 1.0


def test_decode_detr_outputs_filters_background():
    outputs = {
        "pred_logits": torch.zeros(1, 2, 3),
        "pred_boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]]]),
    }
    outputs["pred_logits"][0, 0, 0] = 10.0
    outputs["pred_logits"][0, 1, 2] = 10.0
    decoded = decode_detr_outputs(outputs, num_classes=2, score_threshold=0.5)
    assert decoded[0]["labels"].tolist() == [0]
    assert decoded[0]["boxes"].shape == (1, 4)
