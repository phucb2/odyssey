"""COCO-style mAP metrics for object detection."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from odyssey.training.detection.boxes import box_cxcywh_to_xyxy, box_iou


COCO_IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))


def decode_detr_outputs(
    outputs: dict[str, torch.Tensor],
    num_classes: int,
    *,
    score_threshold: float = 0.05,
) -> list[dict[str, torch.Tensor]]:
    """Convert DETR logits/boxes to scored detections per image."""
    logits = outputs["pred_logits"]
    boxes = outputs["pred_boxes"]
    probs = logits.softmax(-1)[..., :num_classes]
    scores, labels = probs.max(-1)

    decoded: list[dict[str, torch.Tensor]] = []
    for i in range(logits.shape[0]):
        keep = scores[i] > score_threshold
        xyxy = box_cxcywh_to_xyxy(boxes[i][keep]).clamp(0.0, 1.0)
        decoded.append(
            {
                "boxes": xyxy,
                "scores": scores[i][keep],
                "labels": labels[i][keep],
            }
        )
    return decoded


def _match_image_predictions(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_threshold: float,
) -> tuple[list[float], list[bool]]:
    if pred_boxes.numel() == 0:
        return [], []

    order = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[order]
    pred_scores = pred_scores[order]
    matched_gt = torch.zeros(gt_boxes.shape[0], dtype=torch.bool, device=gt_boxes.device)

    scores_out: list[float] = []
    tps_out: list[bool] = []
    for box, score in zip(pred_boxes, pred_scores, strict=True):
        scores_out.append(float(score))
        if gt_boxes.numel() == 0:
            tps_out.append(False)
            continue
        ious, _ = box_iou(box.unsqueeze(0), gt_boxes)
        ious = ious.squeeze(0).masked_fill(matched_gt, -1.0)
        best_idx = int(ious.argmax())
        if float(ious[best_idx]) >= iou_threshold:
            matched_gt[best_idx] = True
            tps_out.append(True)
        else:
            tps_out.append(False)
    return scores_out, tps_out


def _average_precision(scores: list[float], tps: list[bool], num_gt: int) -> float:
    if num_gt == 0:
        return float("nan")
    if not scores:
        return 0.0

    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    tp_cum = 0
    fp_cum = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for idx in order:
        if tps[idx]:
            tp_cum += 1
        else:
            fp_cum += 1
        denom = tp_cum + fp_cum
        precisions.append(tp_cum / denom if denom else 0.0)
        recalls.append(tp_cum / num_gt)

    mrec = [0.0, *recalls, 1.0]
    mpre = [0.0, *precisions, 0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def _ap_for_class(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    class_id: int,
    iou_threshold: float,
) -> float:
    class_scores: list[float] = []
    class_tps: list[bool] = []
    num_gt = 0

    for pred, tgt in zip(predictions, targets, strict=True):
        gt_mask = tgt["labels"] == class_id
        gt_boxes = box_cxcywh_to_xyxy(tgt["boxes"][gt_mask])
        num_gt += gt_boxes.shape[0]

        pred_mask = pred["labels"] == class_id
        scores, tps = _match_image_predictions(
            pred["boxes"][pred_mask],
            pred["scores"][pred_mask],
            gt_boxes,
            iou_threshold,
        )
        class_scores.extend(scores)
        class_tps.extend(tps)

    return _average_precision(class_scores, class_tps, num_gt)


def compute_map_metrics(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    num_classes: int,
    *,
    iou_thresholds: tuple[float, ...] = COCO_IOU_THRESHOLDS,
) -> dict[str, float]:
    aps_50: list[float] = []
    aps_all: list[float] = []

    for class_id in range(num_classes):
        ap50 = _ap_for_class(predictions, targets, class_id, 0.5)
        if ap50 == ap50:
            aps_50.append(ap50)
            class_aps = [
                _ap_for_class(predictions, targets, class_id, thr)
                for thr in iou_thresholds
            ]
            aps_all.append(sum(class_aps) / len(class_aps))

    map50 = sum(aps_50) / len(aps_50) if aps_50 else 0.0
    map50_95 = sum(aps_all) / len(aps_all) if aps_all else 0.0
    return {"map50": map50, "map50_95": map50_95}


@dataclass
class DetectionMapMetric:
    num_classes: int
    score_threshold: float = 0.05
    iou_thresholds: tuple[float, ...] = COCO_IOU_THRESHOLDS
    _predictions: list[dict[str, torch.Tensor]] = field(default_factory=list, init=False, repr=False)
    _targets: list[dict[str, torch.Tensor]] = field(default_factory=list, init=False, repr=False)
    _last: dict[str, float] = field(default_factory=lambda: {"map50": 0.0, "map50_95": 0.0}, init=False, repr=False)

    def reset(self) -> None:
        self._predictions = []
        self._targets = []

    def update(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> None:
        decoded = decode_detr_outputs(
            outputs,
            self.num_classes,
            score_threshold=self.score_threshold,
        )
        for pred, tgt in zip(decoded, targets, strict=True):
            self._predictions.append(
                {
                    "boxes": pred["boxes"].detach().cpu(),
                    "scores": pred["scores"].detach().cpu(),
                    "labels": pred["labels"].detach().cpu(),
                }
            )
            self._targets.append(
                {
                    "boxes": tgt["boxes"].detach().cpu(),
                    "labels": tgt["labels"].detach().cpu(),
                }
            )

    def compute(self) -> dict[str, float]:
        if not self._predictions:
            self._last = {"map50": 0.0, "map50_95": 0.0}
            return dict(self._last)
        self._last = compute_map_metrics(
            self._predictions,
            self._targets,
            self.num_classes,
            iou_thresholds=self.iou_thresholds,
        )
        return dict(self._last)
