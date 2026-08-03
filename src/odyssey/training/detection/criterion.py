"""Hungarian matching and DETR loss (CE + L1 + GIoU)."""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from odyssey.training.detection.boxes import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """Match predicted queries to ground-truth boxes via Hungarian algorithm."""

    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        out_prob = outputs["pred_logits"].softmax(-1)
        out_bbox = outputs["pred_boxes"]

        indices: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i in range(out_prob.shape[0]):
            tgt_ids = targets[i]["labels"]
            tgt_bbox = targets[i]["boxes"]
            if tgt_ids.numel() == 0:
                indices.append(
                    (
                        torch.empty(0, dtype=torch.long, device=out_prob.device),
                        torch.empty(0, dtype=torch.long, device=out_prob.device),
                    )
                )
                continue

            cost_class = -out_prob[i][:, tgt_ids]
            cost_bbox = torch.cdist(out_bbox[i], tgt_bbox, p=1)
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox[i]),
                box_cxcywh_to_xyxy(tgt_bbox),
            )
            cost = (
                self.cost_class * cost_class
                + self.cost_bbox * cost_bbox
                + self.cost_giou * cost_giou
            )
            row_idx, col_idx = linear_sum_assignment(cost.cpu())
            indices.append(
                (
                    torch.as_tensor(row_idx, dtype=torch.long, device=out_prob.device),
                    torch.as_tensor(col_idx, dtype=torch.long, device=out_prob.device),
                )
            )
        return indices


@dataclass
class DetrCriterionConfig:
    num_classes: int = 80
    eos_coef: float = 0.1
    cost_class: float = 1.0
    cost_bbox: float = 5.0
    cost_giou: float = 2.0
    loss_ce_weight: float = 1.0
    loss_bbox_weight: float = 5.0
    loss_giou_weight: float = 2.0


class DetrCriterion(nn.Module):
    """DETR loss with Hungarian matching."""

    def __init__(self, cfg: DetrCriterionConfig | None = None):
        super().__init__()
        self.cfg = cfg or DetrCriterionConfig()
        self.matcher = HungarianMatcher(
            cost_class=self.cfg.cost_class,
            cost_bbox=self.cfg.cost_bbox,
            cost_giou=self.cfg.cost_giou,
        )
        empty_weight = torch.ones(self.cfg.num_classes + 1)
        empty_weight[-1] = self.cfg.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        indices = self.matcher(outputs, targets)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = max(num_boxes, 1)

        loss_ce = self._loss_labels(outputs, targets, indices, num_boxes)
        loss_bbox, loss_giou = self._loss_boxes(outputs, targets, indices, num_boxes)

        loss_dict = {
            "loss_ce": loss_ce,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
        }
        loss = (
            self.cfg.loss_ce_weight * loss_ce
            + self.cfg.loss_bbox_weight * loss_bbox
            + self.cfg.loss_giou_weight * loss_giou
        )
        return loss, loss_dict

    def _loss_labels(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        num_boxes: int,
    ) -> torch.Tensor:
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(
            src_logits.shape[:2],
            self.cfg.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices, strict=True)]
        )
        return F.cross_entropy(
            src_logits.transpose(1, 2),
            target_classes,
            weight=self.empty_weight.to(src_logits.device),
        ) / num_boxes

    def _loss_boxes(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        num_boxes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices, strict=True)],
            dim=0,
        )
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="sum") / num_boxes
        loss_giou = (
            1
            - torch.diag(
                generalized_box_iou(
                    box_cxcywh_to_xyxy(src_boxes),
                    box_cxcywh_to_xyxy(target_boxes),
                )
            ).sum()
            / num_boxes
        )
        return loss_bbox, loss_giou

    @staticmethod
    def _get_src_permutation_idx(
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
