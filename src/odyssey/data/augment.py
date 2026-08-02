"""HLB-style batch augmentation on GPU tensors."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

@dataclass
class AugmentConfig:
    enabled: bool = True
    num_classes: int = 10
    pad_amount: int = 2
    flip_p: float = 0.5
    cutmix_size: int = 3

def make_random_square_masks(inputs: torch.Tensor, mask_size: int) -> torch.Tensor | None:
    "Per-sample square boolean masks centered at random locations."
    if mask_size == 0:
        return None
    is_even = int(mask_size % 2 == 0)
    in_shape = inputs.shape
    low = mask_size // 2 - is_even
    mask_center_y = torch.empty(in_shape[0], dtype=torch.long, device=inputs.device).random_(
        low, in_shape[-2] - mask_size // 2 - is_even,
    )
    mask_center_x = torch.empty(in_shape[0], dtype=torch.long, device=inputs.device).random_(
        low, in_shape[-1] - mask_size // 2 - is_even,
    )
    to_mask_y_dists = torch.arange(in_shape[-2], device=inputs.device).view(1, 1, in_shape[-2], 1) - mask_center_y.view(-1, 1, 1, 1)
    to_mask_x_dists = torch.arange(in_shape[-1], device=inputs.device).view(1, 1, 1, in_shape[-1]) - mask_center_x.view(-1, 1, 1, 1)
    to_mask_y = (to_mask_y_dists >= (-(mask_size // 2) + is_even)) * (to_mask_y_dists <= mask_size // 2)
    to_mask_x = (to_mask_x_dists >= (-(mask_size // 2) + is_even)) * (to_mask_x_dists <= mask_size // 2)
    return to_mask_y * to_mask_x

@torch.no_grad()
def batch_crop(inputs: torch.Tensor, crop_size: int) -> torch.Tensor:
    crop_mask_batch = make_random_square_masks(inputs, crop_size)
    return torch.masked_select(inputs, crop_mask_batch).view(
        inputs.shape[0], inputs.shape[1], crop_size, crop_size,
    )

@torch.no_grad()
def batch_flip_lr(batch_images: torch.Tensor, flip_chance: float = 0.5) -> torch.Tensor:
    flip = torch.rand_like(batch_images[:, 0, 0, 0].view(-1, 1, 1, 1)) < flip_chance
    return torch.where(flip, torch.flip(batch_images, (-1,)), batch_images)

@torch.no_grad()
def batch_cutmix(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_permuted = torch.randperm(inputs.shape[0], device=inputs.device)
    cutmix_batch_mask = make_random_square_masks(inputs, patch_size)
    if cutmix_batch_mask is None:
        return inputs, targets
    cutmix_batch = torch.where(
        cutmix_batch_mask, torch.index_select(inputs, 0, batch_permuted), inputs,
    )
    cutmix_targets = torch.index_select(targets, 0, batch_permuted)
    portion_mixed = float(patch_size ** 2) / float(inputs.shape[-2] * inputs.shape[-1])
    cutmix_labels = portion_mixed * cutmix_targets + (1.0 - portion_mixed) * targets
    return cutmix_batch, cutmix_labels

def soft_cross_entropy(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if targets.ndim == 1 or targets.dtype in (torch.int32, torch.int64):
        return F.cross_entropy(preds, targets.long())
    return -(targets.float() * F.log_softmax(preds, dim=-1)).sum(dim=-1).mean()
