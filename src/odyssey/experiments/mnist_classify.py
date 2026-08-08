"""Zero-shot MNIST digit classification with a trained CLIP checkpoint.

Use after training `clip.py` on mnist-int. Loads the checkpoint, embeds
digit class names ("one" … "nine"), and ranks each MNIST image by cosine
similarity in the shared embedding space. Digit 0 is excluded (tokenizer
never saw "zero" / 'z' in the mnist-int training captions).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor

from odyssey.experiments.clip import (
    CLIP,
    Tokenizer,
    _pad_token_batch,
    int_to_text,
    load_clip_checkpoint,
)
from odyssey.experiments.mnist_int import MNIST_DIGIT_SIZE
from odyssey.paths import DEFAULT_PROJECT, checkpoint_path, dataset_raw_dir

# Exclude 0: checkpoint vocab has no 'z' ("zero" absent from mnist-int captions).
DIGIT_CLASSES = tuple(range(1, 10))
DEFAULT_SIZE = 64
DEFAULT_MARGIN = 4
# Most mnist-int training samples are 4-digit; match that glyph scale.
DEFAULT_REF_DIGITS = 4
DEFAULT_SPACING = 2


def digit_to_clip_image(
    digit: torch.Tensor,
    *,
    size: int = DEFAULT_SIZE,
    margin: int = DEFAULT_MARGIN,
    in_ch: int = 3,
    ref_digits: int = DEFAULT_REF_DIGITS,
    spacing: int = DEFAULT_SPACING,
) -> torch.Tensor:
    """Center a MNIST digit on the CLIP canvas at training glyph scale.

    Training images are mostly multi-digit strips fitted into ``size``; a lone
    digit must not be upscaled to fill the frame. Use the same scale factor as
    a ``ref_digits``-wide strip, then center with padding.
    """
    if digit.ndim == 3:
        digit = digit.squeeze(0)
    if ref_digits < 1:
        raise ValueError(f"ref_digits must be >= 1, got {ref_digits}")

    inner = size - 2 * margin
    if inner <= 0:
        raise ValueError(f"margin {margin} leaves no room inside size {size}")

    ref_w = ref_digits * MNIST_DIGIT_SIZE + max(0, ref_digits - 1) * spacing
    scale = min(inner / MNIST_DIGIT_SIZE, inner / ref_w)
    new_h = max(1, round(MNIST_DIGIT_SIZE * scale))
    new_w = max(1, round(MNIST_DIGIT_SIZE * scale))

    resized = F.interpolate(
        digit.unsqueeze(0).unsqueeze(0),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)

    canvas = torch.zeros(1, size, size)
    top = margin + (inner - new_h) // 2
    left = margin + (inner - new_w) // 2
    canvas[0, top : top + new_h, left : left + new_w] = resized
    if in_ch == 1:
        return canvas
    return canvas.expand(in_ch, -1, -1).contiguous()


@torch.no_grad()
def classify_mnist(
    model: CLIP,
    tokenizer: Tokenizer,
    *,
    split: str = "test",
    batch_size: int = 256,
    size: int = DEFAULT_SIZE,
    margin: int = DEFAULT_MARGIN,
    ref_digits: int = DEFAULT_REF_DIGITS,
    spacing: int = DEFAULT_SPACING,
    device: torch.device | None = None,
    num_workers: int = 0,
) -> dict:
    """Classify MNIST digits 1–9 via nearest English digit-name text embedding."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    in_ch = model.cfg.image_encoder.in_ch

    class_ids = torch.tensor(DIGIT_CLASSES, dtype=torch.long, device=device)
    class_texts = [int_to_text(d) for d in DIGIT_CLASSES]
    class_tokens = _pad_token_batch(
        [tokenizer.encode(t) for t in class_texts],
        pad_id=tokenizer.pad,
    ).to(device)
    text_emb = F.normalize(model.encode_text(class_tokens), dim=-1)

    root = dataset_raw_dir("mnist")
    root.mkdir(parents=True, exist_ok=True)
    train = split == "train"
    full = datasets.MNIST(str(root), train=train, download=True, transform=ToTensor())
    keep = [i for i, (_, y) in enumerate(full) if int(y) != 0]
    ds = Subset(full, keep)

    def collate(batch):
        images, labels = zip(*batch)
        images = torch.stack(
            [
                digit_to_clip_image(
                    img,
                    size=size,
                    margin=margin,
                    in_ch=in_ch,
                    ref_digits=ref_digits,
                    spacing=spacing,
                )
                for img in images
            ]
        )
        labels = torch.tensor(labels, dtype=torch.long)
        return images, labels

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )

    n_cls = len(DIGIT_CLASSES)
    correct = 0
    total = 0
    per_class_correct = torch.zeros(n_cls, dtype=torch.long)
    per_class_total = torch.zeros(n_cls, dtype=torch.long)
    digit_to_slot = {d: i for i, d in enumerate(DIGIT_CLASSES)}

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        img_emb = F.normalize(model.encode_image(images), dim=-1)
        pred_slots = (img_emb @ text_emb.T).argmax(dim=-1)
        preds = class_ids[pred_slots]
        match = preds == labels
        correct += int(match.sum().item())
        total += labels.numel()
        for c in DIGIT_CLASSES:
            slot = digit_to_slot[c]
            mask = labels == c
            per_class_total[slot] += int(mask.sum().item())
            per_class_correct[slot] += int((match & mask).sum().item())

    accuracy = correct / total if total else 0.0
    per_class = {
        int_to_text(c): (
            per_class_correct[digit_to_slot[c]].item() / per_class_total[digit_to_slot[c]].item()
            if per_class_total[digit_to_slot[c]].item()
            else 0.0
        )
        for c in DIGIT_CLASSES
    }
    return {
        "split": split,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "num_classes": n_cls,
        "class_texts": class_texts,
        "per_class_accuracy": per_class,
        "per_class_correct": {
            int_to_text(c): int(per_class_correct[digit_to_slot[c]]) for c in DIGIT_CLASSES
        },
        "per_class_total": {
            int_to_text(c): int(per_class_total[digit_to_slot[c]]) for c in DIGIT_CLASSES
        },
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(checkpoint_path(DEFAULT_PROJECT, "clip_final.pt")),
        help="path to CLIP checkpoint from clip.py",
    )
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--margin", type=int, default=DEFAULT_MARGIN)
    parser.add_argument(
        "--ref-digits",
        type=int,
        default=DEFAULT_REF_DIGITS,
        help="scale glyph like an N-digit training strip (default: 4)",
    )
    parser.add_argument("--spacing", type=int, default=DEFAULT_SPACING)
    parser.add_argument("--device", default=None, help="cpu | cuda | cuda:0 …")
    args = parser.parse_args(argv)

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    device = torch.device(args.device) if args.device else None
    map_location = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, cfg, ckpt_metrics = load_clip_checkpoint(
        args.checkpoint, map_location=map_location
    )
    print(f"checkpoint: {args.checkpoint}")
    print(f"config: embed_dim={cfg.embed_dim} in_ch={cfg.image_encoder.in_ch}")
    print(
        f"canvas: size={args.size} margin={args.margin} "
        f"ref_digits={args.ref_digits} spacing={args.spacing}"
    )
    if ckpt_metrics:
        print(
            "ckpt val image→text: "
            f"top1={ckpt_metrics.get('top1', float('nan')):.2%}  "
            f"top5={ckpt_metrics.get('top5', float('nan')):.2%}  "
            f"classes={ckpt_metrics.get('num_classes')}"
        )

    metrics = classify_mnist(
        model,
        tokenizer,
        split=args.split,
        batch_size=args.batch_size,
        size=args.size,
        margin=args.margin,
        ref_digits=args.ref_digits,
        spacing=args.spacing,
        device=torch.device(map_location) if device is None else device,
    )
    print(
        f"MNIST {metrics['split']}: "
        f"accuracy={metrics['accuracy']:.4%} "
        f"({metrics['correct']}/{metrics['total']})"
    )
    for name, acc in metrics["per_class_accuracy"].items():
        n_ok = metrics["per_class_correct"][name]
        n_tot = metrics["per_class_total"][name]
        print(f"  {name:10s}  {acc:7.2%}  ({n_ok}/{n_tot})")
    return metrics


if __name__ == "__main__":
    main()
