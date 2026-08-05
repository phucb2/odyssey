"""Compose a multi-digit image from MNIST samples for a non-negative integer."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.utils import save_image

from odyssey.paths import dataset_processed_dir, dataset_raw_dir

MNIST_DIGIT_SIZE = 28
DEFAULT_DATASET_NAME = "mnist_int"


@lru_cache(maxsize=1)
def _load_mnist() -> tuple[datasets.MNIST, list[list[int]]]:
    """Download MNIST and build per-digit index lists from the train split."""
    root = dataset_raw_dir("mnist")
    root.mkdir(parents=True, exist_ok=True)
    dataset = datasets.MNIST(str(root), train=True, download=True, transform=ToTensor())

    by_digit: list[list[int]] = [[] for _ in range(10)]
    for idx, (_, label) in enumerate(dataset):
        by_digit[int(label)].append(idx)

    return dataset, by_digit


def _sample_digit(digit: int, rng: torch.Generator | None) -> torch.Tensor:
    """Return one MNIST image tensor of shape (28, 28) for the given digit."""
    if not 0 <= digit <= 9:
        raise ValueError(f"digit must be in 0..9, got {digit}")

    dataset, by_digit = _load_mnist()
    indices = by_digit[digit]
    if not indices:
        raise RuntimeError(f"MNIST has no samples for digit {digit}")

    pick = torch.randint(0, len(indices), (1,), generator=rng).item()
    image, _ = dataset[indices[pick]]
    return image.squeeze(0)


def _compose_digit_strip(digits: str, spacing: int, rng: torch.Generator | None) -> torch.Tensor:
    """Horizontally stack MNIST digit images with configurable spacing."""
    if spacing < 0:
        raise ValueError(f"spacing must be non-negative, got {spacing}")

    parts: list[torch.Tensor] = []
    for i, ch in enumerate(digits):
        parts.append(_sample_digit(int(ch), rng))
        if i < len(digits) - 1 and spacing > 0:
            parts.append(torch.zeros(MNIST_DIGIT_SIZE, spacing))

    return torch.cat(parts, dim=1)


def _fit_with_padding(image: torch.Tensor, size: int, margin: int) -> torch.Tensor:
    """Scale image to fit inside size x size while preserving aspect ratio, then pad."""
    if margin < 0:
        raise ValueError(f"margin must be non-negative, got {margin}")

    inner = size - 2 * margin
    if inner <= 0:
        raise ValueError(f"margin {margin} leaves no room inside size {size}")

    height, width = image.shape
    scale = min(inner / height, inner / width)
    new_h = max(1, round(height * scale))
    new_w = max(1, round(width * scale))

    resized = F.interpolate(
        image.unsqueeze(0).unsqueeze(0),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)

    canvas = torch.zeros(size, size)
    top = margin + (inner - new_h) // 2
    left = margin + (inner - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas.unsqueeze(0)


def number_to_image(
    n: int,
    *,
    spacing: int = 2,
    size: int = 64,
    margin: int = 4,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Return a (1, size, size) float tensor in [0, 1] showing digits of n from MNIST."""
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")

    strip = _compose_digit_strip(str(n), spacing, rng)
    return _fit_with_padding(strip, size, margin)


def default_dataset_dir() -> Path:
    return dataset_processed_dir(DEFAULT_DATASET_NAME)


def generate_dataset(
    n_samples: int = 2000,
    *,
    max_value: int = 9999,
    spacing: int = 2,
    size: int = 64,
    margin: int = 4,
    seed: int = 0,
    output: Path | None = None,
) -> Path:
    """Generate unique (int, image) pairs as PNG files named after each integer."""
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if max_value < 0:
        raise ValueError(f"max_value must be non-negative, got {max_value}")

    pool_size = max_value + 1
    if n_samples > pool_size:
        raise ValueError(f"n_samples {n_samples} exceeds unique values in 0..{max_value}")

    _load_mnist()

    root = Path(output) if output is not None else default_dataset_dir()
    images_dir = root / "images"
    if images_dir.exists():
        for path in images_dir.glob("*.png"):
            path.unlink()
    images_dir.mkdir(parents=True, exist_ok=True)

    rng = torch.Generator().manual_seed(seed)
    labels = torch.randperm(pool_size, generator=rng)[:n_samples].tolist()

    for label in labels:
        image = number_to_image(label, spacing=spacing, size=size, margin=margin, rng=rng)
        save_image(image, images_dir / f"{label}.png")

    manifest = {
        "n_samples": n_samples,
        "max_value": max_value,
        "spacing": spacing,
        "size": size,
        "margin": margin,
        "seed": seed,
        "labels": labels,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return root


def load_dataset(root: Path | str | None = None) -> dict:
    """Load a generated int-image dataset from its image folder."""
    root = Path(root) if root is not None else default_dataset_dir()
    images_dir = root / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"dataset images not found at {images_dir}")

    paths = sorted(images_dir.glob("*.png"), key=lambda p: int(p.stem))
    labels = [int(p.stem) for p in paths]
    meta_path = root / "manifest.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}

    return {"labels": labels, "paths": paths, "root": root, "meta": meta}


def _demo(output: Path | None = None) -> torch.Tensor:
    """Download MNIST, render 123, and optionally save the image."""
    rng = torch.Generator().manual_seed(0)
    randint = torch.randint(0, 10000, (1,), generator=rng).item()
    image = number_to_image(randint, spacing=2, size=64, rng=rng)

    if output is not None:
        import matplotlib.pyplot as plt

        output.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(output, image.squeeze(0).numpy(), cmap="gray", vmin=0.0, vmax=1.0)

    return image


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MNIST multi-digit image compositor")
    parser.add_argument("--generate", action="store_true", help="generate int-image dataset")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--demo-out", type=Path, default=Path("runs/odyssey/mnist_int_demo.png"))
    args = parser.parse_args()

    if args.generate:
        root = generate_dataset(n_samples=args.n_samples, seed=args.seed)
        data = load_dataset(root)
        print(f"Saved {len(data['labels'])} images to {root / 'images'}")
        print(f"  sample: {data['paths'][0].name} -> {data['labels'][0]}")
    else:
        tensor = _demo(args.demo_out)
        print(f"Saved {args.demo_out} with shape {tuple(tensor.shape)}")
