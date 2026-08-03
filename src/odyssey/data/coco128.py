"""coco128 object-detection dataset (Ultralytics YOLO format)."""
from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from odyssey.data.loaders import LoaderConfig
from odyssey.paths import default_data_root

COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def ensure_coco128_downloaded(root: str | Path | None = None) -> Path:
    """Download and extract coco128 if missing."""
    root = Path(root or default_data_root("coco128"))
    images_dir = root / "images" / "train2017"
    if images_dir.is_dir() and any(images_dir.glob("*.jpg")):
        return root

    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "coco128.zip"
    if not zip_path.is_file():
        urlretrieve(COCO128_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Zip contains a top-level coco128/ folder.
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            rel = Path(*Path(member).parts[1:]) if member.startswith("coco128/") else Path(member)
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())
    return root


class Coco128Dataset(Dataset):
    """Ultralytics coco128 in YOLO label format."""

    def __init__(
        self,
        root: str | Path,
        *,
        image_size: int = 320,
        train: bool = True,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.train = train
        self.images_dir = self.root / "images" / "train2017"
        self.labels_dir = self.root / "labels" / "train2017"
        self.image_paths = sorted(self.images_dir.glob("*.jpg"))
        if not self.image_paths:
            raise FileNotFoundError(f"No coco128 images found under {self.images_dir}")

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        img_path = self.image_paths[idx]
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        labels: list[int] = []
        boxes: list[list[float]] = []
        if label_path.is_file():
            for line in label_path.read_text().strip().splitlines():
                if not line.strip():
                    continue
                cls, cx, cy, w, h = line.split()
                labels.append(int(cls))
                boxes.append([float(cx), float(cy), float(w), float(h)])

        target = {
            "labels": torch.tensor(labels, dtype=torch.long),
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            if boxes
            else torch.zeros((0, 4), dtype=torch.float32),
        }
        return image, target


def detection_collate(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    images, targets = zip(*batch, strict=True)
    return torch.stack(images, dim=0), list(targets)


@dataclass
class Coco128DataLoaders:
    train: DataLoader
    valid: DataLoader

    def shutdown(self):
        pass


def create_coco128_dls(
    *,
    bs: int = 4,
    root: str | None = None,
    image_size: int = 320,
    loader: LoaderConfig | None = None,
) -> Coco128DataLoaders:
    root_path = ensure_coco128_downloaded(root)
    loader = loader or LoaderConfig()
    ds = Coco128Dataset(root_path, image_size=image_size, train=True)
    kw = loader.dataloader_kwargs()
    train = DataLoader(
        ds,
        batch_size=bs,
        shuffle=True,
        collate_fn=detection_collate,
        drop_last=True,
        **kw,
    )
    valid = DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        collate_fn=detection_collate,
        drop_last=False,
        **kw,
    )
    return Coco128DataLoaders(train=train, valid=valid)
