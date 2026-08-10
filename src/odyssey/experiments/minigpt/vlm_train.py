"""Train Q-Former + projector for MiniGPT on image-caption pairs.

Uses coco128 images with synthetic captions from YOLO class labels.
Run: ``uv run python -m odyssey.experiments.minigpt.vlm_train --epochs 1``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from odyssey.data.coco128 import ensure_coco128_downloaded
from odyssey.experiments.minigpt.vlm import MiniGPT, MiniGPTConfig
from odyssey.paths import DEFAULT_PROJECT, checkpoint_path, default_data_root

COCO_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


def caption_from_label_file(label_path: Path) -> str:
    """Build a simple caption from YOLO class ids in a label file."""
    if not label_path.is_file():
        return "a photo."
    class_ids: set[int] = set()
    for line in label_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        class_ids.add(int(line.split()[0]))
    if not class_ids:
        return "a photo."
    names = [COCO_CLASSES[cid] for cid in sorted(class_ids) if 0 <= cid < len(COCO_CLASSES)]
    if not names:
        return "a photo."
    if len(names) == 1:
        return f"a photo of a {names[0]}."
    joined = ", ".join(names[:-1]) + f" and {names[-1]}"
    return f"a photo containing {joined}."


class Coco128CaptionDataset(Dataset):
    """Image-caption pairs derived from coco128 detection labels."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.images_dir = self.root / "images" / "train2017"
        self.labels_dir = self.root / "labels" / "train2017"
        self.image_paths = sorted(self.images_dir.glob("*.jpg"))
        if not self.image_paths:
            raise FileNotFoundError(f"No coco128 images found under {self.images_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[Image.Image, str]:
        img_path = self.image_paths[idx]
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        image = Image.open(img_path).convert("RGB")
        caption = caption_from_label_file(label_path)
        return image, caption


def collate_captions(
    batch: list[tuple[Image.Image, str]],
    *,
    image_processor,
    tokenizer,
    prompt_prefix: str = "Caption:",
):
    images, captions = zip(*batch, strict=True)
    pixel_values = image_processor(images=list(images), return_tensors="pt")["pixel_values"]

    # Teacher-forcing on "Caption: {caption}" without chat template for simpler stage-1 training.
    texts = [f"{prompt_prefix} {cap}" for cap in captions]
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    return pixel_values, encoded["input_ids"], encoded["attention_mask"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MiniGPT Q-Former + projector")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="minigpt_qformer.pt")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    data_root = ensure_coco128_downloaded(args.data_root or default_data_root("coco128"))
    dataset = Coco128CaptionDataset(data_root)

    cfg = MiniGPTConfig()
    model = MiniGPT.from_pretrained(cfg, device=device)
    model.freeze_backbone()
    model.train()
    model.qformer.train()
    model.projector.train()

    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model_name)
    image_processor = model.vision.image_processor

    def collate_fn(batch):
        return collate_captions(batch, image_processor=image_processor, tokenizer=tokenizer)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    trainable = [p for p in model.qformer.parameters() if p.requires_grad] + [
        p for p in model.projector.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    for epoch in range(args.epochs):
        total_loss = 0.0
        n_batches = 0
        for pixel_values, input_ids, attention_mask in loader:
            pixel_values = pixel_values.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            optimizer.zero_grad(set_to_none=True)
            _, loss = model(
                pixel_values,
                input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            if loss is None:
                continue
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        avg = total_loss / max(n_batches, 1)
        print(f"epoch {epoch + 1}/{args.epochs}  loss={avg:.4f}")

    out_path = checkpoint_path(DEFAULT_PROJECT, args.checkpoint)
    torch.save(model.trainable_state_dict(), out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
