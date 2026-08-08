import pathlib
from dataclasses import dataclass, field
from functools import partial
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra import compose, initialize_config_module
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.io import read_image

from odyssey.experiments.mnist_int import parse_image_label
from odyssey.paths import DEFAULT_PROJECT, checkpoint_path, dataset_processed_dir
from odyssey.training.callback import Callback
from odyssey.training.callbacks import DeviceCB, TrainCB, default_cbs, make_lr_find
from odyssey.training.learner import Learner

_CLIP_CONFIG_REGISTERED = False


@dataclass
class TokenizerConfig:
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2


@dataclass
class ImageEncoderConfig:
    in_ch: int = 3
    base_ch: int = 64


@dataclass
class TextEncoderConfig:
    max_len: int = 77
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 8
    dim_feedforward: int = 512
    dropout: float = 0.2


@dataclass
class LossConfig:
    temperature: float = 0.07


@dataclass
class DataConfig:
    dataset: str = "mnist_int"
    batch_size: int = 64
    valid_frac: float = 0.1
    seed: int = 0
    num_workers: int = 0
    pin_memory: bool = True
    augment: bool = True
    aug_translate: float = 0.12
    aug_noise: float = 0.05
    aug_jitter: float = 0.2


@dataclass
class FitConfig:
    epochs: int = 5
    lr: float = 3e-4
    weight_decay: float = 0.05
    grad_accum: int = 1
    compile: bool = False
    compile_mode: str = "default"
    project_name: str = "odyssey"
    checkpoint: str = "clip_final.pt"


@dataclass
class ClipConfig:
    """Nested CLIP experiment config (Hydra / OmegaConf structured)."""

    embed_dim: int = 256
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    image_encoder: ImageEncoderConfig = field(default_factory=ImageEncoderConfig)
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    fit: FitConfig = field(default_factory=FitConfig)


def register_clip_config() -> None:
    global _CLIP_CONFIG_REGISTERED
    if _CLIP_CONFIG_REGISTERED:
        return
    ConfigStore.instance().store(name="clip_config", node=ClipConfig)
    _CLIP_CONFIG_REGISTERED = True


register_clip_config()


def compose_clip_config(overrides: Sequence[str] | None = None) -> ClipConfig:
    """Compose ClipConfig from structured defaults plus Hydra-style overrides."""
    register_clip_config()
    with initialize_config_module(config_module="odyssey.conf", version_base="1.3"):
        cfg = compose(config_name="clip", overrides=list(overrides or []))
    return OmegaConf.to_object(cfg)


class ResidualBlock(nn.Module):
    """Basic ResNet block: two 3x3 convs with a shortcut."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))

class ImageEncoder(nn.Module):
    """Small ResNet-style CNN that maps images to a CLIP embedding."""

    def __init__(self, cfg: ImageEncoderConfig, *, embed_dim: int):
        super().__init__()
        in_ch, base_ch = cfg.in_ch, cfg.base_ch
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(base_ch, base_ch, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(base_ch, base_ch * 2, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(base_ch * 2, base_ch * 4, num_blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(base_ch * 4, embed_dim)

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, num_blocks: int, stride: int) -> nn.Sequential:
        blocks = [ResidualBlock(in_ch, out_ch, stride=stride)]
        blocks.extend(ResidualBlock(out_ch, out_ch) for _ in range(1, num_blocks))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)
    
    
def int_to_text(n: int) -> str:
    """Convert an integer to English words, e.g. 112 -> 'one hundred and twelve'."""
    ones = (
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen",
    )
    tens = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
    scales = ("", "thousand", "million", "billion", "trillion")

    def under_1000(x: int) -> str:
        parts: list[str] = []
        hundreds, rem = divmod(x, 100)
        if hundreds:
            parts.append(f"{ones[hundreds]} hundred")
        if rem == 0:
            return " ".join(parts)
        if hundreds:
            parts.append("and")
        if rem < 20:
            parts.append(ones[rem])
        else:
            t, o = divmod(rem, 10)
            parts.append(tens[t] if o == 0 else f"{tens[t]} {ones[o]}")
        return " ".join(parts)

    if n == 0:
        return "zero"
    if n < 0:
        return f"minus {int_to_text(-n)}"

    parts: list[str] = []
    scale = 0
    while n:
        n, chunk = divmod(n, 1000)
        if chunk:
            words = under_1000(chunk)
            if scales[scale]:
                words = f"{words} {scales[scale]}"
            parts.append(words)
        scale += 1
        if scale >= len(scales) and n:
            raise ValueError(f"n is too large to convert, remaining={n}")
    return " ".join(reversed(parts))

class Tokenizer:
    """Char-level tokenizer: pad=_=0, bos=^=1, eos=$=2."""

    PAD_CHAR, BOS_CHAR, EOS_CHAR = "_", "^", "$"

    def __init__(self, cfg: TokenizerConfig | None = None):
        cfg = cfg or TokenizerConfig()
        self.pad, self.bos, self.eos = cfg.pad_id, cfg.bos_id, cfg.eos_id
        self.stoi: dict[str, int] = {
            self.PAD_CHAR: self.pad,
            self.BOS_CHAR: self.bos,
            self.EOS_CHAR: self.eos,
        }
        self.itos: dict[int, str] = {v: k for k, v in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def fit(self, texts: list[str]) -> None:
        chars = sorted({c for text in texts for c in text if c not in self.stoi})
        next_id = max(self.stoi.values()) + 1
        for ch in chars:
            self.stoi[ch] = next_id
            self.itos[next_id] = ch
            next_id += 1

    def encode(self, text: str) -> torch.Tensor:
        try:
            ids = [self.bos] + [self.stoi[c] for c in text] + [self.eos]
        except KeyError as e:
            raise ValueError(f"unknown character {e.args[0]!r}; call fit() first") from e
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str:
        skip = {self.pad, self.bos, self.eos}
        return "".join(self.itos[i] for i in ids.tolist() if i not in skip)
    
class ImageTextPairDataset(Dataset):
    """PNG digit images paired with tokenized English text from the filename."""

    def __init__(
        self,
        cfg: ClipConfig | None = None,
        *,
        root: pathlib.Path | None = None,
        tokenizer: Tokenizer | None = None,
    ):
        cfg = cfg or ClipConfig()
        self.cfg = cfg
        self.root = pathlib.Path(root) if root is not None else dataset_processed_dir(cfg.data.dataset)
        images_dir = self.root / "images"
        self.image_paths = sorted(
            images_dir.glob("*.png"),
            key=lambda p: (parse_image_label(p), p.stem),
        )
        if not self.image_paths:
            raise FileNotFoundError(f"no PNG images found in {images_dir}")
        self.texts = [int_to_text(parse_image_label(p)) for p in self.image_paths]
        self.tokenizer = tokenizer or Tokenizer(cfg.tokenizer)
        self.tokenizer.fit(self.texts)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = read_image(str(self.image_paths[index])).float() / 255.0
        tokens = self.tokenizer.encode(self.texts[index])
        return image, tokens


def collate_image_text(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    pad_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack images and pad token sequences to a rectangular batch."""
    images, texts = zip(*batch)
    images = torch.stack(images, dim=0)
    max_len = max(t.numel() for t in texts)
    ids = torch.full((len(texts), max_len), pad_id, dtype=torch.long)
    for i, t in enumerate(texts):
        ids[i, : t.numel()] = t
    return images, ids

class TextEncoder(nn.Module):
    """Transformer-encoder text tower that maps Tokenizer ids to a CLIP embedding."""

    def __init__(
        self,
        cfg: TextEncoderConfig,
        *,
        vocab_size: int,
        embed_dim: int,
        pad_id: int,
        eos_id: int,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.max_len = cfg.max_len
        self.token_emb = nn.Embedding(vocab_size, cfg.d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            activation="relu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) token ids from Tokenizer (bos … eos, optionally padded)
        if x.size(1) > self.max_len:
            raise ValueError(f"sequence length {x.size(1)} exceeds max_len={self.max_len}")

        b, t = x.shape
        pos = torch.arange(t, device=x.device).unsqueeze(0)
        h = self.token_emb(x) + self.pos_emb(pos)
        h = self.encoder(h, src_key_padding_mask=x.eq(self.pad_id))
        eos_pos = (x == self.eos_id).int().argmax(dim=1)
        pooled = h[torch.arange(b, device=x.device), eos_pos]
        return self.proj(self.norm(pooled))
    
class ContrastiveLoss(nn.Module):
    """CLIP-style symmetric InfoNCE over L2-normalized image/text embeddings."""

    def __init__(self, cfg: LossConfig):
        super().__init__()
        # log-parameterized scale so temperature stays positive (CLIP uses exp(logit_scale))
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / cfg.temperature).log())

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x, y: (B, D) image and text embeddings
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        logits = self.logit_scale.exp().clamp(max=100.0) * x @ y.T
        # logits = x @ y.T
        targets = torch.arange(logits.size(0), device=logits.device)
        return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2
    
class CLIP(nn.Module):
    """Image + text encoders with CLIP contrastive loss."""

    def __init__(self, tokenizer: Tokenizer, cfg: ClipConfig | None = None):
        super().__init__()
        cfg = cfg or ClipConfig()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.image_encoder = ImageEncoder(cfg.image_encoder, embed_dim=cfg.embed_dim)
        self.text_encoder = TextEncoder(
            cfg.text_encoder,
            vocab_size=tokenizer.vocab_size,
            embed_dim=cfg.embed_dim,
            pad_id=tokenizer.pad,
            eos_id=tokenizer.eos,
        )
        self.criterion = ContrastiveLoss(cfg.loss)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(image)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(text)

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        return self.criterion(self.encode_image(image), self.encode_text(text))


@dataclass
class ClipDataLoaders:
    train: DataLoader
    valid: DataLoader


class ClipTrainCB(TrainCB):
    """CLIP forward returns contrastive loss; wire it into Learner predict/get_loss."""

    def predict(self, learn):
        return learn.model(learn.batch[0], learn.batch[1])

    def get_loss(self, learn):
        return learn.preds


class ClipImageAugmentCB(Callback):
    """Digit-preserving train-only aug: translate, intensity jitter, noise."""

    order = DeviceCB.order + 1

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_translate: float = 0.12,
        noise_std: float = 0.05,
        jitter: float = 0.2,
    ):
        self.enabled = enabled
        self.max_translate = max_translate
        self.noise_std = noise_std
        self.jitter = jitter

    def before_batch(self, learn):
        if not self.enabled or not learn.training:
            return
        x = learn.batch[0]
        b = x.size(0)
        device, dtype = x.device, x.dtype

        scale = 1.0 + (torch.rand(b, 1, 1, 1, device=device, dtype=dtype) * 2 - 1) * self.jitter
        shift = (torch.rand(b, 1, 1, 1, device=device, dtype=dtype) * 2 - 1) * (self.jitter * 0.25)
        x = (x * scale + shift).clamp(0, 1)

        if self.max_translate > 0:
            theta = torch.zeros(b, 2, 3, device=device, dtype=dtype)
            theta[:, 0, 0] = 1
            theta[:, 1, 1] = 1
            theta[:, 0, 2] = (torch.rand(b, device=device, dtype=dtype) * 2 - 1) * self.max_translate
            theta[:, 1, 2] = (torch.rand(b, device=device, dtype=dtype) * 2 - 1) * self.max_translate
            grid = F.affine_grid(theta, x.size(), align_corners=False)
            x = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")

        if self.noise_std > 0:
            x = (x + torch.randn_like(x) * self.noise_std).clamp(0, 1)

        learn.batch = (x, *learn.batch[1:])


def create_clip_dls(
    cfg: ClipConfig,
    *,
    tokenizer: Tokenizer | None = None,
) -> tuple[ClipDataLoaders, Tokenizer]:
    """Build train/valid loaders over mnist-int image-text pairs."""
    dataset = ImageTextPairDataset(cfg, tokenizer=tokenizer)
    tokenizer = dataset.tokenizer

    n = len(dataset)
    n_valid = max(1, int(round(n * cfg.data.valid_frac)))
    n_train = n - n_valid
    if n_train < 1:
        raise ValueError(f"valid_frac={cfg.data.valid_frac} leaves no train samples (n={n})")

    generator = torch.Generator().manual_seed(cfg.data.seed)
    train_ds, valid_ds = random_split(dataset, [n_train, n_valid], generator=generator)
    collate = partial(collate_image_text, pad_id=tokenizer.pad)
    loader_kw = {
        "batch_size": cfg.data.batch_size,
        "collate_fn": collate,
        "num_workers": cfg.data.num_workers,
        "pin_memory": cfg.data.pin_memory,
    }
    dls = ClipDataLoaders(
        train=DataLoader(train_ds, shuffle=True, **loader_kw),
        valid=DataLoader(valid_ds, shuffle=False, **loader_kw),
    )
    return dls, tokenizer


def build_clip_learner(cfg: ClipConfig, *, plot_progress: bool = True) -> Learner:
    """Construct a Learner for CLIP using the shared callback stack."""
    dls, tokenizer = create_clip_dls(cfg)
    model = CLIP(tokenizer, cfg)
    cbs = default_cbs(
        train=ClipTrainCB(),
        compile=cfg.fit.compile,
        compile_mode=cfg.fit.compile_mode,
        channels_last=True,
        plot_progress=plot_progress,
        grad_accum=cfg.fit.grad_accum,
        after_device=[
            ClipImageAugmentCB(
                enabled=cfg.data.augment,
                max_translate=cfg.data.aug_translate,
                noise_std=cfg.data.aug_noise,
                jitter=cfg.data.aug_jitter,
            ),
        ],
    )
    opt_func = partial(torch.optim.AdamW, weight_decay=cfg.fit.weight_decay)
    learn = Learner(
        model,
        dls,
        loss_func=lambda preds, _y: preds,
        opt_func=opt_func,
        lr=cfg.fit.lr,
        cbs=cbs,
        lr_find=make_lr_find(show_plot=plot_progress),
        pin_memory=cfg.data.pin_memory,
        project_name=cfg.fit.project_name or DEFAULT_PROJECT,
    )
    learn.tokenizer = tokenizer
    return learn


def _base_image_text_dataset(dataset: Dataset) -> ImageTextPairDataset:
    """Unwrap Subset wrappers to the underlying ImageTextPairDataset."""
    while isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset
    if not isinstance(dataset, ImageTextPairDataset):
        raise TypeError(f"expected ImageTextPairDataset, got {type(dataset)!r}")
    return dataset


def _pad_token_batch(token_lists: list[torch.Tensor], pad_id: int) -> torch.Tensor:
    max_len = max(t.numel() for t in token_lists)
    ids = torch.full((len(token_lists), max_len), pad_id, dtype=torch.long)
    for i, t in enumerate(token_lists):
        ids[i, : t.numel()] = t
    return ids


@torch.no_grad()
def evaluate_image_classification(
    learn: Learner,
    *,
    loader: DataLoader | None = None,
    topk: tuple[int, ...] = (1, 5),
) -> dict:
    """Classify images by nearest text embedding; report top-k string-match accuracy.

    Encodes every unique caption in the dataset as a class prototype, then for each
    validation image ranks captions by cosine similarity via the image encoder.
    """
    model: CLIP = learn.model
    tokenizer: Tokenizer = learn.tokenizer
    device = next(model.parameters()).device
    model.eval()

    base = _base_image_text_dataset((loader or learn.dls.valid).dataset)
    class_texts = sorted(set(base.texts))
    text_to_idx = {t: i for i, t in enumerate(class_texts)}
    class_tokens = _pad_token_batch(
        [tokenizer.encode(t) for t in class_texts],
        pad_id=tokenizer.pad,
    ).to(device)
    text_emb = F.normalize(model.encode_text(class_tokens), dim=-1)

    max_k = min(max(topk), len(class_texts))
    ks = tuple(k for k in topk if k <= max_k)
    correct = {k: 0 for k in ks}
    total = 0
    use_channels_last = getattr(learn, "use_channels_last", False)
    eval_loader = loader or learn.dls.valid

    for images, tokens in eval_loader:
        images = images.to(device, non_blocking=True)
        if use_channels_last:
            images = images.to(memory_format=torch.channels_last)
        img_emb = F.normalize(model.encode_image(images), dim=-1)
        top_idx = (img_emb @ text_emb.T).topk(max_k, dim=-1).indices
        for i, tok in enumerate(tokens):
            true = tokenizer.decode(tok.cpu())
            true_idx = text_to_idx[true]
            ranked = top_idx[i].tolist()
            for k in ks:
                correct[k] += int(true_idx in ranked[:k])
            total += 1

    metrics = {
        "total": total,
        "num_classes": len(class_texts),
    }
    for k in ks:
        metrics[f"top{k}"] = correct[k] / total if total else 0.0
        metrics[f"correct_top{k}"] = correct[k]
    # keep legacy key as top-1
    if 1 in correct:
        metrics["accuracy"] = metrics["top1"]
        metrics["correct"] = metrics["correct_top1"]
    return metrics


def save_clip_checkpoint(
    learn: Learner,
    cfg: ClipConfig,
    *,
    path: str | pathlib.Path | None = None,
) -> pathlib.Path:
    """Save model, tokenizer vocab, config, and eval metrics after training."""
    project = cfg.fit.project_name or DEFAULT_PROJECT
    out = pathlib.Path(path) if path is not None else pathlib.Path(
        checkpoint_path(project, cfg.fit.checkpoint)
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    tokenizer: Tokenizer = learn.tokenizer
    payload = {
        "model": learn.model.state_dict(),
        "tokenizer": {
            "stoi": tokenizer.stoi,
            "itos": {int(k): v for k, v in tokenizer.itos.items()},
            "pad": tokenizer.pad,
            "bos": tokenizer.bos,
            "eos": tokenizer.eos,
        },
        "cfg": OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True),
        "classification_metrics": getattr(learn, "classification_metrics", None),
    }
    torch.save(payload, out)
    return out


def load_clip_checkpoint(
    path: str | pathlib.Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[CLIP, Tokenizer, ClipConfig, dict]:
    """Restore CLIP model + tokenizer from a checkpoint written by `save_clip_checkpoint`."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    cfg = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(ClipConfig), OmegaConf.create(payload["cfg"]))
    )
    tok_state = payload["tokenizer"]
    tokenizer = Tokenizer(cfg.tokenizer)
    tokenizer.stoi = dict(tok_state["stoi"])
    tokenizer.itos = {int(k): v for k, v in tok_state["itos"].items()}
    tokenizer.pad = int(tok_state["pad"])
    tokenizer.bos = int(tok_state["bos"])
    tokenizer.eos = int(tok_state["eos"])
    model = CLIP(tokenizer, cfg)
    model.load_state_dict(payload["model"])
    return model, tokenizer, cfg, payload.get("classification_metrics") or {}


def train(cfg: ClipConfig | None = None, *, plot_progress: bool = True) -> Learner:
    """Build the CLIP learner, fit, eval, then write a checkpoint."""
    cfg = cfg or compose_clip_config()
    learn = build_clip_learner(cfg, plot_progress=plot_progress)
    learn.fit(cfg.fit.epochs)
    metrics = evaluate_image_classification(learn)
    learn.classification_metrics = metrics
    print(
        f"image→text top1: {metrics['top1']:.4%} ({metrics['correct_top1']}/{metrics['total']})  "
        f"top5: {metrics['top5']:.4%} ({metrics['correct_top5']}/{metrics['total']})  "
        f"classes={metrics['num_classes']}"
    )
    ckpt = save_clip_checkpoint(learn, cfg)
    learn.checkpoint_path = str(ckpt)
    print(f"checkpoint: {ckpt}")
    return learn


if __name__ == "__main__":
    import sys

    cfg = compose_clip_config(sys.argv[1:])
    print(OmegaConf.to_yaml(OmegaConf.structured(cfg)))
    learn = train(cfg, plot_progress=False)
    print(f"done: epochs={cfg.fit.epochs} lr={cfg.fit.lr}")
