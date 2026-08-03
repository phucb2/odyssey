"""GPU-preloaded datasets and dataloaders."""
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from odyssey.data.augment import AugmentConfig, batch_crop, batch_cutmix, batch_flip_lr
from odyssey.paths import default_data_root
from odyssey.storage.gpu_cache import (
    default_gpu_cache_path,
    gpu_dataset_keys,
    load_or_build_gpu_dataset,
)


@dataclass
class GpuPreloadConfig:
    "Load entire train/valid sets to GPU, normalize, optional fp16 + one-hot cache."
    dataset: str = "fashion_mnist"
    root: str | None = None
    data_location: str | None = None
    device: str = "cuda"
    one_hot: bool = False
    fp16: bool = True
    num_workers: int = 2

    def __post_init__(self):
        if self.root is None:
            self.root = default_data_root(self.dataset)
        if self.data_location is None:
            self.data_location = default_gpu_cache_path(self.dataset)


def _dataset_normalize_stats(cfg: GpuPreloadConfig, train_images: torch.Tensor):
    "Per-dataset normalization: FMNIST matches learner_v1 (0.5/0.5); CIFAR uses train-set stats."
    if cfg.dataset == "fashion_mnist":
        mean = torch.tensor([0.5], device=train_images.device, dtype=train_images.dtype)
        std = torch.tensor([0.5], device=train_images.device, dtype=train_images.dtype)
        return std, mean
    return torch.std_mean(train_images, dim=(0, 2, 3))


def _batch_normalize_images(images: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (images - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)


def _load_torchvision_split(cfg: GpuPreloadConfig, *, train: bool):
    transform = transforms.Compose([transforms.ToTensor()])
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    if cfg.dataset == "cifar10":
        ds = datasets.CIFAR10(cfg.root, download=True, train=train, transform=transform)
    elif cfg.dataset == "fashion_mnist":
        ds = datasets.FashionMNIST(cfg.root, download=True, train=train, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset {cfg.dataset!r}; choose cifar10 or fashion_mnist")

    loader = DataLoader(
        ds,
        batch_size=len(ds),
        shuffle=train,
        drop_last=True,
        num_workers=cfg.num_workers if train else max(1, cfg.num_workers // 2),
        persistent_workers=False,
    )
    images, targets = next(iter(loader))
    images = images.to(device=device, non_blocking=True)
    targets = targets.to(device=device, non_blocking=True)
    return images, targets


def build_gpu_dataset(cfg: GpuPreloadConfig) -> dict:
    "Load full splits to GPU, normalize from train stats, fp16 images, optional one-hot targets."
    train_key, valid_key = gpu_dataset_keys(cfg.dataset)
    train_images, train_targets = _load_torchvision_split(cfg, train=True)
    valid_images, valid_targets = _load_torchvision_split(cfg, train=False)

    std, mean = _dataset_normalize_stats(cfg, train_images)
    norm = partial(_batch_normalize_images, mean=mean, std=std)
    train_images, valid_images = norm(train_images), norm(valid_images)

    if cfg.fp16:
        train_images = train_images.half().requires_grad_(False)
        valid_images = valid_images.half().requires_grad_(False)
    if cfg.one_hot:
        n_classes = int(train_targets.max().item()) + 1
        train_targets = F.one_hot(train_targets.long(), n_classes).float()
        valid_targets = F.one_hot(valid_targets.long(), n_classes).float()

    return {
        train_key: {"images": train_images, "targets": train_targets},
        valid_key: {"images": valid_images, "targets": valid_targets},
    }


class GpuDataLoader:
    "Yield minibatches via GPU index_select; train epochs apply HLB-style batch aug first."

    def __init__(
        self,
        data: dict,
        key: str,
        batch_size: int,
        *,
        shuffle: bool,
        aug: "AugmentConfig | None" = None,
        crop_size: int | None = None,
    ):
        self.data = data[key]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.aug = aug
        self.crop_size = crop_size if crop_size is not None else self.data["images"].shape[-1]
        self.device = self.data["images"].device

    def __len__(self) -> int:
        return self.data["images"].shape[0] // self.batch_size

    def __iter__(self):
        n = self.data["images"].shape[0]
        n_batches = n // self.batch_size
        images, targets = self.data["images"], self.data["targets"]

        if self.shuffle and self.aug is not None and self.aug.enabled:
            images = batch_crop(images, self.crop_size)
            images = batch_flip_lr(images, flip_chance=self.aug.flip_p)
            if self.aug.cutmix_size > 0:
                images, targets = batch_cutmix(images, targets, self.aug.cutmix_size)

        perm = torch.arange(n, device=self.device)

        for i in range(n_batches):
            idx = perm[i * self.batch_size:(i + 1) * self.batch_size]
            yield images.index_select(0, idx), targets.index_select(0, idx)

    def one_batch(self):
        return next(iter(self))


class GpuDataLoaders:
    def __init__(
        self,
        data: dict,
        train_bs: int,
        valid_bs: int,
        *,
        dataset: str = "cifar10",
        aug: "AugmentConfig | None" = None,
    ):
        train_key, valid_key = gpu_dataset_keys(dataset)
        train_images = data[train_key]["images"]
        crop_size = train_images.shape[-1]
        if aug is not None and aug.enabled and aug.pad_amount > 0:
            data[train_key]["images"] = F.pad(
                train_images, (aug.pad_amount,) * 4, mode="reflect",
            )
            crop_size = train_images.shape[-1]
        self.train = GpuDataLoader(
            data, train_key, train_bs, shuffle=True, aug=aug, crop_size=crop_size,
        )
        self.valid = GpuDataLoader(data, valid_key, valid_bs, shuffle=False)
        self.gpu_data = data

    def shutdown(self):
        pass


@dataclass
class LoaderConfig:
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    def dataloader_kwargs(self) -> dict:
        kw = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if self.num_workers > 0:
            kw["persistent_workers"] = self.persistent_workers
            kw["prefetch_factor"] = self.prefetch_factor
        return kw


def create_gpu_dls(
    bs: int = 128,
    *,
    preload: GpuPreloadConfig | None = None,
    valid_bs: int | None = None,
    aug: AugmentConfig | None = None,
) -> GpuDataLoaders:
    preload = GpuPreloadConfig() if preload is None else preload
    data = load_or_build_gpu_dataset(preload)
    valid_bs = bs if valid_bs is None else valid_bs
    return GpuDataLoaders(data, bs, valid_bs, dataset=preload.dataset, aug=aug)
