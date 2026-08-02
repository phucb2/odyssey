"""GPU dataset cache load/save."""
import os

import torch

from odyssey.paths import default_gpu_cache_path as _default_gpu_cache_path


def default_gpu_cache_path(dataset: str) -> str:
    return _default_gpu_cache_path(dataset)


def gpu_dataset_keys(dataset: str) -> tuple[str, str]:
    return ("train", "eval") if dataset == "cifar10" else ("train", "valid")


def cache_matches_cfg(data: dict, cfg) -> bool:
    train_key, valid_key = gpu_dataset_keys(cfg.dataset)
    if train_key not in data or valid_key not in data:
        return False
    train = data[train_key]
    if "images" not in train or "targets" not in train:
        return False
    targets = train["targets"]
    is_one_hot = targets.ndim > 1
    if is_one_hot != cfg.one_hot:
        return False
    _, channels, height, _ = train["images"].shape
    if cfg.dataset == "cifar10" and (channels, height) != (3, 32):
        return False
    if cfg.dataset == "fashion_mnist" and (channels, height) != (1, 28):
        return False
    return True


def load_or_build_gpu_dataset(cfg):
    cache_path = cfg.data_location or default_gpu_cache_path(cfg.dataset)
    cache_path = str(cache_path)
    if os.path.exists(cache_path):
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        data = torch.load(cache_path, map_location=device)
        if cache_matches_cfg(data, cfg):
            return data
    from odyssey.data.loaders import build_gpu_dataset

    data = build_gpu_dataset(cfg)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    torch.save(data, cache_path)
    return data
