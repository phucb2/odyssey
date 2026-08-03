"""Workspace paths for datasets, checkpoints, and training runs."""
from pathlib import Path

DEFAULT_PROJECT = "odyssey"

DATASET_SLUGS = {
    "fashion_mnist": "fashion-mnist",
    "cifar10": "cifar10",
    "coco128": "coco128",
}


def repo_root() -> Path:
    return Path.cwd()


def dataset_slug(dataset: str) -> str:
    return DATASET_SLUGS.get(dataset, dataset.replace("_", "-"))


def dataset_dir(dataset: str) -> Path:
    return repo_root() / "datasets" / dataset_slug(dataset)


def dataset_raw_dir(dataset: str) -> Path:
    return dataset_dir(dataset) / "raw"


def dataset_processed_dir(dataset: str) -> Path:
    path = dataset_dir(dataset) / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_manifests_dir(dataset: str) -> Path:
    path = dataset_dir(dataset) / "manifests"
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_data_root(dataset: str) -> str:
    return str(dataset_raw_dir(dataset))


def default_gpu_cache_path(dataset: str) -> str:
    return str(dataset_processed_dir(dataset) / f"{dataset}.pt")


def checkpoints_dir(project: str = DEFAULT_PROJECT) -> Path:
    path = repo_root() / "checkpoints" / project
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_path(project: str, filename: str) -> str:
    return str(checkpoints_dir(project) / filename)


def runs_dir(project: str = DEFAULT_PROJECT) -> Path:
    path = repo_root() / "runs" / project
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_path(project: str, *parts: str) -> Path:
    path = runs_dir(project).joinpath(*parts)
    if parts and not str(parts[-1]).endswith((".png", ".pt", ".pth", ".ckpt")):
        path.mkdir(parents=True, exist_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path
