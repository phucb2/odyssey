from pathlib import Path

from odyssey.paths import (
    checkpoint_path,
    checkpoints_dir,
    default_data_root,
    default_gpu_cache_path,
    run_path,
)


def test_dataset_paths() -> None:
    assert Path(default_data_root("fashion_mnist")).parts[-3:] == (
        "datasets",
        "fashion-mnist",
        "raw",
    )
    assert Path(default_data_root("cifar10")).parts[-3:] == (
        "datasets",
        "cifar10",
        "raw",
    )
    cache = Path(default_gpu_cache_path("fashion_mnist"))
    assert cache.parts[-3:] == ("fashion-mnist", "processed", "fashion_mnist.pt")


def test_project_paths() -> None:
    ckpt = checkpoints_dir("odyssey")
    assert ckpt.parts[-2:] == ("checkpoints", "odyssey")
    assert Path(checkpoint_path("odyssey", "best.pt")).name == "best.pt"
    runs = run_path("odyssey", "activations")
    assert runs.parts[-2:] == ("odyssey", "activations")
