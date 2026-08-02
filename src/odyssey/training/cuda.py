"""CUDA performance settings."""
import importlib.util
import sys

import torch


def enable_cuda_speed_settings():
    "Apply cudnn.benchmark and high matmul precision when CUDA is available."
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")


def torch_compile_available() -> bool:
    "True when torch.compile's Inductor backend can run (needs Triton; not on Windows)."
    if sys.platform == "win32":
        return False
    return importlib.util.find_spec("triton") is not None
