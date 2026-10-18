"""Device and dtype selection. Never falls back silently: callers print what was chosen."""

from __future__ import annotations

import torch


def pick_device(requested: str = "auto") -> torch.device:
    """Resolve `--device`. 'auto' means cuda > mps > cpu; anything else is used as given."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    """Compute dtype for training/inference. bf16 on capable CUDA, else fp32.

    fp16 (with GradScaler) is handled by the training loop, not here; mps and cpu stay fp32.
    """
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def describe(device: torch.device, dtype: torch.dtype) -> str:
    return f"device={device} dtype={dtype}"
