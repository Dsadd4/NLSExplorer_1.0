"""Device selection shared by NLSExplorer inference components."""

from __future__ import annotations

import torch


class DeviceUnavailableError(RuntimeError):
    """Raised when a requested accelerator is not available."""


def resolve_device(device: str | torch.device | None = "auto") -> torch.device:
    """Resolve ``auto`` or an explicit device into an available torch device."""

    if device is None or str(device).lower() == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise DeviceUnavailableError(
                "CUDA was requested, but torch.cuda.is_available() is false. "
                "Use device='cpu' or install a CUDA-enabled PyTorch build."
            )
        if resolved.index is not None and resolved.index >= torch.cuda.device_count():
            raise DeviceUnavailableError(
                f"CUDA device {resolved.index} was requested, but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible."
            )
    elif resolved.type == "mps" and not torch.backends.mps.is_available():
        raise DeviceUnavailableError(
            "MPS was requested, but it is not available. Use device='cpu'."
        )
    elif resolved.type not in {"cpu", "cuda", "mps"}:
        raise DeviceUnavailableError(
            f"Unsupported device type {resolved.type!r}; use auto, cpu, cuda, or mps."
        )
    return resolved
