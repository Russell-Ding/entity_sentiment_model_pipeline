"""Utility functions for device management and common operations."""

import torch
from typing import Optional


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the best available device or validate a specified device.

    Priority order when device is None or "auto":
        1. CUDA (NVIDIA GPU)
        2. MPS (Apple Silicon GPU)
        3. CPU

    Args:
        device: Device specification. Options:
            - None or "auto": Auto-detect best available device
            - "cuda" or "cuda:0": Use NVIDIA GPU
            - "mps": Use Apple Silicon GPU
            - "cpu": Use CPU

    Returns:
        torch.device: The selected device

    Raises:
        ValueError: If specified device is not available

    Examples:
        >>> device = get_device()  # Auto-detect
        >>> device = get_device("mps")  # Force MPS
        >>> device = get_device("cuda:0")  # Specific CUDA device
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    # Parse device string
    device_str = str(device).lower()

    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError(
                f"CUDA device '{device}' requested but CUDA is not available. "
                "Available devices: " + _get_available_devices_str()
            )
        return torch.device(device)

    elif device_str == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError(
                "MPS device requested but MPS is not available. "
                "MPS requires macOS 12.3+ and Apple Silicon or AMD GPU. "
                "Available devices: " + _get_available_devices_str()
            )
        return torch.device("mps")

    elif device_str == "cpu":
        return torch.device("cpu")

    else:
        raise ValueError(
            f"Unknown device: '{device}'. "
            f"Valid options: 'auto', 'cuda', 'cuda:N', 'mps', 'cpu'. "
            "Available devices: " + _get_available_devices_str()
        )


def _get_available_devices_str() -> str:
    """Get string listing available devices."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append(f"cuda (count: {torch.cuda.device_count()})")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return ", ".join(devices)


def get_device_info() -> dict:
    """
    Get detailed information about available devices.

    Returns:
        dict: Device availability and details
    """
    info = {
        "cpu": True,
        "cuda": {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "devices": [],
        },
        "mps": {
            "available": torch.backends.mps.is_available(),
            "built": torch.backends.mps.is_built(),
        },
        "recommended": str(get_device()),
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            info["cuda"]["devices"].append({
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total": torch.cuda.get_device_properties(i).total_memory,
            })

    return info


def is_mps_available() -> bool:
    """Check if MPS (Apple Silicon GPU) is available."""
    return torch.backends.mps.is_available()


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def device_to_str(device: torch.device) -> str:
    """
    Convert device to string for serialization.

    Handles MPS device which doesn't have an index.

    Args:
        device: PyTorch device

    Returns:
        String representation
    """
    if device.type == "mps":
        return "mps"
    elif device.type == "cuda":
        return f"cuda:{device.index if device.index is not None else 0}"
    else:
        return "cpu"


def move_to_device(obj, device: torch.device):
    """
    Recursively move tensors in nested structures to device.

    Handles dict, list, tuple containing tensors.

    Args:
        obj: Object to move (tensor, dict, list, tuple)
        device: Target device

    Returns:
        Object with tensors moved to device
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    else:
        return obj
