"""Device utilities for adaptive Bayesian driver."""

import torch


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get optimal compute device.

    Args:
        prefer_cuda: Whether to prefer CUDA if available

    Returns:
        PyTorch device
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():  # Apple Silicon support
        return torch.device('mps')
    else:
        return torch.device('cpu')


def get_device_info() -> str:
    """
    Get information about the current device.

    Returns:
        Device information string
    """
    device = get_device()
    if device.type == 'cuda':
        return f"CUDA device: {torch.cuda.get_device_name()}"
    elif device.type == 'mps':
        return "Apple Silicon MPS device"
    else:
        return "CPU device"
