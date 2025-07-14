"""
Adaptive Bayesian Driver - LC-NE Inspired Learning System

This package implements Locus Coeruleus-Norepinephrine (LC-NE) surprise
detection mechanisms for adaptive Bayesian learning in autonomous driving
applications.

Core modules:
- models: Neural network architectures and learning algorithms
- environment: Task environments and scene generation
- utils: Utilities for visualization and device management
"""

__version__ = "0.2.0"
__author__ = "Azriel Ghadooshahy"

# Core imports for easy access
try:
    import torch

    # Check CUDA availability
    CUDA_AVAILABLE = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")

except ImportError as e:
    print(f"Warning: Core dependencies not available: {e}")
    CUDA_AVAILABLE = False
    DEVICE = torch.device("cpu")

# Expose key classes for convenient imports
from .config import load_config
from .utils.device import get_device

__all__ = [
    "__version__",
    "load_config",
    "get_device",
    "CUDA_AVAILABLE",
    "DEVICE",
]
