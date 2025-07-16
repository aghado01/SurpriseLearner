"""
Adaptive Bayesian Driver - Production ML Package
"""

__version__ = "0.2.0"
__author__ = "Azriel Ghadooshahy"

# Core system imports with robust error handling
try:
    import torch
    import numpy as np

    CUDA_AVAILABLE = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")

    if CUDA_AVAILABLE:
        GPU_COUNT = torch.cuda.device_count()
        GPU_NAME = torch.cuda.get_device_name(0)
    else:
        GPU_COUNT = 0
        GPU_NAME = "No GPU Available"

except ImportError as e:
    print(f"⚠️  Critical dependency missing: {e}")
    CUDA_AVAILABLE = False
    DEVICE = "cpu"
    GPU_COUNT = 0
    GPU_NAME = "Dependencies Missing"

from .config import load_config
from .utils.device import get_device, setup_cuda_environment

__all__ = [
    "__version__",
    "CUDA_AVAILABLE",
    "DEVICE",
    "GPU_COUNT",
    "GPU_NAME",
    "load_config",
    "get_device",
    "setup_cuda_environment"
]
