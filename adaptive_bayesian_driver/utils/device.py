"""Advanced GPU device management for CUDA-first ML development."""

import torch
import platform
import sys
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get optimal compute device with fallback hierarchy.

    Args:
        prefer_cuda: Whether to prefer CUDA if available

    Returns:
        PyTorch device optimized for current hardware
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        return device
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using Apple Silicon MPS device")
        return device
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device")
        return device


def setup_cuda_environment() -> Dict[str, Any]:
    """
    Comprehensive CUDA environment setup and diagnostics.

    Returns:
        Complete device information dictionary
    """
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda,
        'cudnn_version': (torch.backends.cudnn.version()
                          if torch.backends.cudnn.is_available() else None),
        'device_count': (torch.cuda.device_count()
                         if torch.cuda.is_available() else 0),
        'devices': []
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(i)
            major_minor = f"{device_props.major}.{device_props.minor}"
            device_info['devices'].append({
                'index': i,
                'name': device_props.name,
                'total_memory': device_props.total_memory,
                'memory_allocated': torch.cuda.memory_allocated(i),
                'memory_reserved': torch.cuda.memory_reserved(i),
                'compute_capability': major_minor
            })

        # Set memory allocation strategy for better performance
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(0.8)

    # Add system information
    device_info['system'] = {
        'platform': platform.platform(),
        'python_version': sys.version,
        'torch_version': torch.__version__,
        'cpu_count': torch.get_num_threads()
    }

    return device_info


def optimize_cuda_performance():
    """Apply CUDA performance optimizations."""
    if not torch.cuda.is_available():
        logger.warning("CUDA not available - skipping optimizations")
        return

    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Memory management
    torch.cuda.empty_cache()

    logger.info("CUDA performance optimizations applied")


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
