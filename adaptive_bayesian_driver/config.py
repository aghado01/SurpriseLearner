"""Configuration management for adaptive Bayesian driver."""

import yaml
from pathlib import Path
from typing import Dict, Any
import torch


def load_config(config_path: str = "config/experiment.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Add device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['device'] = str(device)

    return config


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
