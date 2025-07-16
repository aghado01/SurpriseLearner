"""Production-grade configuration management."""

import yaml
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import logging

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Configuration-related errors."""
    pass


def load_config(config_path: str = "config/experiment.yaml") -> Dict[str, Any]:
    """
    Load and validate configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Validated configuration dictionary

    Raises:
        ConfigurationError: If configuration is invalid or missing
    """
    config_file = Path(config_path)

    if not config_file.exists():
        alternative_paths = [
            "config/default.yaml",
            "adaptive_bayesian_driver/config/default.yaml",
            "experiments/config.yaml"
        ]

        for alt_path in alternative_paths:
            if Path(alt_path).exists():
                config_file = Path(alt_path)
                logger.warning(f"Using alternative config: {alt_path}")
                break
        else:
            raise ConfigurationError(
                f"No configuration file found. "
                f"Tried: {config_path}, {alternative_paths}"
            )

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in {config_path}: {e}")

    # Validate and enhance configuration
    config = _validate_and_enhance_config(config)
    return config


def _validate_and_enhance_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and enhance configuration with system information."""

    # Add device configuration
    cuda_available = torch.cuda.is_available()
    device_name = str(torch.device("cuda" if cuda_available else "cpu"))
    gpu_count = torch.cuda.device_count() if cuda_available else 0

    config['device'] = {
        'cuda_available': cuda_available,
        'device_name': device_name,
        'gpu_count': gpu_count
    }

    # Add system information
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    cuda_version = torch.version.cuda if cuda_available else None

    config['system'] = {
        'python_version': python_version,
        'torch_version': torch.__version__,
        'cuda_version': cuda_version
    }

    # Validate critical sections
    required_sections = ['model', 'training', 'environment']
    for section in required_sections:
        if section not in config:
            logger.warning(f"Missing configuration section: {section}")
            config[section] = {}

    return config
