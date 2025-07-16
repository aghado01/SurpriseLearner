## Role Assignment

```markdown
You are a Senior Python DevOps Engineer specializing in ML repository hardening.

**Technical Stack:**
- Python 3.11+ with pyproject.toml packaging
- PyTorch CUDA-first development workflows
- Pre-commit hooks with Ruff, mypy, pytest
- Docker containerization

**Approach:**
- Methodical, validation-focused implementation
- Test imports after structural changes
- Verify compatibility before proceeding
- Apply production-grade practices efficiently

Acknowledge this role and confirm you understand the technical requirements.
```


## Execution Framework

### Phase 1: Package Structure Migration

**Create Modern Package Structure**

```powershell
# Create adaptive_bayesian_driver package hierarchy
$PackageDirs = @(
    "adaptive_bayesian_driver",
    "adaptive_bayesian_driver\models",
    "adaptive_bayesian_driver\environment",
    "adaptive_bayesian_driver\utils",
    "adaptive_bayesian_driver\applications"
)

foreach ($Dir in $PackageDirs) {
    New-Item -Path $Dir -ItemType Directory -Force
    Write-Host "✅ Created: $Dir" -ForegroundColor Green
}
```

**Core Package Files**

*adaptive_bayesian_driver\__init__.py*

```python
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
```

*adaptive_bayesian_driver\config.py*

```python
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
            raise ConfigurationError(f"No configuration file found. Tried: {config_path}, {alternative_paths}")

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
    config['device'] = {
        'cuda_available': torch.cuda.is_available(),
        'device_name': str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

    # Add system information
    config['system'] = {
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}",
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
    }

    # Validate critical sections
    required_sections = ['model', 'training', 'environment']
    for section in required_sections:
        if section not in config:
            logger.warning(f"Missing configuration section: {section}")
            config[section] = {}

    return config
```

*adaptive_bayesian_driver\utils\device.py*

```python
"""Advanced GPU device management for CUDA-first ML development."""

import torch
import platform
import sys
from typing import Dict, Optional
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
        'cudnn_version': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'devices': []
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(i)
            device_info['devices'].append({
                'index': i,
                'name': device_props.name,
                'total_memory': device_props.total_memory,
                'memory_allocated': torch.cuda.memory_allocated(i),
                'memory_reserved': torch.cuda.memory_reserved(i),
                'compute_capability': f"{device_props.major}.{device_props.minor}"
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
```


### Phase 2: Modern Python Configuration

**pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "adaptive-bayesian-driver"
version = "0.2.0"
description = "LC-NE inspired adaptive Bayesian learning for autonomous driving"
authors = [{name = "Azriel Ghadooshahy", email = "azriel.ghadooshahy@example.com"}]
readme = "README.md"
requires-python = ">=3.11"
license = {text = "MIT"}
keywords = ["machine-learning", "bayesian", "autonomous-driving", "neuroscience", "cuda"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

dependencies = [
    "torch>=2.2.0,<2.4.0",
    "torchvision>=0.17.0,<0.19.0",
    "numpy>=1.24.0,<1.27.0",
    "matplotlib>=3.6.0,<3.9.0",
    "pillow>=9.0.0,<11.0.0",
    "pyyaml>=6.0.0,<7.0.0",
    "tqdm>=4.66.0,<5.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0,<9.0.0",
    "pytest-cov>=5.0.0,<6.0.0",
    "ruff>=0.4.4,<0.5.0",
    "mypy>=1.10.0,<2.0.0",
    "pre-commit>=3.7.0,<4.0.0",
]

cuda = [
    "cupy-cuda12x>=12.0.0,<13.0.0",
]

production = [
    "mlflow>=2.13.0,<3.0.0",
    "hydra-core>=1.3.0,<2.0.0",
    "wandb>=0.17.0,<1.0.0",
    "fastapi>=0.111.0,<1.0.0",
    "uvicorn[standard]>=0.29.0,<1.0.0",
    "onnx>=1.16.0,<2.0.0",
    "onnxruntime-gpu>=1.18.0,<2.0.0",
]

[project.scripts]
adaptive-demo = "adaptive_bayesian_driver.main:main"
adaptive-train = "adaptive_bayesian_driver.applications.training:main"

[tool.ruff]
line-length = 88
target-version = "py311"
select = ["E", "F", "I", "N", "W", "UP", "B", "C4", "SIM", "TCH"]
ignore = ["E501", "N803", "N806", "B008", "UP007"]
exclude = [".git", "__pycache__", "build", "dist", ".eggs", "_chatbot"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false

[tool.ruff.isort]
known-first-party = ["adaptive_bayesian_driver"]
force-single-line = false

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
ignore_missing_imports = true
exclude = ["_chatbot/", "build/", "dist/", "tests/"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--cov=adaptive_bayesian_driver",
    "--cov-report=term-missing",
    "--cov-report=xml",
    "--cov-fail-under=80"
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "gpu: marks tests as requiring GPU",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
]

[tool.coverage.run]
source = ["adaptive_bayesian_driver"]
omit = ["*/tests/*", "*/_chatbot/*", "*/setup.py", "*/demo.py"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if self.debug:",
    "if settings.DEBUG",
    "raise AssertionError",
    "raise NotImplementedError",
    "if 0:",
    "if __name__ == .__main__.:",
    "class .*\\bProtocol\$$:",
    "@(abc\\.)?abstractmethod",
]
```

**requirements.txt**

```txt
# =============================================================================
# PRODUCTION REQUIREMENTS - CUDA-First ML Development
# =============================================================================

# PyTorch Stack (CUDA-optimized installation)
torch>=2.2.0,<2.4.0
torchvision>=0.17.0,<0.19.0

# Core Scientific Computing
numpy>=1.24.0,<1.27.0
matplotlib>=3.6.0,<3.9.0
pillow>=9.0.0,<11.0.0

# Configuration & Utilities
pyyaml>=6.0.0,<7.0.0
tqdm>=4.66.0,<5.0.0

# Development & Testing
pytest>=8.0.0,<9.0.0
pytest-cov>=5.0.0,<6.0.0

# Code Quality (Modern Python toolchain)
ruff>=0.4.4,<0.5.0
mypy>=1.10.0,<2.0.0
pre-commit>=3.7.0,<4.0.0
```


### Phase 3: Quality Infrastructure

**.pre-commit-config.yaml**

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
        args: [--markdown-linebreak-ext=md]
      - id: end-of-file-fixer
      - id: check-yaml
        args: [--allow-multiple-documents]
      - id: check-json
      - id: check-toml
      - id: check-merge-conflict
      - id: debug-statements
      - id: check-added-large-files
        args: [--maxkb=1000]
      - id: check-case-conflict
      - id: check-docstring-first
      - id: mixed-line-ending
        args: [--fix=lf]

  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.2
    hooks:
      - id: gitleaks
        name: "🔒 security · secrets detection"

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.4
    hooks:
      - id: ruff
        name: "🐍 python · lint and fix"
        args: [--fix, --exit-non-zero-on-fix]
        types_or: [python, pyi, jupyter]
      - id: ruff-format
        name: "🐍 python · format"
        types_or: [python, pyi, jupyter]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy
        name: "🐍 python · type checking"
        additional_dependencies: [
          types-PyYAML,
          types-tqdm,
          types-pillow,
          torch-stubs
        ]
        args: [--ignore-missing-imports, --python-version=3.11]
        exclude: ^(_chatbot/|tests/test_.*\.py|setup\.py)

  - repo: https://github.com/codespell-project/codespell
    rev: v2.2.6
    hooks:
      - id: codespell
        name: "📝 docs · spell check"
        args: ["-L", "nd,iam,thre,ba,pytorch,cuda"]
        exclude: ^(_chatbot/|\.git/|.*\.json$)

ci:
  autofix_commit_msg: |
    [pre-commit.ci] auto fixes from pre-commit hooks
  autofix_prs: true
  autoupdate_schedule: weekly
  skip: [gitleaks]
```


### Phase 4: Installation and Validation

```powershell
# Complete installation and validation sequence
Write-Host "🚀 Starting repository hardening automation..." -ForegroundColor Cyan

# Phase 1: Install package and dependencies
Write-Host "`n📦 Installing package and dependencies..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python -m pip install -r requirements.txt

# Phase 2: Install and setup pre-commit
Write-Host "`n🔧 Setting up pre-commit hooks..." -ForegroundColor Yellow
python -m pip install pre-commit
pre-commit install
pre-commit install --hook-type commit-msg

# Phase 3: Run initial validation
Write-Host "`n✅ Running validation suite..." -ForegroundColor Yellow

# Test package import
Write-Host "Testing package import..." -ForegroundColor White
python -c "
import adaptive_bayesian_driver as abd
print(f'✓ Package v{abd.__version__} imported successfully')
print(f'✓ CUDA Available: {abd.CUDA_AVAILABLE}')
print(f'✓ Device: {abd.DEVICE}')
print(f'✓ GPU Count: {abd.GPU_COUNT}')
"

# Test device utilities
Write-Host "Testing device management..." -ForegroundColor White
python -c "
from adaptive_bayesian_driver.utils.device import setup_cuda_environment
info = setup_cuda_environment()
print(f'✓ CUDA Environment: {info['cuda_available']}')
print(f'✓ Device Count: {info['device_count']}')
"

# Test configuration loading
Write-Host "Testing configuration system..." -ForegroundColor White
python -c "
try:
    from adaptive_bayesian_driver.config import load_config
    from adaptive_bayesian_driver.config import _validate_and_enhance_config
    config = _validate_and_enhance_config({'model': {}, 'training': {}, 'environment': {}})
    print('✓ Configuration system working')
    print(f'✓ Default config has {len(config)} sections')
except Exception as e:
    print(f'⚠️ Config test: {e}')
"

# Run pre-commit validation
Write-Host "Running pre-commit validation..." -ForegroundColor White
pre-commit run --all-files

# Final verification
Write-Host "`n🎉 Repository hardening complete!" -ForegroundColor Green
Write-Host "✅ Modern Python package structure" -ForegroundColor Green
Write-Host "✅ CUDA-first development environment" -ForegroundColor Green
Write-Host "✅ Production-grade quality gates" -ForegroundColor Green
Write-Host "✅ Comprehensive testing framework" -ForegroundColor Green

Write-Host "`n📋 Next Steps:" -ForegroundColor Cyan
Write-Host "1. Run 'python -m pytest tests/ -v' for full test suite"
Write-Host "2. Check 'git status' to review all changes"
Write-Host "3. Commit changes with: git add . && git commit -m 'feat: complete repository hardening'"
```


## Validation Checkpoints

### Critical Success Criteria

- [ ] `python -c "import adaptive_bayesian_driver"` succeeds
- [ ] `python -c "print(adaptive_bayesian_driver.CUDA_AVAILABLE)"` shows GPU status
- [ ] `pre-commit run --all-files` completes without critical errors
- [ ] `python -m pytest tests/ -v` runs without import failures
- [ ] Package structure matches expected layout


### Professional Assessment Points

As the Senior DevOps Engineer, evaluate:

1. **Package Architecture**: Does the structure support both development and production deployment?
2. **CUDA Integration**: Are device management utilities production-ready?
3. **Quality Gates**: Will pre-commit catch issues before production?
4. **Maintainability**: Are configurations future-proof and scalable?

Execute these phases sequentially, validating each before proceeding to the next.
