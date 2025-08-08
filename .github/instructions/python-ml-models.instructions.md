---
applyTo: "adaptive_bayesian_driver/**/*.py"
---

# Python ML Models Instructions for Adaptive Bayesian Driver

## Project Architecture Context
This package implements LC-NE inspired adaptive Bayesian learning for autonomous driving scenarios with:
- **Recursive Bayesian Learning**: Ego vehicle embodied learning with surprise detection
- **CUDA-First Development**: Optimized for GPU acceleration with CPU fallback
- **Production-Grade Standards**: Comprehensive error handling, logging, and type hints

## Core Development Standards
- **Python Version**: Minimum Python 3.11 with full type annotation support
- **Type Hints**: Mandatory type hints for all function signatures using `from typing import Dict, List, Tuple, Optional`
- **Error Handling**: Comprehensive try-except blocks with specific exception types
- **Logging**: Use structured logging with appropriate log levels
- **Docstrings**: Google-style docstrings for all classes and methods

## PyTorch and CUDA Patterns
### Device management - always include fallback
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Memory-efficient tensor operations
with torch.no_grad():
predictions = model(data.to(device))

### Proper CUDA memory management
if torch.cuda.is_available():
torch.cuda.empty_cache()


## Adaptive Bayesian Learning Context
When implementing models, follow these architectural principles:
- **Surprise Detection**: Implement LC-NE inspired adaptive thresholds with continuous uncertainty functions
- **Two-Timescale Dynamics**: Fast latent updates, slow generator parameter adaptation
- **Ego Vehicle Perspective**: Embodied learning approach for rule acquisition and bias emergence
- **Reconstruction Error Patterns**: Use generative prior errors for context uncertainty estimation

## Class Structure Requirements
class ModelName:
"""
Brief description of model purpose.
Key Features:
- Feature 1 with specific technical detail
- Feature 2 with implementation context
- Feature 3 with performance characteristics
"""

def __init__(self, config: Dict[str, Any]):
    self.config = config
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize state tracking variables

def method_name(self, param: torch.Tensor, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Method description with clear purpose.

    Args:
        param: Description with tensor shape expectations
        metadata: Description of required metadata keys

    Returns:
        Dictionary with specific key descriptions
"""

## Integration with Compute Optimizations
- Leverage `IntelComputeOptimizer` for hardware-specific optimizations
- Support both CPU (Intel MKL-DNN) and GPU execution paths
- Use `get_device()` utility for consistent device management
- Implement batch size adaptation based on available memory

## Memory Orchestration Compatibility
- Methods should be stateful to support session continuity
- Include `get_system_state()` methods for checkpoint serialization
- Support memory injection patterns for cross-session learning transfer
- Implement proper cleanup for long-running learning sessions
