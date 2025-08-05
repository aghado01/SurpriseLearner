#!/usr/bin/env python3
"""
Demo script for adaptive-bayesian-driver package.
Tests basic functionality and package import.
"""

import sys
import os

# Add current directory to Python path for development
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import adaptive_bayesian_driver
    print("✓ Package import successful")
    print(f"✓ Version: {adaptive_bayesian_driver.__version__}")
    print(f"✓ Author: {adaptive_bayesian_driver.__author__}")
    print(f"✓ CUDA Available: {adaptive_bayesian_driver.CUDA_AVAILABLE}")
    print(f"✓ Device: {adaptive_bayesian_driver.DEVICE}")

    # Test device utility
    from adaptive_bayesian_driver.utils.device import (
        get_device,
        get_device_info
    )
    device = get_device()
    print(f"✓ Device utility: {device}")
    print(f"✓ Device info: {get_device_info()}")

    print("\n🎉 Demo completed successfully!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
