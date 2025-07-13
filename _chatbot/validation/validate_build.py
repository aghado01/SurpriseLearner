#!/usr/bin/env python3
"""
Quick validation script to test if the package builds and imports correctly.
This helps identify CI/CD issues locally.
"""

import sys
import traceback

def test_imports():
    """Test that all main modules can be imported."""
    print("Testing imports...")

    try:
        import adaptive_bayesian_driver
        print("✓ Main package import successful")
    except Exception as e:
        print(f"✗ Main package import failed: {e}")
        return False

    try:
        from adaptive_bayesian_driver.models import RecursiveBayesianLearner, SurpriseMeter
        print("✓ Core models import successful")
    except Exception as e:
        print(f"✗ Core models import failed: {e}")
        return False

    try:
        from adaptive_bayesian_driver.environment import SceneRenderer, VolatilityController
        print("✓ Environment modules import successful")
    except Exception as e:
        print(f"✗ Environment modules import failed: {e}")
        return False

    try:
        from adaptive_bayesian_driver.config import load_config
        print("✓ Config module import successful")
    except Exception as e:
        print(f"✗ Config module import failed: {e}")
        return False

    return True

def test_basic_functionality():
    """Test basic functionality without full computation."""
    print("\nTesting basic functionality...")

    try:
        from adaptive_bayesian_driver.models import SurpriseMeter
        from adaptive_bayesian_driver.models.surprise import SurpriseType

        # Test SurpriseMeter instantiation
        meter = SurpriseMeter(mode=SurpriseType.EPISTEMIC)
        print("✓ SurpriseMeter instantiation successful")

        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main validation function."""
    print("=== Build Validation ===")

    success = True

    # Test imports
    if not test_imports():
        success = False

    # Test basic functionality
    if not test_basic_functionality():
        success = False

    if success:
        print("\n✅ All validation tests passed!")
        return 0
    else:
        print("\n❌ Some validation tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
