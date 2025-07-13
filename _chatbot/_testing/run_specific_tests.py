#!/usr/bin/env python3
"""Run the specific failing tests directly."""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import unittest
from tests.test_environment import TestSceneRenderer
from tests.test_recursive_bayes import TestSurpriseMeter

def main():
    # Test SceneRenderer
    print("="*50)
    print("Testing SceneRenderer")
    print("="*50)

    try:
        suite = unittest.TestLoader().loadTestsFromTestCase(TestSceneRenderer)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        if result.failures:
            print("\nFAILURES:")
            for test, traceback in result.failures:
                print(f"Test: {test}")
                print(f"Traceback: {traceback}")

        if result.errors:
            print("\nERRORS:")
            for test, traceback in result.errors:
                print(f"Test: {test}")
                print(f"Traceback: {traceback}")

    except Exception as e:
        print(f"Error running SceneRenderer tests: {e}")

    # Test SurpriseMeter
    print("\n" + "="*50)
    print("Testing SurpriseMeter")
    print("="*50)

    try:
        suite = unittest.TestLoader().loadTestsFromTestCase(TestSurpriseMeter)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        if result.failures:
            print("\nFAILURES:")
            for test, traceback in result.failures:
                print(f"Test: {test}")
                print(f"Traceback: {traceback}")

        if result.errors:
            print("\nERRORS:")
            for test, traceback in result.errors:
                print(f"Test: {test}")
                print(f"Traceback: {traceback}")

    except Exception as e:
        print(f"Error running SurpriseMeter tests: {e}")

if __name__ == "__main__":
    main()
