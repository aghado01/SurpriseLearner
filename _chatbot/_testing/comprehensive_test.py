#!/usr/bin/env python3
"""Comprehensive test runner to identify failing tests."""

import sys
import os
import subprocess
import traceback

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

def run_test(test_name):
    """Run a single test and capture output."""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', test_name, '-v'],
            capture_output=True,
            text=True,
            cwd=os.path.abspath('.')
        )

        return {
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    except Exception as e:
        return {
            'returncode': -1,
            'stdout': '',
            'stderr': str(e)
        }

def main():
    """Run all the failing tests mentioned."""
    failing_tests = [
        "tests/test_environment.py::TestSceneRenderer::test_render_scene",
        "tests/test_recursive_bayes.py::TestSurpriseMeter::test_baseline_update",
        "tests/test_recursive_bayes.py::TestSurpriseMeter::test_chi2_surprise",
        "tests/test_recursive_bayes.py::TestSurpriseMeter::test_normalized_surprise",
        "tests/test_recursive_bayes.py::TestSurpriseMeter::test_reconstruction_surprise",
    ]

    results = {}

    for test in failing_tests:
        print(f"\n{'='*60}")
        print(f"Running test: {test}")
        print('='*60)

        result = run_test(test)
        results[test] = result

        if result['returncode'] == 0:
            print("✅ PASSED")
        else:
            print("❌ FAILED")
            print(f"Return code: {result['returncode']}")
            print(f"STDOUT:\n{result['stdout']}")
            print(f"STDERR:\n{result['stderr']}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)

    passed = 0
    failed = 0

    for test, result in results.items():
        status = "✅ PASSED" if result['returncode'] == 0 else "❌ FAILED"
        print(f"{test}: {status}")
        if result['returncode'] == 0:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed")

if __name__ == "__main__":
    main()
