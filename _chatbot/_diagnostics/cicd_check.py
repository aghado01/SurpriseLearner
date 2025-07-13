#!/usr/bin/env python3
"""
Quick CI/CD readiness check.
"""

import subprocess
import sys

def check_cicd_readiness():
    """Quick check for CI/CD readiness."""
    print("🔍 Quick CI/CD Readiness Check\n")

    checks = [
        {
            'name': 'Package Import',
            'cmd': 'python -c "import adaptive_bayesian_driver; print(\'OK\')"',
            'critical': True
        },
        {
            'name': 'Critical Whitespace Issues',
            'cmd': 'python -m flake8 adaptive_bayesian_driver/ --select=W293 --count',
            'critical': True
        },
        {
            'name': 'Basic Syntax Check',
            'cmd': 'python -m py_compile setup.py',
            'critical': True
        },
        {
            'name': 'Requirements Check',
            'cmd': 'python -c "import torch, numpy, matplotlib; print(\'Dependencies OK\')"',
            'critical': False
        }
    ]

    passed = 0
    critical_passed = 0
    critical_total = sum(1 for check in checks if check['critical'])

    for check in checks:
        try:
            result = subprocess.run(check['cmd'], shell=True, capture_output=True, text=True, check=True)
            status = "✅ PASS"
            passed += 1
            if check['critical']:
                critical_passed += 1
        except subprocess.CalledProcessError:
            status = "❌ FAIL"

        critical_marker = "🔴" if check['critical'] else "⚪"
        print(f"{critical_marker} {check['name']}: {status}")

    print(f"\nResults: {passed}/{len(checks)} total, {critical_passed}/{critical_total} critical")

    if critical_passed == critical_total:
        print("🎉 CI/CD READY - All critical checks passed!")
        return True
    else:
        print("⚠️  CI/CD NOT READY - Critical issues remain")
        return False

if __name__ == "__main__":
    success = check_cicd_readiness()
    sys.exit(0 if success else 1)
