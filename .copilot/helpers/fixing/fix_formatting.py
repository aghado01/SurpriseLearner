#!/usr/bin/env python3
"""
Comprehensive formatter to fix all repository formatting issues.
"""

import os
import re
from pathlib import Path

def fix_file_encoding_and_format(file_path):
    """Fix encoding and formatting issues in a file."""
    print(f"Fixing: {file_path}")

    try:
        # Try to read with various encodings
        content = None
        for encoding in ['utf-8', 'iso-8859-1', 'windows-1252', 'macroman']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            print(f"  ❌ Could not decode {file_path}")
            return False

        # Fix common non-ASCII characters in code
        fixes = {
            '->': '->',  # Arrow to dash-greater
            '**2': '**2',  # Superscript 2
            '+/-': '+/-',  # Plus-minus
            'tau': 'tau',  # Greek tau
        }

        original_content = content
        for old, new in fixes.items():
            content = content.replace(old, new)

        # Remove trailing whitespace from each line
        lines = content.splitlines()
        lines = [line.rstrip() for line in lines]
        content = '\n'.join(lines)

        # Ensure file ends with newline if it's not empty
        if content and not content.endswith('\n'):
            content += '\n'

        # Write back as UTF-8
        with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(content)

        if content != original_content:
            print(f"  ✅ Fixed formatting and encoding")
        else:
            print(f"  ✅ Already correct")

        return True

    except Exception as e:
        print(f"  ❌ Error fixing {file_path}: {e}")
        return False

def restore_empty_files():
    """Restore content to important empty files."""
    empty_files_content = {
        'src/__init__.py': '"""Main package initialization."""\n',
        'src/environment/__init__.py': '"""Environment module."""\n',
        'src/models/__init__.py': '"""Models module."""\n',
        'src/utils/__init__.py': '"""Utilities module."""\n',
        'src/tests/__init__.py': '"""Tests module."""\n',
        'adaptive_bayesian_driver/models/utils_particle.py': '''"""
Particle filter utilities for adaptive Bayesian learning.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Callable
import logging

logger = logging.getLogger(__name__)

# This file was restored after being corrupted
# Content needs to be reimplemented based on project needs
''',
        'adaptive_bayesian_driver/utils/visualization.py': '''"""
Visualization utilities for adaptive Bayesian learning.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# This file was restored after being corrupted
# Content needs to be reimplemented based on project needs
''',
    }

    for file_path, content in empty_files_content.items():
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    current = f.read().strip()
                if not current:  # File is empty
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"✅ Restored: {file_path}")
            except Exception as e:
                print(f"❌ Error restoring {file_path}: {e}")

def main():
    """Main formatting fix function."""
    print("🔧 Fixing repository formatting issues...\n")

    # First restore empty files
    print("📝 Restoring empty files...")
    restore_empty_files()
    print()

    # Fix all Python and text files
    patterns = ['*.py', '*.md', '*.txt', '*.yml', '*.yaml']
    project_root = Path(".")

    fixed_count = 0
    total_count = 0

    for pattern in patterns:
        for file_path in project_root.rglob(pattern):
            # Skip certain directories
            if any(part in str(file_path) for part in ['.git', '__pycache__', '.pytest_cache', 'build', 'dist', '.egg-info']):
                continue

            total_count += 1
            if fix_file_encoding_and_format(file_path):
                fixed_count += 1

    print(f"\n🎉 Fixed {fixed_count}/{total_count} files")
    print("\n✅ Repository formatting cleanup complete!")
    print("\nRecommendations:")
    print("1. Run 'git status' to see what changed")
    print("2. Test your code after these fixes")
    print("3. Run 'python -m flake8 . --count' to check for remaining issues")

if __name__ == "__main__":
    main()
