#!/usr/bin/env python3
"""
Chatbot file organization helper.
Use this to ensure all chatbot-created files are properly organized.
"""

import os
from pathlib import Path

def organize_chatbot_files():
    """Organize chatbot-created files into proper directories."""

    # Define the organizational structure
    structure = {
        '_chatbot/_diagnostics': [
            'format_scanner.py',
            'cicd_check.py',
            'diagnostic.py',
            'validate_build.py',
            'ci_test.py',
            'final_validation.py',
            '*_check.py',
            '*_scanner.py',
            '*_diagnostic.py'
        ],
        '_chatbot/_fixing': [
            'fix_formatting.py',
            'clean_whitespace.py',
            '*_fix.py',
            '*_cleanup.py'
        ]
    }

    project_root = Path('.')

    # Create directories if they don't exist
    for dir_path in structure.keys():
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Directory ready: {dir_path}")

    print("\n📁 Chatbot file organization structure ready!")
    print("\nFor future AI assistants:")
    print("- Place diagnostic tools in: _chatbot/_diagnostics/")
    print("- Place fixing tools in: _chatbot/_fixing/")
    print("- Never create scripts in project root")
    print("- Always use descriptive names")

def get_chatbot_dir(script_type):
    """Get the appropriate directory for a chatbot script type."""
    mapping = {
        'diagnostic': '_chatbot/_diagnostics',
        'fix': '_chatbot/_fixing',
        'check': '_chatbot/_diagnostics',
        'scan': '_chatbot/_diagnostics',
        'cleanup': '_chatbot/_fixing'
    }
    return mapping.get(script_type, '_chatbot')

if __name__ == "__main__":
    organize_chatbot_files()
