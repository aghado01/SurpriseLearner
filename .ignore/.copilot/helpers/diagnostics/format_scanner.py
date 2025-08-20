#!/usr/bin/env python3
"""
Comprehensive formatting scanner for the repository.
Detects line ending issues, encoding problems, and whitespace issues.
"""

import os
import re
import chardet
from pathlib import Path

def check_file_encoding(file_path):
    """Check file encoding and detect potential issues."""
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()

        if not raw_data:
            return "empty", None

        detected = chardet.detect(raw_data)
        encoding = detected.get('encoding', 'unknown')
        confidence = detected.get('confidence', 0)

        # Check for BOM
        has_bom = raw_data.startswith(b'\xef\xbb\xbf')

        return encoding, confidence, has_bom
    except Exception as e:
        return f"error: {e}", None, None

def check_line_endings(file_path):
    """Check line ending consistency."""
    try:
        with open(file_path, 'rb') as f:
            content = f.read()

        if not content:
            return "empty"

        crlf_count = content.count(b'\r\n')
        lf_count = content.count(b'\n') - crlf_count
        cr_count = content.count(b'\r') - crlf_count

        endings = []
        if crlf_count > 0:
            endings.append(f"CRLF({crlf_count})")
        if lf_count > 0:
            endings.append(f"LF({lf_count})")
        if cr_count > 0:
            endings.append(f"CR({cr_count})")

        if len(endings) > 1:
            return f"MIXED: {', '.join(endings)}"
        elif endings:
            return endings[0]
        else:
            return "NO_ENDINGS"

    except Exception as e:
        return f"error: {e}"

def check_whitespace_issues(file_path):
    """Check for whitespace problems."""
    issues = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        for i, line in enumerate(lines, 1):
            # Trailing whitespace
            if line.rstrip('\n\r') != line.rstrip():
                issues.append(f"Line {i}: trailing whitespace")

            # Tabs mixed with spaces
            if '\t' in line and '    ' in line:
                issues.append(f"Line {i}: mixed tabs and spaces")

            # Non-ASCII characters that might cause issues
            try:
                line.encode('ascii')
            except UnicodeEncodeError:
                non_ascii = [c for c in line if ord(c) > 127]
                if non_ascii:
                    issues.append(f"Line {i}: non-ASCII chars: {non_ascii[:3]}")

        return issues
    except Exception as e:
        return [f"error reading file: {e}"]

def scan_repository():
    """Scan the entire repository for formatting issues."""
    project_root = Path(".")
    issues_found = {}

    # File patterns to check
    patterns = ['*.py', '*.txt', '*.md', '*.yml', '*.yaml', '*.cfg', '*.ini', '*.json']

    print("🔍 Scanning repository for formatting issues...\n")

    for pattern in patterns:
        for file_path in project_root.rglob(pattern):
            # Skip certain directories
            if any(part in str(file_path) for part in ['.git', '__pycache__', '.pytest_cache', 'build', 'dist']):
                continue

            print(f"Checking: {file_path}")
            file_issues = []

            # Check encoding
            encoding_info = check_file_encoding(file_path)
            if encoding_info[0] not in ['utf-8', 'ascii'] or (len(encoding_info) > 2 and encoding_info[2]):
                file_issues.append(f"Encoding: {encoding_info}")

            # Check line endings
            line_endings = check_line_endings(file_path)
            if 'MIXED' in line_endings or 'CR(' in line_endings:
                file_issues.append(f"Line endings: {line_endings}")

            # Check whitespace issues for text files
            if file_path.suffix in ['.py', '.txt', '.md', '.yml', '.yaml', '.cfg']:
                whitespace_issues = check_whitespace_issues(file_path)
                if whitespace_issues:
                    file_issues.extend(whitespace_issues[:5])  # Limit to first 5 issues

            if file_issues:
                issues_found[str(file_path)] = file_issues

    return issues_found

def main():
    """Main function to run the formatting scan."""
    issues = scan_repository()

    if not issues:
        print("✅ No formatting issues found!")
        return 0

    print(f"\n❌ Found formatting issues in {len(issues)} files:\n")

    for file_path, file_issues in issues.items():
        print(f"📁 {file_path}:")
        for issue in file_issues:
            print(f"   • {issue}")
        print()

    print(f"Total files with issues: {len(issues)}")
    return 1

if __name__ == "__main__":
    exit(main())
