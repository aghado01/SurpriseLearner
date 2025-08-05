# Chatbot Assistant Tools

This directory contains helper scripts and tools created by AI assistants to support development and debugging.

## Directory Structure

### `_diagnostics/`
Scripts for diagnosing issues and checking system health:
- `format_scanner.py` - Scans repository for formatting issues
- `cicd_check.py` - Quick CI/CD readiness verification
- `validate_build.py` - Comprehensive build validation
- Other diagnostic utilities

### `_fixing/`
Scripts for automatically fixing common issues:
- `fix_formatting.py` - Comprehensive formatting repair
- `clean_whitespace.py` - Whitespace cleanup utility
- Other automated fix tools

## Usage Guidelines

- **For Developers**: These tools can be run manually for debugging
- **For AI Assistants**: Use these directories for organizing helper scripts
- **CI/CD**: These scripts are excluded from CI/CD processes

## Best Practices

1. **Organization**: Always place chatbot-created utilities in appropriate subdirectories
2. **Naming**: Use descriptive names that indicate the tool's purpose
3. **Documentation**: Include brief comments explaining what each script does
4. **Cleanup**: Periodically review and remove obsolete scripts

## File Naming Convention

- `*_scanner.py` - Analysis and detection tools
- `*_check.py` - Quick verification scripts
- `*_fix.py` - Automated repair utilities
- `*_diagnostic.py` - Health check tools

---
*This directory structure helps keep the main repository clean while providing useful development tools.*
