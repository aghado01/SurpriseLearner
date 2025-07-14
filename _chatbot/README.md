# `_chatbot`: Coding Assistant Central

This directory contains helper scripts and tools created by AI assistants to support development and debugging.

## Directory Structure

### `_instructions/`
potentially freeform documents containing specific instructions for executing agentic updates

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

## File Naming Conventions

- `*_scanner.py` - Analysis and detection tools
- `*_check.py` - Quick verification scripts
- `*_fix.py` - Automated repair utilities
- `*_diagnostic.py` - Health check tools

- `*_report.md` - for report files - see `reporting\README.md` for more details on reporting filename conventions

## Reminders
- Review other markdown guidance files under the _chatbot directory for additional context and guidance
- when you move a file, don't leave the empty version of the file in its original location
- if a package import fails during your work, a natural thing to do is import it to the environment and suggest adding the package to the requirements file as an on-the-fly adjustment to the current iteration cycle
 - for every file you edit, always check that all packages are imported in the file as needed if the functionality isn't somehow being inherited implicitly
  - try to avoid getting stuck in package import testing loops and focus on completing instruction task archs
  - prompt the human for clarifications or to discuss issues rather than getting stuck in loops

---
*This directory structure helps keep the main repository clean while providing useful development tools.*
