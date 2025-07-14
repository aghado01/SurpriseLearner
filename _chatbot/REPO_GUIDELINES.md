# Repository Guidelines for AI Assistants

## Overview
This document provides guidelines for AI assistants working on the surprise-learning-exploration-exploitation project. These guidelines ensure consistent, professional, and maintainable development practices.

## File Organization

### 🎯 Core Principle: Keep Project Root Clean
- **NEVER** create helper scripts in the project root
- **ALWAYS** use organized subdirectories under `_chatbot/`

### Directory Structure
```
_chatbot/
├── _diagnostics/     # Scanning, checking, validation tools
├── _fixing/          # Automated repair and cleanup tools
├── _testing/         # Test utilities and debugging tools

├── debugging/        # Debug helpers and analysis tools
├── validation/       # CI/CD and quality assurance tools
├── reporting/        # Report generation and documentation tools
└── README.md         # This file
```

## Naming Conventions

### File Naming Patterns
- `*_scanner.py` - Analysis and detection tools
- `*_check.py` - Quick verification scripts
- `*_fix.py` - Automated repair utilities
- `*_diagnostic.py` - Health check tools
- `*_validator.py` - Validation utilities
- `*_reporter.py` - Report generation tools

### Code Quality Standards
- **Always** include proper docstrings
- **Always** handle errors gracefully
- **Always** use type hints where appropriate
- **Always** follow PEP 8 style guidelines

## Agentic Development Workflow

Coding updates in this review are treated as a collaborative and iterative process.

1. *Recieve Instructions*: Instructions will be provided in instruction files under `_chatbot\instructions\{instruction_file-name}.*` in either text or markdown format
2. *Process Instructions*: Analyze the instructions file for quality assurance by checking for flaws in logic, internally inconcistent guidance etc and provide a concise analysis/feedback summary to the user regarding the request.
3. *Execution Planning*: Generate an efficient plan for tackling the request based on your preview of the job and share a concise outline in chat with the user. Iterate with user if necessary.
4. *Apply changes*: After recieving confirmation from user, proceed with making the changes without many requests for user confirmation. Interrupt process and prompt user if you need additional guidance or are stuck. Every few steps, maybe between each segment of your plan, pause in your execution and review your recent actions history in order to detect being stuck in a loop and aborting or asking for user input accordingly.
5. *Review Changes*: After finishing the job, go through your changes and do quality checks like ensuring that updated files with changes have appropriate imports for example. Also review for any code propagations that may have been missed in the plan. Do standard testing like package test build according to guidelines below.
6. *Generate report*: Generate a detailed report of the changes made as well as including feedback to the user regarding challenges encountered and any suggestions regarding the source of the problem and advise on how to prevent in the future e.g. provide feedback on problematic instructions in terms of patterns, language, etc. Place your report in `reporting\refactor` with filename convention `{instructions_file-name}_report.md`


### Before executing job, be it many steps or modifying a single file
1. **Scan** for existing issues: `python _chatbot/_diagnostics/format_scanner.py`
2. **Check** CI/CD readiness: `python _chatbot/_diagnostics/cicd_check.py`
3. **Validate** current state: `python _chatbot/_diagnostics/validate_build.py`

### After Completing an update job / carrying out instructions to completion
1. **Fix** formatting issues: `python _chatbot/_fixing/fix_formatting.py`
2. **Clean** whitespace: `python _chatbot/_fixing/clean_whitespace.py`
3. **Test** the changes: `python -m pytest tests/`
4. **Verify** CI/CD: `python _chatbot/_diagnostics/cicd_check.py`

### Git Workflows

# Committing changes
1. **Stage** changes: `git add --renormalize .` if .gitattributes file is present else `git add .`
2. **Commit** with descriptive message: `git commit -m "{Instructions}"`
3. **Update** changelog if significant changes were made

# Merge workflow
If user requests a merge request to be prepared for the branch:
1. Generate or identify appropriate artifact to attach to the merge request Composea `merge_report` if more than one refactor report is associated with the changes on the current branch based on review of branch change history, in the form of a summary relevant refactor reports.
2. Attach the merge report or a single relevant refactor report, if only one exists for the current merge request, with modified filename for attachments to include appended datetimestamp.


## Package Structure Guidelines

### Main Package: `adaptive_bayesian_driver/`
- Core learning algorithms and models
- Well-tested, production-ready code
- Comprehensive docstrings and type hints

### Testing: `tests/`
- Unit tests for all major functionality
- Integration tests for key workflows
- Clear test names describing what is tested

### Configuration: `config/`
- YAML configuration files
- Environment-specific settings
- Experiment parameters

## Documentation Standards

### README Updates
- Keep installation instructions current
- Update usage examples when APIs change
- Maintain clear project description

### Changelog Maintenance
- Document all significant changes
- Use semantic versioning
- Include migration notes for breaking changes

### Code Documentation
- Include docstrings for all public functions/classes
- Add inline comments for complex logic
- Maintain API documentation

## CI/CD Considerations

### Critical Requirements
- All tests must pass: `python -m pytest`
- No linting errors: `python -m flake8 adaptive_bayesian_driver/`
- Package imports correctly: `python -c "import adaptive_bayesian_driver"`
- No formatting issues: Check with format scanner

### Common Issues to Avoid
- ❌ Non-ASCII characters in source code
- ❌ Trailing whitespace
- ❌ Mixed line endings
- ❌ Import errors
- ❌ Broken package structure

## Emergency Procedures

### If CI/CD Fails
1. Run: `python _chatbot/_diagnostics/format_scanner.py`
2. Fix issues: `python _chatbot/_fixing/fix_formatting.py`
3. Verify: `python _chatbot/_diagnostics/cicd_check.py`

### If Package Import Fails
1. Check `setup.py` configuration
2. Verify directory structure matches package layout
3. Ensure all `__init__.py` files exist

### If Tests Fail
1. Run specific test: `python -m pytest tests/test_specific.py -v`
2. Check for missing dependencies
3. Verify test data and fixtures

## Best Practices

### Code Quality
- Write self-documenting code
- Use meaningful variable names
- Keep functions focused and small
- Handle edge cases gracefully

### Performance
- Profile before optimizing
- Use appropriate data structures
- Cache expensive computations when beneficial
- Consider memory usage for large datasets

### Security
- Never commit sensitive information
- Use environment variables for configuration
- Validate all inputs
- Follow secure coding practices

## Tools and Utilities

### Available Diagnostic Tools
- `format_scanner.py` - Repository health check
- `cicd_check.py` - CI/CD readiness verification
- `validate_build.py` - Comprehensive build validation

### Available Fixing Tools
- `fix_formatting.py` - Automated formatting repair
- `clean_whitespace.py` - Whitespace cleanup

### Integration with IDE
- Use VS Code with Python extension
- Configure linting and formatting
- Set up debugging configurations

---

*These guidelines help maintain code quality and ensure smooth development workflows. Update this document as the project evolves.*
