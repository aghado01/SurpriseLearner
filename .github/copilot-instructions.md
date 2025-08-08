# SurpriseLearner Project Copilot Instructions

## Project Overview
Adaptive Bayesian learning framework combining LC-NE inspired surprise detection with PowerShell Central memory orchestration for AI-assisted development workflows. This project bridges neuroscience principles to computer vision applications in autonomous driving scenarios.

## Memory Orchestration Architecture
This project implements a sophisticated four-tier memory management system:

- **Session Memory**: Current terminal session context and state
- **Sequence Memory**: Post-instruction execution logging and command history
- **Project Memory**: Cross-session continuity and workspace persistence
- **Global Memory**: Pattern learning and insights across multiple projects

All generated code must integrate with this hierarchical memory system for session continuity and AI-augmented workflows.

## Development Standards

### PowerShell 7+ Requirements
- **Version**: Use `#Requires -Version 7.0` for all scripts
- **Syntax**: Complete bracket closure in all nested blocks: `if ($condition) { foreach ($item in $collection) { ... } }`
- **String Interpolation**: Always use `$()` syntax: `"Processing: $($item.Name) in $($item.Directory.FullName)"`
- **Call Operator**: Use `&` for explicit command invocation: `& Get-ChildItem @parameters`
- **Parameter Splatting**: Use proper hashtable patterns with full variable expansion
- **Error Handling**: Comprehensive try-catch blocks with specific exception types
- **Code Completeness**: Provide full replacement functions, never partial snippets
- **Memory Integration**: Include hooks for hierarchical logging and session persistence
- **Safety**: Implement `-WhatIf` parameters for preview mode on destructive operations

### Python ML Standards
- **Type Hints**: Mandatory annotations using `from typing import Dict, List, Optional, Union, Any`
- **PyTorch Patterns**: CUDA-first development with CPU fallback using `torch.device` management
- **Error Handling**: Comprehensive exception handling with specific exception types and logging
- **Docstrings**: Google-style documentation for all classes and methods
- **Production Grade**: Follow reliability, security, and observability standards
- **Bayesian Learning**: Maintain compatibility with RecursiveBayesianLearner patterns
- **Hardware Optimization**: Leverage Intel compute optimizations and device management utilities

## Directory Structure & Organization

### Core Project Layout
```
SurpriseLearner/
├── .copilot/
│   ├── helpers/
│   │   ├── diagnostics/     # Scanning, validation, health checks (*_scanner.ps1, *_check.ps1)
│   │   ├── fixing/          # Automated repair utilities (*_fix.ps1, *_cleanup.ps1)
│   │   ├── testing/         # Test runners and debugging (*_test.ps1, test_*.py)
│   │   └── organize/        # File organization tools (*_organizer.ps1)
│   └── ForReview/           # Files requiring manual classification
├── adaptive_bayesian_driver/  # Main Python package
├── config/                  # YAML configuration files with validation
├── memory/                  # Memory orchestration scope directories
│   ├── session/            # Current session state
│   ├── sequence/           # Execution history
│   ├── project/            # Cross-session continuity
│   └── global/             # Cross-project patterns
└── tests/                  # Comprehensive test suites
```

### File Naming Conventions
- **Diagnostics**: `*_scanner.ps1`, `*_check.py`, `*_diagnostic.ps1`
- **Fixing**: `*_fix.ps1`, `fix_*.py`, `*_cleanup.ps1`
- **Testing**: `*_test.ps1`, `test_*.py`, `comprehensive_test.py`
- **Organization**: `organize*.ps1`, `*_organizer.ps1`

## AI Assistant Behavior Guidelines

### Code Generation Requirements
1. **Memory Architecture Integration**: All code should support the four-tier memory system
2. **Console-Aware Capture**: Include hooks for terminal output integration
3. **Complete Implementation**: Generate full, testable functions with comprehensive error handling
4. **Hardware Compatibility**: Support both CPU and CUDA execution paths
5. **Production Readiness**: Include logging, progress indicators, and robust error recovery
6. **Session Continuity**: Enable state persistence across AI assistant sessions

### Development Workflow Integration
- **Intel Optimizations**: Leverage compute_optimizations.py for 8-core system performance
- **Device Management**: Follow device.py patterns for CUDA/CPU abstraction
- **Configuration**: Integrate with existing YAML-based config management
- **Adaptive Learning**: Maintain compatibility with Bayesian learning components
- **Testing**: Support comprehensive test runners for both PowerShell and Python components

### Operational Workflow
1. **Diagnostics First**: Always run scanning tools before making changes
2. **Fix Systematically**: Use automated repair utilities to normalize code structure
3. **Test Thoroughly**: Run comprehensive tests after modifications
4. **Organize Continuously**: Maintain clean directory structure with organization tools
5. **Document Changes**: Update memory system with session and sequence logging

## Integration Points

### Core Components
- **RecursiveBayesianLearner.py**: Maintain LC-NE inspired surprise detection patterns
- **compute_optimizations.py**: Hardware-aware performance optimization
- **device.py**: Cross-platform CUDA/CPU device management
- **ProductionGrade.md**: Reliability, security, and observability criteria

### Memory Orchestration Features
- **Session Boundary Detection**: Automatic workspace transition handling
- **Context Injection**: Rolling context windows for AI assistance
- **Pattern Learning**: Cross-project insight accumulation
- **Console Integration**: Bidirectional terminal communication
- **State Persistence**: Atomic file operations for memory reliability

This architecture supports scalable, maintainable development workflows with sophisticated AI assistance integration and cross-session continuity.
