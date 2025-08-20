---
applyTo: "**/*.ps1"
---

# PowerShell 7+ Helper Scripts Instructions

## Core PowerShell Standards
- **PowerShell Version**: Require PowerShell 7+ with `#Requires -Version 7.0`
- **Bracket Closure**: Complete bracket closure in all nested blocks - every `{`, `(`, `[` must have matching closing bracket
- **String Interpolation**: Use `$()` syntax for variable expansion in double-quoted strings
- **Call Operator**: Use `&` operator for explicit command invocation, especially with splatted parameters

## Script Structure Requirements
- Include comprehensive comment-based help with `.SYNOPSIS`, `.DESCRIPTION`, `.PARAMETER`, `.EXAMPLE`
- Use `[CmdletBinding()]` for advanced function capabilities
- Implement proper parameter validation with `[Parameter()]` attributes and `[ValidateSet()]` where applicable
- Include `try-catch-finally` blocks for robust error handling

## SurpriseLearner Project Context
This project implements an adaptive Bayesian learning framework with PowerShell automation tools for AI-assisted development workflows. When generating helper scripts:

- **Diagnostics**: Scripts for scanning, validation, and health checks should integrate with the hierarchical logging system
- **Fixing**: Automated repair tools should be non-destructive and use staging/review directories
- **Testing**: Test runners should support both CPU and CUDA execution paths from the adaptive_bayesian_driver package
- **Organization**: File management tools should respect the modern `.copilot/helpers/` directory structure

## Coding Patterns
- **Pipeline Support**: Design functions to work in PowerShell pipelines with proper input/output object types
- **Error Handling**: Implement comprehensive error handling with specific exit codes and detailed error messages
- **Documentation**: Maintain up-to-date help documentation that reflects the current implementation
- **Testing**: Include Pester tests for all non-trivial functions
- **Performance**: Optimize for performance, especially when processing large datasets
- **Security**: Follow security best practices, including input validation and secure credential handling

### Proper parameter splatting
$parameters = @{
Path = $SourcePath
Recurse = $true
Force = $Environment -eq 'Production'
}
$results = & Get-ChildItem @parameters

### String interpolation with complete syntax
$message = "Processing file: $($file.Name) in directory: $($file.Directory.FullName)"

### Nested bracket completion example
if ($condition) {
foreach ($item in $collection) {
if ($item.Property -eq $targetValue) {
Write-Output "Found match: $($item.Name)"}}}

## Integration Requirements
- All scripts must support `-WhatIf` parameter for safe preview mode
- Include progress indicators for long-running operations using `Write-Progress`
- Log operations to both console and file using the project's hierarchical logger patterns
- Ensure compatibility with both Windows and Linux execution environments
- Support the memory orchestration architecture for session continuity
