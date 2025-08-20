# PowerShell 7+ Legacy .copilot Directory Modernization Script
# Brings legacy chatbot/copilot structure in line with current practices

param(
    [string]$SourceDir = $PWD.Path,
    [string]$BackupSuffix = "_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')",
    [switch]$WhatIf,
    [switch]$Verbose
)

# Modern .copilot structure definition
$ModernStructure = @{
    '.copilot' = @{
        'helpers'   = @{
            'diagnostics' = @('*_scanner.py', '*_check.py', '*_diagnostic.py', 'format_scanner.py', 'cicd_check.py', 'validate_*.py')
            'fixing'      = @('*_fix.py', '*_cleanup.py', 'fix_*.py', 'clean_*.py')
            'testing'     = @('*_test.py', 'test_*.py', 'comprehensive_test.py', 'run_*_tests.py')
            'organize'    = @('organize*.py', '*_organizer.py', 'organize.py')
        }
        'ForReview' = @()
    }
}

# File classification rules
$ClassificationRules = @{
    'diagnostics' = @(
        @{ Pattern = '*scanner*'; Weight = 10 }
        @{ Pattern = '*check*'; Weight = 10 }
        @{ Pattern = '*diagnostic*'; Weight = 10 }
        @{ Pattern = '*validate*'; Weight = 8 }
        @{ Pattern = '*ci*'; Weight = 6 }
        @{ Pattern = 'format_scanner.py'; Weight = 15 }
        @{ Pattern = 'cicd_check.py'; Weight = 15 }
    )
    'fixing'      = @(
        @{ Pattern = '*fix*'; Weight = 10 }
        @{ Pattern = '*cleanup*'; Weight = 10 }
        @{ Pattern = '*clean*'; Weight = 8 }
        @{ Pattern = 'fix_formatting.py'; Weight = 15 }
    )
    'testing'     = @(
        @{ Pattern = '*test*'; Weight = 10 }
        @{ Pattern = 'comprehensive_test.py'; Weight = 15 }
        @{ Pattern = 'run_*_tests.py'; Weight = 12 }
    )
    'organize'    = @(
        @{ Pattern = 'organize*'; Weight = 15 }
        @{ Pattern = '*organizer*'; Weight = 10 }
    )
}

function Write-ModernizationLog {
    param(
        [string]$Message,
        [string]$Level = "INFO",
        [ConsoleColor]$Color = "White"
    )

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] [$Level] $Message"

    if ($Verbose -or $Level -eq "ERROR" -or $Level -eq "WARNING") {
        Write-Host $logMessage -ForegroundColor $Color
    }

    # Also log to file
    $logMessage | Out-File -FilePath "$(Join-Path $SourceDir 'modernization.log')" -Append -Encoding UTF8
}

function Get-FileClassification {
    param(
        [System.IO.FileInfo]$File
    )

    $bestMatch = $null
    $highestScore = 0

    foreach ($category in $ClassificationRules.Keys) {
        foreach ($rule in $ClassificationRules[$category]) {
            if ($File.Name -like $rule.Pattern) {
                if ($rule.Weight -gt $highestScore) {
                    $highestScore = $rule.Weight
                    $bestMatch = $category
                }
            }
        }
    }

    return @{
        Category   = $bestMatch
        Confidence = $highestScore
    }
}

function Test-IsRelevantFile {
    param(
        [System.IO.FileInfo]$File
    )

    # Check if file is relevant based on various criteria
    $relevancyScore = 0

    # PowerShell files are generally relevant
    if ($File.Extension -eq '.ps1' -or $File.Extension -eq '.psm1') {
        $relevancyScore += 10
    }

    # Python files with specific patterns
    if ($File.Extension -eq '.py') {
        $relevancyScore += 5
    }

    # Configuration and documentation files
    if ($File.Extension -in @('.md', '.yml', '.yaml', '.json', '.txt')) {
        $relevancyScore += 3
    }

    # Check for helper/utility keywords in filename
    $utilityKeywords = @('helper', 'utility', 'tool', 'script', 'fix', 'test', 'diagnostic', 'scanner', 'check')
    foreach ($keyword in $utilityKeywords) {
        if ($File.BaseName -like "*$keyword*") {
            $relevancyScore += 2
        }
    }

    # Files that are likely irrelevant
    $irrelevantPatterns = @('temp*', 'tmp*', '*.bak', '*.old', '*backup*', '*.cache')
    foreach ($pattern in $irrelevantPatterns) {
        if ($File.Name -like $pattern) {
            $relevancyScore -= 10
        }
    }

    return $relevancyScore -gt 2
}

function New-ModernDirectoryStructure {
    param(
        [string]$BasePath
    )

    Write-ModernizationLog "Creating modern .copilot directory structure..." "INFO" "Cyan"

    $copilotPath = Join-Path $BasePath ".copilot"
    $helpersPath = Join-Path $copilotPath "helpers"

    # Create main directories
    $directories = @(
        $copilotPath,
        $helpersPath,
        (Join-Path $helpersPath "diagnostics"),
        (Join-Path $helpersPath "fixing"),
        (Join-Path $helpersPath "testing"),
        (Join-Path $helpersPath "organize"),
        (Join-Path $copilotPath "ForReview")
    )

    foreach ($dir in $directories) {
        if (!(Test-Path $dir)) {
            if (!$WhatIf) {
                & New-Item -Path $dir -ItemType Directory -Force | Out-Null
            }
            Write-ModernizationLog "Created directory: $dir" "INFO" "Green"
        }
        else {
            Write-ModernizationLog "Directory already exists: $dir" "INFO" "Yellow"
        }
    }

    # Create README files for each directory
    $readmeContent = @{
        'diagnostics' = "# Diagnostic Tools`n`nThis directory contains scripts for scanning, checking, and validating system health.`n`n- Format scanners`n- CI/CD readiness checks`n- Build validation tools`n- Health diagnostics"
        'fixing'      = "# Fixing Tools`n`nThis directory contains automated repair and cleanup utilities.`n`n- Formatting fixers`n- Whitespace cleanup`n- Automated repairs`n- Code normalization"
        'testing'     = "# Testing Utilities`n`nThis directory contains test runners and debugging tools.`n`n- Comprehensive test runners`n- Specific test utilities`n- Debug helpers`n- Test orchestration"
        'organize'    = "# Organization Tools`n`nThis directory contains file organization and structure management tools.`n`n- Directory organizers`n- File classification tools`n- Structure maintenance"
    }

    foreach ($category in $readmeContent.Keys) {
        $readmePath = Join-Path (Join-Path $helpersPath $category) "README.md"
        if (!(Test-Path $readmePath)) {
            if (!$WhatIf) {
                $readmeContent[$category] | Out-File -FilePath $readmePath -Encoding UTF8
            }
            Write-ModernizationLog "Created README: $readmePath" "INFO" "Green"
        }
    }
}

function Move-LegacyFiles {
    param(
        [string]$SourcePath,
        [string]$DestinationBase
    )

    Write-ModernizationLog "Analyzing and moving legacy files..." "INFO" "Cyan"

    # Get all files from legacy directories
    $legacyPaths = @('_chatbot', '_diagnostics', '_fixing', '_testing', 'debugging', 'validation', 'reporting')
    $allFiles = @()

    foreach ($legacyPath in $legacyPaths) {
        $fullLegacyPath = Join-Path $SourcePath $legacyPath
        if (Test-Path $fullLegacyPath) {
            Write-ModernizationLog "Scanning legacy directory: $legacyPath" "INFO" "Gray"
            $files = & Get-ChildItem -Path $fullLegacyPath -Recurse -File
            $allFiles += $files
        }
    }

    # Also scan root directory for orphaned files
    $rootFiles = & Get-ChildItem -Path $SourcePath -File | Where-Object {
        $_.Extension -in @('.py', '.ps1', '.psm1', '.md') -and
        $_.Name -notlike 'README*' -and
        $_.Name -notlike 'modernization*'
    }
    $allFiles += $rootFiles

    Write-ModernizationLog "Found $($allFiles.Count) files to analyze" "INFO" "White"

    $moveOperations = @{
        'diagnostics' = @()
        'fixing'      = @()
        'testing'     = @()
        'organize'    = @()
        'ForReview'   = @()
    }

    # Classify each file
    foreach ($file in $allFiles) {
        $classification = Get-FileClassification -File $file
        $isRelevant = Test-IsRelevantFile -File $file

        if ($classification.Category -and $isRelevant) {
            $moveOperations[$classification.Category] += @{
                File       = $file
                Confidence = $classification.Confidence
                Reason     = "Classified as $($classification.Category) with confidence $($classification.Confidence)"
            }
            Write-ModernizationLog "Classified '$($file.Name)' as $($classification.Category) (confidence: $($classification.Confidence))" "INFO" "Gray"
        }
        else {
            $reason = if (!$isRelevant) { "Low relevancy score" } else { "No clear classification" }
            $moveOperations['ForReview'] += @{
                File       = $file
                Confidence = 0
                Reason     = $reason
            }
            Write-ModernizationLog "Moving '$($file.Name)' to ForReview: $reason" "WARNING" "Yellow"
        }
    }

    # Execute move operations
    foreach ($category in $moveOperations.Keys) {
        $targetDir = if ($category -eq 'ForReview') {
            Join-Path $DestinationBase ".copilot\ForReview"
        }
        else {
            Join-Path $DestinationBase ".copilot\helpers\$category"
        }

        foreach ($operation in $moveOperations[$category]) {
            $sourceFile = $operation.File
            $targetPath = Join-Path $targetDir $sourceFile.Name

            # Handle name conflicts
            if (Test-Path $targetPath) {
                $counter = 1
                $baseName = [System.IO.Path]::GetFileNameWithoutExtension($sourceFile.Name)
                $extension = $sourceFile.Extension
                do {
                    $newName = "$baseName`_$counter$extension"
                    $targetPath = Join-Path $targetDir $newName
                    $counter++
                } while (Test-Path $targetPath)

                Write-ModernizationLog "Renamed due to conflict: $($sourceFile.Name) -> $(Split-Path $targetPath -Leaf)" "WARNING" "Yellow"
            }

            if (!$WhatIf) {
                try {
                    & Move-Item -Path $sourceFile.FullName -Destination $targetPath -Force
                    Write-ModernizationLog "Moved: $($sourceFile.FullName) -> $targetPath" "INFO" "Green"
                }
                catch {
                    Write-ModernizationLog "Failed to move $($sourceFile.FullName): $($_.Exception.Message)" "ERROR" "Red"
                }
            }
            else {
                Write-ModernizationLog "WOULD MOVE: $($sourceFile.FullName) -> $targetPath" "INFO" "Magenta"
            }
        }
    }
}

function Remove-EmptyLegacyDirectories {
    param(
        [string]$BasePath
    )

    Write-ModernizationLog "Cleaning up empty legacy directories..." "INFO" "Cyan"

    $legacyPaths = @('_chatbot', '_diagnostics', '_fixing', '_testing', 'debugging', 'validation', 'reporting')

    foreach ($legacyPath in $legacyPaths) {
        $fullLegacyPath = Join-Path $BasePath $legacyPath
        if (Test-Path $fullLegacyPath) {
            # Check if directory is empty (recursively)
            $remainingItems = & Get-ChildItem -Path $fullLegacyPath -Recurse -Force

            if ($remainingItems.Count -eq 0) {
                if (!$WhatIf) {
                    & Remove-Item -Path $fullLegacyPath -Recurse -Force
                    Write-ModernizationLog "Removed empty legacy directory: $legacyPath" "INFO" "Green"
                }
                else {
                    Write-ModernizationLog "WOULD REMOVE empty directory: $legacyPath" "INFO" "Magenta"
                }
            }
            else {
                Write-ModernizationLog "Legacy directory $legacyPath still contains $($remainingItems.Count) items - keeping" "INFO" "Yellow"
            }
        }
    }
}

function New-CopilotInstructionsFile {
    param(
        [string]$BasePath
    )

    $instructionsPath = Join-Path $BasePath ".copilot\copilot-instructions.md"

    $instructions = @"
# .copilot Directory Instructions

This directory contains modernized helper tools and utilities organized according to current PowerShell Central practices.

## Directory Structure

- **helpers/diagnostics/** - Scanning, checking, and validation tools
- **helpers/fixing/** - Automated repair and cleanup utilities
- **helpers/testing/** - Test runners and debugging tools
- **helpers/organize/** - File organization and structure management
- **ForReview/** - Files requiring manual review and classification

## PowerShell Standards

- Use PowerShell 7+ syntax with complete bracket closure
- Implement proper string interpolation with `$()` syntax
- Include comprehensive error handling and logging
- Follow strict parameter validation and type hints
- Provide full replacement code, not snippets

## Usage Guidelines

1. **Diagnostics First** - Always run diagnostic tools before making changes
2. **Fix Systematically** - Use fixing tools to normalize code formatting and structure
3. **Test Thoroughly** - Run comprehensive tests after any changes
4. **Organize Continuously** - Maintain clean directory structure with organization tools

## File Naming Conventions

- `*_scanner.ps1` - Analysis and detection tools
- `*_check.ps1` - Quick verification scripts
- `*_fix.ps1` - Automated repair utilities
- `*_test.ps1` - Test utilities and runners
- `*_organizer.ps1` - Structure management tools

This structure supports scalable, maintainable PowerShell development workflows with AI assistance integration.
"@

    if (!$WhatIf) {
        $instructions | Out-File -FilePath $instructionsPath -Encoding UTF8
    }

    Write-ModernizationLog "Created copilot instructions file: $instructionsPath" "INFO" "Green"
}

# Main execution
try {
    Write-ModernizationLog "Starting .copilot directory modernization..." "INFO" "Cyan"
    Write-ModernizationLog "Source directory: $SourceDir" "INFO" "White"

    if ($WhatIf) {
        Write-ModernizationLog "Running in WhatIf mode - no changes will be made" "INFO" "Magenta"
    }

    # Step 1: Create modern directory structure
    New-ModernDirectoryStructure -BasePath $SourceDir

    # Step 2: Move and classify legacy files
    Move-LegacyFiles -SourcePath $SourceDir -DestinationBase $SourceDir

    # Step 3: Clean up empty legacy directories
    Remove-EmptyLegacyDirectories -BasePath $SourceDir

    # Step 4: Create copilot instructions
    New-CopilotInstructionsFile -BasePath $SourceDir

    Write-ModernizationLog "Modernization complete!" "INFO" "Green"
    Write-ModernizationLog "Review the ForReview directory for files needing manual classification" "INFO" "Yellow"
    Write-ModernizationLog "Log file created: $(Join-Path $SourceDir 'modernization.log')" "INFO" "Gray"

}
catch {
    Write-ModernizationLog "Error during modernization: $($_.Exception.Message)" "ERROR" "Red"
    throw
}
