<#
.SYNOPSIS
Console enhancements with improved error handling and SmartProfileUtils integration.

.DESCRIPTION
Provides advanced console capture, enhanced prompt, and error detection.
Integrates with logging system and expertise corpus.
#>

function AdvancedConsoleCapture {
    <#
    .SYNOPSIS
    Captures console output with Copilot visibility and memory integration.

    .DESCRIPTION
    Executes commands while capturing output for Copilot context.
    Integrates with workspace context and memory systems.

    .PARAMETER Command
    Command to execute and capture

    .PARAMETER SessionContext
    Session context label for categorization

    .PARAMETER EnableCopilotVisibility
    Make output visible to Copilot via context files

    .PARAMETER EnableMemoryIntegration
    Integrate with memory orchestrator
    #>
    param(
        [Parameter(Mandatory)]
        [string]$Command,
        [string]$SessionContext = 'development',
        [switch]$EnableCopilotVisibility,
        [switch]$EnableMemoryIntegration
    )

    Write-Telemetry "Capturing command: $Command" -Level Host -Color Cyan

    # Execute command and capture all output
    $output = & $Command 2>&1 | Tee-Object -Variable capturedOutput

    # Make output visible to Copilot via workspace context
    if ($EnableCopilotVisibility) {
        # Ensure context directory exists
        $contextDir = "$($script:WorkspaceContext.ContextDirectory)"
        if (-not (Test-Path $contextDir)) {
            New-Item -Path $contextDir -ItemType Directory -Force | Out-Null
        }

        $contextFile = "$contextDir/recent-output.txt"
        $capturedOutput | Out-File $contextFile -Encoding UTF8

        $contextSummary = @{
            timestamp       = Get-Date -f "yyyy-MM-ddTHH:mm:ss"
            command         = $Command
            output          = $capturedOutput -join "`n"
            session_context = $SessionContext
            workspace       = $script:WorkspaceContext.Name
        } | ConvertTo-Json -Depth 3

        $contextSummary | Out-File "$contextDir/command-context.json" -Encoding UTF8
    }

    # Integrate with memory orchestrator for session continuity
    if ($EnableMemoryIntegration -and $global:CopilotMemoryOrchestrator) {
        try {
            $global:CopilotMemoryOrchestrator.Capture($Command, ($capturedOutput -join "`n"))
            $global:CopilotMemoryOrchestrator.Dump()
        }
        catch {
            Write-Verbose "Memory integration failed: $($_.Exception.Message)"
        }
    }

    # Log the command execution
    try {
        Write-CopilotSessionLog -Action "command_captured" -Command $Command -Output ($capturedOutput -join "`n") -Metadata @{
            session_context    = $SessionContext
            copilot_visibility = $EnableCopilotVisibility.IsPresent
            memory_integration = $EnableMemoryIntegration.IsPresent
        }
    }
    catch {
        Write-Verbose "Command logging failed: $($_.Exception.Message)"
    }

    return $output
}

function Get-CopilotEnhancedPrompt {
    <#
    .SYNOPSIS
    Generates enhanced PowerShell prompt with workspace context indicators.

    .DESCRIPTION
    Creates informative prompt showing session, context, git status, and infrastructure state.
    Integrates workspace context and expertise corpus availability.
    #>

    # Path shortening for readability
    $shortPath = if ($PWD.Path.Length -gt 40) {
        "..." + $PWD.Path.Substring($PWD.Path.Length - 37)
    }
    else {
        $PWD.Path
    }

    # Session info with enhanced context
    $sessionInfo = " 🤖$($script:WorkspaceContext.SessionId.Substring(0,4))"

    # Context info with expertise indicator
    $contextInfo = " [$($script:WorkspaceContext.Name):$($script:WorkspaceContext.Type)]"
    if ($script:WorkspaceContext.InfrastructureSetup) { $contextInfo += "🔧" }

    # Check for expertise corpus
    $expertisePath = "$($script:WorkspaceContext.Root)/.copilot/expertise/corpus-index.json"
    if (Test-Path $expertisePath) { $contextInfo += "📚" }

    # Git info
    $gitInfo = if ($script:WorkspaceContext.GitBranch) {
        " ($($script:WorkspaceContext.GitBranch))"
    }
    elseif ($script:WorkspaceContext.IsGitRepo) {
        " (detached)"
    }
    else {
        ""
    }

    # Instructions and logging indicators
    $indicators = ""
    if ($script:WorkspaceContext.HasExistingInstructions) { $indicators += " 📋" }

    # Check for recent activity
    $recentActivity = Get-CopilotRecentActivity -Hours 1
    if ($recentActivity.Count -gt 0) { $indicators += " 🔥" }

    Write-Host "PS" -NoNewline -ForegroundColor Blue
    Write-Host $sessionInfo -NoNewline -ForegroundColor Cyan
    Write-Host $contextInfo -NoNewline -ForegroundColor Yellow
    Write-Host " $shortPath" -NoNewline -ForegroundColor Green
    Write-Host $gitInfo -NoNewline -ForegroundColor Magenta
    Write-Host $indicators -NoNewline -ForegroundColor Gray
    Write-Host ">" -NoNewline -ForegroundColor Blue
    return " "
}

function Test-CommandForErrors {
    <#
    .SYNOPSIS
    Analyzes command output for error patterns and logs findings.

    .DESCRIPTION
    Scans output for common error indicators and logs project events.
    Integrates with project logging system for error tracking.

    .PARAMETER Output
    Command output to analyze
    #>
    param(
        [Parameter(Mandatory)]
        [string]$Output
    )

    $errorPatterns = @("ERROR", "FAILED", "EXCEPTION", "FATAL", "CRITICAL")
    $hasErrors = $errorPatterns | Where-Object { $Output -match $_ }

    if ($hasErrors) {
        Write-Telemetry "Errors detected in command output" -Level Warning -Color Red

        try {
            Write-CopilotProjectLog -Event "command_errors_detected" -Metadata @{
                error_patterns = $hasErrors
                timestamp      = Get-Date
                output_length  = $Output.Length
            }
        }
        catch {
            Write-Verbose "Error logging failed: $($_.Exception.Message)"
        }
    }

    return [bool]$hasErrors
}

function Invoke-CopilotCommandWithCapture {
    <#
    .SYNOPSIS
    Executes command with integrated capture and error detection.

    .DESCRIPTION
    Combines command execution, output capture, error detection, and logging.
    Provides unified interface for enhanced command processing.

    .PARAMETER Command
    Command to execute with full capture
    #>
    param(
        [Parameter(Mandatory)]
        [string]$Command
    )

    Write-Telemetry "Executing with capture: $Command" -Level Host -Color Green

    $output = & $Command 2>&1
    $hasErrors = Test-CommandForErrors ($output -join "`n")

    try {
        Write-CopilotSessionLog -Action "command_executed" -Command $Command -Output ($output -join "`n") -Metadata @{
            has_errors   = $hasErrors
            output_lines = $output.Count
        }
    }
    catch {
        Write-Verbose "Command execution logging failed: $($_.Exception.Message)"
    }

    return $output
}

# Set alias with proper function reference
Set-Alias -Name "capture" -Value "AdvancedConsoleCapture" -Scope Global -Force
