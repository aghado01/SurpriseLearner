#Requires -Version 7.0

<#
.SYNOPSIS
    Executes PowerShell code in a completely isolated sub-instance with clean environment

.DESCRIPTION
    Creates a fresh PowerShell sub-instance that bypasses all profile loading, function caching,
    and environment variable interference. Perfect for testing parallel processing functions
    without conflicts from the main PowerShell session.

.PARAMETER ScriptBlock
    The script block to execute in the isolated environment

.PARAMETER ModulesToLoad
    Array of PowerShell script files to load (e.g., ParallelLoop.ps1, batch processing scripts)

.PARAMETER EnvironmentOverrides
    Hashtable of environment variables to set in the isolated instance

.PARAMETER ShowOutput
    Display the output from the isolated instance

.PARAMETER PassThru
    Return the results from the isolated execution

.PARAMETER TimeoutSeconds
    Maximum time to wait for the isolated instance (default: 60 seconds)

.EXAMPLE
    # Test parallel processing in complete isolation
    Invoke-IsolatedSubInstance -ScriptBlock {
        $data = 1..10
        $result = $data | parforeach { param($x) $x * 2 }
        Write-Output "Results: $($result -join ', ')"
    } -ModulesToLoad @('ParallelLoop.ps1') -ShowOutput

.EXAMPLE
    # Test with custom environment and return results
    $results = Invoke-IsolatedSubInstance -ScriptBlock {
        Invoke-ParallelBatchProcessing -InputData (1..5) -ScriptBlock { param($n) $n * $n } -ShowProgress
    } -ModulesToLoad @('ParallelLoop.ps1', 'Invoke-ParallelBatchProcessing.ps1') -PassThru

.AUTHOR
    Azriel Ghadooshahy - Using Invoke-WithTempEnv for isolation
#>
function Invoke-IsolatedSubInstance {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [ScriptBlock]$ScriptBlock,

        [string[]]$ModulesToLoad = @(),

        [hashtable]$EnvironmentOverrides = @{},

        [switch]$ShowOutput,

        [switch]$PassThru,

        [ValidateRange(10, 300)]
        [int]$TimeoutSeconds = 60,

        [switch]$ShowDetails
    )

    # Load Invoke-WithTempEnv if not available
    $tempEnvPath = "$env:PSLibraryPath\Functions\Invoke-WithTempEnv.ps1"
    if (-not (Get-Command 'Invoke-WithTempEnv' -ErrorAction SilentlyContinue) -and (Test-Path $tempEnvPath)) {
        . $tempEnvPath
        if ($ShowDetails) { Write-Host "✅ Loaded Invoke-WithTempEnv for isolation" -ForegroundColor Green }
    }

    if (-not (Get-Command 'Invoke-WithTempEnv' -ErrorAction SilentlyContinue)) {
        throw "Invoke-WithTempEnv function is required but not available. Check path: $tempEnvPath"
    }

    # Prepare isolated environment variables
    $isolatedEnv = @{
        # Clear profile-related variables to prevent loading
        'PROFILE' = ''
        'PSLibraryPath' = ''
        'PORTABLE_HOME' = ''

        # Force clean execution
        'PSModulePath' = [System.Environment]::GetEnvironmentVariable('PSModulePath', 'Machine')
        'PSExecutionPolicyPreference' = 'Bypass'

        # Disable other potential interference
        'POWERSHELL_TELEMETRY_OPTOUT' = '1'
        'DISABLE_PARALLEL_ALIASES' = 'true'
    }

    # Merge with user-provided overrides
    foreach ($override in $EnvironmentOverrides.GetEnumerator()) {
        $isolatedEnv[$override.Key] = $override.Value
    }

    # Build the script to execute in isolation
    $isolatedScript = @"
# PowerShell Isolated Sub-Instance Execution
Write-Host "🔬 Starting isolated PowerShell sub-instance..." -ForegroundColor Cyan
Write-Host "📍 PowerShell Version: `$(`$PSVersionTable.PSVersion)" -ForegroundColor Gray
Write-Host "🧹 Clean Environment: Profile disabled, functions cleared" -ForegroundColor Gray

# Ensure we're starting clean
`$ErrorActionPreference = 'Stop'

# Load required modules in isolation
"@

    # Add module loading logic
    if ($ModulesToLoad.Count -gt 0) {
        $isolatedScript += @"

Write-Host "`n📚 Loading required modules in isolation..." -ForegroundColor Yellow
"@
        foreach ($module in $ModulesToLoad) {
            # Determine full path for module
            $modulePath = if ([System.IO.Path]::IsPathRooted($module)) {
                $module
            } else {
                "C:\Users\azriy\Development\PowerShellLib\Functions\$module"
            }

            $isolatedScript += @"

# Load $module
`$modulePath = '$modulePath'
if (Test-Path `$modulePath) {
    try {
        . `$modulePath
        Write-Host "   ✅ Loaded $module" -ForegroundColor Green
    } catch {
        Write-Host "   ❌ Failed to load $module`: `$(`$_.Exception.Message)" -ForegroundColor Red
        throw "Module loading failed: $module"
    }
} else {
    Write-Host "   ⚠️ Module not found: `$modulePath" -ForegroundColor Yellow
    throw "Module not found: $module"
}
"@
        }
    }

    # Add the user's script block
    $isolatedScript += @"

Write-Host "`n🚀 Executing user script block..." -ForegroundColor Magenta

# Execute user script
try {
    `$userResult = & {
$($ScriptBlock.ToString())
    }

    Write-Host "✅ User script execution completed successfully" -ForegroundColor Green

    # Return results in a structured format
    return @{
        Success = `$true
        Result = `$userResult
        Error = `$null
        ExecutionTime = 0  # Will be calculated by parent
    }
} catch {
    Write-Host "❌ User script execution failed: `$(`$_.Exception.Message)" -ForegroundColor Red
    return @{
        Success = `$false
        Result = `$null
        Error = `$_.Exception.Message
        ExecutionTime = 0
    }
}
"@

    if ($ShowDetails) {
        Write-Host "🔧 Prepared isolated script with $($ModulesToLoad.Count) modules" -ForegroundColor Cyan
        Write-Host "🌍 Environment overrides: $($isolatedEnv.Count) variables" -ForegroundColor Cyan
    }

    # Execute in isolated environment using Invoke-WithTempEnv
    $startTime = Get-Date

    try {
        if ($ShowDetails) {
            Write-Host "`n🔄 Launching isolated sub-instance..." -ForegroundColor Yellow
        }

        $isolatedResult = Invoke-WithTempEnv -EnvironmentVariables $isolatedEnv -ScriptBlock {
            # Create a completely fresh PowerShell instance
            $powershellArgs = @(
                '-NoProfile',
                '-NoLogo',
                '-ExecutionPolicy', 'Bypass',
                '-Command', $using:isolatedScript
            )

            if ($using:ShowDetails) {
                Write-Host "🚀 Starting fresh PowerShell process with args: $($powershellArgs -join ' ')" -ForegroundColor Gray
            }

            # Start fresh PowerShell process
            $processInfo = @{
                FileName = 'pwsh.exe'
                Arguments = $powershellArgs
                UseShellExecute = $false
                RedirectStandardOutput = $true
                RedirectStandardError = $true
                CreateNoWindow = $true
            }

            $process = Start-Process @processInfo -PassThru

            # Wait for completion with timeout
            $completed = $process.WaitForExit($using:TimeoutSeconds * 1000)

            if (-not $completed) {
                $process.Kill()
                throw "Isolated sub-instance timed out after $($using:TimeoutSeconds) seconds"
            }

            # Capture outputs
            $stdout = $process.StandardOutput.ReadToEnd()
            $stderr = $process.StandardError.ReadToEnd()
            $exitCode = $process.ExitCode

            return @{
                ExitCode = $exitCode
                StandardOutput = $stdout
                StandardError = $stderr
                Success = ($exitCode -eq 0)
            }
        }

        $executionTime = (Get-Date) - $startTime

        # Process results
        if ($isolatedResult.Success) {
            if ($ShowOutput -and $isolatedResult.StandardOutput) {
                Write-Host "`n📋 Isolated Instance Output:" -ForegroundColor Cyan
                Write-Host $isolatedResult.StandardOutput -ForegroundColor White
            }

            if ($isolatedResult.StandardError) {
                Write-Host "`n⚠️ Isolated Instance Warnings/Errors:" -ForegroundColor Yellow
                Write-Host $isolatedResult.StandardError -ForegroundColor Red
            }

            Write-Host "`n✅ Isolated sub-instance completed successfully in $([math]::Round($executionTime.TotalMilliseconds, 1))ms" -ForegroundColor Green

            if ($PassThru) {
                # Try to parse the result from stdout if possible
                # This is a simple approach - could be enhanced with structured data return
                return @{
                    Success = $true
                    Output = $isolatedResult.StandardOutput
                    Error = $isolatedResult.StandardError
                    ExecutionTime = $executionTime
                    ExitCode = $isolatedResult.ExitCode
                }
            }
        } else {
            Write-Host "`n❌ Isolated sub-instance failed with exit code: $($isolatedResult.ExitCode)" -ForegroundColor Red
            if ($isolatedResult.StandardError) {
                Write-Host "Error details: $($isolatedResult.StandardError)" -ForegroundColor Red
            }

            if ($PassThru) {
                return @{
                    Success = $false
                    Output = $isolatedResult.StandardOutput
                    Error = $isolatedResult.StandardError
                    ExecutionTime = $executionTime
                    ExitCode = $isolatedResult.ExitCode
                }
            }
        }

    } catch {
        $executionTime = (Get-Date) - $startTime
        Write-Host "`n💥 Isolated sub-instance execution failed: $($_.Exception.Message)" -ForegroundColor Red

        if ($PassThru) {
            return @{
                Success = $false
                Output = ""
                Error = $_.Exception.Message
                ExecutionTime = $executionTime
                ExitCode = -1
            }
        }

        throw
    }
}

# Export the function for use
if ($env:DISABLE_PARALLEL_ALIASES -ne "true") {
    Set-Alias -Name "isolated" -Value "Invoke-IsolatedSubInstance" -Scope Global
    Set-Alias -Name "subinstance" -Value "Invoke-IsolatedSubInstance" -Scope Global
}
