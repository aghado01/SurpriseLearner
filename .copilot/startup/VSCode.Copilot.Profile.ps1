#Requires -Version 7.0
# filepath: c:/Users/azriy/PortDenv/.denv/Profiles/VSCode.Copilot.Profile.ps1
<#
.SYNOPSIS
VSCode Copilot terminal profile with modular loader architecture.

.DESCRIPTION
Clean orchestrating profile that delegates all functionality to modular loaders.
Removes all duplicate functions and uses SmartProfileUtils for common operations.
#>

[CmdletBinding()]
param(
  [switch]$Quiet,
  [string]$SessionId = (New-Guid).ToString().Substring(0, 8)
)

Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process

# 2) Use helper to load other files
# May need to add conditional logic for copilot file loading dependng on which functionalities are enabled
# Memory system orchestration engine is intialized in SmartProfileLoader
# Resolve core paths based on PORTABLE_ROOT
$denvRoot = "$env:PORTABLE_ROOT/.denv"
# $denvProfiles = "$denvRoot/Profiles"
$copilotHome = "$denvRoot/Copilot"
$copilotCore = "$copilotHome/Core"
$loadFiles = @(
  "$copilotCore/CopilotFileSystemInfrastructure.ps1"
  , "$copilotCore/CopilotLoggingMemorySystem.ps1"
  , "$copilotCore/CopilotSelfMonitoringSystem.ps1"
  , "$copilotCore/CopilotConsoleEnhancments.ps1"
)
Import-FilesList -Paths $loadFiles


# 5) Session environment variables
$env:COPILOT_SESSION_ID = $SessionId

# 6) Initialize workspace context
$gitRoot = $null
$gitBranch = $null
try { $gitRoot = & git rev-parse --show-toplevel 2>$null } catch {}
try { $gitBranch = & git branch --show-current 2>$null } catch {}

$initialRoot = if ($gitRoot) { $gitRoot } else { $PWD.Path }
$initialType = if ($initialRoot -like "*Projects*") { "project" } else { "infrastructure" }
$initialName = if ($initialType -eq "project") { Split-Path $initialRoot -Leaf } else { "portdenv" }

$script:WorkspaceContext = @{
  Root                    = $initialRoot
  Name                    = $initialName
  Type                    = $initialType
  GitRepo                 = $gitRoot
  GitBranch               = $gitBranch
  IsGitRepo               = [bool]$gitRoot
  SessionId               = $SessionId
  LogDirectory            = Join-Path $initialRoot ".copilot/logs"
  ContextDirectory        = Join-Path $initialRoot ".copilot/context"
  FunctionsDirectory      = Join-Path $initialRoot ".copilot/functions"
  InstructionsPath        = $null
  HasExistingInstructions = $false
  InfrastructureSetup     = $false
}

# 7) Detect instruction files (prefer workspace > repo > root > global)
# NOTE: Need to revisit these conventions
$instructionCandidates = @(
  "$($script:WorkspaceContext.Root)/.github/copilot-instructions.md",
  if ($script:WorkspaceContext.GitRepo) { "$($script:WorkspaceContext.GitRepo)/.github/copilot-instructions.md" },
  "$($script:WorkspaceContext.Root)/.copilot-instructions.md",
  "$($env:PORTABLE_ROOT)/.github/copilot-instructions.md"
) | Where-Object { $_ }

foreach ($candidate in $instructionCandidates)
{
  if (Test-Path $candidate)
  {
    $script:WorkspaceContext.InstructionsPath = $candidate
    $script:WorkspaceContext.HasExistingInstructions = $true
    Write-Telemetry "Found Copilot instructions: $candidate" -Level Host -Color Cyan
    break
  }
}

# 14) Set up aliases using SmartProfileUtils pattern
$copilotAliases = @{
  "clog"  = "Write-CopilotSessionLog"
  "cseq"  = "Write-CopilotSequenceLog"
  "cws"   = "Get-CopilotWorkspaceContext"
  "cinst" = "Show-CopilotInstructions"
}


# 15) Final status report
if (-not $Quiet)
{
  Write-Host "🤖 Copilot Enhanced Terminal Ready" -ForegroundColor Cyan
  Write-Host "   Session: $($SessionId.Substring(0,8))" -ForegroundColor Green
  Write-Host "   Context: $($script:WorkspaceContext.Name):$($script:WorkspaceContext.Type)" -ForegroundColor Green
  Write-Host "   Infrastructure: $(if ($script:WorkspaceContext.InfrastructureSetup) { 'Auto-Setup Complete' } else { 'Existing' })" -ForegroundColor Green
  Write-Host "   Instructions: $(if ($script:WorkspaceContext.HasExistingInstructions) { 'Found Existing' } else { 'Global Only' })" -ForegroundColor Yellow
  Write-Host "   Commands: $(($copilotAliases.Keys | Sort-Object) -join ', ')" -ForegroundColor Yellow
}


# Convenience aliases
Set-Alias -Name "clog" -Value "Write-CopilotSessionLog" -Scope Global -Force
Set-Alias -Name "cseq" -Value "Write-CopilotSequenceLog" -Scope Global -Force
Set-Alias -Name "cws" -Value "Get-CopilotWorkspaceContext" -Scope Global -Force
Set-Alias -Name "cinst" -Value "Show-CopilotInstructions" -Scope Global -Force


# Mark profile as loaded at the end
$global:COPILOT_PROFILE_LOADED = $true


if (-not $Quiet)
{
  Write-Host "🤖 Copilot Enhanced Terminal Ready" -ForegroundColor Cyan
  Write-Host "   Session: $($SessionId.Substring(0,8))" -ForegroundColor Green
  Write-Host "   Context: $($script:WorkspaceContext.Name):$($script:WorkspaceContext.Type)" -ForegroundColor Green
  Write-Host "   Infrastructure: $(if ($script:WorkspaceContext.InfrastructureSetup) { 'Auto-Setup Complete' } else { 'Existing' })" -ForegroundColor Green
  Write-Host "   Instructions: $(if ($script:WorkspaceContext.HasExistingInstructions) { 'Found Existing' } else { 'Global Only' })" -ForegroundColor Yellow
  Write-Host "   Commands: clog, cseq, cws, cinst" -ForegroundColor Yellow
}
