<#
.SYNOPSIS
Consolidated filesystem infrastructure loader for Copilot augmentation system.

.DESCRIPTION
Sets up minimal Copilot directory structure and workspace context files.
Removes duplicate Initialize-CopilotInfrastructure definitions and adds expertise corpus support.
#>

function Initialize-CopilotInfrastructure {
  <#
    .SYNOPSIS
    Creates minimal Copilot directory structure and workspace context.

    .DESCRIPTION
    Establishes .copilot directories, workspace context JSON, and expertise corpus structure.
    Integrates with hierarchical logging system preparation.

    .PARAMETER WorkspaceRoot
    Root directory for Copilot infrastructure setup

    .PARAMETER Force
    Force recreation of existing infrastructure
    #>
  [CmdletBinding(SupportsShouldProcess)]
  param(
    [string]$WorkspaceRoot = $script:WorkspaceContext.Root,
    [switch]$Force
  )

  $infrastructureNeeded = @()

  # Basic directory structure focused on core functionality
  $copilotDirs = @(
    "$WorkspaceRoot/.copilot",
    "$WorkspaceRoot/.copilot/logs",
    "$WorkspaceRoot/.copilot/logs/session",
    "$WorkspaceRoot/.copilot/logs/sequences",
    "$WorkspaceRoot/.copilot/logs/project",
    "$WorkspaceRoot/.copilot/context",
    "$WorkspaceRoot/.copilot/functions"
  )

  foreach ($dir in $copilotDirs) {
    if (-not (Test-Path $dir)) {
      $infrastructureNeeded += $dir
    }
  }

  if ($infrastructureNeeded.Count -gt 0 -or $Force) {
    Write-Host "🔧 Setting up Copilot infrastructure..." -ForegroundColor Cyan

    foreach ($dir in $copilotDirs) {
      if (-not (Test-Path $dir)) {
        if ($PSCmdlet.ShouldProcess($dir, "Create directory")) {
          New-Item -ItemType Directory -Path $dir -Force | Out-Null
          Write-Host "   Created: $($dir.Replace($WorkspaceRoot, '.'))" -ForegroundColor Green
        }
      }
    }

    # Global repository structure (with safety check)
    if ($env:PORTABLE_ROOT) {
      $globalPath = "$env:PORTABLE_ROOT/Logs/AI-Repository"
      @("project_events", "session_patterns", "sequence_analytics") | ForEach-Object {
        $globalDir = "$globalPath/$_"
        if (-not (Test-Path $globalDir)) {
          New-Item -Path $globalDir -ItemType Directory -Force -ErrorAction SilentlyContinue | Out-Null
        }
      }
    }

    # Create enhanced workspace context file
    $contextPath = "$WorkspaceRoot/.copilot/context/workspace-context.json"
    if ($PSCmdlet.ShouldProcess($contextPath, "Create workspace context")) {
      $contextData = @{
        workspace_name            = $script:WorkspaceContext.Name
        workspace_type            = $script:WorkspaceContext.Type
        root_path                 = $script:WorkspaceContext.Root
        is_git_repo               = $script:WorkspaceContext.IsGitRepo
        git_branch                = $script:WorkspaceContext.GitBranch
        setup_timestamp           = Get-Date -Format "yyyy-MM-ddTHH:mm:ss.fffZ"
        session_logging           = "enabled"
        sequence_logging          = "enabled"
        has_existing_instructions = $script:WorkspaceContext.HasExistingInstructions
        instructions_path         = $script:WorkspaceContext.InstructionsPath
      }

      $contextData | ConvertTo-Json -Depth 5 | Out-File -FilePath $contextPath -Encoding UTF8
      Write-Host "   Created: .copilot/context/workspace-context.json" -ForegroundColor Green
    }

    $script:WorkspaceContext.InfrastructureSetup = $true
    Write-Host "✅ Copilot infrastructure setup complete" -ForegroundColor Green
    Write-Host "   Instructions: $($script:WorkspaceContext.InstructionsPath ?? 'Global only')" -ForegroundColor Yellow
  }
  else {
    Write-Verbose "Copilot infrastructure already exists"
  }
}
