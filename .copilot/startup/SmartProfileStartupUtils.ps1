# Telemetry/logging helper that includes the calling function name
function Write-Telemetry
<#
.SYNOPSIS
Enhanced logging wrapper that automatically identifies the calling function.

.DESCRIPTION
Writes telemetry messages with automatic caller detection and customizable output levels.
Wraps standard PowerShell cmdlets (Write-Host, Write-Warning, Write-Error) with context.

.PARAMETER Message
The message to log.

.PARAMETER Level
Output level: 'Host' (default), 'Warning', or 'Error'.

.PARAMETER Color
Console color for Host level messages. Default is White.

.PARAMETER Caller
Override automatic caller detection with custom name.

.EXAMPLE
Write-Telemetry "Profile loading started"
# Output: [Telemetry:MyFunction] Profile loading started

.EXAMPLE
Write-Telemetry "Config loaded successfully" -Level Host -Color Green
# Output: [Telemetry:LoadConfig] Config loaded successfully (in green)

.EXAMPLE
Write-Telemetry "Missing required file" -Level Warning
# Output: WARNING: [Telemetry:ValidateFiles] Missing required file

.EXAMPLE
Write-Telemetry "Critical failure" -Level Error
# Output: [Telemetry:ProcessData] Critical failure (as error)

.EXAMPLE
Write-Telemetry "Custom context message" -Caller "CustomLoader"
# Output: [Telemetry:CustomLoader] Custom context message

.NOTES
Automatically detects calling function name from call stack.
Fallback hierarchy: FunctionName → InvocationName → 'ScriptBlock'
#>

{
    param(
        [string]$Message,
        [string]$Level = 'Host',
        [ConsoleColor]$Color = 'White',
        [string]$Caller = $null
    )
    if (-not $Caller)
    {
        $callStack = Get-PSCallStack
        foreach ($frame in $callStack)
        {
            if ($frame.FunctionName -and $frame.FunctionName -ne '<ScriptBlock>')
            {
                $Caller = $frame.FunctionName
                break
            }
        }
        if (-not $Caller)
        {
            $Caller = $MyInvocation.InvocationName
            if (-not $Caller) { $Caller = 'ScriptBlock' }
        }
    }
    $prefix = "[Telemetry:$Caller]"
    switch ($Level)
    {
        'Host' { Write-Host "$prefix $Message" -ForegroundColor $Color }
        'Warning' { Write-Warning "$prefix $Message" }
        'Error' { Write-Error "$prefix $Message" }
        default { Write-Host "$prefix $Message" -ForegroundColor $Color }
    }
}

function Test-PortablePaths
{
    param(
        [hashtable]$Paths = @($PortableRoot, $ProjectsRoot, $AppsRoot, $PsLibraryPath)
    )

    foreach ($path in $Paths)
    {
        if (-not (Test-Path $path))
        {
            Write-Warning "Critical path missing: $path"
            if ($memorySystemAvailable -and (Get-Command Write-MemoryLog -ErrorAction SilentlyContinue))
            {
                Write-MemoryLog -Scope Session -Type error -Message "Missing critical path: $path"
            }
        }
    }
}

function Update-EnvPathSmart
{

    [CmdletBinding()]
    param(
        [Parameter(ParameterSetName = "DirectPaths", Mandatory)]
        [string[]]$Paths,
        [ValidateSet("Prepend", "Append", "Replace")]
        [string]$Strategy = "Prepend",
        [ValidateSet("Process", "User", "Machine")]
        [string]$Scope = "Process",
        [switch]$ValidateExecutables,
        [switch]$WhatIf
    )

    # Build paths array based on parameter set
    $pathsToProcess = @()

    $pathsToProcess = $Paths | Where-Object { $_ -and $_.Trim() }

    # Get current PATH and normalize
    $currentPath = $env:PATH -split ';' | Where-Object { $_ -and $_.Trim() }

    # Process each path with validation
    $validPaths = @()
    $skippedPaths = @()
    $duplicatePaths = @()

    foreach ($path in $pathsToProcess)
    {
        $normalizedPath = $path.TrimEnd('\')

        # Check if path exists
        if (-not (Test-Path $normalizedPath))
        {
            $skippedPaths += @{ Path = $normalizedPath; Reason = "Path does not exist" }
            continue
        }

        # Check for executables if requested
        if ($ValidateExecutables)
        {
            $executables = Get-ChildItem $normalizedPath -File |
                Where-Object { $_.Extension -in @('.exe', '.ps1', '.bat', '.cmd', '.com') }

            if ($executables.Count -eq 0)
            {
                $skippedPaths += @{ Path = $normalizedPath; Reason = "No executables found" }
                continue
            }
        }

        # Check for duplicates (case-insensitive, handle trailing slashes)
        $isDuplicate = $currentPath | Where-Object {
            ($_.TrimEnd('\') -eq $normalizedPath) -or
            ($_.TrimEnd('\').ToLower() -eq $normalizedPath.ToLower())
        }

        if ($isDuplicate)
        {
            $duplicatePaths += $normalizedPath
            Write-Host "⚠️ Already in PATH: $normalizedPath" -ForegroundColor Yellow
        }
        else
        {
            $validPaths += $normalizedPath
            if ($WhatIf)
            {
                Write-Host "Would add to PATH: $normalizedPath" -ForegroundColor Cyan
            }
            else
            {
                Write-Host "✅ Adding to PATH: $normalizedPath" -ForegroundColor Green
            }
        }

    }
}

function Set-EnvPathSorted
{
    param (
        [string]$RootPath,
        [bool]$SortOthers = $false
    )

    $paths = $env:Path -split ';' | Where-Object { $_ -ne '' }

    $portablePaths = $paths | Where-Object { $_ -like "$RootPath*" }
    $otherPaths = $paths | Where-Object { $_ -notlike "$RootPath*" }

    $sortedPortable = $portablePaths | Sort-Object
    $sortedOthers = $SortOthers ? ($otherPaths | Sort-Object) : $otherPaths

    $env:Path = ($sortedPortable + $sortedOthers) -join ';'
}

function Import-FilesList
{
    [CmdletBinding()]
    param (
        [Parameter(Mandatory)]
        [string[]]$Paths,

        [switch]$Quiet
    )

    $loadedCount = 0
    $errorCount = 0

    foreach ($filePath in $Paths)
    {
        $label = Split-Path $filePath -Leaf

        if (Test-Path $filePath)
        {
            try
            {
                . $filePath
                $loadedCount++
                if (-not $Quiet)
                {
                    Write-Host "✅ Loaded: $label" -ForegroundColor Green
                }
            }
            catch
            {
                $errorCount++
                if (-not $Quiet)
                {
                    Write-Host "❌ Failed to load $label`: $($_.Exception.Message)" -ForegroundColor Red
                }
            }
        }
        else
        {
            $errorCount++
            if (-not $Quiet)
            {
                Write-Host "⚠️ File not found: $label" -ForegroundColor Yellow
            }
        }
    }

    if (-not $Quiet)
    {
        Write-Host "📦 Scripts loaded: $loadedCount | Errors: $errorCount | Total: $($Paths.Count)" -ForegroundColor Cyan
    }

    return @{
        Loaded = $loadedCount
        Errors = $errorCount
        Total  = $Paths.Count
    }
}

function Set-EnvVarsScoped
{
    param(
        [hashtable]$Vars,
        [ValidateSet("Process", "User", "Machine")]
        [string]$Scope = "Process"
    )

    # 1. Export env vars with specified scope
    foreach ($key in $Vars.Keys)
    {
        [Environment]::SetEnvironmentVariable($key, $Vars[$key], $Scope)
    }
}

function Set-SessionAliases
{

    foreach ($alias in $safeAliases.GetEnumerator())
    {
        try
        {
            $existingCommand = Get-Command $alias.Key -ErrorAction SilentlyContinue
            if (-not $existingCommand -or $existingCommand.CommandType -eq 'Alias')
            {
                Set-Alias -Name $alias.Key -Value $alias.Value -Force
            }
            elseif ($Verbose)
            {
                Write-Host "  Skipping alias '$($alias.Key)' - conflicts with existing $($existingCommand.CommandType)" -ForegroundColor Yellow
            }
        }
        catch
        {
            Write-Warning "Failed to create alias '$($alias.Key)': $($_.Exception.Message)"
        }
    }
}

function Set-CustomUserPrompt
{
    param(
        [string]$customPrefix = "[pdenv]:",
        [string]$portableRoot = $env:PORTABLE_ROOT
    )
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal] $identity
    $adminRole = [Security.Principal.WindowsBuiltInRole]::Administrator

    $PortableRootLeaf = Split-Path $portableRoot -Leaf
    $prefix = ""
    if (Test-Path Variable:/PSDebugContext) { $prefix += "[DBG]:" }
    if ($principal.IsInRole($adminRole)) { $prefix += "[ADMIN]:" }
    if ($env:TERMINAL_USER -eq 'User')
    {
        $prefix += ""

        $portableRoot = $env:COPILOT_WORKSPACE_ROOT ?? "$portableRoot"
        $currentPath = $PWD.Path

        if ($currentPath.StartsWith($portableRoot, [StringComparison]::OrdinalIgnoreCase))
        {
            $relativePath = $currentPath.Substring($portableRoot.Length).TrimStart('\')
            $displayPath = if ([string]::IsNullOrEmpty($relativePath))
            {
                "$pDenvRootName"
            }
            else
            {
                "$PortableRootLeaf\$relativePath"
            }
        }
        else
        {
            $displayPath = "PS $($PWD.Path)"
        }

        return "$prefix$displayPath> "
    }

    $body = "PS $($PWD.Path)"
    $suffix = $(if ($NestedPromptLevel -ge 1) { '>>' }) + '> '
    return "$prefix$body$suffix"
}

function Get-PDenvStatus
{
    Write-Host "🔧 PDenv Status" -ForegroundColor Cyan
    $script:CoreEnvironmentVars | ForEach-Object {
        $value = Get-Item -Path "env:$_" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Value
        $status = if ($value) { "✅" } else { "❌" }
        Write-Host "$status $_ = $value" -ForegroundColor $(if ($value) { "Green" } else { "Red" })
    }
}

function Set-DotNetVersion {
    param(
        [string]$Version = "9.0.304",
        [string]$BasePath = "$HOME\PortDenv\Apps\dotnet",
        [switch]$LogActivation
    )

    $DotNetRoot = Join-Path $BasePath $Version
    $DotNetExe  = Join-Path $DotNetRoot "dotnet.exe"

    if (-not (Test-Path $DotNetExe)) {
        Write-Warning "dotnet.exe not found at $DotNetExe"
        return $false
    }

    # Set environment variables
    $env:DOTNET_ROOT              = $DotNetRoot
    $env:DOTNET_ROOT_USER         = $DotNetRoot
    $env:DOTNET_ROOT_USER_RELEASE = $DotNetRoot
    $env:PATH = "$DotNetRoot;$env:PATH"

    # Override shell binding
    function dotnet {
        param([Parameter(ValueFromRemainingArguments=$true)]$Args)
        & "$env:DOTNET_ROOT\dotnet.exe" @Args
    }

    # Optional corpus logging
    if ($LogActivation) {
        $logEntry = @{
            timestamp = (Get-Date).ToString("s")
            version   = $Version
            path      = $DotNetRoot
            user      = $env:USERNAME
            host      = $env:COMPUTERNAME
        }
        $logPath = Join-Path $BasePath "..\Logs\dotnet_activation.jsonl"
        $logEntry | ConvertTo-Json -Compress | Add-Content -Path $logPath
    }

    Write-Host "✅ dotnet $Version activated from $DotNetRoot"
    return $true
}

function Import-CopilotCustomConfig {
    $configPath = "$env:USERPROFILE/PortDenv/share/copilot-custom-config.json"
    if (Test-Path $configPath) {
        $config = Get-Content $configPath | ConvertFrom-Json
        $global:CopilotConfig = $config
        Write-Telemetry "Custom Copilot config loaded" -Level Host -Color Green
    }
}
