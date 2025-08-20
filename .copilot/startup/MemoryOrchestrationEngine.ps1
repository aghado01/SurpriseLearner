#Requires -Version 7.0
<#
.SYNOPSIS
Complete Memory Orchestration and Session Management

.DESCRIPTION
Handles all memory system initialization, Copilot session continuity, and session management
in a single function call. Replaces the large initialization blocks in profiles.

.PARAMETER SessionId
Session identifier for memory correlation

.PARAMETER MemoryContext
Context for memory system (Development, Testing, etc.)

.PARAMETER LoadingContext
Loading environment context (VSCode.User, VSCode.Copilot, Nominal)

.PARAMETER PortableRoot
Root path for portable environment

.EXAMPLE
Initialize-MemoryOrchestration -SessionId $SessionId -MemoryContext $MemoryContext -LoadingContext $loadingContext -PortableRoot $PortableRoot
#>

function Initialize-MemoryOrchestration {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$SessionId,

        [Parameter(Mandatory)]
        [string]$MemoryContext,

        [Parameter(Mandatory)]
        [string]$LoadingContext,

        [Parameter()]
        [string]$PortableRoot = $env:PORTABLE_ROOT,

        [Parameter()]
        [switch]$EnableAutoSave,

        [Parameter()]
        [int]$AutoSaveInterval = 300
    )

    Write-Verbose "Initializing memory orchestration for session: $SessionId"

    #region Load Required Systems
    $memorySystemAvailable = $false
    $smartUtilsLoaded = $false

    try {
        # Load SmartProfileUtils first
        $smartUtilsPath = "$PortableRoot/.denv/startup/SmartProfileUtils.ps1"
        if (Test-Path $smartUtilsPath) {
            . $smartUtilsPath
            $smartUtilsLoaded = $true
            Write-Verbose "SmartProfileUtils loaded successfully"
        }

        # Load Memory System
        $memorySystemPath = "$PortableRoot/Projects/CopilotAugmentation/Prototype/MemorySystem.ps1"
        if (Test-Path $memorySystemPath) {
            . $memorySystemPath
            $memorySystemAvailable = $true
            Write-Verbose "Memory system loaded successfully"
        }
    }
    catch {
        Write-Verbose "Failed to load core systems: $($_.Exception.Message)"
    }
    #endregion

    #region Initialize Memory Session
    if ($memorySystemAvailable) {
        try {
            # Register this session with the memory system
            if (Get-Command Initialize-MemorySession -ErrorAction SilentlyContinue) {
                Initialize-MemorySession -SessionId $SessionId -Context $MemoryContext -LoadingContext $LoadingContext
            }

            # Set memory-aware environment variables
            $env:MEMORY_SESSION_ID = $SessionId
            $env:MEMORY_CONTEXT = $MemoryContext
            $env:MEMORY_SYSTEM_AVAILABLE = "true"

            Write-Verbose "Memory system session initialized: $SessionId"
        }
        catch {
            Write-Verbose "Failed to initialize memory session: $($_.Exception.Message)"
            $memorySystemAvailable = $false
            $env:MEMORY_SYSTEM_AVAILABLE = "false"
        }
    }
    else {
        $env:MEMORY_SYSTEM_AVAILABLE = "false"
    }
    #endregion

    #region Initialize Copilot Session Continuity
    $copilotResult = $null

    if ($memorySystemAvailable -and $LoadingContext -eq "VSCode.Copilot") {
        try {
            # Load Copilot Session Manager
            $copilotManagerPath = "$PortableRoot/.denv/startup/CopilotSessionManager.ps1"
            if (Test-Path $copilotManagerPath) {
                . $copilotManagerPath

                # Initialize Copilot session with auto-save enabled
                $copilotResult = Initialize-CopilotSession -SessionId $SessionId -LoadingContext $LoadingContext -MemorySystemAvailable $memorySystemAvailable -EnableAutoSave:$EnableAutoSave -AutoSaveInterval $AutoSaveInterval

                Write-Verbose "Copilot session manager result: $($copilotResult.Message)"
            }
        }
        catch {
            Write-Warning "Failed to initialize Copilot session manager: $($_.Exception.Message)"
        }
    }
    elseif ($memorySystemAvailable) {
        # Standard session for non-Copilot contexts
        $env:COPILOT_SESSION_ACTIVE = "false"
        $env:COPILOT_MEMORY_ENABLED = "false"
        $env:SESSION_TYPE = "Standard"
    }
    else {
        # Basic session - no memory system
        $env:COPILOT_SESSION_ACTIVE = "false"
        $env:COPILOT_MEMORY_ENABLED = "false"
        $env:SESSION_TYPE = "Basic"
    }
    #endregion

    #region Return Orchestration Status
    $orchestrationStatus = @{
        SessionId             = $SessionId
        MemoryContext         = $MemoryContext
        LoadingContext        = $LoadingContext
        MemorySystemAvailable = $memorySystemAvailable
        SmartUtilsLoaded      = $smartUtilsLoaded
        CopilotResult         = $copilotResult
        EnvironmentVariables  = @{
            MEMORY_SESSION_ID       = $env:MEMORY_SESSION_ID
            MEMORY_CONTEXT          = $env:MEMORY_CONTEXT
            MEMORY_SYSTEM_AVAILABLE = $env:MEMORY_SYSTEM_AVAILABLE
            COPILOT_SESSION_ACTIVE  = $env:COPILOT_SESSION_ACTIVE
            SESSION_TYPE            = $env:SESSION_TYPE
        }
        Timestamp             = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"
    }

    Write-Verbose "Memory orchestration completed for session: $SessionId"
    return $orchestrationStatus
    #endregion
}


function Get-MemoryOrchestrationStatus {
    [CmdletBinding()]
    param()

    return @{
        MemorySystemAvailable = $env:MEMORY_SYSTEM_AVAILABLE -eq "true"
        SessionId             = $env:MEMORY_SESSION_ID
        MemoryContext         = $env:MEMORY_CONTEXT
        CopilotActive         = $env:COPILOT_SESSION_ACTIVE -eq "true"
        SessionType           = $env:SESSION_TYPE
        LoadingContext        = $env:TERMINAL_PROFILE
    }
}

# Export functions
# Export-ModuleMember -Function Initialize-MemoryOrchestration, Get-MemoryOrchestrationStatus

#region Auto-Execute if Called Directly from Profile
# Check if this script was called with session parameters
if ($SessionId -and $MemoryContext -and $loadingContext) {
    Write-Verbose "Auto-executing memory orchestration from profile context"

    # Auto-execute with passed parameters
    $global:MemoryOrchestrationResult = Initialize-MemoryOrchestration -SessionId $SessionId -MemoryContext $MemoryContext -LoadingContext $loadingContext -EnableAutoSave -Verbose:$VerbosePreference

    Write-Verbose "Memory orchestration auto-execution completed: $($global:MemoryOrchestrationResult.Timestamp)"
}
#endregion
