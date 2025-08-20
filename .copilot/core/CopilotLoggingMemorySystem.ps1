<#
.SYNOPSIS
Enhanced logging and memory system with JSONL preparation and SmartProfileUtils integration.

.DESCRIPTION
Provides session, sequence, and project logging with hierarchical structure.
Removes duplicate functions and integrates telemetry from SmartProfileUtils.
#>


function Initialize-CopilotAssistantContext {
    <#
    .SYNOPSIS
    Detects and configures assistant context from terminal profile information.

    .DESCRIPTION
    Parses VSCODE_TERM_PROFILE to extract assistant type and context.
    Sets environment variables for session management.
    #>

    $profilepath = $env:VSCODE_TERM_PROFILE
    if ($profilepath -match '^([^(]+)\s*\(([^)]+)\)') {
        $env:PRIMARY_ASSISTANT = $matches[1].Trim().ToLower()
        $env:ASSISTANT_CONTEXT = $matches[2].Trim().ToLower()
    }
    else {
        $env:PRIMARY_ASSISTANT = 'copilot'
        $env:ASSISTANT_CONTEXT = 'dedicated'
    }
    $env:AI_SESSION_TYPE = "vscode_terminal"

    # Use SmartProfileUtils telemetry
    Write-Telemetry "Assistant context detected: $($env:PRIMARY_ASSISTANT) ($($env:ASSISTANT_CONTEXT))" -Level Host -Color Cyan

    # Log the context detection
    try {
        Write-CopilotSessionLog -Action "assistant_context_detected" -Metadata @{
            primary_assistant = $env:PRIMARY_ASSISTANT
            assistant_context = $env:ASSISTANT_CONTEXT
            session_type      = $env:AI_SESSION_TYPE
        }
    }
    catch {
        Write-Verbose "Context logging failed: $($_.Exception.Message)"
    }
}


function Write-CopilotSessionLog {
    <#
    .SYNOPSIS
    Writes session-level events to hierarchical JSONL log structure.

    .DESCRIPTION
    Creates daily session logs in JSONL format with rich metadata.
    Ensures directory structure exists before writing.

    .PARAMETER Action
    Action being logged (e.g., profile_loaded, command_executed)

    .PARAMETER Metadata
    Additional metadata hashtable

    .PARAMETER Command
    Command being executed (optional)

    .PARAMETER Output
    Command output (optional, truncated to 500 chars)
    #>
    param(
        [Parameter(Mandatory)]
        [string]$Action,
        [hashtable]$Metadata = @{},
        [string]$Command = "",
        [string]$Output = ""
    )

    # Ensure directory structure exists
    $sessionLogDir = "$($script:WorkspaceContext.LogDirectory)/session"
    if (-not (Test-Path $sessionLogDir)) {
        New-Item -Path $sessionLogDir -ItemType Directory -Force | Out-Null
    }

    $sessionLog = "$sessionLogDir/$(Get-Date -f yyyy-MM-dd).jsonl"

    $entry = @{
        timestamp  = Get-Date -f "yyyy-MM-ddTHH:mm:ss.fffZ"
        type       = "session"
        action     = $Action
        session_id = $script:WorkspaceContext.SessionId
        metadata   = $Metadata
    }

    if ($Command) { $entry.command = $Command }
    if ($Output) { $entry.output = $Output.Substring(0, [Math]::Min(500, $Output.Length)) }

    try {
        $jsonEntry = $entry | ConvertTo-Json -Compress -Depth 5
        Add-Content -Path $sessionLog -Value $jsonEntry
        Write-Verbose "Session logged: $Action"
    }
    catch {
        Write-Warning "Session logging failed: $($_.Exception.Message)"
    }
}

function Write-CopilotSequenceLog {
    <#
    .SYNOPSIS
    Logs multi-step sequence execution with hierarchical directory structure.

    .DESCRIPTION
    Creates sequence-specific logs for complex operations tracking.
    Each sequence gets its own directory under logs/sequences.

    .PARAMETER Sequence
    Name of the sequence being executed

    .PARAMETER Status
    Current status (started, in_progress, completed, failed)

    .PARAMETER Results
    Results hashtable with step outcomes

    .PARAMETER Step
    Current step name (optional)
    #>
    param(
        [Parameter(Mandatory)]
        [string]$Sequence,
        [Parameter(Mandatory)]
        [string]$Status,
        [hashtable]$Results = @{},
        [string]$Step = ""
    )

    # Ensure hierarchical directory structure
    $sequenceLogDir = "$($script:WorkspaceContext.LogDirectory)/sequences/$Sequence"
    if (-not (Test-Path $sequenceLogDir)) {
        New-Item -Path $sequenceLogDir -ItemType Directory -Force | Out-Null
    }

    $sequenceLog = "$sequenceLogDir/execution.jsonl"

    $entry = @{
        timestamp  = Get-Date -f "yyyy-MM-ddTHH:mm:ss.fffZ"
        type       = "sequence_step"
        sequence   = $Sequence
        status     = $Status
        session_id = $script:WorkspaceContext.SessionId
        results    = $Results
    }

    if ($Step) { $entry.step = $Step }

    try {
        $jsonEntry = $entry | ConvertTo-Json -Compress -Depth 5
        Add-Content -Path $sequenceLog -Value $jsonEntry
        Write-Verbose "Sequence logged: $Sequence/$Status"
    }
    catch {
        Write-Warning "Sequence logging failed: $($_.Exception.Message)"
    }
}

function Write-CopilotProjectLog {
    <#
    .SYNOPSIS
    Logs project-level events with global archiving capability.

    .DESCRIPTION
    Creates project event logs and archives to global repository.
    Integrates with hierarchical logging structure.

    .PARAMETER Event
    Event name (e.g., project_started, error_detected)

    .PARAMETER Metadata
    Event metadata hashtable
    #>
    param(
        [Parameter(Mandatory)]
        [string]$Event,
        [hashtable]$Metadata = @{}
    )

    # Ensure directory structure exists
    $projectLogDir = "$($script:WorkspaceContext.LogDirectory)/project"
    if (-not (Test-Path $projectLogDir)) {
        New-Item -Path $projectLogDir -ItemType Directory -Force | Out-Null
    }

    $projectLog = "$projectLogDir/events.jsonl"

    $entry = @{
        timestamp  = Get-Date -f "yyyy-MM-ddTHH:mm:ss.fffZ"
        type       = "project_event"
        event      = $Event
        workspace  = $script:WorkspaceContext.Name
        session_id = $script:WorkspaceContext.SessionId
        metadata   = $Metadata
    }

    try {
        $jsonEntry = $entry | ConvertTo-Json -Compress -Depth 5
        Add-Content -Path $projectLog -Value $jsonEntry

        # Archive to global (with safety check)
        Write-CopilotGlobalArchive -Entry $jsonEntry -Category "project_events"
        Write-Verbose "Project logged: $Event"
    }
    catch {
        Write-Warning "Project logging failed: $($_.Exception.Message)"
    }
}

function Write-CopilotGlobalArchive {
    <#
    .SYNOPSIS
    Archives entries to global AI repository for cross-session analysis.

    .DESCRIPTION
    Creates monthly JSONL files in global repository structure.
    Provides centralized logging for pattern analysis.

    .PARAMETER Entry
    JSONL entry to archive

    .PARAMETER Category
    Archive category (project_events, session_patterns, etc.)
    #>
    param(
        [Parameter(Mandatory)]
        [string]$Entry,
        [Parameter(Mandatory)]
        [string]$Category
    )

    if (-not $env:PORTABLE_ROOT) {
        Write-Verbose "No PORTABLE_ROOT defined, skipping global archive"
        return
    }

    try {
        $globalLog = "$env:PORTABLE_ROOT/Logs/AI-Repository/$Category/$(Get-Date -f yyyy-MM).jsonl"
        $globalDir = Split-Path $globalLog

        if (-not (Test-Path $globalDir)) {
            New-Item -Path $globalDir -ItemType Directory -Force | Out-Null
        }

        Add-Content -Path $globalLog -Value $Entry
        Write-Verbose "Archived to global: $Category"
    }
    catch {
        Write-Verbose "Global archiving failed: $($_.Exception.Message)"
    }
}

function Get-CopilotWorkspaceMemory {
    <#
    .SYNOPSIS
    Retrieves current workspace memory context for session continuity.

    .DESCRIPTION
    Aggregates recent activity, project context, and session information.
    Provides memory context for assistant operations.
    #>
    return @{
        recent_commands     = Get-CopilotRecentActivity -Hours 2
        project_context     = $script:WorkspaceContext
        active_session      = $script:WorkspaceContext.SessionId
        expertise_available = Test-Path "$($script:WorkspaceContext.Root)/.copilot/expertise/corpus-index.json"
    }
}

function Get-CopilotRecentActivity {
    <#
    .SYNOPSIS
    Queries recent session activity from JSONL logs.

    .DESCRIPTION
    Parses session logs to extract recent commands and activities.
    Supports time-based filtering for context relevance.

    .PARAMETER Hours
    Number of hours to look back for activity
    #>
    param([int]$Hours = 24)

    $cutoffTime = (Get-Date).AddHours(-$Hours)
    $sessionLogDir = "$($script:WorkspaceContext.LogDirectory)/session"

    if (-not (Test-Path $sessionLogDir)) {
        return @()
    }

    $recentLogs = Get-ChildItem $sessionLogDir -Filter "*.jsonl" |
    Where-Object { $_.LastWriteTime -gt $cutoffTime } |
    Sort-Object LastWriteTime -Descending

    $activities = @()
    foreach ($logFile in $recentLogs) {
        try {
            $entries = Get-Content $logFile.FullName | ForEach-Object {
                $_ | ConvertFrom-Json
            }
            $activities += $entries | Where-Object {
                [DateTime]::Parse($_.timestamp) -gt $cutoffTime
            }
        }
        catch {
            Write-Verbose "Failed to parse log file: $($logFile.Name)"
        }
    }

    return $activities | Sort-Object timestamp -Descending
}


function Import-CopilotMemoryOrchestrationConfig {
    $configPath = "$env:USERPROFILE/PortDenv/share/copilot-memory-orchestration.json"
    if (Test-Path $configPath) {
        $global:CopilotMemoryConfig = Get-Content $configPath | ConvertFrom-Json
        Write-Telemetry "Memory orchestration config loaded" -Level Host -Color Green

        # Apply your custom terminal allowlist logic here
        # Set up memory logging based on config
        # Initialize session tracking
    }
}
