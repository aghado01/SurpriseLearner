<#
.SYNOPSIS
Export-CopilotChatLogs - Timestamped JSONL export of VS Code Copilot chat sessions

.DESCRIPTION
Minimal, high-fidelity conversion of Copilot chat sessions to JSONL format.
Optimized for prosthetic memory systems with FLAC-level semantic preservation.
Generates timestamped extracts with comprehensive telemetry and loss tracking.

.PARAMETER Path
VS Code workspaceStorage path (single workspace or root directory)

.PARAMETER Output
Output JSONL file path (timestamp will be added automatically)

.PARAMETER OutputDirectory
Directory for output files. Defaults to current working directory.

.PARAMETER MaxTokens
Token budget per conversation (rough estimate: 4 chars/token)

.PARAMETER StartDate
Start date for filtering chats (inclusive). Accepts date strings or keywords: 'today', 'yesterday', 'week', 'month'. Defaults to all dates.

.PARAMETER EndDate
End date for filtering chats (inclusive). Accepts date strings or keywords: 'today', 'yesterday', 'week', 'month'. Defaults to all dates.

.EXAMPLE
Export-CopilotChatLogs

.EXAMPLE
Export-CopilotChatLogs -StartDate "today"

.EXAMPLE
Export-CopilotChatLogs -OutputDirectory "extracts"

.EXAMPLE
Export-CopilotChatLogs -StartDate "week" -OutputDirectory "extracts"

.EXAMPLE
Export-CopilotChatLogs -StartDate "2025-08-01" -EndDate "2025-08-10"
#>

function Export-CopilotChatLogs {
    [CmdletBinding()]
    param(
        [Parameter(ValueFromPipeline)]
        [string]$Path = "$env:APPDATA\Code\User\workspaceStorage",

        [string]$Output = "copilot-chatlog-extract.jsonl",

        [string]$OutputDirectory,

        [int]$MaxTokens = 6000,

        [string]$StartDate,

        [string]$EndDate
    )

    # Helper function to parse date keywords
    function ConvertTo-DateFromKeyword {
        param([string]$DateInput)

        if ([string]::IsNullOrEmpty($DateInput)) { return $null }

        switch ($DateInput.ToLower()) {
            "today" { return (Get-Date).Date }
            "yesterday" { return (Get-Date).Date.AddDays(-1) }
            "week" { return (Get-Date).Date.AddDays(-7) }
            "month" { return (Get-Date).Date.AddMonths(-1) }
            default {
                try {
                    return [DateTime]::Parse($DateInput)
                }
                catch {
                    Write-Warning "Invalid date format: $DateInput"
                    return $null
                }
            }
        }
    }

    # Parse date parameters
    $parsedStartDate = ConvertTo-DateFromKeyword $StartDate
    $parsedEndDate = ConvertTo-DateFromKeyword $EndDate

    # Find chat session files
    $sessions = if (Test-Path (Join-Path $Path "chatSessions")) {
        Get-ChildItem (Join-Path $Path "chatSessions") -Filter "*.json"
    }
    else {
        Get-ChildItem $Path -Recurse -Filter "*.json" | Where-Object { $_.Directory.Name -eq "chatSessions" }
    }

    Write-Host "🔄 Processing $($sessions.Count) chat sessions..." -ForegroundColor Cyan

    # Generate timestamped filename
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($Output)
    $extension = [System.IO.Path]::GetExtension($Output)
    $timestampedOutput = "${baseName}_${timestamp}${extension}"

    # Apply OutputDirectory if specified
    if ($OutputDirectory) {
        # Create output directory if it doesn't exist
        if (-not (Test-Path $OutputDirectory)) {
            New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
            Write-Host "📁 Created directory: $OutputDirectory" -ForegroundColor Green
        }
        $timestampedOutput = Join-Path $OutputDirectory $timestampedOutput
    }

    # Clear output file and get full path
    if (Test-Path $timestampedOutput) { Remove-Item $timestampedOutput }

    # Get absolute output path
    if ([System.IO.Path]::IsPathRooted($timestampedOutput)) {
        $fullOutputPath = $timestampedOutput
    }
    else {
        $fullOutputPath = Join-Path (Get-Location).Path $timestampedOutput
    }

    # Normalize to universal forward slashes
    $fullOutputPath = $fullOutputPath.Replace('\', '/')
    $normalizedSourcePath = $Path.Replace('\', '/')

    Write-Host "📁 Output destination: $fullOutputPath" -ForegroundColor Yellow

    # Write chatlog extract header with loss tracking
    $header = @{
        extract_type   = "copilot_chatlog_sessions"
        export_date    = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ")
        total_sessions = $sessions.Count
        source_path    = $normalizedSourcePath
        output_path    = $fullOutputPath
        token_budget   = $MaxTokens
        date_filter    = @{
            start_date = if ($parsedStartDate) { $parsedStartDate.ToString("yyyy-MM-dd") } else { $null }
            end_date   = if ($parsedEndDate) { $parsedEndDate.ToString("yyyy-MM-dd") } else { $null }
        }
        format_version = "1.0"
    }
    $header | ConvertTo-Json -Depth 3 -Compress | Add-Content $timestampedOutput -Encoding UTF8

    $processedCount = 0
    $skippedCount = 0
    $truncatedCount = 0
    $totalMessages = 0
    $totalUserMessages = 0
    $totalAssistantMessages = 0

    $sessions | ForEach-Object {
        try {
            $data = Get-Content $_.FullName -Raw | ConvertFrom-Json

            # Apply date filtering if specified
            if ($data.creationDate) {
                $sessionDate = [DateTimeOffset]::FromUnixTimeMilliseconds($data.creationDate).DateTime

                # Check start date filter
                if ($parsedStartDate -and $sessionDate -lt $parsedStartDate) {
                    $skippedCount++
                    return
                }

                # Check end date filter
                if ($parsedEndDate -and $sessionDate -gt $parsedEndDate.AddDays(1).AddMilliseconds(-1)) {
                    $skippedCount++
                    return
                }
            }

            # Extract messages safely
            $messages = @()
            foreach ($req in $data.requests) {
                if ($req.message -and $req.message.text) {
                    $messages += @{ role = "user"; content = $req.message.text.Trim() }
                }
                if ($req.response -and $req.response.message -and $req.response.message.text) {
                    $messages += @{ role = "assistant"; content = $req.response.message.text.Trim() }
                }
            }

            if ($messages.Count -eq 0) {
                $skippedCount++
                return
            }

            # Generate simple summary
            $userMessages = @($messages | Where-Object { $_.role -eq "user" })
            $assistantMessages = @($messages | Where-Object { $_.role -eq "assistant" })

            $firstQuery = if ($userMessages.Count -gt 0) {
                $userMessages[0].content -split '[.?!\n\r]' | Select-Object -First 1 | ForEach-Object { $_.Trim() }
            }
            else { "" }

            # Safe string handling
            $firstQuerySafe = if ($firstQuery.Length -gt 0) {
                $firstQuery.Substring(0, [Math]::Min(100, $firstQuery.Length))
            }
            else { "No query" }

            $summary = @{
                first_query        = $firstQuerySafe
                user_messages      = $userMessages.Count
                assistant_messages = $assistantMessages.Count
                total_messages     = $messages.Count
                duration_days      = if ($data.lastMessageDate -and $data.creationDate) {
                    [math]::Round(($data.lastMessageDate - $data.creationDate) / (1000 * 60 * 60 * 24), 1)
                }
                else { 0 }
            }

            # Track message counts
            $totalMessages += $messages.Count
            $totalUserMessages += $userMessages.Count
            $totalAssistantMessages += $assistantMessages.Count

            # Build chatlog entry
            $entry = @{
                id       = $data.sessionId
                created  = ([DateTimeOffset]::FromUnixTimeMilliseconds($data.creationDate)).ToString("yyyy-MM-dd")
                summary  = $summary
                messages = $messages
            }

            # Token-aware truncation with tracking
            $json = $entry | ConvertTo-Json -Depth 5 -Compress
            if ($json.Length -gt ($MaxTokens * 4)) {
                $keepCount = [math]::Floor($messages.Count * 0.6)
                $entry.messages = $messages[0..($keepCount - 1)]
                $entry.truncated = $true
                $truncatedCount++
            }

            # Export as JSONL with compressed format
            $entry | ConvertTo-Json -Depth 5 -Compress | Add-Content $timestampedOutput -Encoding UTF8
            $processedCount++

            $truncationFlag = if ($entry.truncated) { " [TRUNCATED]" } else { "" }
            Write-Host "✓ $($data.sessionId.Substring(0,8))... ($($messages.Count) msgs) - $($firstQuerySafe.Substring(0, [Math]::Min(30, $firstQuerySafe.Length)))$truncationFlag" -ForegroundColor Green

        }
        catch {
            Write-Warning "Failed: $($_.Name) - $($_.Exception.Message)"
            $skippedCount++
        }
    }

    # Enhanced header with loss tracking summary
    $finalHeader = @{
        extract_type    = "copilot_chatlog_extract"
        export_date     = $timestamp
        source_path     = $normalizedSourcePath
        output_path     = $fullOutputPath
        token_budget    = $MaxTokens
        session_count   = $sessions.Count
        processed_count = $processedCount
        message_summary = @{
            total_conversations     = $processedCount
            total_messages          = $totalMessages
            user_messages           = $totalUserMessages
            assistant_messages      = $totalAssistantMessages
            truncated_conversations = $truncatedCount
        }
        format_version  = "1.0"
    }

    # Replace header with final version containing loss tracking
    $content = Get-Content $timestampedOutput | Select-Object -Skip 1
    $finalHeader | ConvertTo-Json -Depth 3 -Compress | Set-Content $timestampedOutput -Encoding UTF8
    $content | Add-Content $timestampedOutput -Encoding UTF8

    # Final telemetry with loss tracking
    $size = if (Test-Path $timestampedOutput) { [math]::Round((Get-Item $timestampedOutput).Length / 1KB, 1) } else { 0 }

    Write-Host "`n📊 Export Summary" -ForegroundColor Cyan
    Write-Host "Processed: $processedCount/$($sessions.Count) sessions" -ForegroundColor Green
    Write-Host "Total messages: $totalMessages ($totalUserMessages user, $totalAssistantMessages assistant)" -ForegroundColor Yellow
    Write-Host "Truncated conversations: $truncatedCount" -ForegroundColor $(if ($truncatedCount -gt 0) { "Yellow" } else { "Green" })
    Write-Host "📄 File: $fullOutputPath" -ForegroundColor Cyan
    Write-Host "📊 Size: $size KB | Header: ✓" -ForegroundColor Cyan
}

# Auto-run for current user if script is executed directly
if ($MyInvocation.InvocationName -ne '.') {
    $vscodePath = "$env:APPDATA\Code\User\workspaceStorage"
    if (Test-Path $vscodePath) {
        $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        Export-CopilotChatLogs -Path $vscodePath -Output "copilot-chatlog-extract-$timestamp.jsonl"
    }
    else {
        Write-Host "❌ VS Code storage not found: $vscodePath" -ForegroundColor Red
    }
}
