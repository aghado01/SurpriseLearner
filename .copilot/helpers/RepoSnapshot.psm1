<#
.SYNOPSIS
    RepoSnapshot.Helpers - Advanced Repository Analysis and Filtering Infrastructure

.DESCRIPTION
    A comprehensive PowerShell module providing vendor-agnostic repository snapshot
    capabilities with intelligent path filtering, gitignore-compliant pattern matching,
    and batch-optimized git integration. Built for reliable ML engineering workflows
    and automated development pipelines.

    Core Philosophy:
    - Gitignore-semantic compatibility with ordered rule evaluation
    - Batch processing for performance at scale
    - Vendor-agnostic tokenization hooks for budget-aware content trimming
    - Fail-safe fallbacks when external tools (git) are unavailable
    - Scripting ergonomics: predictable, composable, pipeline-friendly

    Key Capabilities:
    - External ignore file discovery with parent-directory inheritance
    - High-performance glob-to-regex conversion with ** recursive matching
    - Batched git check-ignore for authoritative ignore decisions
    - Configurable precedence hierarchy: ForceInclude → Allowlist → Exclude → Ignore
    - Token-aware content trimming with pluggable tokenizer CLI support
    - Cross-platform path normalization and case-sensitivity handling

.NOTES
    Module:      RepoSnapshot.Helpers
    Version:     2.2.0
    Author:      Azriel Ghadooshahy
    Purpose:     Generate highly configurable high-fidelity repository snapshots for various purposes including delivering LLM-consumable whole-repo views in compact format
    Requires:    PowerShell 7.0+, .NET 6.0+
    Compatibility: Cross-platform (Windows, Linux, macOS)

    Git Integration: Uses batch git check-ignore when available; graceful fallback
    Performance:     Optimized for repositories with 10K+ files via batching and caching
    Parallelism:     Native PowerShell 7+ ForEach-Object -Parallel with intelligent throttling

.EXAMPLE
    # Simple usage - current directory with auto-generated filename
    Get-RepoSnapshot

    # Use ML preset with custom output
    Get-RepoSnapshot -Preset 'python-ml' -OutputFile 'ml-snapshot.json'

    # Power user with all options
    Get-RepoSnapshot -IncludePatterns "src/**" -ExtraIncludePatterns "models/final.pth" -TokenBudget 8000 -OutputFile 'custom.json'

.LINK
    https://docs.microsoft.com/powershell/scripting/whats-new/what-s-new-in-powershell-70
    https://git-scm.com/docs/gitignore - Git Ignore Pattern Reference
#>
# Requires PowerShell 7

using namespace System.Text
using namespace System.Text.RegularExpressions
using namespace System.Collections.Generic

# ==================== CORE IGNORE PROCESSING ====================

function Resolve-RelPath {
    param(
        [Parameter(Mandatory)] [string]$Root,
        [Parameter(Mandatory)] [string]$Path
    )
    $full = [System.IO.Path]::GetFullPath($Path)
    $rootFull = [System.IO.Path]::GetFullPath($Root)
    $rel = $full.Substring($rootFull.Length).TrimStart('\', '/')
    return ($rel -replace '\\', '/')
}

function Read-GitIgnoreRules {
    param(
        [Parameter(Mandatory)] [string]$FilePath,
        [Parameter()] [string]$Source = $null
    )

    if (-not (Test-Path $FilePath)) { return @() }

    try {
        $content = [System.IO.File]::ReadAllText($FilePath, [System.Text.UTF8Encoding]::new($false))
    }
    catch {
        Write-Warning "Failed to read ignore file: $FilePath"
        return @()
    }

    $rules = [List[object]]::new()
    $lineNum = 0

    foreach ($rawLine in ($content -split "`r?`n")) {
        $lineNum++
        $line = $rawLine.Trim()

        if ([string]::IsNullOrWhiteSpace($line) -or $line.StartsWith('#')) {
            continue
        }

        $isNegation = $false
        if ($line.StartsWith('!')) {
            $isNegation = $true
            $line = $line.Substring(1).Trim()
            if ([string]::IsNullOrWhiteSpace($line)) { continue }
        }

        $anchored = $line.StartsWith('/')
        $dirOnly = $line.EndsWith('/')

        $pattern = $line
        if ($anchored) { $pattern = $pattern.Substring(1) }
        if ($dirOnly) { $pattern = $pattern.TrimEnd('/') }

        $rules.Add([pscustomobject]@{
                Pattern      = $pattern
                IsNegation   = $isNegation
                Anchored     = $anchored
                DirOnly      = $dirOnly
                Source       = $Source ?? $FilePath
                LineNumber   = $lineNum
                OriginalLine = $rawLine
            })
    }

    return $rules
}

function Convert-GitIgnoreGlobToRegex {
    param(
        [Parameter(Mandatory)] [string]$Pattern,
        [Parameter()] [bool]$Anchored = $false,
        [Parameter()] [bool]$DirOnly = $false
    )

    $sb = [StringBuilder]::new()
    $i = 0
    $len = $Pattern.Length

    if ($Anchored) {
        [void]$sb.Append('^')
    }
    else {
        [void]$sb.Append('^(?:.*\/)?')
    }

    while ($i -lt $len) {
        $c = $Pattern[$i]

        switch ($c) {
            '.' { [void]$sb.Append('\.'); $i++ }
            '+' { [void]$sb.Append('\+'); $i++ }
            '(' { [void]$sb.Append('\('); $i++ }
            ')' { [void]$sb.Append('\)'); $i++ }
            '|' { [void]$sb.Append('\|'); $i++ }
            '^' { [void]$sb.Append('\^'); $i++ }
            '$' { [void]$sb.Append('\$'); $i++ }
            '{' { [void]$sb.Append('\{'); $i++ }
            '}' { [void]$sb.Append('\}'); $i++ }
            '[' { [void]$sb.Append('\['); $i++ }
            ']' { [void]$sb.Append('\]'); $i++ }
            '\' { [void]$sb.Append('\\'); $i++ }

            '*' {
                if (($i + 1) -lt $len -and $Pattern[$i + 1] -eq '*') {
                    $i += 2
                    if ($i -lt $len -and $Pattern[$i] -eq '/') {
                        [void]$sb.Append('(?:.*\/)?')
                        $i++
                    }
                    else {
                        [void]$sb.Append('.*')
                    }
                }
                else {
                    [void]$sb.Append('[^\/]*')
                    $i++
                }
            }

            '?' {
                [void]$sb.Append('[^\/]')
                $i++
            }

            '/' {
                [void]$sb.Append('\/')
                $i++
            }

            default {
                [void]$sb.Append([Regex]::Escape([string]$c))
                $i++
            }
        }
    }

    if ($DirOnly) {
        [void]$sb.Append('(?:\/.*)?$')
    }
    else {
        [void]$sb.Append('(?:\/.*)?$')
    }

    return $sb.ToString()
}

function Build-GitIgnoreMatcher {
    param(
        [Parameter(Mandatory)] [AllowEmptyCollection()] [object[]]$Rules,
        [Parameter()] [bool]$CaseSensitive = $false
    )

    if ($Rules.Count -eq 0) {
        return { param($p, $d) $false }
    }

    $compiledRules = foreach ($rule in $Rules) {
        try {
            $regexPattern = Convert-GitIgnoreGlobToRegex -Pattern $rule.Pattern -Anchored $rule.Anchored -DirOnly $rule.DirOnly
            $regexOptions = if ($CaseSensitive) { [RegexOptions]::Compiled } else { [RegexOptions]::Compiled -bor [RegexOptions]::IgnoreCase }

            [pscustomobject]@{
                Regex        = [Regex]::new($regexPattern, $regexOptions)
                IsNegation   = $rule.IsNegation
                DirOnly      = $rule.DirOnly
                Source       = $rule.Source
                Pattern      = $rule.Pattern
                OriginalLine = $rule.OriginalLine
            }
        }
        catch {
            Write-Warning "Failed to compile regex for pattern '$($rule.Pattern)' from $($rule.Source):$($rule.LineNumber) - $_"
            continue
        }
    }

    return {
        param([string]$RelativePath, [bool]$IsDirectory)

        $normalizedPath = $RelativePath -replace '\\', '/'
        $decision = $null

        foreach ($compiledRule in $compiledRules) {
            if ($compiledRule.DirOnly -and -not $IsDirectory) {
                $parentMatch = $compiledRule.Regex.IsMatch($normalizedPath)
                if (-not $parentMatch) { continue }
            }

            if ($compiledRule.Regex.IsMatch($normalizedPath)) {
                $decision = if ($compiledRule.IsNegation) { $false } else { $true }
            }
        }

        return [bool]($decision -eq $true)
    }
}

function Find-ExternalIgnoreRules {
    param(
        [Parameter(Mandatory)] [string]$Root,
        [string]$IgnoreFileName = ".ignore",
        [Parameter()] [switch]$UseParentIgnore
    )

    if ([string]::IsNullOrWhiteSpace($IgnoreFileName)) { return @() }

    $rootFull = [IO.Path]::GetFullPath($Root)
    $dir = $rootFull
    $all = [List[object]]::new()
    while ($true) {
        $candidate = Join-Path $dir $IgnoreFileName
        if (Test-Path $candidate) {
            $rules = Read-GitIgnoreRules -FilePath $candidate
            if ($rules.Count -gt 0) {
                $all.AddRange($rules)
            }
        }
        $parent = [IO.Directory]::GetParent($dir)
        if (-not $UseParentIgnore) { break }
        if ($null -eq $parent) { break }
        if ($parent.FullName -eq $dir) { break }
        $dir = $parent.FullName
    }

    $result = [System.Linq.Enumerable]::Reverse($all)
    return @($result)
}

function Normalize-PatternArray {
    param(
        [Parameter()] [AllowEmptyCollection()] [string[]]$Patterns
    )

    if (-not $Patterns -or $Patterns.Count -eq 0) {
        return @()
    }

    $normalized = @()
    foreach ($pattern in $Patterns) {
        if ([string]::IsNullOrWhiteSpace($pattern)) { continue }

        # Convert .ext to *.ext for file extension patterns
        if ($pattern -match '^\.[\w]+$') {
            $normalized += "*$pattern"
        }
        else {
            $normalized += $pattern
        }
    }

    return $normalized
}

function New-PathInclusionTester {
    param(
        [Parameter()] [scriptblock]$GitIgnoreMatcher = { param($p, $d) $false },
        [Parameter()] [scriptblock]$ExternalIgnoreMatcher = { param($p, $d) $false },
        [Parameter()] [AllowEmptyCollection()] [string[]]$ExtraIgnorePatterns = @(),
        [Parameter()] [AllowEmptyCollection()] [string[]]$ExtraIncludePatterns = @(),
        [Parameter()] [AllowEmptyCollection()] [string[]]$IncludePatterns = @(),
        [Parameter()] [AllowEmptyCollection()] [string[]]$ExcludePatterns = @(),
        [Parameter()] [HashSet[string]]$GitIgnoredPaths = $null,
        [Parameter()] [bool]$CaseSensitive = $false
    )

    # Normalize all pattern arrays
    $ExtraIgnorePatterns = Normalize-PatternArray -Patterns $ExtraIgnorePatterns
    $ExtraIncludePatterns = Normalize-PatternArray -Patterns $ExtraIncludePatterns
    $IncludePatterns = Normalize-PatternArray -Patterns $IncludePatterns
    $ExcludePatterns = Normalize-PatternArray -Patterns $ExcludePatterns

    # SIMPLIFIED LOGIC: Only create matchers for non-empty, non-wildcard patterns
    $hasRealIncludePatterns = ($IncludePatterns.Count -gt 0 -and $IncludePatterns -notcontains "*")
    $hasExcludePatterns = ($ExcludePatterns.Count -gt 0)
    $hasExtraIgnore = ($ExtraIgnorePatterns.Count -gt 0)
    $hasExtraInclude = ($ExtraIncludePatterns.Count -gt 0)

    # Build matchers only when needed
    $includeMatcher = if ($hasRealIncludePatterns) {
        $rules = $IncludePatterns | ForEach-Object {
            [pscustomobject]@{ Pattern = $_; IsNegation = $false; Anchored = $_.StartsWith('/'); DirOnly = $_.EndsWith('/'); Source = '<cli>' }
        }
        Build-GitIgnoreMatcher -Rules $rules -CaseSensitive $CaseSensitive
    } else {
        { param($p, $d) $true }  # Allow all when no real include patterns
    }

    $excludeMatcher = if ($hasExcludePatterns) {
        $rules = $ExcludePatterns | ForEach-Object {
            [pscustomobject]@{ Pattern = $_; IsNegation = $false; Anchored = $_.StartsWith('/'); DirOnly = $_.EndsWith('/'); Source = '<cli>' }
        }
        Build-GitIgnoreMatcher -Rules $rules -CaseSensitive $CaseSensitive
    } else {
        { param($p, $d) $false }
    }

    $extraIgnoreMatcher = if ($hasExtraIgnore) {
        $rules = $ExtraIgnorePatterns | ForEach-Object {
            [pscustomobject]@{ Pattern = $_; IsNegation = $false; Anchored = $_.StartsWith('/'); DirOnly = $_.EndsWith('/'); Source = '<cli>' }
        }
        Build-GitIgnoreMatcher -Rules $rules -CaseSensitive $CaseSensitive
    } else {
        { param($p, $d) $false }
    }

    $extraIncludeMatcher = if ($hasExtraInclude) {
        $rules = $ExtraIncludePatterns | ForEach-Object {
            [pscustomobject]@{ Pattern = $_; IsNegation = $false; Anchored = $_.StartsWith('/'); DirOnly = $_.EndsWith('/'); Source = '<cli>' }
        }
        Build-GitIgnoreMatcher -Rules $rules -CaseSensitive $CaseSensitive
    } else {
        { param($p, $d) $false }
    }

    $gitIgnored = $GitIgnoredPaths ?? [HashSet[string]]::new()

    # CLEAR, SIMPLE DECISION LOGIC
    $tester = {
        param(
            [Parameter(Mandatory)] [string]$RelativePath,
            [Parameter(Mandatory)] [bool]$IsDirectory
        )

        $normalizedPath = $RelativePath -replace '\\', '/'

        # 1. FORCE INCLUDES always win
        if ($extraIncludeMatcher.InvokeReturnAsIs($normalizedPath, $IsDirectory)) {
            return $true
        }

        # 2. If we have specific include patterns (not just "*"), file must match
        if ($hasRealIncludePatterns -and -not ($includeMatcher.InvokeReturnAsIs($normalizedPath, $IsDirectory))) {
            return $false
        }

        # 3. Check all exclusion sources
        if ($excludeMatcher.InvokeReturnAsIs($normalizedPath, $IsDirectory)) { return $false }
        if ($extraIgnoreMatcher.InvokeReturnAsIs($normalizedPath, $IsDirectory)) { return $false }
        if ($ExternalIgnoreMatcher.InvokeReturnAsIs($normalizedPath, $IsDirectory)) { return $false }
        if ($GitIgnoreMatcher.InvokeReturnAsIs($normalizedPath, $IsDirectory)) { return $false }
        if ($gitIgnored.Contains($normalizedPath)) { return $false }

        # 4. Default: include
        return $true
    }.GetNewClosure()

    return $tester
}

function Test-IsBinaryFile {
    param([Parameter(Mandatory)] [string]$Path)

    try {
        $fileInfo = [System.IO.FileInfo]::new($Path)
        if ($fileInfo.Length -eq 0) { return $false }

        if ($fileInfo.Length -le 1024) {
            $bytes = [System.IO.File]::ReadAllBytes($Path)
            return ([Array]::IndexOf($bytes, 0) -ge 0)
        }

        $buffer = New-Object byte[] 4096
        $stream = [System.IO.File]::OpenRead($Path)
        try {
            $bytesRead = $stream.Read($buffer, 0, 4096)
            return ([Array]::IndexOf($buffer[0..($bytesRead - 1)], 0) -ge 0)
        }
        finally {
            $stream.Close()
        }
    }
    catch {
        return $true
    }
}

function Build-DirectoryTree {
    param(
        [Parameter(Mandatory)] [string]$Root,
        [Parameter(Mandatory)] [AllowEmptyCollection()] [string[]]$IncludedPaths,
        [Parameter()] [int]$MaxDepth = 6,
        [Parameter()] [int]$MaxFilesPerDir = 100
    )

    $node = [pscustomobject]@{
        name           = [IO.Path]::GetFileName($Root)
        path           = ''
        type           = 'dir'
        children       = @()
        omitted_counts = [pscustomobject]@{ files = 0; dirs = 0 }
    }

    # Handle empty or null paths
    if (-not $IncludedPaths -or $IncludedPaths.Count -eq 0) {
        return $node
    }

    $byDir = $IncludedPaths | Group-Object { [IO.Path]::GetDirectoryName($_) -replace '\\', '/' }
    $map = @{ '' = $node }

    foreach ($g in $byDir) {
        $dirRel = ($g.Name ?? '') -replace '^/', ''
        $parts = @()
        if ($dirRel) { $parts = $dirRel.Split('/') }
        $cursorKey = ''
        $cursor = $node

        foreach ($p in $parts) {
            $cursorKey = if ($cursorKey) { "$cursorKey/$p" } else { $p }
            if (-not $map.ContainsKey($cursorKey)) {
                $newDir = [pscustomobject]@{ name = $p; path = $cursorKey; type = 'dir'; children = @(); omitted_counts = [pscustomobject]@{files = 0; dirs = 0 } }
                $cursor.children += , $newDir
                $map[$cursorKey] = $newDir
            }
            $cursor = $map[$cursorKey]
        }

        $files = $g.Group | ForEach-Object { [IO.Path]::GetFileName($_) } | Sort-Object
        $count = 0
        foreach ($f in $files) {
            if ($count -lt $MaxFilesPerDir) {
                $cursor.children += , ([pscustomobject]@{
                        name = $f
                        path = if ($dirRel) { "$dirRel/$f" } else { $f }
                        type = 'file'
                    })
                $count++
            }
            else {
                $cursor.omitted_counts.files++
            }
        }
    }

    return $node
}

function Build-AsciiTree {
    param([Parameter(Mandatory)] $Tree)

    $lines = [List[string]]::new()

    function Write-TreeNode($node, $prefix) {
        $lines.Add("$prefix$($node.name)")
        if ($node.type -eq 'dir') {
            $children = @($node.children | Sort-Object { $_.type }, { $_.name })
            for ($i = 0; $i -lt $children.Count; $i++) {
                $last = ($i -eq $children.Count - 1)
                $pre = $prefix + ($last ? '└── ' : '├── ')
                $nextPrefix = $prefix + ($last ? '    ' : '│   ')
                Write-TreeNode $children[$i] $pre
            }
            if ($node.omitted_counts.files -gt 0) {
                $lines.Add("$prefix... ($($node.omitted_counts.files) files omitted)")
            }
        }
    }

    Write-TreeNode $Tree ''
    return ($lines -join "`n")
}

function Get-GitIgnoredPaths {
    param(
        [Parameter(Mandatory)] [string]$RepositoryRoot,
        [Parameter(Mandatory)] [string[]]$RelativePaths,
        [Parameter()] [int]$BatchSize = 5000,
        [Parameter()] [switch]$VerboseOutput
    )

    $ignoredSet = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)

    # Resolve a single git executable explicitly
    $gitCmd = Get-Command git -CommandType Application -All -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $gitCmd) {
        if ($VerboseOutput) { Write-Verbose "Git command not found - skipping git ignore check" }
        return $ignoredSet
    }

    $isGitRepo = $false
    Push-Location $RepositoryRoot
    try {
        $result = & $gitCmd.Path 'rev-parse' '--is-inside-work-tree' 2>$null
        $isGitRepo = ($LASTEXITCODE -eq 0 -and $result -eq 'true')
    }
    catch {
        if ($VerboseOutput) { Write-Verbose "Git repo check failed: $_" }
        return $ignoredSet
    }
    finally {
        Pop-Location
    }

    if (-not $isGitRepo) {
        if ($VerboseOutput) { Write-Verbose "Not a git repository: $RepositoryRoot" }
        return $ignoredSet
    }

    for ($i = 0; $i -lt $RelativePaths.Count; $i += $BatchSize) {
        $endIndex = [Math]::Min($i + $BatchSize - 1, $RelativePaths.Count - 1)
        $batch = $RelativePaths[$i..$endIndex]

        try {
            $sb = [System.Text.StringBuilder]::new()
            foreach ($path in $batch) {
                [void]$sb.AppendLine($path)
            }

            $tempFile = [System.IO.Path]::GetTempFileName()
            [System.IO.File]::WriteAllText($tempFile, $sb.ToString(), [System.Text.UTF8Encoding]::new($false))

            $psi = [System.Diagnostics.ProcessStartInfo]::new()
            $psi.FileName = $gitCmd.Path
            $psi.WorkingDirectory = $RepositoryRoot
            $psi.UseShellExecute = $false
            $psi.RedirectStandardInput = $true
            $psi.RedirectStandardOutput = $true
            $psi.RedirectStandardError = $true
            $psi.CreateNoWindow = $true
            [void]$psi.ArgumentList.Add('check-ignore')
            [void]$psi.ArgumentList.Add('--stdin')
            [void]$psi.ArgumentList.Add('--quiet')
            [void]$psi.ArgumentList.Add('--no-index')

            $process = [System.Diagnostics.Process]::Start($psi)
            $process.StandardInput.Write($sb.ToString())
            $process.StandardInput.Close()

            $output = $process.StandardOutput.ReadToEnd()
            $process.WaitForExit()
            $exitCode = $process.ExitCode
            $process.Dispose()

            if ($exitCode -eq 0) {
                foreach ($line in ($output -split "`r?`n")) {
                    $trimmed = $line.Trim()
                    if ($trimmed) {
                        $pathname = $trimmed -replace '\\', '/'
                        [void]$ignoredSet.Add($pathname)
                    }
                }
            }
        }
        catch {
            Write-Warning "Git check-ignore failed for batch starting at $i`: $_"
        }
        finally {
            if ($tempFile -and (Test-Path $tempFile)) {
                Remove-Item $tempFile -Force -ErrorAction SilentlyContinue
            }
        }
    }

    return $ignoredSet
}

function Convert-LineEndings {
    param([Parameter(Mandatory)] [string]$Text)
    return ($Text -replace "`r`n", "`n" -replace "`r", "`n")
}

function Read-TextPreview {
    param(
        [Parameter(Mandatory)] [string]$Path,
        [Parameter()] [int]$MaxChars = 400,
        [Parameter()] [switch]$MinifyWhitespace
    )

    try {
        $raw = [System.IO.File]::ReadAllText($Path, [System.Text.UTF8Encoding]::new($false))
    }
    catch {
        return ''
    }

    $raw = Convert-LineEndings $raw
    if ($MinifyWhitespace) {
        $raw = ($raw -replace '[ \t]{2,}', ' ') -replace '(\n){3,}', "`n`n"
    }

    if ($raw.Length -le $MaxChars) { return $raw }

    $headLen = [Math]::Min([Math]::Floor($MaxChars * 0.6), $raw.Length)
    $tailLen = [Math]::Min($MaxChars - $headLen - 20, [Math]::Max(0, $raw.Length - $headLen - 20))
    $head = $raw.Substring(0, $headLen)
    $tail = if ($tailLen -gt 0) { $raw.Substring($raw.Length - $tailLen) } else { '' }
    return ($head + "`n<...omitted...>`n" + $tail)
}

function Measure-TokenCount {
    param(
        [Parameter(Mandatory)] $Object,
        [Parameter()] [string]$TokenizerCLI = '',
        [Parameter()] [int]$Depth = 20
    )

    if ($TokenizerCLI) {
        $json = $Object | ConvertTo-Json -Depth $Depth -Compress
        $psi = [System.Diagnostics.ProcessStartInfo]::new()
        $parts = $TokenizerCLI -split '\s+'
        $psi.FileName = $parts[0]
        foreach ($arg in $parts[1..($parts.Length - 1)]) { $psi.ArgumentList.Add($arg) }
        $psi.UseShellExecute = $false
        $psi.RedirectStandardInput = $true
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true

        $p = [System.Diagnostics.Process]::Start($psi)
        try {
            $p.StandardInput.Write($json)
            $p.StandardInput.Close()
            while (-not $p.HasExited) { Start-Sleep -Milliseconds 10 }
            $out = $p.StandardOutput.ReadToEnd().Trim()
            if ([int]::TryParse($out, [ref]([int]$null))) {
                return [int]$out
            }
            else {
                return [math]::Ceiling($json.Length / 4.0)
            }
        }
        finally {
            if (-not $p.HasExited) { $p.Kill() }
            $p.Dispose()
        }
    }
    else {
        $json = $Object | ConvertTo-Json -Depth $Depth -Compress
        return [math]::Ceiling($json.Length / 4.0)
    }
}

# ==================== HELPER FUNCTIONS ====================


# ==================== OPTIMIZED FILE ENUMERATION ====================

function Get-FilteredFiles {
    param(
        [string]$Root,
        [string[]]$ExcludeDirectories = @(),
        [string[]]$ExcludePatterns = @(),
        [int]$MaxFileCount = 50000
    )

    $files = [System.Collections.Generic.List[string]]::new()
    $stack = [System.Collections.Generic.Stack[string]]::new()
    $stack.Push($Root)

    while ($stack.Count -gt 0 -and $files.Count -lt $MaxFileCount) {
        $currentDir = $stack.Pop()

        try {
            # Get directory name for exclusion check
            $dirName = Split-Path $currentDir -Leaf
            $relativePath = $currentDir.Substring($Root.Length).TrimStart('\', '/').Replace('\', '/')

            # SKIP ENTIRE DIRECTORY if it matches exclusion patterns
            $shouldSkipDir = $false
            foreach ($excludeDir in $ExcludeDirectories) {
                if ($dirName -eq $excludeDir -or $relativePath -like "*/$excludeDir" -or $relativePath -eq $excludeDir) {
                    $shouldSkipDir = $true
                    Write-Verbose "Skipping entire directory: $relativePath"
                    break
                }
            }

            if ($shouldSkipDir) { continue }

            # Process files in current directory
            Get-ChildItem -LiteralPath $currentDir -File -ErrorAction SilentlyContinue | ForEach-Object {
                if ($files.Count -lt $MaxFileCount) {
                    $files.Add($_.FullName)
                }
            }

            # Add subdirectories to stack for processing
            Get-ChildItem -LiteralPath $currentDir -Directory -ErrorAction SilentlyContinue | ForEach-Object {
                $stack.Push($_.FullName)
            }
        }
        catch {
            Write-Verbose "Skipping inaccessible directory: $currentDir"
        }
    }

    return $files.ToArray()
}

# Helper functions for dynamic defaults
function _GetDefaultOutFileName {
    param ([string]$BasePath)
    return "$(Split-Path -Leaf $BasePath)_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
}
$PWDLeaf = _GetDefaultOutFileName "$PWD"

function _GetDefaultThrottleLimit {
    try {
        $cpuCount = [Environment]::ProcessorCount
        return [math]::Max(1, [math]::Floor($cpuCount / 2))
    }
    catch {
        Write-Warning "Failed to detect CPU count, defaulting to 1 core"
        return 1
    }
}
$HalfNumCores = _GetDefaultThrottleLimit


# ==================== MAIN PUBLIC FUNCTION ====================
function Get-RepoSnapshot {
    <#
    .SYNOPSIS
        Generate a comprehensive repository snapshot with preset support and smart defaults

    .PARAMETER Path
        Directory to snapshot (default: current directory)

    .PARAMETER Preset
        Preset name to load from preset file

    .PARAMETER PresetFile
        Path to preset file (.json or .jsonl format - auto-detected)

    .PARAMETER OutputFile
        If specified, saves JSON to this path; otherwise auto-generates timestamped filename

    .PARAMETER RespectGitIgnore
        Use git check-ignore for authoritative ignore decisions (default: $true)

    .PARAMETER UseParentIgnore
        Search parent directories for .ignore files (default: $true)

    .PARAMETER IgnoreFileName
        Name of external ignore files to search for (default: ".ignore")

    .PARAMETER UseParallelism
        Enable PowerShell 7+ parallel processing (default: $true)

    .PARAMETER IncludeFileContent
        Include file content in snapshot (default: $true)

    .PARAMETER MinifyWhitespace
        Compress whitespace in file previews (default: $false)

    .PARAMETER TokenizerCLI
        External tokenizer command for precise token counting (default: "")

    .PARAMETER IncludePatterns
        Include patterns (glob syntax) - treated as allowlist when specified

    .PARAMETER ExcludePatterns
        Exclude patterns (glob syntax)

    .PARAMETER ExtraIgnorePatterns
        Additional ignore patterns beyond gitignore

    .PARAMETER ExtraIncludePatterns
        Force-include patterns that override ignores

    .PARAMETER TokenBudget
        Maximum tokens for content optimization (default: 4000)

    .PARAMETER GitHistoryCount
        Number of recent commits to include (default: 5)

    .PARAMETER MaxCoreCount
        Maximum parallel threads (default: 0 = auto-detect)

    .PARAMETER MaxFileSizeKB
        Maximum file size for content inclusion in KB (default: 1024)

    .PARAMETER MaxFileCount
        Maximum number of files to process (default: 50000)

    .EXAMPLE
        Get-RepoSnapshot
        Get-RepoSnapshot -Preset 'python-ml'
        Get-RepoSnapshot -OutputFile "snapshot.json" -TokenBudget 8000
        Get-RepoSnapshot -MaxFileSizeKB 512 -MinifyWhitespace
        Get-RepoSnapshot -WhatIf
    #>

    [CmdletBinding(SupportsShouldProcess)]
    param(
        [string]$Path = "$PWD",
        [string]$OutputFile = "$PWD/.snapshot/$PWDLeaf.json",
        [bool]$ExportTree = $true,
        [bool]$UseParallelism = $true,
        [int]$ThrottleLimit = $HalfNumCores,
        [int]$GitHistoryCount = 20,
        [int]$MaxFileSizeKB = 2048,
        [int]$MaxFileCount = 10000,
        [int]$TokenBudget = 0,
        [string]$TokenizerCLI = "",
        [bool]$MinifyWhitespace = $true,
        [bool]$UseParentIgnore = $true,
        [bool]$RespectGitIgnore = $true,
        [bool]$IncludeFileContent = $true,
        [string]$NativeIgnoreFile = ".snapshotignore",
        # Smart defaults - reasonable "built-in preset" behavior
        [string[]]$IncludePatterns = @(),  # Empty = include all by default
        [string[]]$ExcludePatterns = @(
            "_depr/",".snapshot/**", "*.log", "*.tmp", ".git/**", "node_modules/**",
            "__pycache__/**", "*.pyc", ".venv/**", "venv/**",
            "dist/**", "build/**", "*.egg-info/**", ".DS_Store"
        ),
        [string[]]$ExtraIgnorePatterns = @(),
        [string[]]$ExtraIncludePatterns = @()
    )

    $ErrorActionPreference = 'Stop'
    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    # Resolve target directory
    $rootPath = [IO.Path]::GetFullPath($Path)
    if (-not (Test-Path $rootPath)) { throw "Root not found: $rootPath" }

    Write-Verbose "Starting RepoSnapshot for: $rootPath"

    # Create the output directory if it doesn't exist (smart default behavior)
    if ([string]::IsNullOrWhiteSpace($OutputFile)) {
        $pathLeaf = Split-Path -Leaf $Path
        $timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
        $defaultFileName = "${pathLeaf}_${timestamp}.json"
        $OutputFile = Join-Path $Path ".snapshot" $defaultFileName
    }

    $outputDir = Split-Path -Parent $OutputFile
    if (-not (Test-Path -Path $outputDir)) {
        New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
        Write-Verbose "Created output directory: $outputDir"
    }

    # Auto-detect optimal parallelism for PowerShell 7+
    if ($ThrottleLimit -le 0) {
        $cpuCount = [Environment]::ProcessorCount
        $ThrottleLimit = [Math]::Max(1, [Math]::Floor($cpuCount / 2))
        Write-Verbose "Auto-detected parallelism: $ThrottleLimit cores (CPU count: $cpuCount)"
    }

    # Discover external ignore rules
    Write-Verbose "Discovering external ignore rules..."
    $externalRules = if ([string]::IsNullOrWhiteSpace($NativeIgnoreFile)) {
        @()
    } else {
        Find-ExternalIgnoreRules -Root $rootPath -IgnoreFileName $NativeIgnoreFile -UseParentIgnore:$UseParentIgnore
    }
    Write-Verbose "Found $($externalRules.Count) external ignore rules"

    $externalMatcher = if ($externalRules.Count -gt 0) {
        Build-GitIgnoreMatcher -Rules $externalRules
    } else {
        { param($p, $d) $false }
    }

    # Enhanced file enumeration with directory-first exclusion
    Write-Verbose "Enumerating files with directory-first optimization..."

    # Extract directory names from exclude patterns
    $excludeDirectories = @()
    $excludeFilePatterns = @()

    # Combine all exclusion sources
    $allExcludePatterns = $ExcludePatterns + $ExtraIgnorePatterns
    foreach ($pattern in $allExcludePatterns) {
        if ($pattern -match '^([^*?\/]+)/?(?:\*\*)?/?$') {
            # Simple directory pattern like "__pycache__" or "__pycache__/**"
            $excludeDirectories += $matches[1]
        } else {
            # Complex pattern that needs regex matching
            $excludeFilePatterns += $pattern
        }
    }

    # Add common directory exclusions
    $excludeDirectories += @('.git', 'node_modules', '__pycache__', '.venv', 'venv', 'dist', 'build', '.snapshot')
    $excludeDirectories = $excludeDirectories | Select-Object -Unique

    Write-Verbose "Directory exclusions: $($excludeDirectories -join ', ')"
    Write-Verbose "Pattern exclusions: $($excludeFilePatterns -join ', ')"

    # Get files with directory-first filtering
    $allFiles = Get-FilteredFiles -Root $rootPath -ExcludeDirectories $excludeDirectories -MaxFileCount $MaxFileCount

    # Apply remaining pattern-based exclusions only to the reduced file set
    if ($excludeFilePatterns.Count -gt 0) {
        $allFiles = $allFiles | Where-Object {
            $relativePath = $_.Substring($rootPath.Length).TrimStart('\', '/').Replace('\', '/')
            $shouldExclude = $false

            foreach ($pattern in $excludeFilePatterns) {
                if ($relativePath -like $pattern) {
                    $shouldExclude = $true
                    break
                }
            }

            -not $shouldExclude
        }
    }

    Write-Verbose "Found $($allFiles.Count) files after directory-first optimization"

    if ($allFiles.Count -eq 0) {
        Write-Warning "No files found in $rootPath"
        return [pscustomobject]@{
            metadata = @{ error = "No files found"; root = $rootPath; execution_time_ms = $sw.ElapsedMilliseconds }
            files    = @()
        }
    }

    # Ensure arrays (even when a single item)
    $allFiles = @($allFiles)
    $allDirs  = @(Get-ChildItem -LiteralPath $rootPath -Recurse -Force -Directory | ForEach-Object { $_.FullName })

    # Compute relative paths from absolute lists
    $relFiles = @($allFiles | ForEach-Object { Resolve-RelPath -Root $rootPath -Path $_ })
    $relDirs  = @($allDirs  | ForEach-Object { Resolve-RelPath -Root $rootPath -Path $_ })

    # Git ignored set
    $gitIgnored = [HashSet[string]]::new()
    if ($RespectGitIgnore -and $allFiles.Count -gt 0) {
        Write-Verbose "Running git check-ignore on $($relFiles.Count) files..."
        $gitIgnored = Get-GitIgnoredPaths -RepositoryRoot $rootPath -RelativePaths ($relFiles + $relDirs) -VerboseOutput:$($PSCmdlet.MyInvocation.BoundParameters.ContainsKey('Verbose'))
        Write-Verbose "Git ignored $($gitIgnored.Count) files"
    }

    # Build path inclusion tester
    Write-Verbose "Creating path inclusion tester..."
    Write-Verbose "Include patterns: $($IncludePatterns -join ', ')"
    Write-Verbose "Exclude patterns: $($ExcludePatterns -join ', ')"

    $includePath = New-PathInclusionTester -ExternalIgnoreMatcher $externalMatcher -ExtraIgnorePatterns $ExtraIgnorePatterns -ExtraIncludePatterns $ExtraIncludePatterns -IncludePatterns $IncludePatterns -ExcludePatterns $ExcludePatterns -GitIgnoredPaths $gitIgnored

    # Null-safety: ensure $includePath is an invokable scriptblock
    if (-not $includePath -or -not ($includePath -is [scriptblock])) {
        $includePath = { param($RelativePath, $IsDirectory) $true }
    }

    # Filter files
    $keptFiles = [List[string]]::new()
    for ($i = 0; $i -lt $relFiles.Count; $i++) {
        $rel = $relFiles[$i]
        $abs = $allFiles[$i]
        if ($includePath.InvokeReturnAsIs($rel, $false)) {
            $keptFiles.Add($abs)
        }
    }

    Write-Verbose "Processing $($keptFiles.Count) kept files"

    # Debug: if no files kept, show some examples of what was filtered
    if ($keptFiles.Count -eq 0 -and $relFiles.Count -gt 0) {
        Write-Warning "No files passed inclusion filter. Sample paths checked:"
        $samplePaths = $relFiles | Select-Object -First 5
        foreach ($sample in $samplePaths) {
            Write-Warning "  - $sample"
        }
        Write-Warning "Include patterns: $($IncludePatterns -join ', ')"
        Write-Warning "Exclude patterns: $($ExcludePatterns -join ', ')"
    }

    # Process file entries with PowerShell 7+ parallel processing
    $entries = [List[object]]::new()

    if ($UseParallelism -and $keptFiles.Count -gt 10) {
        $cpuCount = [Environment]::ProcessorCount
        $baseThrottle = [Math]::Min($ThrottleLimit, $cpuCount)

        if ($keptFiles.Count -lt 100) {
            $throttle = [Math]::Max(2, [Math]::Min($baseThrottle, 4))
        }
        elseif ($keptFiles.Count -lt 1000) {
            $throttle = [Math]::Max(2, $baseThrottle)
        }
        else {
            $throttle = [Math]::Max(4, [Math]::Min($baseThrottle, 12))
        }

        Write-Verbose "Processing with $throttle parallel threads (PowerShell 7+ native parallelism)"

        $fileData = for ($i = 0; $i -lt $keptFiles.Count; $i++) {
            @{
                AbsPath = $keptFiles[$i]
                RelPath = Resolve-RelPath -Root $rootPath -Path $keptFiles[$i]
            }
        }

        $entries = $fileData | ForEach-Object -Parallel {
            $item = $_
            $absPath = $item.AbsPath
            $relPath = $item.RelPath

            $fi = [System.IO.FileInfo]::new($absPath)
            $size = $fi.Length

            # Binary detection
            $isBinary = $false
            if ($size -gt 0 -and $size -le 1024) {
                try {
                    $bytes = [System.IO.File]::ReadAllBytes($absPath)
                    $isBinary = ([Array]::IndexOf($bytes, 0) -ge 0)
                }
                catch { $isBinary = $true }
            }
            elseif ($size -gt 1024) {
                try {
                    $buffer = New-Object byte[] 4096
                    $fs = [System.IO.File]::OpenRead($absPath)
                    $bytesRead = $fs.Read($buffer, 0, 4096)
                    $fs.Close()
                    $isBinary = ([Array]::IndexOf($buffer[0..($bytesRead - 1)], 0) -ge 0)
                }
                catch { $isBinary = $true }
            }

            $preview = $null
            if ($using:IncludeFileContent -and -not $isBinary -and $size -le ($using:MaxFileSizeKB * 1024)) {
                try {
                    $raw = [System.IO.File]::ReadAllText($absPath, [System.Text.UTF8Encoding]::new($false))
                    $raw = $raw -replace "`r`n", "`n" -replace "`r", "`n"

                    if ($using:MinifyWhitespace) {
                        $raw = ($raw -replace '[ \t]{2,}', ' ') -replace '(\n){3,}', "`n`n"
                    }

                    if ($raw.Length -gt 400) {
                        $headLen = [Math]::Min(240, $raw.Length)
                        $tailLen = [Math]::Min(140, [Math]::Max(0, $raw.Length - $headLen - 20))
                        $head = $raw.Substring(0, $headLen)
                        $tail = if ($tailLen -gt 0) { $raw.Substring($raw.Length - $tailLen) } else { '' }
                        $preview = "$head`n<...omitted...>`n$tail"
                    }
                    else {
                        $preview = $raw
                    }
                }
                catch {
                    $preview = $null
                }
            }

            [pscustomobject]@{
                path       = $relPath
                size       = $size
                binary     = $isBinary
                preview    = $preview
                last_write = $fi.LastWriteTimeUtc.ToString('o')
            }
        } -ThrottleLimit $throttle
    }
    else {
        # Sequential processing
        Write-Verbose "Processing $($keptFiles.Count) files sequentially"
        for ($i = 0; $i -lt $keptFiles.Count; $i++) {
            $abs = $keptFiles[$i]
            $rel = Resolve-RelPath -Root $rootPath -Path $abs

            $fi = Get-Item -LiteralPath $abs
            $size = $fi.Length
            $isBinary = Test-IsBinaryFile -Path $abs
            $preview = $null
            if ($IncludeFileContent -and -not $isBinary -and $size -le ($MaxFileSizeKB * 1024)) {
                $preview = Read-TextPreview -Path $abs -MaxChars 400 -MinifyWhitespace:$MinifyWhitespace
            }

            $entries.Add([pscustomobject]@{
                    path       = $rel
                    size       = $size
                    binary     = $isBinary
                    preview    = $preview
                    last_write = $fi.LastWriteTimeUtc.ToString('o')
                })
        }
    }

    # Build directory structure and ASCII tree
    $includedRel = @($entries | ForEach-Object { $_.path })
    if ($includedRel.Count -eq 0) {
        $includedRel = @()  # Ensure empty array, not null
    }
    $tree = Build-DirectoryTree -Root $rootPath -IncludedPaths $includedRel -MaxDepth 6 -MaxFilesPerDir 100
    $ascii = Build-AsciiTree -Tree $tree

    # Build metadata
    $metadata = [pscustomobject]@{
        export_date         = (Get-Date).ToString('o')
        root                = $rootPath
        file_count          = $entries.Count
        ps_version          = $PSVersionTable.PSVersion.ToString()
        parallel_processing = $UseParallelism
        max_parallelism     = $ThrottleLimit
        execution_time_ms   = $sw.ElapsedMilliseconds
    }

    # Add git history if requested
    $gitCmd = Get-Command git -CommandType Application -All -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($GitHistoryCount -gt 0 -and $gitCmd) {
        Push-Location $rootPath
        try {
            $log = & $gitCmd.Path 'log' '-n' $GitHistoryCount "--pretty=format:%H|%an|%ad|%s" '--date=iso-strict' 2>$null
            if ($LASTEXITCODE -eq 0) {
                $metadata | Add-Member -NotePropertyName git_history -NotePropertyValue (
                    $log -split "`n" | ForEach-Object {
                        $parts = $_ -split '\|', 4
                        if ($parts.Count -ge 4) {
                            [pscustomobject]@{ hash = $parts[0]; author = $parts[1]; date = $parts[2]; subject = $parts[3] }
                        }
                    }
                )
            }
        }
        finally {
            Pop-Location
        }
    }

    # Create snapshot object
    $snapshot = [pscustomobject]@{
        metadata           = $metadata
        DirectoryStructure = $tree
        AsciiTree          = $ascii
        files              = $entries
    }

    # Apply token budget optimization if specified
    if ($TokenBudget -gt 0) {
        $iter = 0
        while ($iter -lt 6) {
            $iter++
            $tokens = Measure-TokenCount -Object $snapshot -TokenizerCLI $TokenizerCLI
            if ($tokens -le $TokenBudget) { break }

            # Reduce preview sizes
            if ($snapshot.files) {
                foreach ($f in $snapshot.files) {
                    if ($f.preview) {
                        $len = $f.preview.Length
                        if ($len -gt 120) {
                            $f.preview = Read-TextPreview -Path $null -MaxChars ([Math]::Max(80, [Math]::Floor($len * 0.6))) -MinifyWhitespace
                        }
                    }
                }
            }

            $tokens = Measure-TokenCount -Object $snapshot -TokenizerCLI $TokenizerCLI
            if ($tokens -le $TokenBudget) { break }

            # Remove all previews if still over budget
            if ($snapshot.files) {
                foreach ($f in $snapshot.files) { $f.preview = $null }
            }
        }

        $metadata | Add-Member -NotePropertyName token_budget -NotePropertyValue $TokenBudget -Force
        $metadata | Add-Member -NotePropertyName token_estimate -NotePropertyValue (Measure-TokenCount -Object $snapshot -TokenizerCLI $TokenizerCLI) -Force
    }

    $sw.Stop()
    Write-Host "RepoSnapshot completed in $($sw.ElapsedMilliseconds)ms"

    # Handle output
    if ($PSCmdlet.ShouldProcess($OutputFile, "Write repository snapshot")) {
        $snapshot | ConvertTo-Json -Depth 30 | Set-Content -Path $OutputFile -Encoding UTF8
        Write-Host "Repository snapshot saved to: $OutputFile" -ForegroundColor Green

        # Export coordinated tree file
        if ($ExportTree) {
            $treeFile = $OutputFile -replace '\.json$', '_tree.txt'
            $snapshot.AsciiTree | Set-Content -Path $treeFile -Encoding UTF8
            Write-Host "Tree exported to: $treeFile" -ForegroundColor Cyan
        }

        return $OutputFile
    }
    else {
        Write-Host "Operation cancelled by user" -ForegroundColor Yellow
        return $null
    }
}

# ==================== PRESET HANDLING FUNCTIONS ====================

function Get-PresetArguments {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [ValidateNotNullOrEmpty()]
        [string]$PresetName,

        [Parameter()]
        [string]$PresetFile = (Join-Path $PSScriptRoot "snapshot-presets.json"),

        [Parameter()]
        [string]$SchemaFile = (Join-Path $PSScriptRoot "snapshot-presets.schema.json")
    )

    Write-Verbose "Using preset file: $PresetFile"
    Write-Verbose "Using schema file: $SchemaFile"

    # Validate JSON against schema
    try {
        $schemaContent = Get-Content $SchemaFile -Raw -ErrorAction Stop
        Test-Json -Path $PresetFile -Schema $schemaContent -ErrorAction Stop
        Write-Verbose "Schema validation passed for '$PresetFile'"
    } catch {
        throw "Schema validation failed for '$PresetFile': $($_.Exception.Message)"
    }

    # Load and extract preset
    Write-Verbose "Loading preset '$PresetName' from '$PresetFile'"
    $raw = Get-Content $PresetFile -Raw -ErrorAction Stop
    $parsed = $raw | ConvertFrom-Json -ErrorAction Stop

    # Check if preset exists
    $availablePresets = $parsed.PSObject.Properties.Name
    if (-not ($availablePresets -contains $PresetName)) {
        $availableList = $availablePresets -join ', '
        throw "Preset '$PresetName' not found. Available presets: $availableList"
    }

    Write-Verbose "Found preset '$PresetName'"
    $preset = $parsed.$PresetName

    # Convert to hashtable for splatting
    $args = @{}
    foreach ($property in $preset.PSObject.Properties) {
        $key = $property.Name
        $value = $property.Value

        Write-Verbose "Processing property: $key (type: $($value.GetType().Name), value: $value)"

        if ($null -ne $value) {
            # Handle type conversions
            switch ($value.GetType().Name) {
                'Int64' {
                    $args[$key] = [int]$value
                    Write-Verbose "Converted Int64 to Int32: $key = $($args[$key])"
                }
                'Boolean' {
                    $args[$key] = [bool]$value
                    Write-Verbose "Boolean: $key = $($args[$key])"
                }
                'String' {
                    $args[$key] = [string]$value
                    Write-Verbose "String: $key = $($args[$key])"
                }
                'Object[]' {
                    $args[$key] = [string[]]$value
                    Write-Verbose "Array: $key = [$($value -join ', ')]"
                }
                default {
                    $args[$key] = $value
                    Write-Verbose "Default: $key = $value"
                }
            }
        }
    }

    Write-Verbose "Successfully loaded preset with $($args.Keys.Count) parameters"
    Write-Verbose "Parameter keys: $($args.Keys -join ', ')"
    return $args
}

function Invoke-RepoSnapshotWithPreset {
    [CmdletBinding(SupportsShouldProcess)]
    param(
        [Parameter(Mandatory)]
        [ValidateNotNullOrEmpty()]
        [string]$PresetName,

        [string]$PresetFile = (Join-Path $PSScriptRoot "snapshot-presets.json"),
        [string]$SchemaFile = (Join-Path $PSScriptRoot "snapshot-presets.schema.json")
    )

    # FAST FAIL: Check file existence immediately
    if (-not (Test-Path $PresetFile -PathType Leaf)) {
        Write-Error "Preset file not found: $PresetFile" -ErrorAction Stop
    }
    if (-not (Test-Path $SchemaFile -PathType Leaf)) {
        Write-Error "Schema file not found: $SchemaFile" -ErrorAction Stop
    }

    # Load preset arguments
    $presetArgs = Get-PresetArguments -PresetName $PresetName -PresetFile $PresetFile -SchemaFile $SchemaFile

    # Create combined arguments: preset + user overrides
    $combinedArgs = $presetArgs.Clone()

    # Override with any user-provided parameters (excluding our preset-specific ones)
    $excludeParams = @('PresetName', 'PresetFile', 'SchemaFile')
    foreach ($param in $PSBoundParameters.Keys) {
        if ($param -notin $excludeParams) {
            $combinedArgs[$param] = $PSBoundParameters[$param]
            Write-Verbose "Override: $param = $($PSBoundParameters[$param])"
        }
    }

    # Ensure Path is set
    if (-not $combinedArgs.ContainsKey('Path')) {
        $combinedArgs['Path'] = $PWD
    }

    Write-Verbose "Calling Get-RepoSnapshot with $($combinedArgs.Keys.Count) parameters"

    # Forward all parameters via splatting
    Get-RepoSnapshot @combinedArgs
}




# ==================== MODULE EXPORTS ====================

# Updated aliases
Set-Alias -Name rs -Value Get-RepoSnapshot              # Direct snapshot, no presets
Set-Alias -Name rsp -Value Invoke-RepoSnapshotWithPreset # Preset-based snapshot

# Export the new functions
Export-ModuleMember -Function Get-RepoSnapshot, Invoke-RepoSnapshotWithPreset -Alias rs, rsp
