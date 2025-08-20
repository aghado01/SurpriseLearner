# Requires PowerShell 7.0

using namespace System.Management.Automation.Language
using namespace System.Collections.Generic
using namespace System.Text.RegularExpressions

$script:SchemaVersion = '1.0.0'
$script:TargetPowerShell = '7.5.2'

# Self-contained configuration in lieu of external JSON
$script:PSLinterSchema = @{
    SchemaVersion = '1.0.0'
    Settings = @{
        MaxLineLength = @{ Type = 'Int'; Min = 40; Max = 400; Default = 120 }
        RequiredIndent = @{ Type = 'Int'; Enum = @(2, 4, 8); Default = 4 }
        MaxFileBytes = @{ Type = 'Int64'; Min = 1024; Max = 50MB; Default = 5MB }
        RegexTimeoutSec = @{ Type = 'Int'; Min = 1; Max = 10; Default = 2 }
        MinimumSeverity = @{ Type = 'Enum'; Values = @('Error', 'Warning', 'Info'); Default = 'Info' }
        EnforceTabsAsSpaces = @{ Type = 'Bool'; Default = $true }
    }
    Rules = @{
        RegexValidation = @{
            Enabled = $true
            Severity = 'Error'
            Options = @{
                engine = 'ECMAScript'
                maxBacktracking = 10000
            }
        }
        TokenSanity = @{ Enabled = $true; Severity = 'Error' }
        HereStringIntegrity = @{ Enabled = $true; Severity = 'Error' }
        StringInterpolation = @{ Enabled = $true; Severity = 'Warning' }
        CmdletUsage = @{ Enabled = $true; Severity = 'Warning' }
        DateTimeLiteral = @{ Enabled = $true; Severity = 'Warning' }
        FunctionNaming = @{ Enabled = $true; Severity = 'Warning' }
        Indentation = @{
            Enabled = $true
            Severity = 'Info'
            Options = @{ AllowTabs = $false }
        }
        LineLength = @{ Enabled = $true; Severity = 'Info' }
        TrailingWhitespace = @{ Enabled = $true; Severity = 'Info' }
        UnusedVariables = @{ Enabled = $true; Severity = 'Info' }
        BestPractices = @{ Enabled = $true; Severity = 'Info' }
    }
}

# Regex caching with timeout to prevent catastrophic backtracking
class PSRegexCache {
    static [hashtable] $Cache = @{}
    static [TimeSpan] $DefaultTimeout = [TimeSpan]::FromSeconds(2)

    static [regex] Get([string] $Pattern, [RegexOptions] $Options = [RegexOptions]::None) {
        $key = "$Options::$Pattern"
        if (-not [PSRegexCache]::Cache.ContainsKey($key)) {
            [PSRegexCache]::Cache[$key] = [regex]::new($Pattern, $Options, [PSRegexCache]::DefaultTimeout)
        }
        return [PSRegexCache]::Cache[$key]
    }

    static [void] Clear() {
        [PSRegexCache]::Cache.Clear()
    }
}

# Lint context passed to each rule
class LintContext {
    [string] $FilePath
    [string[]] $Lines
    [Ast] $Ast
    [Token[]] $Tokens
    [ParseError[]] $ParseErrors
    [hashtable] $Settings

    LintContext([string] $FilePath, [string[]] $Lines, [Ast] $Ast, [Token[]] $Tokens, [ParseError[]] $ParseErrors, [hashtable] $Settings) {
        $this.FilePath = $FilePath
        $this.Lines = $Lines
        $this.Ast = $Ast
        $this.Tokens = $Tokens
        $this.ParseErrors = $ParseErrors
        $this.Settings = $Settings
    }
}

# Core linter
class PSLinter {
    [List[pscustomobject]] $Issues
    [hashtable] $RuleRegistry
    [hashtable] $Settings
    [hashtable] $Statistics
    [string] $FilePath
    [string[]] $Lines
    [Ast] $Ast
    [Token[]] $Tokens
    [ParseError[]] $ParseErrors

    # Constructor
    PSLinter() {
        $this.Issues = [List[pscustomobject]]::new()
        $this.Settings = @{
            MaxLineLength = 120
            RequiredIndent = 4
            MaxFileBytes = 5MB
            RegexTimeoutSec = 2
            EnforceTabsAsSpaces = $true
            DiscouragedCmdlets = @{
                'Write-Host' = 'Prefer Write-Output, Write-Information, or Write-Verbose'
                'Invoke-Expression' = 'Avoid due to security risk; consider splatting or safer alternatives'
                'ConvertTo-SecureString -AsPlainText' = 'Avoid plaintext secure strings; use SecretManagement or vault-backed flows'
            }
            ApprovedVerbs = @(
                'Get', 'Set', 'New', 'Remove', 'Add', 'Clear', 'Copy', 'Move', 'Rename', 'Test', 'Start', 'Stop', 'Restart', 'Enable', 'Disable',
                'Install', 'Uninstall', 'Import', 'Export', 'Select', 'Where', 'Sort', 'Group', 'Measure', 'Compare', 'Convert', 'Join', 'Split',
                'Format', 'Out', 'Write', 'Read', 'Show', 'Hide', 'Wait', 'Find', 'Search', 'Invoke', 'Enter', 'Exit', 'Push', 'Pop', 'Send',
                'Receive', 'Request', 'Register', 'Unregister', 'Update', 'Sync', 'Lock', 'Unlock', 'Grant', 'Revoke', 'Protect', 'Unprotect',
                'Backup', 'Restore', 'Mount', 'Dismount', 'Debug', 'Trace', 'Connect', 'Disconnect', 'Publish', 'Unpublish', 'Save', 'Suspend',
                'Resume', 'Submit', 'Block', 'Unblock', 'Limit', 'Resolve', 'Assert', 'Complete', 'Approve', 'Deny', 'Expand', 'Compress',
                'Optimize', 'Reset', 'Undo', 'Redo', 'Confirm', 'Step', 'Build', 'Deploy'
            )
        }
        [PSRegexCache]::DefaultTimeout = [TimeSpan]::FromSeconds($this.Settings.RegexTimeoutSec)
        $this.InitializeStatistics()
        $this.InitializeRules()
    }

    hidden [void] InitializeStatistics() {
        $this.Statistics = @{
            TotalLines = 0
            EmptyLines = 0
            CommentLines = 0
            CodeLines = 0
            Functions = 0
            Parameters = 0
        }
    }

    hidden [void] InitializeRules() {
        # Registry maps RuleId -> ScriptBlock accepting (ctx, addIssue)
        $this.RuleRegistry = @{
            'ParseErrors' = {
                param($ctx, $add)
                foreach ($e in ($ctx.ParseErrors ?? @())) {
                    $severity = 'Error'
                    $category = switch -Regex ($e.ErrorId) {
                        'MissingEndCurlyBrace|MissingEndParenthesis|MissingEndSquareBracket' { 'Bracket Mismatch'; break }
                        'MissingStringTerminator|MissingEndQuote|InvalidStringEscape' { 'String'; break }
                        'InvalidNumericConstant' { 'Number'; break }
                        default { 'Syntax' }
                    }
                    # Fix: Use explicit concatenation to satisfy PSScriptAnalyzer
                    $message = $category + ': ' + $e.Message
                    $add.Invoke($severity, $e.Extent.StartLineNumber, $message, 'ParseError', $e.Extent.Text)
                }
            }
            'TokenSanity' = {
                param($ctx, $add)
                if (-not $ctx.Tokens) { return }
                foreach ($t in $ctx.Tokens) {
                    if ($t.Kind -eq [TokenKind]::Unknown) {
                        $add.Invoke('Error', $t.Extent.StartLineNumber, "Unknown token '$($t.Text)'", 'UnknownToken', $t.Text)
                    }
                    if ($t.Kind -eq [TokenKind]::Number) {
                        $text = $t.Text
                        if ([PSRegexCache]::Get('^0x[^0-9a-fA-F]').IsMatch($text)) {
                            $add.Invoke('Error', $t.Extent.StartLineNumber, "Malformed hex literal: '$text'", 'MalformedHex', $text)
                        }
                        if ([PSRegexCache]::Get('(?i)\b\d+(\.\d+)?e[+-]?$').IsMatch($text) -or
                            [PSRegexCache]::Get('(?i)\be[+-]?[^0-9]').IsMatch($text)) {
                            $add.Invoke('Error', $t.Extent.StartLineNumber, "Invalid scientific notation: '$text'", 'InvalidSci', $text)
                        }
                        # Fix: Use proper split method instead of regex split
                        if ($text.Split('.').Count -gt 2) {
                            $add.Invoke('Error', $t.Extent.StartLineNumber, "Multiple decimal points in number: '$text'", 'MultiDotNumber', $text)
                        }
                    }
                }
            }
            'HereStringIntegrity' = {
                param($ctx, $add)
                if (-not $ctx.Tokens) { return }
                $stack = New-Object System.Collections.Stack
                foreach ($t in $ctx.Tokens) {
                    if ($t.Kind -eq [TokenKind]::HereStringLiteral -or $t.Kind -eq [TokenKind]::HereStringExpandable) {
                        continue
                    }
                    if ($t.Kind -eq [TokenKind]::HereStringBegin) { $stack.Push($t); continue }
                    if ($t.Kind -eq [TokenKind]::HereStringEnd) { if ($stack.Count -gt 0) { $null = $stack.Pop() } }
                }
                while ($stack.Count -gt 0) {
                    $beginTok = $stack.Pop()
                    $add.Invoke('Error', $beginTok.Extent.StartLineNumber, "Unclosed here-string starting on this line", 'UnclosedHereString', $beginTok.Extent.Text)
                }
            }
            'RegexValidation' = {
                param($ctx, $add)
                if (-not $ctx.Ast) { return }
                $binary = $ctx.Ast.FindAll({ param($n) $n -is [BinaryExpressionAst] }, $true)
                foreach ($b in $binary) {
                    if ($b.Operator -in @([TokenKind]::Match, [TokenKind]::NotMatch, [TokenKind]::Imatch, [TokenKind]::Inotmatch)) {
                        if ($b.Right -is [StringConstantExpressionAst]) {
                            $pattern = $b.Right.Value
                            try {
                                $rx = [regex]::new($pattern, [RegexOptions]::None, [PSRegexCache]::DefaultTimeout)
                                $null = $rx.Match('sample')
                            }
                            catch {
                                $add.Invoke('Error', $b.Extent.StartLineNumber, "Invalid regex: $($_.Exception.Message)", 'InvalidRegex', $pattern)
                            }
                        }
                    }
                }
            }
            'StringInterpolation' = {
                param($ctx, $add)
                if (-not $ctx.Tokens) { return }
                $rxProp = [PSRegexCache]::Get('\$[a-zA-Z_]\w*\.[a-zA-Z_]\w*')
                $rxIndex = [PSRegexCache]::Get('\$[a-zA-Z_]\w*\[[^\]]+\]')
                $rxSubExpr = [PSRegexCache]::Get('\$\([^)]+\)')
                for ($i = 0; $i -lt $ctx.Lines.Count; $i++) {
                    $line = $ctx.Lines[$i]
                    if ($line.Trim().StartsWith('#') -or -not $line.Contains('"')) { continue }
                    if ($rxProp.IsMatch($line) -and -not $rxSubExpr.IsMatch($line)) {
                        $add.Invoke('Warning', $i + 1, 'Property access in expandable string should use $() for clarity', 'StringPropertyAccess', $line.Trim())
                    }
                    if ($rxIndex.IsMatch($line) -and -not $rxSubExpr.IsMatch($line)) {
                        $add.Invoke('Warning', $i + 1, 'Indexing in expandable string should use $() for clarity', 'StringIndexAccess', $line.Trim())
                    }
                }
            }
            'Indentation' = {
                param($ctx, $add)
                if ($ctx.Settings.RequiredIndent -le 0) { return }
                $indentSize = [int]$ctx.Settings.RequiredIndent
                $expected = 0
                for ($i = 0; $i -lt $ctx.Lines.Count; $i++) {
                    $line = $ctx.Lines[$i]
                    $trim = $line.Trim()
                    if ($trim -eq '' -or $trim.StartsWith('#')) { continue }
                    # Dedent if line starts with a closing brace
                    if ($trim.StartsWith('}')) { $expected = [math]::Max(0, $expected - $indentSize) }
                    # Count current indent (spaces + tabs)
                    $current = 0
                    $tabFound = $false
                    for ($c = 0; $c -lt $line.Length; $c++) {
                        $ch = $line[$c]
                        if ($ch -eq ' ') { $current++ }
                        elseif ($ch -eq "`t") {
                            $tabFound = $true
                            $current += $indentSize
                        }
                        else { break }
                    }
                    if ($ctx.Settings.EnforceTabsAsSpaces -and $tabFound) {
                        $add.Invoke('Warning', $i + 1, 'Use spaces instead of tabs for indentation', 'TabIndentation', $line.Substring(0, [Math]::Min(20, $line.Length)))
                    }
                    if ([math]::Abs($current - $expected) -gt 1 -and $current -ne 0) {
                        $add.Invoke('Info', $i + 1, "Inconsistent indentation: expected $expected spaces, found $current", 'Indentation', $line)
                    }
                    # Now increase indent if line ends with opening brace
                    if ($trim.EndsWith('{')) { $expected += $indentSize }
                }
            }
            'LineLength' = {
                param($ctx, $add)
                $max = [int]$ctx.Settings.MaxLineLength
                if ($max -le 0) { return }
                for ($i = 0; $i -lt $ctx.Lines.Count; $i++) {
                    $line = $ctx.Lines[$i]
                    if ($line.Length -gt $max) {
                        $snippet = if ($line.Length -gt 80) { $line.Substring(0, 80) + '...' } else { $line }
                        $add.Invoke('Info', $i + 1, "Line exceeds $max chars ($($line.Length))", 'LineLength', $snippet)
                    }
                }
            }
            'TrailingWhitespace' = {
                param($ctx, $add)
                $rx = [PSRegexCache]::Get('\S\s+$')
                for ($i = 0; $i -lt $ctx.Lines.Count; $i++) {
                    $line = $ctx.Lines[$i]
                    if ($rx.IsMatch($line)) {
                        $add.Invoke('Info', $i + 1, 'Trailing whitespace', 'TrailingWhitespace', $line)
                    }
                }
            }
            'CmdletUsage' = {
                param($ctx, $add)
                $discouraged = $ctx.Settings.DiscouragedCmdlets
                for ($i = 0; $i -lt $ctx.Lines.Count; $i++) {
                    $line = $ctx.Lines[$i]
                    if ($line.Trim().StartsWith('#')) { continue }
                    foreach ($k in $discouraged.Keys) {
                        if ($line -like "*$k*") {
                            $add.Invoke('Warning', $i + 1, "Discouraged: $($discouraged[$k])", 'DiscouragedCmdlet', $line.Trim())
                        }
                    }
                    if ($line -like '*| Out-Null*' -and $line -notmatch '-ErrorAction') {
                        $add.Invoke('Info', $i + 1, 'Prefer -ErrorAction/-WarningAction over piping to Out-Null for control flow', 'OutNullUsage', $line.Trim())
                    }
                }
            }
            'DateTimeLiteral' = {
                param($ctx, $add)
                if (-not $ctx.Ast) { return }
                $converts = $ctx.Ast.FindAll({ param($n) $n -is [ConvertExpressionAst] }, $true)
                foreach ($c in $converts) {
                    $typeName = $c.Type.TypeName?.FullName
                    if ($typeName -and $typeName -match '^(?i)datetime$|^System\.DateTime$') {
                        if ($c.Child -is [StringConstantExpressionAst]) {
                            $text = $c.Child.Value
                            try { [datetime]::Parse($text) | Out-Null } catch {
                                $add.Invoke('Warning', $c.Extent.StartLineNumber, "Potentially invalid date literal: '$text'", 'DateTimeParse', $text)
                            }
                        }
                    }
                }
            }
            'FunctionNaming' = {
                param($ctx, $add)
                if (-not $ctx.Ast) { return }
                $funcs = $ctx.Ast.FindAll({ param($n) $n -is [FunctionDefinitionAst] }, $true)
                foreach ($f in $funcs) {
                    $name = $f.Name
                    if (-not $name) { continue }
                    $parts = $name -split '-', 2
                    if ($parts.Count -eq 2) {
                        $verb = $parts[0]
                        if ($ctx.Settings.ApprovedVerbs -notcontains $verb) {
                            $add.Invoke('Warning', $f.Extent.StartLineNumber, "Use approved verb for function name: '$name'", 'FunctionVerb', $name)
                        }
                        foreach ($seg in $parts) {
                            if ($seg -cmatch '^[a-z]') {
                                $add.Invoke('Info', $f.Extent.StartLineNumber, "Use PascalCase for function segments: '$name'", 'FunctionCasing', $name)
                                break
                            }
                        }
                    }
                    else {
                        $add.Invoke('Warning', $f.Extent.StartLineNumber, "Function name should be Verb-Noun: '$name'", 'FunctionVerbNoun', $name)
                    }
                }
            }
            'UnusedVariables' = {
                param($ctx, $add)
                if (-not $ctx.Ast) { return }
                $defined = @{}
                $used = @{}
                # Parameters
                $paramBlocks = $ctx.Ast.FindAll({ param($n) $n -is [ParamBlockAst] }, $true)
                foreach ($pb in $paramBlocks) {
                    foreach ($p in ($pb.Parameters ?? @())) {
                        $n = $p.Name?.VariablePath?.UserPath
                        if ($n) { $defined[$n] = $p.Extent.StartLineNumber }
                    }
                }
                # Assignments (left side)
                $assigns = $ctx.Ast.FindAll({ param($n) $n -is [AssignmentStatementAst] }, $true)
                foreach ($as in $assigns) {
                    $lhs = $as.Left
                    if ($lhs -is [VariableExpressionAst]) {
                        $n = $lhs.VariablePath?.UserPath
                        if ($n) { $defined[$n] = $as.Extent.StartLineNumber }
                    }
                }
                # References (any variable expression)
                $vars = $ctx.Ast.FindAll({ param($n) $n -is [VariableExpressionAst] }, $true)
                foreach ($v in $vars) {
                    $n = $v.VariablePath?.UserPath
                    if ($n) {
                        $used[$n] = $true
                    }
                }
                foreach ($name in $defined.Keys) {
                    if (-not $used.ContainsKey($name) -and $name -notin @('_', 'PSBoundParameters', 'PSCmdlet', 'MyInvocation')) {
                        $add.Invoke('Info', [int]$defined[$name], "Variable `$${name} is defined but never used", 'UnusedVariable', "`$${name}")
                    }
                }
            }
            'BestPractices' = {
                param($ctx, $add)
                if (-not $ctx.Ast) { return }
                # Suggest #Requires when using [CmdletBinding()]
                $hasCmdletBinding = $ctx.Ast.FindAll({ param($n) $n -is [AttributeAst] -and $n.TypeName.FullName -eq 'CmdletBinding' }, $true).Count -gt 0
                if ($hasCmdletBinding -and $ctx.Lines.Count -gt 0 -and -not $ctx.Lines[0].StartsWith('#Requires')) {
                    $add.Invoke('Info', 1, 'Consider adding #Requires -Version X.Y for compatibility', 'RequiresSuggestion')
                }
                # Missing comment-based help near functions
                $funcs = $ctx.Ast.FindAll({ param($n) $n -is [FunctionDefinitionAst] }, $true)
                foreach ($f in $funcs) {
                    $line = $f.Extent.StartLineNumber
                    $start = [Math]::Max(0, $line - 6)
                    $window = $ctx.Lines[$start..([Math]::Min($ctx.Lines.Count - 1, $line - 1))]
                    if ($window -notmatch '^\s*<#' -and $window -notmatch '\.SYNOPSIS') {
                        $add.Invoke('Info', $line, 'Function missing comment-based help', 'MissingHelp', $f.Name)
                    }
                }
                # Hardcoded Windows path hint
                for ($i = 0; $i -lt $ctx.Lines.Count; $i++) {
                    $line = $ctx.Lines[$i]
                    if ($line.Trim().StartsWith('#')) { continue }
                    if ([PSRegexCache]::Get('(?i)\b[A-Z]:\\').IsMatch($line)) {
                        $add.Invoke('Warning', $i + 1, 'Hardcoded absolute path detected; prefer env vars or Join-Path', 'HardcodedPath', $line.Trim())
                    }
                }
            }
        }
    }

    [pscustomobject[]] LintFile([string] $Path, [string[]] $IncludeRules = @(), [string[]] $ExcludeRules = @()) {
        if ([string]::IsNullOrWhiteSpace($Path)) {
            throw [ArgumentException]::new("Path cannot be null or empty", "Path")
        }
        if (-not (Test-Path -LiteralPath $Path)) {
            throw [FileNotFoundException]::new("File not found: $Path")
        }
        $fi = Get-Item -LiteralPath $Path
        if ($fi.Length -gt [int64]$this.Settings.MaxFileBytes) {
            $this.AddIssue('Error', 1, "File too large ($($fi.Length) bytes). Max allowed: $($this.Settings.MaxFileBytes).", 'FileTooLarge', "$Path")
            return $this.Issues.ToArray()
        }
        $content = Get-Content -LiteralPath $Path -Raw
        $lines = $content -split "`r?`n", -1
        return $this.LintContentCore($lines, $Path, $IncludeRules, $ExcludeRules)
    }

    [pscustomobject[]] LintContent([string] $Content, [string[]] $IncludeRules = @(), [string[]] $ExcludeRules = @()) {
        # Content can be empty/null, so just ensure it's a string
        $Content = $Content ?? ''
        $lines = $Content -split "`r?`n", -1
        return $this.LintContentCore($lines, '', $IncludeRules, $ExcludeRules)
    }

    hidden [pscustomobject[]] LintContentCore([string[]] $Lines, [string] $Path, [string[]] $IncludeRules, [string[]] $ExcludeRules) {
        $this.Issues.Clear()
        $this.InitializeStatistics()
        $this.FilePath = $Path
        $this.Lines = $Lines ?? @() # Fix: Ensure Lines is never null

        # Stats
        $this.Statistics.TotalLines = $this.Lines.Count
        foreach ($l in $this.Lines) {
            $trimmed = $l?.Trim() ?? '' # Fix: Handle potential null lines
            if ($trimmed -eq '') { $this.Statistics.EmptyLines++ }
            elseif ($trimmed.StartsWith('#')) { $this.Statistics.CommentLines++ }
            else { $this.Statistics.CodeLines++ }
        }

        # Parse the content
        $content = $this.Lines -join "`n"
        try {
            $this.Tokens = $null
            $this.ParseErrors = $null
            $this.Ast = [Parser]::ParseInput($content, [ref]$this.Tokens, [ref]$this.ParseErrors)
        }
        catch {
            $this.AddIssue('Error', 1, "Critical parse failure: $($_.Exception.Message)", 'CriticalParseError', $content.Substring(0, [Math]::Min(100, $content.Length)))
            return $this.Issues.ToArray()
        }

        # Update statistics with AST info
        if ($this.Ast) {
            $functions = $this.Ast.FindAll({ param($n) $n -is [FunctionDefinitionAst] }, $true)
            $this.Statistics.Functions = $functions.Count
            $params = $this.Ast.FindAll({ param($n) $n -is [ParameterAst] }, $true)
            $this.Statistics.Parameters = $params.Count
        }

        # Create context for rules
        $context = [LintContext]::new($this.FilePath, $this.Lines, $this.Ast, $this.Tokens, $this.ParseErrors, $this.Settings)

        # Determine which rules to run
        $rulesToRun = if ($IncludeRules.Count -gt 0) {
            $IncludeRules | Where-Object { $this.RuleRegistry.ContainsKey($_) }
        }
        else {
            $this.RuleRegistry.Keys | Where-Object { $ExcludeRules -notcontains $_ }
        }

        # Run rules
        foreach ($ruleId in $rulesToRun) {
            try {
                $ruleScript = $this.RuleRegistry[$ruleId]
                $addIssue = {
                    param($severity, $line, $message, $rule = $ruleId, $code = '')
                    $this.AddIssue($severity, $line, $message, $rule, $code)
                }.GetNewClosure()
                & $ruleScript $context $addIssue
            }
            catch {
                $this.AddIssue('Error', 1, "Rule '$ruleId' failed: $($_.Exception.Message)", 'RuleExecutionError', $ruleId)
            }
        }

        return $this.Issues.ToArray()
    }

    [void] AddIssue([string] $Severity, [int] $Line, [string] $Message, [string] $Rule, [string] $Code = '') {
        $obj = [pscustomobject]@{
            Severity = $Severity
            Line = [math]::Max(1, $Line)
            Message = $Message
            Rule = $Rule
            Code = $Code
            File = $this.FilePath
        }
        $this.Issues.Add($obj)
    }

    [void] PrintReport() {
        Write-Host "`n=== PSLinter Report ===" -ForegroundColor Green
        Write-Host "File: $($this.FilePath)" -ForegroundColor Cyan
        Write-Host ""

        if ($this.ParseErrors -and $this.ParseErrors.Count -gt 0) {
            Write-Host "Parse Errors: $($this.ParseErrors.Count)" -ForegroundColor Red
            foreach ($e in $this.ParseErrors) {
                Write-Host " Line $($e.Extent.StartLineNumber): $($e.Message)" -ForegroundColor Red
                Write-Host " Code: $($e.Extent.Text)" -ForegroundColor Gray
            }
            Write-Host ""
        }

        if ($this.Tokens -and $this.Tokens.Count -gt 0) {
            Write-Host "Token Analysis:" -ForegroundColor Yellow
            Write-Host " Total Tokens: $($this.Tokens.Count)"
            $top = $this.Tokens.Kind | Group-Object | Sort-Object Count -Descending | Select-Object -First 5
            if ($top) {
                $desc = $top | ForEach-Object { "$($_.Name): $($_.Count)" } -join ', '
                Write-Host " Top Kinds: $desc"
            }
            Write-Host ""
        }

        Write-Host "Statistics:" -ForegroundColor Yellow
        Write-Host " Total Lines: $($this.Statistics.TotalLines)"
        Write-Host " Code Lines: $($this.Statistics.CodeLines)"
        Write-Host " Comment Lines: $($this.Statistics.CommentLines)"
        Write-Host " Empty Lines: $($this.Statistics.EmptyLines)"
        Write-Host " Functions: $($this.Statistics.Functions)"
        Write-Host " Parameters: $($this.Statistics.Parameters)"
        Write-Host ""

        $errs = @($this.Issues | Where-Object Severity -eq 'Error')
        $warn = @($this.Issues | Where-Object Severity -eq 'Warning')
        $info = @($this.Issues | Where-Object Severity -eq 'Info')

        Write-Host "Issues:" -ForegroundColor Yellow
        Write-Host " Errors: $($errs.Count)" -ForegroundColor Red
        Write-Host " Warnings: $($warn.Count)" -ForegroundColor Yellow
        Write-Host " Info: $($info.Count)" -ForegroundColor Blue
        Write-Host ""

        $bySeverity = $this.Issues | Group-Object Severity | Sort-Object { switch ($_.Name) { 'Error' { 1 }; 'Warning' { 2 }; 'Info' { 3 }; default { 4 } } }
        foreach ($g in $bySeverity) {
            $color = switch ($g.Name) { 'Error' { 'Red' } 'Warning' { 'Yellow' } 'Info' { 'Blue' } default { 'White' } }
            Write-Host "$($g.Name) Issues:" -ForegroundColor $color
            foreach ($i in ($g.Group | Sort-Object Line, Rule)) {
                Write-Host " Line $($i.Line): $($i.Message)" -ForegroundColor $color
                if ($i.Code) { Write-Host " Code: $($i.Code)" -ForegroundColor Gray }
                Write-Host " Rule: $($i.Rule)" -ForegroundColor DarkGray
            }
            Write-Host ""
        }

        if (-not $this.Issues.Count) {
            Write-Host "No issues found! ✓" -ForegroundColor Green
        }
        Write-Host "=== End Report ===`n" -ForegroundColor Green
    }

    [pscustomobject] GetStatistics() { return [pscustomobject]$this.Statistics }
    [Token[]] GetTokens() { return $this.Tokens }
    [ParseError[]] GetParseErrors() { return $this.ParseErrors }
    [Ast] GetScriptAst() { return $this.Ast }
    [hashtable] GetSettings() { return $this.Settings }
    [void] SetSetting([string] $Name, [object] $Value) { $this.Settings[$Name] = $Value }
}

function Invoke-PSLinter {
    [CmdletBinding()]
    param(
        [Parameter(ParameterSetName = 'File', Mandatory, ValueFromPipeline)][string] $FilePath,
        [Parameter(ParameterSetName = 'Content', Mandatory)][string] $Content,
        [ValidateSet('Error', 'Warning', 'Info')][string] $MinimumSeverity = 'Info',
        [string[]] $IncludeRules = @(),
        [string[]] $ExcludeRules = @(),
        [switch] $ShowReport,
        [switch] $ShowTokens,
        [switch] $ShowAst
    )

    $l = [PSLinter]::new()
    $issues = if ($PSCmdlet.ParameterSetName -eq 'File') {
        $l.LintFile($FilePath, $IncludeRules, $ExcludeRules)
    }
    else {
        $l.LintContent($Content, $IncludeRules, $ExcludeRules)
    }

    $order = @{ 'Error' = 1; 'Warning' = 2; 'Info' = 3 }
    $threshold = $order[$MinimumSeverity]
    $filtered = $issues | Where-Object { $order[$_.Severity] -le $threshold }

    if ($ShowReport) { $l.PrintReport() }

    if ($ShowTokens) {
        $tokens = $l.GetTokens()
        if ($tokens -and $tokens.Count -gt 0) {
            Write-Host "`n=== Token Sample ===" -ForegroundColor Green
            $max = [Math]::Min(50, $tokens.Count)
            for ($i = 0; $i -lt $max; $i++) {
                $t = $tokens[$i]
                Write-Host "[$($t.Kind)] '$($t.Text)' at Line $($t.Extent.StartLineNumber)" -ForegroundColor Cyan
            }
            if ($tokens.Count -gt $max) {
                Write-Host "... and $($tokens.Count - $max) more tokens" -ForegroundColor Gray
            }
        }
    }

    if ($ShowAst) {
        $ast = $l.GetScriptAst()
        Write-Host "`n=== AST ===" -ForegroundColor Green
        if ($null -ne $ast) {
            $stmtCount = $ast.EndBlock?.Statements?.Count
            Write-Host "AST Type: $($ast.GetType().Name)" -ForegroundColor Cyan
            if ($stmtCount -ne $null) { Write-Host "Body: $stmtCount statements" -ForegroundColor Cyan }
        }
        else {
            Write-Host "No AST (parse failure)" -ForegroundColor Red
        }
    }

    [pscustomobject]@{
        Issues = $filtered
        ParseErrors = $l.GetParseErrors()
        Statistics = $l.GetStatistics()
        Tokens = $l.GetTokens()
        ScriptAst = $l.GetScriptAst()
    }
}

function Test-PowerShellSyntax {
    [CmdletBinding()]
    param(
        [Parameter(ValueFromPipeline, Mandatory)][string] $ScriptContent
    )

    try {
        $tokens = $null
        $errors = $null
        $ast = [Parser]::ParseInput($ScriptContent, [ref]$tokens, [ref]$errors)
        [pscustomobject]@{
            IsValid = (-not $errors -or $errors.Count -eq 0)
            ParseErrors = $errors
            Tokens = $tokens
            Ast = $ast
        }
    }
    catch {
        [pscustomobject]@{
            IsValid = $false
            ParseErrors = @([pscustomobject]@{ Message = $_.Exception.Message; Line = 0 })
            Tokens = @()
            Ast = $null
        }
    }
}

function Test-LinterConfig {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][hashtable] $Config,
        [hashtable] $Schema = $script:PSLinterSchema
    )

    $errors = [List[pscustomobject]]::new()

    function Add-Err($msg, $path) {
        $errors.Add([pscustomobject]@{ Message = $msg; Path = $path })
    }

    # Check schema version
    if (-not $Config.ContainsKey('SchemaVersion')) {
        Add-Err "Missing required field 'SchemaVersion'." '$.SchemaVersion'
    }
    elseif ($Config.SchemaVersion -ne $Schema.SchemaVersion) {
        Add-Err "Schema version mismatch. Expected '$($Schema.SchemaVersion)', got '$($Config.SchemaVersion)'." '$.SchemaVersion'
    }

    # Validate settings
    $settingsDesc = $Schema.Settings
    $settings = $Config.Settings ?? @{}
    foreach ($settingName in $settingsDesc.Keys) {
        $rule = $settingsDesc[$settingName]
        $val = if ($settings.ContainsKey($settingName)) { $settings[$settingName] } else { $rule.Default }
        if ($null -eq $val) {
            Add-Err "Missing setting '$settingName'." "$.Settings.$settingName"
            continue
        }

        # Type validation
        switch ($rule.Type) {
            'Int' { if (-not ($val -is [int])) { Add-Err "Setting '$settingName' must be Int." "$.Settings.$settingName" } }
            'Int64' { if (-not ($val -is [int64] -or $val -is [int])) { Add-Err "Setting '$settingName' must be Int64." "$.Settings.$settingName" } }
            'Bool' { if (-not ($val -is [bool])) { Add-Err "Setting '$settingName' must be Bool." "$.Settings.$settingName" } }
            'Enum' { if ($rule.Values -notcontains $val) { Add-Err "Setting '$settingName' must be one of: $($rule.Values -join ', ')." "$.Settings.$settingName" } }
        }

        # Range validation
        if ($rule.ContainsKey('Enum') -and $rule.Enum -notcontains $val) {
            Add-Err "Setting '$settingName' must be one of: $($rule.Enum -join ', ')." "$.Settings.$settingName"
        }
        if ($rule.ContainsKey('Min') -and $val -lt $rule.Min) {
            Add-Err "Setting '$settingName' below minimum $($rule.Min)." "$.Settings.$settingName"
        }
        if ($rule.ContainsKey('Max') -and $val -gt $rule.Max) {
            Add-Err "Setting '$settingName' above maximum $($rule.Max)." "$.Settings.$settingName"
        }
    }

    # Check rule conflicts
    $inc = @($Config.includeRules ?? @())
    $exc = @($Config.excludeRules ?? @())
    $overlap = $inc | Where-Object { $exc -contains $_ }
    if ($overlap.Count -gt 0) {
        Add-Err "includeRules and excludeRules overlap: $($overlap -join ', ')." '$'
    }

    # Validate individual rule configs
    if ($Config.ContainsKey('rules')) {
        foreach ($ruleName in $Config.rules.Keys) {
            $ruleConfig = $Config.rules[$ruleName]
            if ($ruleConfig.ContainsKey('severity') -and $ruleConfig.severity -notin @('Error', 'Warning', 'Info')) {
                Add-Err "Rule '$ruleName' severity must be Error | Warning | Info." "$.rules.$ruleName.severity"
            }
            if ($ruleConfig.ContainsKey('enabled') -and $ruleConfig.enabled -isnot [bool]) {
                Add-Err "Rule '$ruleName' enabled must be boolean." "$.rules.$ruleName.enabled"
            }
        }
    }

    return [pscustomobject]@{
        IsValid = ($errors.Count -eq 0)
        Errors = $errors.ToArray()
        Config = $Config
    }
}

Set-Alias -Name pslint -Value Invoke-PSLinter
Set-Alias -Name pslintcheck -Value Test-PowerShellSyntax
Set-Alias -Name pslintcfg -Value Test-LinterConfig
