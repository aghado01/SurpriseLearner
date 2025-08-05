# Repository to JSON Exporter - Environment Agnostic
# Usage: .\Export-JsonRepo.ps1 -RepositoryPath "." -OutputPath "backup.json"
# If no OutputPath specified, auto-generates: repo_export_{reponame}_{timestamp}.json


# Repository to JSON Exporter - Environment Agnostic
param(
    [string]$RepositoryPath = ".",
    [string]$OutputPath = "",  # Will auto-generate if not provided
    [switch]$IncludeAllFiles = $false,
    [int]$MaxFileSizeKB = 10000,  # Skip files larger than
    [switch]$Verbose = $true,   # Show excluded files
    [switch]$Debug = $false      # Show detailed filtering info
)

# Define file extensions to include (when not using -IncludeAllFiles)
$IncludedExtensions = @(
    '.py', '.md', '.yaml', '.yml', '.ipynb', '.cfg', '.env',
    '.xml', '.html', '.css', '.js', '.ts', '.sql', '.sh',
    '.bat', '.ps1', '.ini', '.conf', '.log', '.csv', '.toml', '.lock',
    '.gitignore', '.dockerignore', '.editorconfig', '.json'
    # '.txt',
)

# Directories to exclude
#
$ExcludedDirs = @('_snapshot', 'Modules', 'site-packages', '_backup', '_dev', '_cop', '_depr', '_dev', '_gem', '_prp', 'txt', '.git', '.vscode', '__pycache__', 'node_modules', '.pytest_cache', '_mypy_cache', '.mypy_cache', 'venv', 'env', '.env', 'dist', 'build', '.ruff_cache', '.black_cache')

function Test-IsBinaryFile {
    param([string]$FilePath)

    try {
        $bytes = [System.IO.File]::ReadAllBytes($FilePath)
        if ($bytes.Length -eq 0) { return $false }

        # Check first 1024 bytes for null characters (common in binary files)
        $checkLength = [Math]::Min(1024, $bytes.Length)
        for ($i = 0; $i -lt $checkLength; $i++) {
            if ($bytes[$i] -eq 0) { return $true }
        }
        return $false
    }
    catch {
        return $true  # If we can't read it, assume it's binary
    }
}

function Get-FileContent {
    param([string]$FilePath)

    try {
        if (Test-IsBinaryFile -FilePath $FilePath) {
            return "[BINARY FILE - Content not included]"
        }

        # Try UTF-8 first, then fall back to default encoding
        try {
            return [System.IO.File]::ReadAllText($FilePath, [System.Text.Encoding]::UTF8)
        }
        catch {
            return [System.IO.File]::ReadAllText($FilePath)
        }
    }
    catch {
        return "[ERROR: Could not read file - $($_.Exception.Message)]"
    }
}

# Resolve repository path to absolute path
try {
    $resolvedRepoPath = (Resolve-Path $RepositoryPath -ErrorAction Stop).Path
}
catch {
    Write-Host "Error: Cannot resolve repository path '$RepositoryPath'" -ForegroundColor Red
    exit 1
}

# Auto-generate output path if not provided
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $repoName = Split-Path $resolvedRepoPath -Leaf
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputPath = Join-Path $resolvedRepoPath "${repoName}_${timestamp}.json"
}
else {
    # If OutputPath is relative, make it relative to the repository
    if (-not [System.IO.Path]::IsPathRooted($OutputPath)) {
        $OutputPath = Join-Path $resolvedRepoPath $OutputPath
    }
}

# Ensure output directory exists
$outputDir = Split-Path $OutputPath -Parent
if (-not (Test-Path $outputDir)) {
    try {
        New-Item -ItemType Directory -Path $outputDir -Force -ErrorAction Stop | Out-Null
        Write-Host "Created output directory: $outputDir" -ForegroundColor Cyan
    }
    catch {
        Write-Host "Error: Cannot create output directory '$outputDir': $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
}

Write-Host "Scanning repository: $resolvedRepoPath" -ForegroundColor Green
Write-Host "Output file: $OutputPath" -ForegroundColor Green

# Initialize the export structure
$exportData = @{
    metadata = @{
        export_date        = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        repository_path    = $resolvedRepoPath
        repository_name    = Split-Path $resolvedRepoPath -Leaf
        total_files        = 0
        exported_files     = 0
        skipped_files      = 0
        script_version     = "1.1"
        powershell_version = $PSVersionTable.PSVersion.ToString()
    }
    files    = @()
}

# Get all files recursively with improved exclusion logic
$allFiles = Get-ChildItem -Path $resolvedRepoPath -Recurse -File -ErrorAction SilentlyContinue | Where-Object {
    # Exclude files in excluded directories - improved logic
    $relativePath = $_.FullName.Replace($resolvedRepoPath, "").TrimStart('\', '/').Replace('\', '/')
    $inExcludedDir = $false

    foreach ($excludedDir in $ExcludedDirs) {
        # Check multiple patterns to catch all variations
        $patterns = @(
            "$excludedDir/*",           # Direct child: mypy_cache/file.txt
            "*/$excludedDir/*",         # Nested: something/mypy_cache/file.txt
            "$excludedDir\*",           # Windows separator
            "*\$excludedDir\*",         # Windows nested
            "*/$excludedDir",           # Directory itself
            "*\$excludedDir"            # Directory itself (Windows)
        )

        foreach ($pattern in $patterns) {
            if ($relativePath -like $pattern) {
                if ($Debug) { Write-Host "EXCLUDED: $relativePath (matched pattern: $pattern for dir: $excludedDir)" -ForegroundColor Red }
                elseif ($Verbose) { Write-Host "Excluding: $relativePath" -ForegroundColor Yellow }
                $inExcludedDir = $true
                break
            }
        }
        if ($inExcludedDir) { break }
    }

    return -not $inExcludedDir
}

$exportData.metadata.total_files = $allFiles.Count
Write-Host "Found $($allFiles.Count) files to process" -ForegroundColor Cyan

if ($Debug) {
    Write-Host "`nExcluded directories: $($ExcludedDirs -join ', ')" -ForegroundColor Yellow
    $testFiles = Get-ChildItem -Path $resolvedRepoPath -Recurse -File -ErrorAction SilentlyContinue | Where-Object { $_.FullName -like "*mypy_cache*" } | Select-Object -First 3
    if ($testFiles) {
        Write-Host "`nFound mypy_cache files:" -ForegroundColor Magenta
        foreach ($tf in $testFiles) {
            $testRelPath = $tf.FullName.Replace($resolvedRepoPath, "").TrimStart('\', '/').Replace('\', '/')
            Write-Host "  $testRelPath" -ForegroundColor Magenta
        }
    }

    # Count excluded vs included files
    $allFilesDebug = Get-ChildItem -Path $resolvedRepoPath -Recurse -File -ErrorAction SilentlyContinue
    Write-Host "`nTotal files before filtering: $($allFilesDebug.Count)" -ForegroundColor Cyan
    Write-Host "Files after exclusion filtering: $($allFiles.Count)" -ForegroundColor Cyan
    Write-Host "Files filtered out: $($allFilesDebug.Count - $allFiles.Count)" -ForegroundColor Yellow
}

foreach ($file in $allFiles) {
    $relativePath = $file.FullName.Replace($resolvedRepoPath, "").TrimStart('\', '/').Replace('\', '/')
    $extension = $file.Extension.ToLower()
    $fileSizeKB = [Math]::Round($file.Length / 1KB, 2)

    # Skip large files
    if ($fileSizeKB -gt $MaxFileSizeKB) {
        Write-Host "Skipping large file: $relativePath ($fileSizeKB KB)" -ForegroundColor Yellow
        $exportData.metadata.skipped_files++
        continue
    }

    # Check if file should be included
    $shouldInclude = $IncludeAllFiles -or ($extension -in $IncludedExtensions) -or ($file.Name -in @('.gitignore', '.dockerignore', '.editorconfig', 'Dockerfile', 'Makefile', 'README'))

    if (-not $shouldInclude) {
        $exportData.metadata.skipped_files++
        continue
    }

    Write-Host "Processing: $relativePath" -ForegroundColor Gray

    # Get file content
    $content = Get-FileContent -FilePath $file.FullName

    # Create file object
    $fileObject = @{
        path            = $relativePath
        directory       = Split-Path $relativePath -Parent
        filename        = $file.Name
        extension       = $extension
        size_bytes      = $file.Length
        size_kb         = $fileSizeKB
        last_modified   = $file.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
        content         = $content
        content_preview = if ($content.Length -gt 200) { $content.Substring(0, 200) + "..." } else { $content }
        is_binary       = Test-IsBinaryFile -FilePath $file.FullName
    }

    $exportData.files += $fileObject
    $exportData.metadata.exported_files++
}

# Sort files by path for better organization
$exportData.files = $exportData.files | Sort-Object path

Write-Host "`nExport Summary:" -ForegroundColor Green
Write-Host "Total files found: $($exportData.metadata.total_files)" -ForegroundColor White
Write-Host "Files exported: $($exportData.metadata.exported_files)" -ForegroundColor Green
Write-Host "Files skipped: $($exportData.metadata.skipped_files)" -ForegroundColor Yellow

# Export to JSON with enhanced error handling
try {
    $jsonOutput = $exportData | ConvertTo-Json -Depth 10 -Compress:$false
    [System.IO.File]::WriteAllText($OutputPath, $jsonOutput, [System.Text.Encoding]::UTF8)
    Write-Host "`nRepository exported successfully to: $OutputPath" -ForegroundColor Green

    # Show file size
    $outputSize = [Math]::Round((Get-Item $OutputPath).Length / 1KB, 2)
    Write-Host "Output file size: $outputSize KB" -ForegroundColor Cyan

    # Return the output path for potential pipeline use
    return $OutputPath
}
catch {
    Write-Host "Error exporting to JSON: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Attempted output path: $OutputPath" -ForegroundColor Yellow
    exit 1
}
