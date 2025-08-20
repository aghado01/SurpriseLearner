#Requires -Version 7.0


<#
.SYNOPSIS
Primary portable environment profile with memory orchestration integration.

.DESCRIPTION
Establishes the complete portable development environment with hierarchical memory
system integration, proper Python/pyenv configuration, and structured path management.
Designed for both PowerShell extension and terminal usage.
#>

Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
$profilesPath = "$env:PORTABLE_ROOT/.denv/Profiles"

# Check if profile is already loaded
if (-not $env:USER_PROFILE_LOADED)
{
    . "$profilesPath/SmartProfileStartupUtils.ps1"
    Write-Telemetry "USER_PROFILE_LOADED is false, Setting PORTABLE_ROOT to $env:PORTABLE_ROOT and loading profile" -Level Host -Color Green
}

# Load essential helper functions for profile loading used below

$profilesPath = "$env:PORTABLE_ROOT/.denv/Profiles"

$SessionId = (New-Guid).ToString().Substring(0, 8)

$loadingContext = if ($env:TERMINAL_PROFILE) { $env:TERMINAL_PROFILE; $MemoryContext = "Development" } else { $env:TERMINAL_PROFILE = "Nominal"; $MemoryContext = "Nominal" }


# begin region constructing portable paths based on hierarchical structure
# Note: $env:PORTABLE_ROOT and $env:PSHOME are now set as a user-level environmental variable

# $script:PortableRoot = "$env:PORTABLE_ROOT"
$script:PortableRoot = "$env:PORTABLE_ROOT"
$script:AppsRoot = "$PortableRoot/Apps"
$script:Corpii = "$PortableRoot/Corpii"
$script:ProjectsRoot = "$PortableRoot/Projects"
$script:PythonLibUser = "$PortableRoot/PythonLib"
$script:PsLibraryPath = "$PortableRoot/PowerShellLib"
$script:denvPath = "$PortableRoot/.denv"

# Denv
$script:denvHelpersPath = "$denvPath/helpers"
$script:denvFunctionsPath = "$denvPath/functions"
$script:CopilotHomePath = "$denvPath/Copilot"
$script:CopilotHomeCorePath = "$CopilotHomePath/Copilot/Core"
$script:denvProfilesPath = "$denvPath/Profiles"
$script:UserProfile = "$denvProfilesPath/PrimaryEnvProfileNew.ps1"
$script:CopilotProfile = "$denvProfilesPath/VSCode.Copilot.Path.ps1"


#region PsLibrary-Secific Paths
$script:PsProfilesPath = "$PsLibraryPath/Profiles"
$script:PsFunctionsPath = "$PsLibraryPath/Functions"
$script:PsScriptsPath = "$PsLibraryPath/Scripts"
$script:PsTestsPath = "$PsLibraryPath/Tests"
$script:PsModulesPath = "$PsModulesPath/Modules"
# endregion

#region Project-Specific Paths
$script:LocalLlm = "$ProjectsRoot/LocalLLM"
$script:RepoSnapShot = "$ProjectsRoot/RepoSnapShot"
$script:ParForPowerShell = "$ProjectsRoot/ParForPowerShell"
$script:CopilotAugmentation = "$ProjectsRoot/CopilotAugmentation"
$script:SurpriseLearner = "$ProjectsRoot/SurpriseLearner"
$script:MemoryProsthetic = "$ProjectsRoot/MemoryProsthetic"
$script:ChatHistoryExport = "$ProjectsRoot/ChatHistoryExport"
$script:CorpusCollections = "$ProjectsRoot/CorpusCollections"
$script:CorpusCollectionsScripts = "$CorpusCollections/Scripts"
$script:CorpusCollectionsPowerShell = "$CorpusCollections/Domain-PowerShell"
$script:CorpusCollectionsPytorch = "$CorpusCollections/Domain-Pytorch"
#endregion


#region Application Paths
$script:AppsExtra = "$AppsRoot/AppsExtra"
$script:VsCode = "$AppsRoot/VSCode"
$script:CliLib = "$AppsRoot/clilib"
$script:GitHome = "$AppsRoot/Git"
$script:GoRoot = "$AppsRoot/go/Go"
$script:DotNetRoot = "$AppsRoot/dotnet/9.0.304"
$script:NpmRoot = "$AppsRoot/npm"
$script:PsRoot = "$AppsRoot/PowerShell/7"
#endregion

#region Portable Pyenv/Python/Pip Configuration
$script:PyVersion = "3.12.6"
$script:PyenvRoot = "$ProjectsRoot/.pyenv"
$script:PyenvWinRoot = "$PyenvRoot/pyenv-win-$PyVersion"
$script:PyenvBin = "$PyenvWinRoot/bin"
$script:PyenvShims = "$PyenvWinRoot/shims"
$script:PythonHome = "$PyenvWinRoot/versions/$PyVersion"
$script:PythonExecutable = "$PythonHome/python.exe"
$script:PythonBin = "$PythonHome/bin"
$script:PythonScripts = "$PythonHome/Scripts"
$script:PythonLibs = "$PythonHome/libs"
$script:PythonLib = "$PythonHome/Lib"
$script:PythonPipCacheDir = "$PythonLib/pip_cache"
$script:PythonSitePackages = "$PythonLib/site-packages"

#endregion


#region PATH Management
# Define paths in priority order (most important first)
$script:PortablePaths = @(
    # Python - CRITICAL: pyenv shims must come first for proper Python interception

    # Root
    $PortableRoot,
    $AppsRoot,

    # Python
    $PyenvShims,
    $PyenvBin,
    $PythonHome,
    $PythonLib,
    $PythonPipCacheDir,
    $PythonSitePackages,
    $PythonScripts,  # This is where pip.exe lives!

    # denv
    $denvPath,
    $denvHelpersPath,
    $denvFunctionsPath,
    $denvProfilesPath,
    $CopilotHomePath,
    $CopilotHomeCorePath,
    $UserProfile,
    $CopilotProfile,

    # PowerShell Library
    $PsLibraryPath,
    $PsProfilesPath,
    $PsFunctionsPath,
    $PsScriptsPath,
    $PsModulesPath,

    # Projects
    $ProjectsRoot,
    $CorpusCollections,
    $CorpusCollectionsScripts,
    $CopilotAugmentation,
    $LocalLlm,
    $MemoryProsthetic,
    $RepoSnapShot,
    $SurpriseLearner,

    # Applications
    $CliLib,
    "$CliLib/bin",
    "$CliLib/lib",
    $VSCode,
    $GoRoot,
    $NpmRoot,
    $DotNetRoot,
    $PsRoot,

    # Git
    $GitHome,
    "$GitHome/bin",
    "$GitHome/cmd"
)
Write-Telemetry "Prepending pdenv paths at process level..." -Level Host -Color Cyan
Update-EnvPathSmart -Paths $PortablePaths -Strategy "Prepend" -Scope Process
Write-Telemetry "Sorting pdenv PATH..." -Level Host -Color Cyan
Set-EnvPathSorted -RootPath $PortableRoot -PathVariable "PATH" -RemoveEmpty -RemoveDuplicates

# endregion



#Begin block of delineating portable paths

Write-Telemetry "Sending Portable env path and other variables..." -Level Host -Color Cyan
$script:PortableEnvVars = @{
    HOME                     = $PortableRoot
    PORTABLE_ROOT            = $PortableRoot
    PORTABLE_HOME            = $PortableRoot
    PROJECTS_ROOT            = $ProjectsRoot
    PRIMARY_ENV_PROFILE      = $PSCommandPath
    APPS_ROOT                = $AppsRoot
    PSHOME                   = $PsRoot
    PSModulePath             = "$PsLibraryPath/Modules"
    PsLibraryPath            = $PsLibraryPath
    PsProfilesPath           = $PsProfilesPath
    PsFunctionsPath          = $PsFunctionsPath
    PsScriptsPath            = $PsScriptsPath
    PsCopilotPath            = $denvCopilotHomePath
    denvPath                 = $denvPath
    denvHelpers              = $denvHelpers
    denvFunctionsPath        = $denvFunctionsPath
    denvProfilesPath         = $denvProfilesPath
    CopilotHomePath          = $CopilotHomePath
    CopilotHomeCorePath      = $CopilotHomeCorePath
    UserProfile              = $UserProfile
    CopilotProfile           = $CopilotProfile
    ProjectsRoot             = $ProjectsRoot
    ParForPowerShell         = $ParForPowerShell
    ChatHistoryExport        = $ChatHistoryExport
    CopilotAugmentation      = $CopilotAugmentation
    LocalLlm                 = $LocalLlm
    MemoryProsthetic         = $MemoryProsthetic
    RepoSnapShot             = $RepoSnapShot
    SurpriseLearner          = $SurpriseLearner
    CorpusCollections        = $CorpusCollections
    CorpusCollectionsScripts = $CorpusCollectionsScripts
    LANG                     = "en_US.UTF-8"
    PRIMARY_ASSISTANT        = "Copilot"
    ACTIVE_PERSONALITY       = "Helpful" # not beig used
    LOADING_CONTEXT          = $loadingContext
}

Set-EnvVarsScoped -Vars $PortableEnvVars -Scope Process

# Begin block of defining environmental variables via Simple approach
Write-Telemetry "Sending Portable env pyenv/python path env variables..." -Level Host -Color Cyan
$script:PyenvPythonPaths = @{
    PYTHON_VERSION   = $PyVersion
    PYENV            = $PyenvWinRoot
    PYENV_ROOT       = $PyenvWinRoot
    PYENV_HOME       = $PyenvWinRoot
    # PYTHONPATH       = $PythonSitePackages
    PYTHON_HOME      = $PythonHome
    PIP_CACHE_DIR    = $PipCacheDir
    PYENV_VERSION    = $PyVersion
    PYTHONEXECUTABLE = "$PythonHome/python.exe"
    PYTHONNOUSERSITE = "1"
    PIP_USER         = "0"
}
Set-EnvVarsScoped -Vars $PyenvPythonPaths -Scope Process


Write-Telemetry "Sending Portable env git/gh path and config env variabless..." -Level Host -Color Cyan
# Git Environment Variables
$script:GitEnvVars = @{
    GIT_HOME               = $GitHome
    GIT_EXEC_PATH          = "$GitHome/mingw64/libexec/git-core"
    GIT_CMD_PATH           = "$GitHome/cmd/git.exe"
    XDG_CONFIG_HOME        = "$PortableRoot/.config"
    GIT_CONFIG_SYSTEM      = "$PortableRoot/.config/.gitconfig"
    GITHUB_USERNAME        = "aghado01"
    GITHUB_EMAIL           = "azriel.ghadoooshahy@gmail.com"
    GCM_AUTHORITY          = "https://github.com"
    GCM_GITHUB_AUTHMODES   = "personal_access_token,oauth"
    GIT_TERMINAL_PROMPT    = "0"
    GCM_CREDENTIAL_STORE   = "wincred"
    GIT_CREDENTIAL_MANAGER = "wincred"

}
Set-EnvVarsScoped -Vars $GitEnvVars -Scope Process

# Go Environment Variables
Write-Telemetry "Sending pdenv go path env variables..." -Level Host -Color Cyan
$script:GoPaths = @{
    GOROOT = $GoRoot
    GOPATH = "$GoRoot/workspace"
    GOBIN  = "$CliLib/bin"
}
Set-EnvVarsScoped -Vars $GoPaths -Scope Process

# Define portable env var hash tables by category
Write-Telemetry "Sending pdenv dotnet env variables..." -Level Host -Color Cyan

$script:DotNetVars = @{
    DOTNET_ROOT              = $DotNetRoot
    DOTNET_ROOT_USER         = $DotNetRoot
    DOTNET_ROOT_USER_RELEASE = $DotNetRoot
    DOTNET_MULTILEVEL_LOOKUP = "0"
}
Set-EnvVarsScoped -Vars $DotNetVars -Scope Process

# NPM Environment Variables

Write-Telemetry "Sending NPM path env variables..." -Level Host -Color Cyan
$script:NpmEnvVars = @{
    NPM_CONFIG_PREFIX       = $NpmRoot
    NPM_CONFIG_CACHE        = "$NpmRoot/cache"
    NPM_CONFIG_USERCONFIG   = "$NpmRoot/.npmrc"
    NPM_CONFIG_GLOBALCONFIG = "$NpmRoot/.npmrc"
}
Set-EnvVarsScoped -Vars $NpmEnvVars -Scope Process

Write-Verbose "Environment variables configured successfully"

# Set default Python version via pyenv
# pyenv global 3.12.6

# region set handy shortcut functions and aliases
Write-Telemetry "Sending convient function wrappers..." -Level Host -Color Cyan

function get-env { Get-ChildItem env: | Format-Table Name, Value -AutoSize }

function genv { param($name) (Get-Item "Env:$name").Value }

function cdg { param($name) cd $(genv $name) }

Import-Module "C:\Users\azriy\PortDenv\Projects\RepoSnapShot\RepoSnapshot.v2.2.psm1"

function ftypes
{
    Get-ChildItem -Recurse -File "." |
    Select-Object -ExpandProperty Extension |
    Where-Object { $_ } |
    Sort-Object -Unique
}

# Assign it to the prompt function
$function:prompt = { Set-CustomUserPrompt }

$global:USER_PROFILE_LOADED = $true
Write-Telemetry "PortableEnvProfile loaded successfully..." -Level Host -Color Cyan
