# Smart profile loader that detects context
param()

# Detect if we're in VS Code and which terminal type
# $isVSCode = $env:VSCODE_PID -or $env:TERM_PROGRAM -eq "vscode"
# $terminalUser = $env:TERMINAL_USER
# $terminalProfile =

# Exit early if portable environment already loaded
if ($global:PORTABLE_ENV_LOADED)
{
    Write-Verbose "Portable environment already loaded, skipping profile"
    return
}

# 1) Load SmartProfileUtils first for every profile
$profilesPath = "$env:PORTABLE_ROOT/.denv/Profiles"
. "$profilesPath/SmartProfileStartupUtils.ps1"
. "$profilesPath/MemoryOrchestrationEngine.ps1"

Set-DotNetVersion -Version "9.0.304"

# Copilot profile loads PrimaryEnv indepently during load

$primaryProfile = "PrimaryEnvProfileNew"
$userPrimary = "VSCode.User.Primary"
$userPrimary = "VSCode.User.MiniPC"

switch ($true)
{
    ($env:TERMINAL_PROFILE -eq "VSCode.Copilot")
    {
        Write-Host "🤖 Loading Copilot Profile..." -ForegroundColor Cyan
        . "$profilesPath/VSCode.Copilot.Profile.ps1" -Quiet:$false
        break
    }
    ($env:TERMINAL_PROFILE -eq "$userPrimary") {
        Write-Host "🤖 Loading $userPrimary Profile ..." -ForegroundColor Cyan
        . "$profilesPath/$primaryProfile" -Quiet:$false
        break
    }
    ($env:TERMINAL_PROFILE -eq "$userPrimary") {
        Write-Host "🤖 Loading $userPrimary Profile..." -ForegroundColor Cyan
        . "$profilesPath/$primaryProfileMiniPC" -Quiet:$false
        break
    }
    default
    {
        Write-Host "🔧 TERMINAL_PROFILE not recognize. Defaulting Primary Profile..." -ForegroundColor Green
        . "$profilesPath/$primaryProfile" -Quiet:$false
        break
    }
}
