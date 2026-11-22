<#
.SYNOPSIS
    Build the Windows installer for Audio from Video, with FFmpeg bundled.

.DESCRIPTION
    Creates a build virtual environment, downloads and verifies the pinned FFmpeg
    build, freezes the app with PyInstaller, writes license notices, and compiles
    the Inno Setup installer. Everything goes under build\windows\ in the repo.

    Requirements on the build machine:
      * Python 3.11 or newer, 64-bit (3.13 recommended)
      * Inno Setup 6 (winget install JRSoftware.InnoSetup), unless -SkipInstaller
      * Git (optional; used to stamp the commit and archive the source)

.PARAMETER Python
    Python launcher used to create the build environment, for example "python"
    or "py -3.13". Defaults to "python".

.PARAMETER Constraints
    Optional pip constraints file that pins exact dependency versions, for a
    reproducible build (for example, the pip-freeze.txt saved by an earlier build).

.PARAMETER Clean
    Delete build\windows\ (except the download cache) before building.

.PARAMETER SkipInstaller
    Stop after creating the app folder; do not run Inno Setup.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
#>
[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$Constraints,
    [switch]$Clean,
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Exe $($Arguments -join ' ')"
    }
}

function Find-Iscc {
    $command = Get-Command "iscc.exe" -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $candidates = @(
        (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"),
        (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe")
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) { return $candidate }
    }
    throw "Inno Setup 6 was not found. Install it with: winget install JRSoftware.InnoSetup"
}

$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Build = Join-Path $Repo "build\windows"
$Venv = Join-Path $Build "venv"
$Py = Join-Path $Venv "Scripts\python.exe"
$Cache = Join-Path $Build "cache"
$FFmpegStage = Join-Path $Build "ffmpeg"
$Dist = Join-Path $Build "dist"
$App = Join-Path $Dist "Audio from Video"
$Release = Join-Path $Build "release"
$Icon = Join-Path $Build "app_icon.ico"

Set-Location $Repo

if ($Clean -and (Test-Path $Build)) {
    Write-Host "==> Cleaning $Build (keeping the download cache)"
    Get-ChildItem $Build | Where-Object { $_.Name -ne "cache" } | Remove-Item -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $Build, $Release | Out-Null

# --- 1. Build environment ---------------------------------------------------
if (-not (Test-Path $Py)) {
    Write-Host "==> Creating build environment with '$Python'"
    $launcher = $Python.Split(" ")
    $launcherArgs = @()
    if ($launcher.Length -gt 1) { $launcherArgs = $launcher[1..($launcher.Length - 1)] }
    Invoke-Checked $launcher[0] ($launcherArgs + @("-m", "venv", $Venv))
}
Invoke-Checked $Py @("-c", "import sys, struct; assert sys.version_info >= (3, 11), 'Python 3.11+ is required'; assert struct.calcsize('P') == 8, '64-bit Python is required'; print('Python', sys.version)")

Write-Host "==> Installing the app and PyInstaller into the build environment"
Invoke-Checked $Py @("-m", "pip", "install", "--quiet", "--upgrade", "pip")
$pipArgs = @("-m", "pip", "install", "--quiet", "$($Repo)[gui]", "pyinstaller==6.22.3")
if ($Constraints) { $pipArgs += @("--constraint", (Resolve-Path $Constraints).Path) }
Invoke-Checked $Py $pipArgs
# Always reinstall the app itself so the build picks up local source changes.
Invoke-Checked $Py @("-m", "pip", "install", "--quiet", "--no-deps", "--force-reinstall", $Repo)

$Version = (& $Py -c "import audio_from_video; print(audio_from_video.__version__)").Trim()
$Commit = "unknown"
$Dirty = $false
if (Get-Command "git" -ErrorAction SilentlyContinue) {
    $Commit = (git rev-parse HEAD).Trim()
    $Dirty = [bool](git status --porcelain --untracked-files=no)
}
Write-Host "==> Building Audio from Video $Version (commit $Commit$(if ($Dirty) { ', with uncommitted changes' }))"

# --- 2. FFmpeg ---------------------------------------------------------------
Write-Host "==> Fetching and verifying FFmpeg"
Invoke-Checked $Py @(
    "packaging\fetch_ffmpeg.py", "--target", "windows-x64",
    "--dest", $FFmpegStage, "--cache-dir", $Cache, "--release-dest", $Release
)

# --- 3. Freeze the app -------------------------------------------------------
Write-Host "==> Rendering the app icon"
Invoke-Checked $Py @("packaging\make_icon.py", "--out", $Icon)

Write-Host "==> Running PyInstaller"
$env:AFV_ICON = $Icon
Invoke-Checked $Py @(
    "-m", "PyInstaller", "--noconfirm", "--clean",
    "--distpath", $Dist, "--workpath", (Join-Path $Build "work"),
    "packaging\windows\afv.spec"
)

Write-Host "==> Adding FFmpeg and license notices to the app folder"
Copy-Item -Recurse -Force $FFmpegStage (Join-Path $App "ffmpeg")
Invoke-Checked $Py @(
    "packaging\gen_notices.py", "--out", (Join-Path $App "licenses"),
    "--ffmpeg-dir", $FFmpegStage, "--commit", $Commit, "--sdist-dest", $Release
)
# Record the exact package versions that went into this build (usable with -Constraints).
& $Py -m pip freeze --exclude-editable |
    Where-Object { $_ -notmatch "^audio[-_]from[-_]video" } |
    Set-Content -Path (Join-Path $Release "pip-freeze.txt") -Encoding ascii

Write-Host "==> Quick check of the frozen command-line tool"
Invoke-Checked (Join-Path $App "afv.exe") @("--version")

# --- 4. Source archive -------------------------------------------------------
if ($Commit -ne "unknown") {
    if ($Dirty) {
        Write-Warning "Uncommitted changes are not included in the source archive. Commit first for a release build."
    }
    Invoke-Checked "git" @("archive", "--format=zip", "--output", (Join-Path $Release "audio-from-video-src-$Commit.zip"), "HEAD")
}

# --- 5. Installer ------------------------------------------------------------
if ($SkipInstaller) {
    Write-Host "==> Skipping the installer. App folder: $App"
} else {
    $Iscc = Find-Iscc
    Write-Host "==> Compiling the installer with $Iscc"
    Invoke-Checked $Iscc @(
        "/Q",
        "/DAppVersion=$Version",
        "/DSourceDir=$App",
        "/DOutputDir=$Release",
        "/DIconFile=$Icon",
        (Join-Path $Repo "packaging\windows\installer.iss")
    )
}

# --- 6. Checksums ------------------------------------------------------------
$sums = Get-ChildItem $Release -File | Where-Object { $_.Name -ne "SHA256SUMS.txt" } | Sort-Object Name | ForEach-Object {
    "$((Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower())  $($_.Name)"
}
Set-Content -Path (Join-Path $Release "SHA256SUMS.txt") -Value $sums -Encoding ascii

Write-Host ""
Write-Host "Done. Release files are in $Release"
Get-ChildItem $Release -File | ForEach-Object { Write-Host ("  {0,-70} {1,8:N1} MB" -f $_.Name, ($_.Length / 1MB)) }
