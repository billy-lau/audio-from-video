<#
.SYNOPSIS
    Install, exercise, and uninstall a built Audio from Video installer.

.DESCRIPTION
    Checks that the installed app works with no FFmpeg on PATH, so it must use the
    copy bundled inside the app:
      1. Silent install into a temporary folder.
      2. "afv --version" and "afv doctor" (doctor must report the bundled FFmpeg).
      3. Offline extraction of a generated test video to MP3.
      4. The desktop app starts and stays running.
      5. Silent uninstall removes the program files.

    Uses a temporary config.toml, so your own settings are not touched. The
    desktop app window opens for a few seconds during step 4.

.PARAMETER Installer
    Path to AudioFromVideo-Setup-<version>-x64.exe. Defaults to the newest one in
    build\windows\release\.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File packaging\windows\smoke_test.ps1
#>
[CmdletBinding()]
param([string]$Installer)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Assert-True {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw "FAILED: $Message" }
    Write-Host "  ok: $Message"
}

$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $Installer) {
    $found = Get-ChildItem (Join-Path $Repo "build\windows\release") -Filter "AudioFromVideo-Setup-*.exe" |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $found) { throw "No installer found. Run packaging\build_windows.ps1 first, or pass -Installer." }
    $Installer = $found.FullName
}
Write-Host "==> Testing $Installer"

$Work = Join-Path ([System.IO.Path]::GetTempPath()) ("afv-smoke-" + [guid]::NewGuid().ToString("N").Substring(0, 8))
$InstallDir = Join-Path $Work "Audio from Video"
$InputDir = Join-Path $Work "input"
$OutputDir = Join-Path $Work "output"
New-Item -ItemType Directory -Force -Path $Work, $InputDir, $OutputDir | Out-Null

# Keep the test away from real settings, and make sure no system FFmpeg is visible.
$env:AFV_CONFIG_PATH = Join-Path $Work "config.toml"
$env:AFV_FFMPEG_PATH = ""
$env:AFV_FFPROBE_PATH = ""
$env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
$env:COLUMNS = "300"

try {
    Write-Host "==> 1. Silent install"
    $proc = Start-Process -FilePath $Installer -Wait -PassThru -ArgumentList @(
        "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/DIR=`"$InstallDir`""
    )
    Assert-True ($proc.ExitCode -eq 0) "installer exit code is 0 (was $($proc.ExitCode))"
    $Afv = Join-Path $InstallDir "afv.exe"
    $Gui = Join-Path $InstallDir "Audio from Video.exe"
    $FFmpeg = Join-Path $InstallDir "ffmpeg\bin\ffmpeg.exe"
    $FFprobe = Join-Path $InstallDir "ffmpeg\bin\ffprobe.exe"
    foreach ($file in @($Afv, $Gui, $FFmpeg, $FFprobe, (Join-Path $InstallDir "licenses\THIRD_PARTY_NOTICES.txt"))) {
        Assert-True (Test-Path $file) "installed $([System.IO.Path]::GetFileName($file))"
    }
    Assert-True (-not (Get-Command "ffmpeg.exe" -ErrorAction SilentlyContinue)) "no ffmpeg on PATH"

    Write-Host "==> 2. Command-line checks"
    $versionText = (& $Afv --version | Out-String)
    Assert-True ($LASTEXITCODE -eq 0 -and $versionText -match "audio-from-video") "afv --version ($($versionText.Trim()))"
    $doctorText = (& $Afv doctor | Out-String)
    $doctorExit = $LASTEXITCODE
    Write-Host $doctorText
    Assert-True ($doctorExit -eq 0) "afv doctor exit code is 0 (was $doctorExit)"
    Assert-True ($doctorText -match "\(bundled\)") "afv doctor uses the bundled FFmpeg"

    Write-Host "==> 3. Offline extraction"
    $Sample = Join-Path $InputDir "Smoke Test Artist - Smoke Test Song.mp4"
    & $FFmpeg -hide_banner -loglevel error -y `
        -f lavfi -i "sine=frequency=440:duration=3" `
        -f lavfi -i "color=c=blue:s=320x240:d=3" `
        -shortest -c:v mpeg4 -c:a aac $Sample
    Assert-True ($LASTEXITCODE -eq 0 -and (Test-Path $Sample)) "generated a test video with the bundled FFmpeg"
    & $Afv extract $InputDir --offline --non-interactive --output-dir $OutputDir
    Assert-True ($LASTEXITCODE -eq 0) "afv extract exit code is 0 (was $LASTEXITCODE)"
    $mp3 = Get-ChildItem $OutputDir -Recurse -Filter "*.mp3" | Select-Object -First 1
    Assert-True ($null -ne $mp3) "an MP3 was written"
    $codec = (& $FFprobe -v error -select_streams a:0 -show_entries stream=codec_name -of "default=nw=1:nk=1" $mp3.FullName | Out-String).Trim()
    Assert-True ($codec -eq "mp3") "the output audio codec is mp3 ($($mp3.Name))"

    Write-Host "==> 4. Desktop app launch"
    $guiProc = Start-Process -FilePath $Gui -PassThru
    Start-Sleep -Seconds 10
    Assert-True (-not $guiProc.HasExited) "the desktop app is still running after 10 seconds"
    [void]$guiProc.CloseMainWindow()
    if (-not $guiProc.WaitForExit(15000)) { Stop-Process -Id $guiProc.Id -Force }

    Write-Host "==> 5. Silent uninstall"
    $uninstaller = Join-Path $InstallDir "unins000.exe"
    Start-Process -FilePath $uninstaller -ArgumentList @("/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART") | Out-Null
    # The uninstaller re-launches itself from a temp folder, so wait for the files to go.
    $deadline = (Get-Date).AddSeconds(90)
    while ((Test-Path $Gui) -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 2 }
    Assert-True (-not (Test-Path $Gui)) "uninstall removed the program files"

    Write-Host ""
    Write-Host "All smoke tests passed."
} finally {
    Remove-Item -Recurse -Force $Work -ErrorAction SilentlyContinue
}
