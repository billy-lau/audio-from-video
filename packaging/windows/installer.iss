; Inno Setup script for the Windows installer of Audio from Video.
;
; Built by packaging/build_windows.ps1, which passes:
;   /DAppVersion=0.1.0
;   /DSourceDir=<PyInstaller output folder, with ffmpeg\ and licenses\ added>
;   /DOutputDir=<folder for the finished installer>
;   /DIconFile=<app_icon.ico>
;
; The installer is per-user: it needs no admin rights and installs to
; %LOCALAPPDATA%\Programs\Audio from Video. It does not change PATH.

#ifndef AppVersion
  #error AppVersion must be defined (/DAppVersion=...)
#endif
#ifndef SourceDir
  #error SourceDir must be defined (/DSourceDir=...)
#endif
#ifndef OutputDir
  #define OutputDir "."
#endif

#define AppName "Audio from Video"
#define AppExeName "Audio from Video.exe"

[Setup]
; Keep AppId fixed forever: Windows uses it to recognize upgrades and uninstalls.
AppId={{B0615091-F8CB-470E-A6E4-0C4DFC5BECCA}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=Audio from Video
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
OutputDir={#OutputDir}
OutputBaseFilename=AudioFromVideo-Setup-{#AppVersion}-x64
#ifdef IconFile
SetupIconFile={#IconFile}
#endif
UninstallDisplayIcon={app}\{#AppExeName}
UninstallDisplayName={#AppName}
LicenseFile={#SourceDir}\licenses\LICENSE_SUMMARY.txt
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[InstallDelete]
; Remove files left by an older version before copying the new ones.
Type: filesandordirs; Name: "{app}\_internal"
Type: filesandordirs; Name: "{app}\ffmpeg"
Type: filesandordirs; Name: "{app}\licenses"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

; User settings, cache, and logs (outside {app}) are intentionally kept on uninstall.
