#ifndef AppVersion
  #error AppVersion must be passed by the build script
#endif
#ifndef BundleDir
  #error BundleDir must point to the frozen viewer bundle
#endif

[Setup]
AppId={{592FF0F4-125E-466C-9E16-D104F495054A}
AppName=TRPL Viewer
AppVersion={#AppVersion}
AppPublisher=TRPL Viewer contributors
VersionInfoVersion={#AppVersion}
DefaultDirName={localappdata}\Programs\TRPL Viewer
DefaultGroupName=TRPL Viewer
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
MinVersion=10.0.17763
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\TRPL-Viewer.exe
CloseApplications=yes
RestartApplications=no
WizardStyle=modern
Compression=lzma2
SolidCompression=yes
OutputBaseFilename=TRPL-Viewer-Setup-{#AppVersion}-win-x64

[Files]
Source: "{#BundleDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\TRPL Viewer"; Filename: "{app}\TRPL-Viewer.exe"; WorkingDir: "{app}"
Name: "{group}\Third-party licenses"; Filename: "{app}\_internal\THIRD-PARTY-LICENSES\NOTICE.txt"

[Run]
Filename: "{app}\TRPL-Viewer.exe"; Description: "{cm:LaunchProgram,TRPL Viewer}"; Flags: nowait postinstall skipifsilent
