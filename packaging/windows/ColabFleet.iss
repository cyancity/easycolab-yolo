; Inno Setup 6 安装脚本，由 packaging/windows/build.py 调用，版本号经 /DAppVersion= 传入
; 装到当前用户目录，不需要管理员权限；账号数据在 %LOCALAPPDATA%\EasyColab，卸载时保留

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
#define AppName "Colab Fleet"
#define AppExe "ColabFleet.exe"

[Setup]
AppId={{8C1F4E2A-6B7D-4C35-9A0E-3F5D2B7C9E14}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=cyancity
AppPublisherURL=https://github.com/cyancity/easycolab-yolo
DefaultDirName={localappdata}\Programs\Colab Fleet
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\..\dist
OutputBaseFilename=ColabFleet-Setup-{#AppVersion}
SetupIconFile=..\..\build\app.ico
UninstallDisplayIcon={app}\{#AppExe}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=no
RestartApplications=no

[Languages]
; 简体中文语言文件取自 jrsoftware/issrc 标签 is-6_7_1（Unofficial），Inno 6.7 安装包不自带
Name: "chs"; MessagesFile: "ChineseSimplified.isl"

[Tasks]
Name: "desktopicon"; Description: "在桌面创建快捷方式"
Name: "autostart"; Description: "开机自动启动（在托盘运行，本机网关一直可用）"

[InstallDelete]
; 覆盖安装前清掉旧版的运行时文件，上一版删掉的模块和网页文件不会残留
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "..\..\dist\ColabFleet\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Registry]
; 与托盘菜单「开机自启」写的是同一个值
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "ColabFleet"; ValueData: """{app}\{#AppExe}"" --tray"; Flags: uninsdeletevalue; Tasks: autostart
; 安装时没勾、之后在托盘里打开了自启：卸载时也删掉
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: none; ValueName: "ColabFleet"; Flags: uninsdeletevalue dontcreatekey

[Run]
Filename: "{app}\{#AppExe}"; Description: "启动 Colab Fleet"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "{sys}\taskkill.exe"; Parameters: "/IM {#AppExe} /F"; Flags: runhidden; RunOnceId: "StopApp"

[Code]
// 托盘进程没有普通窗口，Restart Manager 关不掉它；覆盖安装前先结束，否则 exe 被占用
function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  Code: Integer;
begin
  Exec(ExpandConstant('{sys}\taskkill.exe'), '/IM {#AppExe} /F', '', SW_HIDE, ewWaitUntilTerminated, Code);
  Result := '';
end;
