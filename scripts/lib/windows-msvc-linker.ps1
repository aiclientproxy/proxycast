param(
    [Parameter(Mandatory = $true)]
    [string]$Target
)

$ErrorActionPreference = "Stop"

if ($Target -ne "x86_64-pc-windows-msvc") {
    throw "Unsupported Windows MSVC target: $Target"
}
if ([string]::IsNullOrWhiteSpace($env:GITHUB_ENV)) {
    throw "GITHUB_ENV is required"
}

$VsWhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $VsWhere)) {
    throw "vswhere.exe not found at $VsWhere"
}

$RequiredComponent = "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"
$InstallPath = & $VsWhere -latest -products * -requires $RequiredComponent -property installationPath 2>$null
if (-not $InstallPath) {
    throw "Visual Studio with $RequiredComponent was not found"
}

$VsDevCmd = Join-Path $InstallPath "Common7\Tools\VsDevCmd.bat"
if (-not (Test-Path $VsDevCmd)) {
    throw "VsDevCmd.bat not found at $VsDevCmd"
}

$VariablesToExport = @(
    "INCLUDE",
    "LIB",
    "LIBPATH",
    "PATH",
    "UCRTVersion",
    "UniversalCRTSdkDir",
    "VCINSTALLDIR",
    "VCToolsInstallDir",
    "WindowsLibPath",
    "WindowsSdkBinPath",
    "WindowsSdkDir",
    "WindowsSDKLibVersion",
    "WindowsSDKVersion"
)

$EnvironmentLines = & cmd.exe /c ('"{0}" -no_logo -arch=x64 -host_arch=x64 >nul && set' -f $VsDevCmd)
$VsDevCmdExitCode = $LASTEXITCODE
if ($VsDevCmdExitCode -ne 0) {
    throw "VsDevCmd.bat failed with exit code $VsDevCmdExitCode"
}

$ExportedVariables = @{}
foreach ($Line in $EnvironmentLines) {
    if ($Line -notmatch "^(.*?)=(.*)$") {
        continue
    }

    $Name = $Matches[1]
    $Value = $Matches[2]
    if ($VariablesToExport -contains $Name) {
        if ($Name -ieq "Path") {
            $Name = "PATH"
        }
        $ExportedVariables[$Name] = $Value
        "$Name=$Value" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
    }
}

foreach ($RequiredVariable in @("INCLUDE", "LIB", "LIBPATH", "PATH", "UCRTVersion", "VCToolsInstallDir", "WindowsSdkDir")) {
    if ([string]::IsNullOrWhiteSpace($ExportedVariables[$RequiredVariable])) {
        throw "VsDevCmd.bat did not export $RequiredVariable"
    }
}

$RustSysroot = (& rustc --print sysroot).Trim()
$RustHostLine = & rustc -vV | Select-String "^host: "
if (-not $RustHostLine) {
    throw "rustc did not report a host target"
}
$RustHost = $RustHostLine.Line.Substring(6).Trim()
$Linker = Join-Path $RustSysroot "lib\rustlib\$RustHost\bin\rust-lld.exe"
if (-not (Test-Path $Linker)) {
    throw "rust-lld.exe not found at $Linker"
}

Write-Output "Using Windows linker: $Linker"
"CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER=$Linker" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
