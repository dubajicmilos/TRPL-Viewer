param(
    [Parameter(Mandatory)]
    [ValidatePattern('^\d+\.\d+\.\d+$')]
    [string]$Version,
    [string]$Python = 'python',
    [string]$Iscc,
    [string]$OutputDir = (Join-Path $PWD 'dist'),
    [switch]$BundleOnly,
    [switch]$VerifyInstall,
    [string]$PreviousInstaller
)

$ErrorActionPreference = 'Stop'
$sourceRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
if (-not (Test-Path -LiteralPath (Join-Path $sourceRoot '.viewer-export') -PathType Leaf) -or
    (Test-Path -LiteralPath (Join-Path $sourceRoot 'hsapp/__main__.py'))) {
    throw 'Build from the exported viewer tree so acquisition code is outside the build environment.'
}
if ($BundleOnly -and $VerifyInstall) {
    throw 'Install validation requires an installer.'
}
& $Python -c "import platform, sys; sys.exit(platform.machine().upper() not in ('AMD64', 'X86_64') or sys.maxsize <= 2**32)"
if ($LASTEXITCODE -ne 0) {
    throw 'The viewer build requires x64 Python.'
}

if (-not $BundleOnly -and -not $Iscc) {
    $command = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($command) {
        $Iscc = $command.Source
    } else {
        foreach ($location in @(
            [Environment]::GetFolderPath('ProgramFilesX86'),
            [Environment]::GetFolderPath('ProgramFiles'),
            (Join-Path $env:LOCALAPPDATA 'Programs')
        )) {
            $candidate = Join-Path $location 'Inno Setup 6/ISCC.exe'
            if (Test-Path -LiteralPath $candidate -PathType Leaf) {
                $Iscc = $candidate
                break
            }
        }
    }
}
if (-not $BundleOnly -and (-not $Iscc -or -not (Test-Path -LiteralPath $Iscc -PathType Leaf))) {
    throw 'Install Inno Setup 6 or pass the path to ISCC.exe using -Iscc.'
}

function Invoke-CheckedProcess($Program, $Arguments, $Directory) {
    $proc = Start-Process -FilePath $Program -ArgumentList $Arguments -WorkingDirectory $Directory -WindowStyle Hidden -PassThru
    if (-not $proc.WaitForExit(180000)) {
        $proc.Kill()
        throw "Timed out running $Program"
    }
    if ($proc.ExitCode -ne 0) {
        throw "$Program returned exit code $($proc.ExitCode)"
    }
}

function Test-Viewer($Exe, $Marker) {
    # Launch from outside the source tree and without the build Python or Qt environment.
    $saved = @{}
    foreach ($key in @(
        'PATH', 'PYTHONPATH', 'PYTHONHOME', 'CONDA_PREFIX', 'QT_PLUGIN_PATH',
        'QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_QPA_PLATFORM', 'TCL_LIBRARY', 'TK_LIBRARY'
    )) {
        $saved[$key] = [Environment]::GetEnvironmentVariable($key)
        [Environment]::SetEnvironmentVariable($key, $null)
    }
    $env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
    try {
        try {
            Invoke-CheckedProcess $Exe ('--smoke-test "{0}"' -f $Marker) $buildDir
        } catch {
            if (Test-Path -LiteralPath $Marker -PathType Leaf) {
                Get-Content -LiteralPath $Marker | Write-Host
            }
            throw
        }
        $result = Get-Content -LiteralPath $Marker -Raw | ConvertFrom-Json
        if ($result.status -ne 'passed' -or $result.version -ne $Version -or -not $result.frozen) {
            throw "Viewer smoke test failed. Result: $($result | ConvertTo-Json -Compress)"
        }
    } finally {
        foreach ($key in $saved.Keys) {
            [Environment]::SetEnvironmentVariable($key, $saved[$key])
        }
    }
}

New-Item -Path $OutputDir -ItemType Directory -Force | Out-Null
$OutputDir = (Resolve-Path -LiteralPath $OutputDir).Path
$tempRoot = (Resolve-Path -LiteralPath ([IO.Path]::GetTempPath())).Path.TrimEnd('\')
$buildDir = Join-Path $tempRoot ('trpl-viewer-build-' + [guid]::NewGuid().ToString('N'))
New-Item -Path $buildDir -ItemType Directory | Out-Null
$priorVersion = $env:TRPL_VIEWER_BUILD_VERSION
$env:TRPL_VIEWER_BUILD_VERSION = $Version
try {
    $distRoot = if ($BundleOnly) { $OutputDir } else { Join-Path $buildDir 'frozen' }
    & $Python -m PyInstaller --noconfirm --workpath (Join-Path $buildDir 'work') --distpath $distRoot (Join-Path $PSScriptRoot 'viewer.spec')
    if ($LASTEXITCODE -ne 0) {
        throw 'PyInstaller failed.'
    }
    $bundle = Join-Path $distRoot 'TRPL-Viewer'
    Test-Viewer (Join-Path $bundle 'TRPL-Viewer.exe') (Join-Path $buildDir 'frozen.json')
    if ($BundleOnly) {
        Write-Host "Verified frozen viewer: $bundle"
        return
    }
    & $Iscc "/DAppVersion=$Version" "/DBundleDir=$bundle" "/O$OutputDir" (Join-Path $PSScriptRoot 'viewer.iss')
    if ($LASTEXITCODE -ne 0) {
        throw 'Inno Setup failed.'
    }
    $installer = Join-Path $OutputDir "TRPL-Viewer-Setup-$Version-win-x64.exe"
    if (-not (Test-Path -LiteralPath $installer -PathType Leaf)) {
        throw 'Installer output is missing.'
    }
    if ($VerifyInstall) {
        $modes = if ($PreviousInstaller) { @('fresh', 'upgraded') } else { @('fresh') }
        foreach ($mode in $modes) {
            $installDir = Join-Path $buildDir "$mode viewer"
            $log = Join-Path $buildDir "$mode setup.log"
            $setupArgs = '/VERYSILENT /SUPPRESSMSGBOXES /NORESTART /SP- /DIR="{0}" /LOG="{1}"' -f $installDir, $log
            try {
                if ($mode -eq 'upgraded') {
                    $baseline = (Resolve-Path -LiteralPath $PreviousInstaller).Path
                    Invoke-CheckedProcess $baseline $setupArgs $buildDir
                }
                Invoke-CheckedProcess $installer $setupArgs $buildDir
                Test-Viewer (Join-Path $installDir 'TRPL-Viewer.exe') (Join-Path $buildDir "$mode.json")
            } catch {
                if (Test-Path -LiteralPath $log -PathType Leaf) {
                    Get-Content -LiteralPath $log -Tail 30 | Write-Host
                }
                throw
            } finally {
                $uninstaller = Join-Path $installDir 'unins000.exe'
                if (Test-Path -LiteralPath $uninstaller -PathType Leaf) {
                    Invoke-CheckedProcess $uninstaller '/VERYSILENT /SUPPRESSMSGBOXES /NORESTART' $buildDir
                }
            }
        }
    }
    Write-Host "Verified Windows installer: $installer"
} finally {
    $env:TRPL_VIEWER_BUILD_VERSION = $priorVersion
    $resolvedBuild = (Resolve-Path -LiteralPath $buildDir).Path
    if ((Split-Path -Parent $resolvedBuild) -ne $tempRoot -or (Split-Path -Leaf $resolvedBuild) -notmatch '^trpl-viewer-build-[a-f0-9]{32}$') {
        throw "Refusing to remove unexpected build directory $resolvedBuild"
    }
    Remove-Item -LiteralPath $resolvedBuild -Recurse -Force
}
