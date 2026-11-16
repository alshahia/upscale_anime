<#
.SYNOPSIS
  Clean-room rehearsal of the built GUI wheel on Windows.

.DESCRIPTION
  Creates a throwaway venv outside the repo, installs the newest
  dist/anime_upscaler_gui-*.whl with its dependencies, then runs the installed
  anime-upscaler-gui.exe --self-check. Exit 0 proves the wheel carries its own
  package data, needs no repo checkout and no anime_sr installation, and can
  find ffmpeg/ffprobe on a clean machine.

.PARAMETER RepoRoot
  Repository (or clean git-archive) root that contains dist/. Defaults to two
  levels up from this script.

.PARAMETER Python
  Interpreter used to create the throwaway venv. Defaults to "python" on PATH;
  pass an explicit path when the machine has no python on PATH.
#>
param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../../")).Path,
    [string]$Python = "python"
)
$ErrorActionPreference = "Stop"

$work = Join-Path $env:TEMP ("anime-sr-rehearsal-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $work | Out-Null
Write-Host "Rehearsal dir: $work"

$wheel = Get-ChildItem (Join-Path $RepoRoot "dist/anime_upscaler_gui-*.whl") |
    Sort-Object LastWriteTime | Select-Object -Last 1
if (-not $wheel) { throw "no GUI wheel in dist/ - build it first" }
Write-Host "Wheel under test: $($wheel.FullName)"

$venv = Join-Path $work "venv"
& $Python -m venv $venv
if ($LASTEXITCODE -ne 0) { throw "venv creation failed for $Python" }
$vpy = Join-Path $venv "Scripts/python.exe"
& $vpy -m pip install --upgrade pip | Out-Null

& $vpy -m pip install $wheel.FullName
if ($LASTEXITCODE -ne 0) { throw "wheel install failed in clean venv" }

# The console script must prove its own environment: JSON report + exit 0.
$exe = Join-Path $venv "Scripts/anime-upscaler-gui.exe"
& $exe --self-check
if ($LASTEXITCODE -ne 0) { throw "self-check failed in clean venv" }

& $vpy -c "from anime_upscaler_gui.arch_registry import registered_kinds as k; print(sorted(k()))"
if ($LASTEXITCODE -ne 0) { throw "arch registry import failed in clean venv" }
$hasAnimeSr = (& $vpy -c "import importlib.util; print(importlib.util.find_spec('anime_sr') is not None)").Trim()
if ($hasAnimeSr -ne "False") { throw "clean-room venv unexpectedly has anime_sr installed" }

Write-Host "REHEARSAL OK: $work"
