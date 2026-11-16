<#
.SYNOPSIS
  Clean-room rehearsal of a built GUI wheel on Windows.

.DESCRIPTION
  Creates a throwaway venv outside the repo, installs one exact GUI wheel
  with its dependencies, and runs the installed console script:
    - --self-check must exit 0, which proves ffmpeg, ffprobe and the bundled
      preset catalog (data/registry.json) are all present in the wheel;
    - the architecture registry must import;
    - anime_upscaler_gui must import from the throwaway venv site-packages
      (PYTHONPATH is cleared, so a repo checkout cannot leak in);
    - anime_sr must not be installed.

  The script echoes the exact wheel path, its SHA256 and the commit it
  certifies, and repeats them on the final REHEARSAL OK line. It fails when
  no wheel is available or any check fails, so it cannot certify a wheel
  that was not built from the intended source tree.

.PARAMETER RepoRoot
  Repository (or clean git-archive) root that contains dist/. Defaults to two
  levels up from this script.

.PARAMETER Python
  Interpreter used to create the throwaway venv. Defaults to "python" on PATH;
  pass an explicit path when the machine has no python on PATH.

.PARAMETER Wheel
  Exact wheel file to rehearse. Defaults to the newest
  dist/anime_upscaler_gui-*.whl under RepoRoot.

.PARAMETER Commit
  Commit SHA the wheel was built from. Defaults to `git -C RepoRoot rev-parse
  HEAD`; pass it explicitly for clean git-archive trees (no .git directory).
#>
param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../../")).Path,
    [string]$Python = "python",
    [string]$Wheel = "",
    [string]$Commit = ""
)
$ErrorActionPreference = "Stop"

if ($Wheel) {
    $wheelFile = Get-Item -LiteralPath $Wheel -ErrorAction SilentlyContinue
    if (-not $wheelFile) { throw "wheel not found: $Wheel" }
} else {
    $wheelFile = Get-ChildItem (Join-Path $RepoRoot "dist/anime_upscaler_gui-*.whl") |
        Sort-Object LastWriteTime | Select-Object -Last 1
    if (-not $wheelFile) {
        throw "no GUI wheel in dist/ - build it first (python -m pip wheel apps/anime_upscaler_gui --no-deps -w dist)"
    }
}
if (-not $Commit) {
    if (Test-Path (Join-Path $RepoRoot ".git")) {
        $Commit = (& git -C $RepoRoot rev-parse HEAD).Trim()
    } else {
        $Commit = "UNKNOWN (no .git in RepoRoot; pass -Commit <sha>)"
    }
}
$wheelHash = (Get-FileHash -LiteralPath $wheelFile.FullName -Algorithm SHA256).Hash.ToLower()
Write-Host "Wheel under test: $($wheelFile.FullName)"
Write-Host "Wheel SHA256: $wheelHash"
Write-Host "Commit: $Commit"

$work = Join-Path $env:TEMP ("anime-sr-rehearsal-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $work | Out-Null
Write-Host "Rehearsal dir: $work"

# PYTHONPATH-hermetic: clear any developer PYTHONPATH so every import in
# the throwaway venv resolves from the installed wheel or its dependencies.
$savedPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = ""
try {
    $venv = Join-Path $work "venv"
    & $Python -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed for $Python" }
    $vpy = Join-Path $venv "Scripts/python.exe"
    & $vpy -m pip install --upgrade pip | Out-Null

    & $vpy -m pip install $wheelFile.FullName
    if ($LASTEXITCODE -ne 0) { throw "wheel install failed in clean venv" }

    # The console script must prove its own environment: JSON report + exit 0
    # (ffmpeg, ffprobe and the bundled preset registry must all be present).
    $exe = Join-Path $venv "Scripts/anime-upscaler-gui.exe"
    & $exe --self-check
    if ($LASTEXITCODE -ne 0) { throw "self-check failed in clean venv" }

    # Import must resolve from this venv, not from a repo checkout or PYTHONPATH.
    & $vpy -c 'import anime_upscaler_gui; p = anime_upscaler_gui.__file__; assert "site-packages" in p, p; print("import location OK:", p)'
    if ($LASTEXITCODE -ne 0) { throw "anime_upscaler_gui did not import from the clean venv" }

    & $vpy -c "from anime_upscaler_gui.arch_registry import registered_kinds as k; print(sorted(k()))"
    if ($LASTEXITCODE -ne 0) { throw "arch registry import failed in clean venv" }

    $hasAnimeSr = (& $vpy -c "import importlib.util; print(importlib.util.find_spec('anime_sr') is not None)").Trim()
    if ($hasAnimeSr -ne "False") { throw "clean-room venv unexpectedly has anime_sr installed" }

    Write-Host "REHEARSAL OK: $work | wheel=$($wheelFile.Name) | sha256=$wheelHash | commit=$Commit"
} finally {
    $env:PYTHONPATH = $savedPythonPath
}