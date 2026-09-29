# scripts/run_distill_v3.ps1
# Phase 5 -- full 40-epoch v3 distillation run.
#
# Invoke from REPO ROOT (where 'anime_upscaler/' lives):
#   powershell -ExecutionPolicy Bypass -File scripts\run_distill_v3.ps1
#
# Optional flags:
#   -Smoke                        Run the 2-epoch smoke gate at runs/distill_smoke
#                                 (recommended once per session; overrides -Epochs/-OutDir).
#   -Epochs <int>                 Total epochs (default 40).
#   -OutDir <path>                Output dir (default runs/distill_v3).
#   -Resume <path>                Continue training from a student_last.pt / student_best.pt.
#   -Degradation apsisr_v1|none   Train split LR degradation (default apsisr_v1).
#   -Python <path>                Python interpreter (default .venv at repo root).
#
# CRITICAL: distill.py is invoked as a SCRIPT (python <abs path>), NOT as a module
# (python -m anime_upscaler.distill). Module mode fails with
# ModuleNotFoundError: No module named 'dataset' because the absolute imports
# inside distill.py need 'anime_upscaler/' on sys.path[0], which script mode
# provides and module mode does not.
#
# IMPORTANT: we set $ErrorActionPreference = "Continue" and use direct `1>`/`2>`
# file redirects. PowerShell maps a python stderr line into ErrorRecords when
# routed through any pipe or via the bare `&` operator -- even harmless
# UserWarnings (e.g. 'Failed to import from src') will trip
# $ErrorActionPreference = "Stop". Direct file redirects bypass that and
# faithfully propagate $LASTEXITCODE from python.

[CmdletBinding()]
param(
    [switch]$Smoke,
    [int]$Epochs = 40,
    [string]$OutDir = "runs/distill_v3",
    [string]$Resume = $null,
    [string]$Degradation = "apsisr_v1",
    [string]$Python = "E:\python projects\upscale_anime\.venv\Scripts\python.exe",
    [switch]$NoLpips,
    [switch]$NoMsssim
)

$ErrorActionPreference = "Continue"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

if (-not (Test-Path $Python)) {
    Write-Host "[run_distill_v3] FATAL: python not found at $Python" -ForegroundColor Red
    exit 2
}
$distill = Join-Path $repoRoot "anime_upscaler\distill.py"
if (-not (Test-Path $distill)) {
    Write-Host "[run_distill_v3] FATAL: distill.py not found at $distill" -ForegroundColor Red
    exit 2
}

# Boolean ablation toggles (forwarded as --no-lpips / --no-msssim)
if ($NoLpips) { $scriptArgs += @("--no-lpips") }
if ($NoMsssim) { $scriptArgs += @("--no-msssim") }

$scriptArgs = @()
$scriptArgs += (Resolve-Path $distill).Path

if ($Smoke) {
    $scriptArgs += @("--smoke")  # overrides epochs/out-dir inside distill.py
} else {
    $OutDirAbs = if (-not [System.IO.Path]::IsPathRooted($OutDir)) {
        Join-Path $repoRoot $OutDir
    } else { $OutDir }
    New-Item -ItemType Directory -Path $OutDirAbs -Force | Out-Null
    $scriptArgs += @("--epochs", $Epochs,
                     "--out-dir", $OutDirAbs,
                     "--degradation", $Degradation)
    if ($Resume) {
        $ResumeAbs = if (-not [System.IO.Path]::IsPathRooted($Resume)) {
            Join-Path $repoRoot $Resume
        } else { $Resume }
        if (Test-Path $ResumeAbs) {
            $scriptArgs += @("--resume", $ResumeAbs)
        } else {
            Write-Warning "--resume path not found: $ResumeAbs -- ignoring."
        }
    }
}

$resolvedOutDir = if ($Smoke) { Join-Path $repoRoot "runs\distill_smoke" }
                  elseif ([System.IO.Path]::IsPathRooted($OutDir)) { $OutDir }
                  else { Join-Path $repoRoot $OutDir }
New-Item -ItemType Directory -Path $resolvedOutDir -Force | Out-Null

$outLog = Join-Path $resolvedOutDir "train_stdout.log"
$errLog = Join-Path $resolvedOutDir "train_stderr.log"
$pidFile = Join-Path $resolvedOutDir "train.pid"

$myPid | Out-File -FilePath $pidFile -Encoding ascii -Force

Write-Host "[run_distill_v3] repo       : $repoRoot"
Write-Host "[run_distill_v3] python     : $Python"
Write-Host "[run_distill_v3] out-dir    : $resolvedOutDir"
Write-Host "[run_distill_v3] stdout-log : $outLog"
Write-Host "[run_distill_v3] stderr-log : $errLog"
if ($Smoke) { Write-Host "[run_distill_v3] mode       : SMOKE (2 epochs, tiny subset)" }

# Build quoted arg list safely (each arg wrapped in "...").
$quotedArgs = foreach ($a in $scriptArgs) { '"' + ($a -replace '"','""') + '"' }
$quotedCmd = ($quotedArgs -join ' ')

Write-Host "[run_distill_v3] invoking: $Python $quotedCmd"
Write-Host "[run_distill_v3] streams  : 1> $outLog  2> $errLog"

# Direct file redirects on `&` (no pipe, no ScriptBlock).
& $Python @scriptArgs 1> $outLog 2> $errLog
$exitCode = $LASTEXITCODE

if (Test-Path $outLog) {
    Write-Host "---- train_stdout.log (tail 60) ----"
    Get-Content $outLog -Tail 60 | ForEach-Object { Write-Host $_ }
}
if ((Test-Path $errLog) -and ((Get-Item $errLog).Length -gt 0)) {
    Write-Host "---- train_stderr.log (full) ----"
    Get-Content $errLog | ForEach-Object { Write-Host $_ }
}

if ($exitCode -ne 0) {
    Write-Host "[run_distill_v3] FAILED exit=$exitCode" -ForegroundColor Red
    Remove-Item $pidFile -ErrorAction SilentlyContinue
    exit $exitCode
}

Write-Host "[run_distill_v3] DONE" -ForegroundColor Green
Write-Host "Inspect:"
Write-Host "  - $resolvedOutDir\train_log.csv        (per-epoch metrics)"
Write-Host "  - $resolvedOutDir\results_table.md    (final test set)"
Write-Host "  - $resolvedOutDir\student_best.pt     (EMA state at best epoch)"
Write-Host "  - $resolvedOutDir\student_last.pt     (raw state at last epoch)"
Remove-Item $pidFile -ErrorAction SilentlyContinue
