# scripts/run_rank1_warmstart.ps1
# Phase 5 Rank #1 -- warm-start v1 RFDN -> SRVGG student, no adversarial.
#
# Recipe (from docs/research/anime_sr_2026/RECOMMENDED_PATH.md §"Expected recipe"):
#   Warm-start from pretrained\RFDN_distill_v1_4x_student.pth (315K RFDN)
#   into a TinySRVGGStudent (317K SRVGG body). Cross-arch partial load via
#   --warm-start-mode partial: copies body.0 (head) + body.{26} (tail) from
#   v1 RFDN, leaves the 12-conv middle stack at random init.
#   Teacher: animevideov3 (621K SRVGG, frozen). No adversarial.
#   40 epochs, batch 16, lr 5e-5. Expected wall-time ~21-26 min.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\run_rank1_warmstart.ps1
#
# Optional flags:
#   -Smoke                        Run the 2-epoch smoke gate at runs/distill_v3_4x_srvgg_warmstart_smoke
#                                 (overrides -Epochs/-OutDir). Verifies infra only.
#   -Epochs <int>                 Total epochs (default 40).
#   -BatchSize <int>              Batch size (default 16).
#   -Lr <float>                   Learning rate (default 5e-5).
#   -OutDir <path>                Output dir (default runs/distill_v3_4x_srvgg_warmstart).
#   -V1Ckpt <path>                Source checkpoint for warm-start (default pretrained\RFDN_distill_v1_4x_student.pth).
#   -Python <path>                Python interpreter (default repo .venv).
#   -SkipWarmStart                Skip --resume (from-scratch run for ablation; not the recommended path).
#
# Halt conditions (from docs/plans/student_phase4_nearest_adv_srvgg_plan.md §5):
#   - Val PSNR < 20 dB at any epoch  -> mode collapse  -> abort
#   - Val PSNR < 26 dB after epoch 10 -> slow convergence -> abort
#   - EMA PSNR not improving for 10 consecutive epochs -> plateau -> abort
#   (distill.py does not auto-halt; inspect runs/.../train_log.csv manually.)
#
# Promotion gates (from PROJECT_MEMORY §6):
#   D.1 (binding): PSNR >= 29.0 dB on held-out test set
#   D.2           : lap_var >= 35 on full-frame harness (target: tmp/real_video_1sec.mp4 frame 8)
#   latency       : <= 80 ms/frame (vs v1 RFDN 104 ms)
#
# CRITICAL: invoke distill.py as a script (python <abs path>) NOT as a module.
# Module mode fails with ModuleNotFoundError: 'dataset' (see run_distill_v3.ps1
# header for the long version of why).

[CmdletBinding()]
param(
    [switch]$Smoke,
    [int]$Epochs = 40,
    [int]$BatchSize = 16,
    [double]$Lr = 5e-5,
    [string]$OutDir = "runs/distill_v3_4x_srvgg_warmstart",
    [string]$V1Ckpt = "pretrained\RFDN_distill_v1_4x_student.pth",
    [string]$Python = "E:\python projects\upscale_anime\.venv\Scripts\python.exe",
    [switch]$SkipWarmStart
)

$ErrorActionPreference = "Continue"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

if (-not (Test-Path $Python)) {
    Write-Host "[run_rank1] FATAL: python not found at $Python" -ForegroundColor Red
    exit 2
}
$distill = Join-Path $repoRoot "anime_upscaler\distill.py"
if (-not (Test-Path $distill)) {
    Write-Host "[run_rank1] FATAL: distill.py not found at $distill" -ForegroundColor Red
    exit 2
}

# Resolve V1 checkpoint path relative to repo root
$V1CkptAbs = if (-not [System.IO.Path]::IsPathRooted($V1Ckpt)) {
    Join-Path $repoRoot $V1Ckpt
} else { $V1Ckpt }
if (-not $SkipWarmStart -and -not (Test-Path $V1CkptAbs)) {
    Write-Host "[run_rank1] FATAL: v1 checkpoint not found at $V1CkptAbs" -ForegroundColor Red
    Write-Host "  pass -V1Ckpt <path> or -SkipWarmStart to override"
    exit 2
}

# Resolve out-dir relative to repo root
$OutDirAbs = if (-not [System.IO.Path]::IsPathRooted($OutDir)) {
    Join-Path $repoRoot $OutDir
} else { $OutDir }
New-Item -ItemType Directory -Path $OutDirAbs -Force | Out-Null

# Build argument list
$scriptArgs = @()
$scriptArgs += (Resolve-Path $distill).Path
$scriptArgs += @("--teacher", "animevideov3")
$scriptArgs += @("--arch", "srvgg")
$scriptArgs += @("--lambda-adv", "0")
$scriptArgs += @("--feat-weight", "1.0")
$scriptArgs += @("--shortcut-anneal", "off")
if (-not $SkipWarmStart) {
    $scriptArgs += @("--resume", $V1CkptAbs,
                     "--fresh-epoch",
                     "--warm-start-mode", "partial")
}

if ($Smoke) {
    # Use a separate smoke subdir so we don't clobber the production out-dir.
    $smokeOutDir = Join-Path $OutDirAbs "smoke"
    New-Item -ItemType Directory -Path $smokeOutDir -Force | Out-Null
    $scriptArgs += @("--epochs", "2",
                     "--batch-size", "4",
                     "--num-workers", "2",
                     "--val-batches", "2",
                     "--out-dir", $smokeOutDir)
} else {
    $scriptArgs += @("--epochs", $Epochs,
                     "--batch-size", $BatchSize,
                     "--lr", $Lr,
                     "--out-dir", $OutDirAbs)
}

# Stdout/stderr capture (use smoke subdir if -Smoke to keep prod out-dir clean)
$logDir = if ($Smoke) { Join-Path $OutDirAbs "smoke" } else { $OutDirAbs }
$outLog = Join-Path $logDir "train_stdout.log"
$errLog = Join-Path $logDir "train_stderr.log"
$pidFile = Join-Path $logDir "train.pid"

$myPid | Out-File -FilePath $pidFile -Encoding ascii -Force

Write-Host "[run_rank1] repo       : $repoRoot"
Write-Host "[run_rank1] python     : $Python"
Write-Host "[run_rank1] v1-ckpt    : $V1CkptAbs"
Write-Host "[run_rank1] warm-start : $([bool](-not $SkipWarmStart))"
Write-Host "[run_rank1] out-dir    : $OutDirAbs"
Write-Host "[run_rank1] stdout-log : $outLog"
Write-Host "[run_rank1] stderr-log : $errLog"
if ($Smoke) { Write-Host "[run_rank1] mode       : SMOKE (2 epochs, batch 4)" }

# Quote args safely
$quotedArgs = foreach ($a in $scriptArgs) { '"' + ($a -replace '"','""') + '"' }
$quotedCmd = ($quotedArgs -join ' ')
Write-Host "[run_rank1] invoking: $Python $quotedCmd"

# Direct file redirects (no pipe, no ScriptBlock) -- see run_distill_v3.ps1 header
& $Python @scriptArgs 1> $outLog 2> $errLog
$exitCode = $LASTEXITCODE

if (Test-Path $outLog) {
    Write-Host "---- train_stdout.log (tail 40) ----"
    Get-Content $outLog -Tail 40 | ForEach-Object { Write-Host $_ }
}
if ((Test-Path $errLog) -and ((Get-Item $errLog).Length -gt 0)) {
    Write-Host "---- train_stderr.log (full) ----"
    Get-Content $errLog | ForEach-Object { Write-Host $_ }
}

if ($exitCode -ne 0) {
    Write-Host "[run_rank1] FAILED exit=$exitCode" -ForegroundColor Red
    Remove-Item $pidFile -ErrorAction SilentlyContinue
    exit $exitCode
}

Write-Host "[run_rank1] DONE" -ForegroundColor Green
Write-Host "Inspect:"
Write-Host "  - $OutDirAbs\train_log.csv        (per-epoch metrics)"
Write-Host "  - $OutDirAbs\results_table.md    (final test set)"
Write-Host "  - $OutDirAbs\student_best.pt     (EMA state at best epoch)"
Write-Host "  - $OutDirAbs\student_last.pt     (raw state at last epoch)"
Write-Host ""
Write-Host "Post-run promotion gates:"
Write-Host "  D.1 (binding): PSNR >= 29.0 dB on held-out test"
Write-Host "  D.2           : lap_var >= 35 on tmp/real_video_1sec.mp4 frame 8"
Write-Host "  latency       : <= 80 ms/frame"
Remove-Item $pidFile -ErrorAction SilentlyContinue
