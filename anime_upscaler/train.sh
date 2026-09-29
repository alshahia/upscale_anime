#!/usr/bin/env bash
# =============================================================================
# End-to-end teacher-student distillation pipeline for the 4x anime upscaler.
#
# Stages:
#   1. dataset self-test          (bicubic LR pairs, splits, augmentations)
#   2. teacher load check         (frozen SPAN V7, feature taps, parity)
#   3. student module self-test   (<600K params, shapes, GPU latency)
#   4. distillation smoke         (2 epochs, tiny subset -> runs/distill_smoke)
#   5. FULL distillation          (EPOCHS epochs -> runs/distill_v1)
#   6. export                     (fp32/fp16/calibrated-int8 ONNX + report)
#   7. validation                 (side-by-side strips + metrics in output/)
#
# Usage:
#   bash train.sh                  # defaults: EPOCHS=40
#   EPOCHS=60 bash train.sh        # longer run
#   FAST=1 bash train.sh           # skip self-tests + smoke, go straight to 5-7
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
PKG="$ROOT/anime_upscaler"
PY="$ROOT/.venv/Scripts/python.exe"

EPOCHS="${EPOCHS:-40}"
RUN_DIR="$ROOT/runs/distill_v1"

cd "$ROOT"
# torch bundles the cuDNN/CUDA DLLs onnxruntime's GPU EP needs; not on PATH by default.
export PATH="$ROOT/.venv/Lib/site-packages/torch/lib:$PATH"

banner() { printf '\n================ %s ================\n' "$*"; }

if [ "${FAST:-0}" != "1" ]; then
    banner "[1/7] dataset self-test"
    "$PY" "$PKG/dataset.py"

    banner "[2/7] teacher load + parity check"
    "$PY" "$PKG/teacher.py"

    banner "[3/7] student module self-test"
    "$PY" "$PKG/student.py"

    banner "[4/7] distillation SMOKE (2 epochs)"
    "$PY" "$PKG/distill.py" --smoke --out-dir "$ROOT/runs/distill_smoke"
fi

banner "[5/7] FULL distillation ($EPOCHS epochs) -> $RUN_DIR"
"$PY" "$PKG/distill.py" \
    --epochs "$EPOCHS" --batch-size 16 --lr 5e-5 \
    --out-dir "$RUN_DIR"

banner "[6/7] export fp32/fp16/int8 + latency report"
"$PY" "$PKG/export.py" --run-dir "$RUN_DIR"

banner "[7/7] side-by-side validation strips -> output/"
"$PY" "$PKG/validate.py" --run-dir "$RUN_DIR" --frames 5

echo ""
echo "DONE."
echo "  metrics table : $RUN_DIR/results_table.md"
echo "  export report : $PKG/export/export_report.csv"
echo "  visual strips : $ROOT/output/"
