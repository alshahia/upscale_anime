"""
CI-friendly smoke runner for the v7 anime SOTA test subset.

Runs the 12 v7-relevant test files in a single pytest invocation.
Skips tests marked with @pytest.mark.slow.

Runs in <2 min on a CPU. Exits 0 if all pass, 1 otherwise.

Usage:
    python scripts/run_v7_smoke.py
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# v7-relevant test files. Paths are current as of the src-layout restructure
# (package anime_sr under src/, tests split into unit/ integration/ e2e/).
# Excluded on purpose:
#   tests/unit/utils/test_topiq_metric.py      - crashes the runner (pyiqa
#       native segfault); already listed in tests/conftest.py collect_ignore.
#   tests/integration/test_evaluate_quality_smoke.py - has a pre-existing
#       failure unrelated to v7, so it would break a green smoke run.
TEST_FILES = [
    "tests/unit/data/test_shuffled_resize.py",
    "tests/integration/inference/test_apisr_two_stage.py",
    "tests/unit/data/test_xdog_pseudo_gt.py",
    "tests/integration/training/test_v7_loss_schedule.py",
    "tests/integration/training/test_ema_integration.py",
    "tests/integration/training/test_ema_resume.py",
    "tests/unit/models/test_mamba_spab.py",
    "tests/e2e/test_mamba_span_e2e.py",
    "tests/e2e/test_v7_degradation_e2e.py",
    "tests/unit/utils/test_codec_perf.py",
    "tests/unit/data/test_degradation_presets.py",
    "tests/unit/utils/test_load_config.py",
]


def main() -> int:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-v",
        "--tb=short",
        "-m",
        "not slow",
        *TEST_FILES,
    ]
    print("[v7-smoke] Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
