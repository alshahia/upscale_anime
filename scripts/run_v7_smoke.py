"""
CI-friendly smoke runner for the v7 anime SOTA test subset.

Runs the 12 v7-relevant test files in a single pytest invocation.
Skips tests marked with @pytest.mark.slow and the known-flaky
test_topiq_higher_for_cleaner_image (already slow-marked, but
deselected explicitly for self-documentation).

Runs in <2 min on a CPU. Exits 0 if all pass, 1 otherwise.

Usage:
    python scripts/run_v7_smoke.py
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_FILES = [
    "tests/test_shuffled_resize.py",
    "tests/test_apisr_two_stage.py",
    "tests/test_xdog_pseudo_gt.py",
    "tests/test_v7_loss_schedule.py",
    "tests/test_ema_integration.py",
    "tests/test_ema_resume.py",
    "tests/test_topiq_metric.py",
    "tests/test_evaluate_quality_smoke.py",
    "tests/test_mamba_spab.py",
    "tests/test_mamba_span_e2e.py",
    "tests/test_degradation_presets.py",
    "tests/test_load_config.py",
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
        "--deselect",
        "tests/test_topiq_metric.py::test_topiq_higher_for_cleaner_image",
        *TEST_FILES,
    ]
    print("[v7-smoke] Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
