"""Meta-test: verify the v7 smoke runner script works end-to-end."""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_v7_smoke_runner_exits_zero():
    """Run scripts/run_v7_smoke.py in a subprocess and assert exit 0."""
    result = subprocess.run(
        [sys.executable, "scripts/run_v7_smoke.py"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"Smoke runner failed (rc={result.returncode}):\n"
        f"STDOUT (tail):\n{result.stdout[-2000:]}\n"
        f"STDERR (tail):\n{result.stderr[-2000:]}"
    )
