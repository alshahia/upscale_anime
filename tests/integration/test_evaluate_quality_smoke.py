"""
Smoke tests for scripts/evaluate_quality.py.

The smoke test:
  1. Generates 2 small dummy PNGs in a tmp_path
  2. Invokes evaluate_quality.py with --smoke 2 and a fast metric (clipiqa)
  3. Asserts exit code 0 and that the CSV has the expected columns

Skipped if pyiqa is not installed (CLIPIQA depends on it).
"""
import csv
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
SCRIPT = PROJECT_ROOT / "scripts" / "evaluate_quality.py"

# Make pyiqa importable check
try:
    import pyiqa  # noqa: F401
    HAS_PYIQA = True
except ImportError:
    HAS_PYIQA = False


def _make_dummy_image(path: Path, h: int = 96, w: int = 96) -> None:
    """Write a small RGB PNG. Uses PIL if available, else falls back to OpenCV."""
    try:
        from PIL import Image
        import numpy as np
        arr = (np.random.rand(h, w, 3) * 255).astype("uint8")
        Image.fromarray(arr).save(path)
    except ImportError:
        import cv2
        import numpy as np
        arr = (np.random.rand(h, w, 3) * 255).astype("uint8")
        cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def _has_cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _yaml_safe_load(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _check_yaml_or_skip(path: Path):
    try:
        _yaml_safe_load(path)
    except ImportError:
        pytest.skip("PyYAML not installed")


@pytest.mark.skipif(not HAS_PYIQA, reason="pyiqa not installed")
@pytest.mark.skipif(not VENV_PYTHON.exists(), reason="local venv missing")
def test_smoke_flag_runs_fast(tmp_path):
    """
    Run scripts/evaluate_quality.py with --smoke 2 on 2 dummy images and
    confirm it exits 0 and produces a CSV with at least 2 data rows.
    """
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    _make_dummy_image(img_dir / "a.png")
    _make_dummy_image(img_dir / "b.png")

    out_csv = tmp_path / "report.csv"

    cmd = [
        str(VENV_PYTHON),
        str(SCRIPT),
        "--input", str(img_dir),
        "--output", str(out_csv),
        "--smoke", "2",
        "--metrics", "clipiqa", "niqe",
        "--device", "cuda" if _has_cuda() else "cpu",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,  # 10 min; first-run pyiqa download can be slow
        cwd=str(PROJECT_ROOT),
    )

    assert result.returncode == 0, (
        f"evaluate_quality.py failed (rc={result.returncode})\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert out_csv.exists(), f"Expected CSV at {out_csv}, got stdout:\n{result.stdout}"

    # Read CSV and assert at least 2 data rows (one per image, plus the AVERAGE row).
    with out_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # 2 image rows + 1 AVERAGE row = 3 total
    assert len(rows) >= 2, f"Expected >= 2 rows, got {len(rows)}: {rows}"
    filename_rows = [r for r in rows if r.get("filename") not in ("AVERAGE", "", None)]
    assert len(filename_rows) == 2, (
        f"Expected 2 image rows, got {len(filename_rows)}: {filename_rows}"
    )


@pytest.mark.skipif(not HAS_PYIQA, reason="pyiqa not installed")
@pytest.mark.skipif(not VENV_PYTHON.exists(), reason="local venv missing")
def test_csv_columns_include_all_metrics(tmp_path):
    """
    Verify the output CSV header contains the requested metric columns
    in addition to 'filename'.
    """
    img_dir = tmp_path / "imgs2"
    img_dir.mkdir()
    _make_dummy_image(img_dir / "c.png")

    out_csv = tmp_path / "report2.csv"

    cmd = [
        str(VENV_PYTHON),
        str(SCRIPT),
        "--input", str(img_dir),
        "--output", str(out_csv),
        "--smoke", "1",
        "--metrics", "clipiqa", "niqe", "topiq_nr",
        "--device", "cuda" if _has_cuda() else "cpu",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0, (
        f"evaluate_quality.py failed (rc={result.returncode})\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    with out_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        # Strip BOM if present (Windows).
        header = [h.lstrip("\ufeff") for h in header]

    assert "filename" in header
    for metric in ("clipiqa", "niqe", "topiq_nr"):
        assert metric in header, (
            f"Expected column '{metric}' in CSV header, got: {header}"
        )


def test_script_compiles():
    """verify the script is syntactically valid (py_compile gate)."""
    import py_compile
    py_compile.compile(str(SCRIPT), doraise=True)


def test_metrics_module_compiles():
    """verify src/utils/metrics.py is syntactically valid (py_compile gate)."""
    import py_compile
    metrics_py = PROJECT_ROOT / "src" / "utils" / "metrics.py"
    py_compile.compile(str(metrics_py), doraise=True)
