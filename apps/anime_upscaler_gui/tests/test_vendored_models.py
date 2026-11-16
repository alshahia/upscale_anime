"""The bundled model tree must be generated, in sync, and self-contained."""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
GUI = Path(__file__).resolve().parents[1] / "anime_upscaler_gui"
SCRIPT = REPO / "scripts" / "sync_vendored_models.py"

spec = importlib.util.spec_from_file_location("sync_vendored_models", SCRIPT)
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


def test_vendored_models_match_upstream_transform():
    problems = []
    for dst_rel, src_rel in sync.FILES.items():
        expected = sync.transform((sync.SRC / src_rel).read_text(encoding="utf-8"))
        actual = (sync.DST / dst_rel).read_text(encoding="utf-8")
        if expected != actual:
            problems.append(f"{dst_rel} differs from src/anime_sr/models/{src_rel}")
    assert not problems, (
        "vendored models drifted; run python scripts/sync_vendored_models.py\n"
        + "\n".join(problems))


def test_gui_package_does_not_import_anime_sr():
    offenders = []
    for path in GUI.rglob("*.py"):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(("import anime_sr", "from anime_sr")):
                offenders.append(f"{path.relative_to(GUI)}:{number}: {stripped}")
    assert not offenders, "GUI must be standalone:\n" + "\n".join(offenders)
