"""The bundled model tree must be generated, exact, and self-contained."""
from __future__ import annotations

import importlib.util
import shutil
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


def test_vendored_tree_is_exact():
    problems = sync.tree_problems()
    assert not problems, (
        "vendored _models tree drifted from the generated file set; run "
        "python scripts/sync_vendored_models.py\n" + "\n".join(problems))


def test_vendored_notice_attributes_upstream_projects():
    notice_path = sync.DST / sync.NOTICE
    assert notice_path.is_file(), "missing _models/NOTICE.md attribution file"
    notice = notice_path.read_text(encoding="utf-8")
    for upstream in ("MambaIR", "neosr", "ERANet", "MSRSWVSR", "spandrel", "RFDN"):
        assert upstream in notice, f"NOTICE.md does not mention upstream project: {upstream}"
    assert "licensing and attribution review" in notice.lower(), (
        "NOTICE.md must state that licensing/redistribution review is required")
    assert sync.NOTICE in sync.expected_paths()


def test_planted_stale_file_fails_exact_tree_check():
    planted = sync.DST / "_stale_probe.py"
    planted.write_text("raise SystemExit('stale module')\n", encoding="utf-8")
    try:
        problems = sync.tree_problems()
    finally:
        planted.unlink(missing_ok=True)
    assert any("_stale_probe.py" in problem for problem in problems), (
        "exact-tree guard did not detect a planted stale vendored module")


def test_planted_hand_edited_init_fails_exact_tree_check():
    init = sync.DST / "span" / "__init__.py"
    original = init.read_text(encoding="utf-8")
    init.write_text(original + "# hand edit\n", encoding="utf-8")
    try:
        problems = sync.tree_problems()
    finally:
        init.write_text(original, encoding="utf-8")
    assert any("span/__init__.py" in problem for problem in problems), (
        "exact-tree guard did not detect a hand-edited package init")


def test_sync_tree_prunes_stale_files_and_regenerates(tmp_path):
    dst = tmp_path / "_models"
    dst.mkdir()
    shutil.copyfile(sync.DST / sync.NOTICE, dst / sync.NOTICE)
    stale = dst / "stale_probe.py"
    stale.write_text("raise SystemExit('stale module')\n", encoding="utf-8")
    hand_edited = dst / "eranet.py"
    hand_edited.write_text("# hand edit\n", encoding="utf-8")
    hand_edited_init = dst / "span" / "__init__.py"
    hand_edited_init.parent.mkdir(parents=True)
    hand_edited_init.write_text("# hand edit\n", encoding="utf-8")

    removed = sync.sync_tree(dst=dst)

    assert Path("stale_probe.py") in removed
    assert not stale.exists(), "sync did not prune the stale vendored file"
    assert (dst / "eranet.py").read_text(encoding="utf-8") == sync.transform(
        (sync.SRC / sync.FILES["eranet.py"]).read_text(encoding="utf-8"))
    assert hand_edited_init.read_text(encoding="utf-8") == ""
    assert sync.tree_problems(dst) == [], "synced tree is not exact"


def test_gui_package_does_not_import_anime_sr():
    offenders = []
    for path in GUI.rglob("*.py"):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(("import anime_sr", "from anime_sr")):
                offenders.append(f"{path.relative_to(GUI)}:{number}: {stripped}")
    assert not offenders, "GUI must be standalone:\n" + "\n".join(offenders)
