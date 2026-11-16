"""Regenerate the GUI's bundled model code from src/anime_sr/models.

The GUI wheel must not import anime_sr. These files are copied with the
import rewrites listed in REWRITES; the GUI test suite compares the vendored
bytes against this transform, so upstream drift fails CI until re-synced.

Closure note (Task 12): the brief measured seven torch-only model files.
Hunting every anime_sr reference in the GUI package found one extra lazy
import: pipeline/backends.py `_tta_forward` imports anime_sr.inference.tta,
which is also torch-only, so inference/tta.py is part of the bundled
closure (8 files). It mirrors the upstream path under _models/inference/.

Tree rule (Task 12b): the vendored tree is generated AND exact. sync_tree()
prunes any file under the destination directory that the generator does not
own. The one hand-authored exception is _models/NOTICE.md (upstream
attribution documentation), which is preserved. tests/test_vendored_models.py
asserts the exact file set and that a planted stale file is detected/pruned.

Run: python scripts/sync_vendored_models.py
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "anime_sr" / "models"
DST = REPO / "apps" / "anime_upscaler_gui" / "anime_upscaler_gui" / "_models"

FILES = {
    "students/__init__.py": "students/__init__.py",
    "students/mambair.py": "students/mambair.py",
    "span/neosr_span.py": "span/neosr_span.py",
    "span/mambair_v2.py": "span/mambair_v2.py",
    "eranet.py": "eranet.py",
    "msrsr.py": "msrsr.py",
    "teachers/spandrel.py": "teachers/spandrel.py",
    # Lazy TTA import in pipeline/backends.py: torch-only, lives in the
    # sibling inference package, so the source is reached by traversal.
    "inference/tta.py": "../inference/tta.py",
}

REWRITES = [
    ("from anime_sr.models.span.mambair_v2", "from .mambair_v2"),
    ("from anime_sr.models.span.neosr_span", "from ..span.neosr_span"),
]

# Hand-authored attribution documentation; never generated, never pruned.
NOTICE = Path("NOTICE.md")

# Every package directory in the vendored tree. `students/__init__.py` is a
# real generated module (it is in FILES), so only the remaining package
# directories get an empty __init__.py.
PACKAGES = ("", "students", "span", "teachers", "inference")


def transform(text: str) -> str:
    for old, new in REWRITES:
        text = text.replace(old, new)
    return text


def empty_init_paths() -> list[Path]:
    """Package inits the generator writes as empty files (FILES excluded)."""
    paths = []
    for pkg in PACKAGES:
        rel = Path(pkg) / "__init__.py" if pkg else Path("__init__.py")
        if rel.as_posix() not in FILES:
            paths.append(rel)
    return paths


def expected_paths() -> set[Path]:
    """Every path (relative to DST) that the generated tree must contain."""
    paths = {Path(rel) for rel in FILES}
    paths.update(empty_init_paths())
    paths.add(NOTICE)
    return paths


def tree_problems(dst: Path = DST) -> list[str]:
    """Describe any drift between *dst* and the expected generated tree."""
    expected = expected_paths()
    if not dst.is_dir():
        return [f"missing generated tree: {dst}"]
    actual = {
        path.relative_to(dst)
        for path in dst.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    problems = [f"missing: {rel.as_posix()}" for rel in sorted(expected - actual)]
    problems += [f"unexpected: {rel.as_posix()}" for rel in sorted(actual - expected)]
    for rel in empty_init_paths():
        if rel in actual and (dst / rel).read_text(encoding="utf-8") != "":
            problems.append(f"hand-edited package init: {rel.as_posix()}")
    return problems


def prune(dst: Path = DST) -> list[Path]:
    """Delete files under *dst* that the generated tree does not own."""
    if not dst.is_dir():
        return []
    keep = expected_paths()
    removed: list[Path] = []
    for path in sorted(dst.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(dst)
        if rel not in keep:
            path.unlink()
            removed.append(rel)
    for directory in sorted(
        (p for p in dst.rglob("*") if p.is_dir() and "__pycache__" not in p.parts),
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        try:
            directory.rmdir()
        except OSError:
            pass
    return removed


def write_tree(dst: Path = DST, src: Path = SRC) -> None:
    """Regenerate every owned file (the empty package inits included)."""
    for rel in empty_init_paths():
        init = dst / rel
        init.parent.mkdir(parents=True, exist_ok=True)
        init.write_text("", encoding="utf-8", newline="\n")
    for dst_rel, src_rel in FILES.items():
        target = dst / dst_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            transform((src / src_rel).read_text(encoding="utf-8")),
            encoding="utf-8",
            newline="\n",
        )


def sync_tree(dst: Path = DST, src: Path = SRC) -> list[Path]:
    """Prune stale files, then write the generated tree; return pruned paths."""
    dst.mkdir(parents=True, exist_ok=True)
    removed = prune(dst)
    write_tree(dst, src)
    return removed


def main() -> int:
    removed = sync_tree()
    for rel in removed:
        print(f"pruned stale file: {rel.as_posix()}")
    problems = tree_problems()
    if problems:
        print("vendored tree is still inconsistent:")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print(f"synced {len(FILES)} model files into {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())