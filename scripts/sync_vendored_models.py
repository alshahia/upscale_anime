"""Regenerate the GUI's bundled model code from src/anime_sr/models.

The GUI wheel must not import anime_sr. These files are copied with the
import rewrites listed in REWRITES; the GUI test suite compares the vendored
bytes against this transform, so upstream drift fails CI until re-synced.

Closure note (Task 12): the brief measured seven torch-only model files.
Hunting every anime_sr reference in the GUI package found one extra lazy
import: pipeline/backends.py `_tta_forward` imports anime_sr.inference.tta,
which is also torch-only, so inference/tta.py is part of the bundled
closure (8 files). It mirrors the upstream path under _models/inference/.

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


def transform(text: str) -> str:
    for old, new in REWRITES:
        text = text.replace(old, new)
    return text


def main() -> int:
    for rel in FILES:
        (DST / rel).parent.mkdir(parents=True, exist_ok=True)
    for pkg in ("", "students", "span", "teachers", "inference"):
        init = (DST / pkg / "__init__.py") if pkg else (DST / "__init__.py")
        init.parent.mkdir(parents=True, exist_ok=True)
        if not init.exists():
            init.write_text("", encoding="utf-8", newline="\n")
    for dst_rel, src_rel in FILES.items():
        src = SRC / src_rel
        text = transform(src.read_text(encoding="utf-8"))
        (DST / dst_rel).write_text(text, encoding="utf-8", newline="\n")
    print(f"synced {len(FILES)} model files into {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
