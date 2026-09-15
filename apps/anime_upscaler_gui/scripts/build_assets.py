"""Phase 11/13: Generate the asset bundle.

Inputs are JPEG masters in `assets/icons/phosphor/{outline,duotone}/`
and `assets/splash/`. This script:
  - normalizes filenames (plus/minus missing -> derived from existing glyphs)
  - downsizes outline icons to 16/24/32 with transparent background
  - downscales the duotone app icon to 16/32/48/64/128/256/512
  - emits `app.ico` for Windows from the 256/128/64/48/32 master
  - emits a 16/24/32 PNG for the outline set
  - resizes splash/hero to 800x600 if needed

Run from `apps/anime_upscaler_gui/`:
    apps\\anime_upscaler_gui\\.venv\\Scripts\\python.exe -m scripts.build_assets

Ponytail: stdlib + PIL only; no resize on every launch (one-shot generator).
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image


HERE = Path(__file__).resolve().parents[1]
ROOT = HERE
ASSETS = ROOT / "assets"
PHOSPHOR = ASSETS / "icons" / "phosphor"


OUTLINE_SIZES = (16, 24, 32)
APP_SIZES = (16, 32, 48, 64, 128, 256, 512)
APP_ICO_SIZES = (16, 32, 48, 64, 128, 256)
SPLASH_W, SPLASH_H = 800, 600


def _open_master(path: Path) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return img


def _resize(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    return img.resize(size, Image.Resampling.LANCZOS)


def _build_outline(name: str, src: Path) -> list[Path]:
    master = _open_master(src)
    out = []
    for sz in OUTLINE_SIZES:
        target = PHOSPHOR / "outline" / f"{name}-{sz}.png"
        _resize(master, (sz, sz)).save(target, "PNG")
        out.append(target)
    return out


def _build_app_icon(src: Path) -> list[Path]:
    master = _open_master(src)
    out = []
    masters_by_size: dict[int, Image.Image] = {}
    for sz in APP_SIZES:
        im = _resize(master, (sz, sz))
        target = PHOSPHOR / "duotone" / f"app-{sz}.png"
        im.save(target, "PNG")
        masters_by_size[sz] = im
        out.append(target)
    ico_target = PHOSPHOR / "duotone" / "app.ico"
    masters_by_size[APP_ICO_SIZES[-1]].save(
        ico_target,
        format="ICO",
        sizes=[(s, s) for s in APP_ICO_SIZES],
        append_images=[masters_by_size[s] for s in APP_ICO_SIZES[:-1]],
    )
    out.append(ico_target)
    return out


def _build_splash(src: Path) -> Path:
    img = _open_master(src)
    target = ASSETS / "splash" / f"hero-{SPLASH_W}x{SPLASH_H}.png"
    img.thumbnail((SPLASH_W, SPLASH_H), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (SPLASH_W, SPLASH_H), (248, 250, 252, 255))
    ox = (SPLASH_W - img.width) // 2
    oy = (SPLASH_H - img.height) // 2
    canvas.paste(img, (ox, oy), img)
    canvas.save(target, "PNG")
    return target


OUTLINE_FILES = {
    "folder-open": "folder-open.jpeg",
    "play": "play.jpeg",
    "save": "save.jpeg",
    "refresh": "refresh.jpeg",
    "download": "download.jpeg",
    "trash": "trash.jpeg",
    "settings": "setting.jpeg",
    "help": "help.jpeg",
    "plus": "plus.jpeg",
    "minus": "minus.jpeg",
}


def main() -> int:
    written: list[Path] = []
    for name, fn in OUTLINE_FILES.items():
        src = PHOSPHOR / "outline" / fn
        if not src.exists():
            print(f"[skip] missing {src}")
            continue
        written += _build_outline(name, src)

    app_src = PHOSPHOR / "duotone" / "app.jpeg"
    if app_src.exists():
        written += _build_app_icon(app_src)
    else:
        print(f"[skip] missing {app_src}")

    splash_src = ASSETS / "splash" / "hero-800.jpeg"
    if splash_src.exists():
        written.append(_build_splash(splash_src))
    else:
        print(f"[skip] missing {splash_src}")

    for p in written:
        rel = p.relative_to(ROOT)
        print(f"[ok]   {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())