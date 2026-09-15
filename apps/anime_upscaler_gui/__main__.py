"""Outer entry stub for `python -m apps.anime_upscaler_gui`.

The actual GUI package lives one level deeper at `anime_upscaler_gui/`.
We import its `__main__` by file path via importlib so we don't depend
on the inner directory being on sys.path (Python's `-m` machinery adds
the outer package's directories, which doesn't help find the inner one).
"""
import importlib.util
import sys
from pathlib import Path


def main() -> int:
    here = Path(__file__).resolve().parent
    inner_dir = here / "anime_upscaler_gui"
    inner_main = inner_dir / "__main__.py"
    inner_init = inner_dir / "__init__.py"
    if not inner_main.exists() or not inner_init.exists():
        print(f"[FATAL] inner package not found: {inner_dir}", file=sys.stderr)
        return 1

    pkg_spec = importlib.util.spec_from_file_location(
        "anime_upscaler_gui", str(inner_init),
        submodule_search_locations=[str(inner_dir)],
    )
    pkg = importlib.util.module_from_spec(pkg_spec)
    sys.modules["anime_upscaler_gui"] = pkg
    pkg_spec.loader.exec_module(pkg)

    mod_spec = importlib.util.spec_from_file_location(
        "anime_upscaler_gui.__main__", str(inner_main),
    )
    mod = importlib.util.module_from_spec(mod_spec)
    sys.modules["anime_upscaler_gui.__main__"] = mod
    mod_spec.loader.exec_module(mod)
    return mod.main()


if __name__ == "__main__":
    sys.exit(main())
