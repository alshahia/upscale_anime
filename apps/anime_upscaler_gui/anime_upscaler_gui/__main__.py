"""Entry point: `python -m anime_upscaler_gui [--no-splash] [--self-check]`.

Single Tk root pattern: the `UpscaleGUI` IS the root, just withdrawn until
the splash finishes. This avoids the "can't use pyimage as iconphoto" bug
where images created in a hidden boot root don't transfer to a second Tk.
"""
import argparse
import json
import shutil
import sys


def _self_check() -> int:
    """Print an environment report as JSON and exit without starting Tk.

    This is the clean-room smoke test for the shipped console script: it must
    keep working on a machine where only the wheel (no repo checkout, no
    anime_sr) is installed. Exit 0 needs both ffmpeg and ffprobe on PATH.
    """
    from . import __version__
    from .arch_registry import registered_kinds

    report = {
        "version": __version__,
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "ffprobe": shutil.which("ffprobe") is not None,
        "architectures": sorted(registered_kinds()),
    }
    print(json.dumps(report, indent=2))
    return 0 if report["ffmpeg"] and report["ffprobe"] else 1


def _run_with_splash() -> int:
    from .app import UpscaleGUI
    from .widgets.splash import Splash

    app = UpscaleGUI()
    app.withdraw()

    def _reveal():
        try:
            app.deiconify()
        except Exception:
            pass

    Splash(app, display_ms=1500, fade_ms=200, on_done=_reveal)
    app.mainloop()
    return 0


def _run_without_splash() -> int:
    from .app import UpscaleGUI
    UpscaleGUI().mainloop()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="anime-upscaler-gui")
    parser.add_argument("--no-splash", action="store_true",
                        help="Skip the splash screen on launch.")
    parser.add_argument("--self-check", action="store_true",
                        help="Print an environment report as JSON and exit.")
    args = parser.parse_args()

    if args.self_check:
        return _self_check()

    try:
        return _run_without_splash() if args.no_splash else _run_with_splash()
    except Exception as e:
        print(f"[FATAL] cannot start GUI: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
