"""Entry point: `python -m anime_upscaler_gui [--no-splash]`.

Single Tk root pattern: the `UpscaleGUI` IS the root, just withdrawn until
the splash finishes. This avoids the "can't use pyimage as iconphoto" bug
where images created in a hidden boot root don't transfer to a second Tk.
"""
import argparse
import sys


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
    args = parser.parse_args()

    try:
        return _run_without_splash() if args.no_splash else _run_with_splash()
    except Exception as e:
        print(f"[FATAL] cannot start GUI: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
