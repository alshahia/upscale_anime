#!/usr/bin/env bash
# Double-click launcher for Linux/macOS.
# The package lives at apps/anime_upscaler_gui/ but Python needs the
# REPO ROOT on sys.path so it can import the `apps.` namespace.
# This script cd's two levels up, then uses the same local .venv.
# Pass --no-splash (or any module flags) through: ./LAUNCHER.sh --no-splash
HERE="$(cd "$(dirname "$(readlink -f "$0" 2>/dev/null || echo "$0")")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"
if [ -x "$HERE/.venv/bin/python" ]; then
    "$HERE/.venv/bin/python" -m apps.anime_upscaler_gui "$@"
else
    python3 -m apps.anime_upscaler_gui "$@"
fi
