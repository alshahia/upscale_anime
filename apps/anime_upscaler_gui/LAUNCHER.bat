@echo off
REM Double-click launcher for Windows.
REM The package lives at apps/anime_upscaler_gui/ but Python needs the
REM REPO ROOT on sys.path so it can import the `apps.` namespace.
REM This script cd's two levels up, then uses the same local .venv.
REM Pass --no-splash (or any module flags) through: LAUNCHER.bat --no-splash
setlocal
set HERE=%~dp0
set ROOT=%HERE%..\..
for %%I in ("%ROOT%") do set ROOT=%%~fI
cd /d "%ROOT%"
if exist "%HERE%\.venv\Scripts\python.exe" (
    "%HERE%\.venv\Scripts\python.exe" -m apps.anime_upscaler_gui %*
) else (
    python -m apps.anime_upscaler_gui %*
)
endlocal
