"""Logging configuration: a `logging.Logger` for the whole app + a rotating file handler.

Call `setup_logging(logs_dir)` once at app startup. The returned logger
is the package-level logger for `apps.anime_upscaler_gui`.
"""
import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional


# Package-level logger name. Modules use:
#   logger = logging.getLogger("anime_upscaler_gui")
LOGGER_NAME = "anime_upscaler_gui"

_initialized = False
_file_handler: Optional[logging.handlers.RotatingFileHandler] = None
_panel_handler: Optional["_LogPanelHandler"] = None  # forward ref to avoid circular import


def setup_logging(logs_dir: Path, *, level: int = logging.INFO,
                  max_bytes: int = 1_000_000, backup_count: int = 3) -> logging.Logger:
    """Configure the package logger.

    - StreamHandler: WARNING+ to stderr (so we don't spam the user's terminal).
    - RotatingFileHandler: INFO+ to logs_dir/app.log (preserved across restarts).
    - _LogPanelHandler: attached only after the GUI creates its log panel
      (see `attach_panel`).
    """
    global _initialized, _file_handler
    if _initialized:
        return logging.getLogger(LOGGER_NAME)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "app.log"

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False  # don't bubble to the root logger

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    # File: INFO+, rotating.
    _file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    _file_handler.setLevel(logging.INFO)
    _file_handler.setFormatter(fmt)
    logger.addHandler(_file_handler)

    # Console: WARNING+ (so a normal session isn't noisy).
    stream = logging.StreamHandler()
    stream.setLevel(logging.WARNING)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    _initialized = True
    logger.info("logging initialized; file=%s", log_path)
    return logger


def attach_panel(handler) -> None:
    """Attach a `_LogPanelHandler` so log records also show in the GUI panel."""
    global _panel_handler
    _panel_handler = handler
    logger = logging.getLogger(LOGGER_NAME)
    logger.addHandler(handler)
