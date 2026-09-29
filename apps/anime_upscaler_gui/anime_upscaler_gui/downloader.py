"""Model downloader: registry preset + URL + local import with progress polling."""
import shutil
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from .registry import _ModelRegistry


class _DownloadError(Exception):
    pass


class ModelDownloader:
    """Thin wrapper around ModelRegistry.download() with progress callbacks.

    The GUI polls this from a thread and updates a ttk.Progressbar.
    """

    def __init__(self, registry: _ModelRegistry):
        self.registry = registry

    def download_url(self, url: str, dest_filename: str,
                     on_progress: Optional[Callable[[int, int], None]] = None,
                     on_done: Optional[Callable[[Path], None]] = None,
                     on_error: Optional[Callable[[Exception], None]] = None,
                     expected_sha256: str = None,
                     cancel_event: Optional[threading.Event] = None) -> threading.Thread:
        def _run():
            try:
                self.registry.download(url, dest_filename, on_progress=on_progress,
                                       expected_sha256=expected_sha256,
                                       cancel_event=cancel_event)
                if on_done:
                    on_done(self.registry.pretrained_dir / dest_filename)
            except Exception as e:
                if on_error:
                    on_error(e)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t

    def import_local(self, src: Path,
                     on_done: Optional[Callable[[object], None]] = None,
                     on_error: Optional[Callable[[Exception], None]] = None) -> threading.Thread:
        def _run():
            try:
                m = self.registry.import_local(src)
                if on_done:
                    on_done(m)
            except Exception as e:
                if on_error:
                    on_error(e)
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t