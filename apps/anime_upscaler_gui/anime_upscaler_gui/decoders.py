"""Video and image readers (cv2 + PyAV) with optional async prefetch."""
import queue
import threading
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2

try:
    import av
    _HAS_PYAV = True
except Exception:
    _HAS_PYAV = False


# ============================================================================ #
# Sync readers
# ============================================================================ #
class _Cv2Reader:
    """Yields RGB uint8 (H, W, 3) frames + metadata."""

    def __init__(self, path):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 24.0
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __iter__(self):
        while True:
            ok, bgr = self.cap.read()
            if not ok:
                break
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass

    def read_one(self, frame_idx: int):
        """Decode a single frame at the given index (for video preview scrubbing)."""
        cap = cv2.VideoCapture(self.path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
        ok, bgr = cap.read()
        cap.release()
        if not ok:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class _PyAvReader:
    """PyAV reader; tries NVDEC hwaccel but falls back silently."""

    def __init__(self, path):
        if not _HAS_PYAV:
            raise RuntimeError("PyAV not installed")
        self.path = str(path)
        self.container = av.open(self.path)
        self.stream = self.container.streams.video[0]
        try:
            self.stream.codec_context.options = {"hwaccel": "cuda"}
        except Exception:
            pass
        self.fps = float(self.stream.average_rate) or float(self.stream.base_rate) or 24.0
        self.total = self.stream.frames or 0
        self.w = self.stream.codec_context.width
        self.h = self.stream.codec_context.height

    def __iter__(self):
        for frame in self.container.decode(self.stream):
            yield frame.to_ndarray(format="rgb24")

    def release(self):
        try:
            self.container.close()
        except Exception:
            pass


class _SkipFirstFrames:
    """Reader wrapper: yields frames after skipping the first N (cut start).

    Works for both cv2 and PyAV sync readers; PyAV has no reliable absolute
    seek on all containers, so consuming-and-dropping is the portable way to
    reach a cut-start offset.
    """

    def __init__(self, reader, skip_frames: int):
        self._reader = reader
        self._skip = max(0, int(skip_frames))

    def __iter__(self):
        it = iter(self._reader)
        for _ in range(self._skip):
            next(it, None)
        for frame in it:
            yield frame

    def release(self):
        self._reader.release()


# ============================================================================ #
# Async reader (background-thread producer, bounded queue consumer)
# ============================================================================ #
_SENTINEL = object()


class _AsyncReader:
    """Wrap a sync reader so frame decoding happens on a background thread.

    Producer is a daemon thread that pushes into a bounded queue. The consumer
    (main loop) pulls frames. Queue maxsize provides natural backpressure when
    the GPU is slower than the decoder; otherwise the queue fills ahead of the
    consumer and decode overlaps with model compute.
    """

    def __init__(self, sync_reader, prefetch: int = 8):
        self._sync = sync_reader
        self._q: "queue.Queue" = queue.Queue(maxsize=prefetch)

    def start(self, max_frames: int = 0):
        def _run():
            n = 0
            try:
                for frame in self._sync:
                    self._q.put(frame)
                    n += 1
                    if max_frames and n >= max_frames:
                        break
            except Exception as e:
                print(f"  [AsyncReader] producer error: {e}")
            finally:
                self._q.put(_SENTINEL)
        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def __iter__(self):
        while True:
            item = self._q.get()
            if item is _SENTINEL:
                return
            yield item

    def release(self):
        self._sync.release()


# ============================================================================ #
# Image helpers
# ============================================================================ #
def load_image_rgb(path) -> "np.ndarray":
    """Load an image as RGB uint8 (H, W, 3). Returns None on failure."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_image_rgb(rgb: "np.ndarray", path) -> bool:
    """Save an RGB uint8 (H, W, 3) array as BGR via cv2. Returns True on success."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bool(cv2.imwrite(str(p), bgr))


def is_image_path(path) -> bool:
    return Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def is_video_path(path) -> bool:
    return Path(path).suffix.lower() in {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}


def open_reader(path, decode: str = "cv2"):
    """Factory: returns a sync reader for the chosen backend."""
    if decode == "pyav":
        return _PyAvReader(path)
    return _Cv2Reader(path)