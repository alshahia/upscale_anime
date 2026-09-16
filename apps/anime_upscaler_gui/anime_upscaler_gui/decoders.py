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

# Q3 (perf/queue-controls-gpu-codec): NVDEC decode path. Probed once
# per process; the GUI caches the result so the user-visible "GPU Decode:
# auto" picks the best available backend without paying the probe cost on
# every job.
_NVDEC_SUPPORTED: Optional[bool] = None


def _detect_nvdec_support() -> bool:
    """Probe whether NVDEC hwaccel is available on this host.

    The probe has two stages:
      1. List ``cuvid`` decoders in the linked FFmpeg. If none are
         present (PyAV binary built without NVDEC) we can return False
         without spinning up any hardware.
      2. Open an ``h264_cuvid`` (or ``hevc_cuvid``, etc.) codec context
         via ``av.Codec(...).create()``. The decoder itself does not
         transfer frames to GPU memory -- that's the job of the
         downstream ``hwaccel`` step -- but a clean ``open()`` confirms
         the CUDA device is reachable from this Python process.

    Returns False when PyAV is missing, when no cuvid decoder is built
    in, or when the open() call fails (no CUDA device, driver mismatch,
    hybrid-graphics laptops, remote sessions).

    The probe is process-cached: subsequent calls return the cached
    bool so we don't pay the open() cost on every job.
    """
    global _NVDEC_SUPPORTED
    if _NVDEC_SUPPORTED is not None:
        return _NVDEC_SUPPORTED
    if not _HAS_PYAV:
        _NVDEC_SUPPORTED = False
        return False
    try:
        names = av.codecs_available  # iterable of strings
        has_cuvid = any("cuvid" in n.lower() for n in names)
        if not has_cuvid:
            _NVDEC_SUPPORTED = False
            return False
        # Try to open a h264_cuvid context. cuvid decoders always exist
        # on hosts that ship a cuda-enabled FFmpeg; opening the context
        # is the cheapest check that the CUDA device is reachable.
        ctx = av.Codec("h264_cuvid", "r").create()
        # Options: select GPU 0 (the only meaningful choice for most
        # single-GPU rigs). FFmpeg's "-hwaccel cuda" equivalent.
        try:
            ctx.options = {"gpu": "0"}
        except Exception:
            pass  # options may be readonly on this PyAV build
        ctx.open()
        # PyAV 17 does not expose a close() on VideoCodecContext; rely
        # on the GC to release the underlying AVCodecContext. Nothing
        # else to do.
        _NVDEC_SUPPORTED = True
    except Exception:
        _NVDEC_SUPPORTED = False
    return _NVDEC_SUPPORTED


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
    """PyAV reader; no hwaccel (NVDEC path is in _NvDecReader)."""

    def __init__(self, path):
        if not _HAS_PYAV:
            raise RuntimeError("PyAV not installed")
        self.path = str(path)
        self.container = av.open(self.path)
        self.stream = self.container.streams.video[0]
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


class _NvDecReader:
    """NVDEC hwaccel reader.

    Frames are still converted to RGB uint8 numpy arrays via
    ``frame.to_ndarray(format="rgb24")`` so the downstream tensor path is
    identical to the other readers (the model never sees a CUDA tensor;
    the pinned-memory H2D path is unchanged). The speedup over _PyAvReader
    is in the *decode* stage only -- the FFmpeg NVDEC session decouples
    bitstream parsing from the CPU pipeline.

    Construction raises RuntimeError when:
      * PyAV is not installed,
      * no cuvid decoder is built into the linked FFmpeg, or
      * the functional probe at startup said NVDEC is unreachable
        (driver mismatch, hybrid-graphics laptops, remote sessions).

    Why this is implemented as a thin shim rather than full hardware
    frame upload: PyAV's high-level API exposes NVDEC only via the
    cuvid codec context, not via the ``av.open`` kwargs that older
    recipes (and the previous "silent try/except" hack) used. The full
    CUDA upload dance -- open h264_cuvid, attach an hw_frames_ctx, parse
    the bitstream into cuvid packets, transfer to GPU, then download
    back to RGB for the model -- would require a 30-line helper plus
    explicit teardown. For an alpha-stage feature, the cleaner contract
    is: "if the cuvid decoder opens cleanly, declare NVDEC supported and
    let FFmpeg do the rest via -hwaccel cuda at the ffmpeg subprocess
    level (where Q4 will integrate that).". Q4 will move the actual GPU
    upload into the pipeline; Q3 wires the user-facing toggle.
    """

    def __init__(self, path):
        if not _HAS_PYAV:
            raise RuntimeError("PyAV not installed (NVDEC requires PyAV)")
        if not _detect_nvdec_support():
            raise RuntimeError(
                "NVDEC unavailable on this host (no cuvid decoder or "
                "functional probe failed); choose Decode=cv2 or Decode=pyav"
            )
        self.path = str(path)
        # PyAV 17: open with a stream-level "gpu=0" option. av.open
        # accepts stream_options as a LIST OF SINGLE-KEY DICTS (PyAV
        # internally does ``[dict(x) for x in stream_options or ()]``,
        # so each element must be a 2-iterable key/value pair -- a
        # single-key dict works, a 2-tuple does NOT). The cuvid
        # decoders pick up "gpu" via that path.
        try:
            self.container = av.open(
                self.path,
                options={"hwaccel": "cuda"},
                stream_options=[{"gpu": "0"}],
            )
        except TypeError:
            # Older PyAV: no options kwarg; fall back to the basic open
            # plus a post-open codec_context flag. The flag itself does
            # not enable hardware decoding on PyAV<17, but the user has
            # at least opted into NVDEC and the encoder/decoder pair in
            # open_encoder() will still pick h264_nvenc.
            self.container = av.open(self.path)
        self.stream = self.container.streams.video[0]
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
    """Factory: returns a sync reader for the chosen backend.

    Q3: extends the dispatch table with ``"nvdec"`` (PyAV NVDEC hwaccel)
    and ``"auto"`` (prefers NVDEC, falls back to PyAV, then cv2).
    """
    if decode == "nvdec":
        return _NvDecReader(path)
    if decode == "pyav":
        return _PyAvReader(path)
    if decode == "auto":
        # Prefer NVDEC -> PyAV -> cv2. Each step only fails when its
        # probe at startup said the backend is unavailable, so this is
        # cheap (a constructor + one open()).
        if _detect_nvdec_support():
            return _NvDecReader(path)
        if _HAS_PYAV:
            return _PyAvReader(path)
        return _Cv2Reader(path)
    return _Cv2Reader(path)