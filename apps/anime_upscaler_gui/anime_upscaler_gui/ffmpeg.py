"""ffmpeg helpers: pipe raw frames, stream-copy cut extraction, duration probe.

Supports NVENC (NVIDIA hardware H.264 encoder) via the `use_nvenc` flag.
Falls back to libx264 when NVENC is requested but unavailable.
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


# --- NVENC support --------------------------------------------------------

_NVENC_SUPPORTED: Optional[bool] = None  # None = not probed yet

# --- libx264 memory guard (used only when NVENC is unavailable) ------------
# x264 defaults spawn one worker thread per CPU core and keep a deep lookahead
# queue per thread. At 4K on many-core machines that adds up to gigabytes of
# committed RAM, full CPU and disk saturation without any visible quality
# benefit at these sizes (CRF still governs quality).
_X264_MAX_ENCODER_THREADS = 8     # encoder worker threads
_X264_RC_LOOKAHEAD = 20           # buffered lookahead frames (default ~40)
_X264_LOOKAHEAD_THREADS = 2       # lookahead analysis threads (default = cores)
_NVENC_FALLBACK_WARNED = False
_NVENC_PROBE_TIMEOUT = 15  # seconds (functional encode probe can take a few seconds)
# How long to watch a freshly spawned h264_nvenc ffmpeg for synchronous init
# failure. A healthy encoder sits alive waiting for stdin frames, so a poll
# that sees an exit code means the encoder never opened. Near-zero cost
# when NVENC works.
_NVENC_INIT_GRACE_S = 4.0


def _detect_nvenc_support() -> bool:
    """Functionally probe ffmpeg for h264_nvenc. Cached after first call.

    A build can list h264_nvenc among its encoders yet fail to open it at
    runtime (driver mismatch, no capable GPU device, laptop hybrid graphics).
    So we actually encode 3 black test frames through h264_nvenc and only
    treat that as "available" when it exits cleanly. Returns False when
    ffmpeg is missing or the probe fails/times out.
    """
    global _NVENC_SUPPORTED
    if _NVENC_SUPPORTED is not None:
        return _NVENC_SUPPORTED
    if not ffmpeg_available():
        _NVENC_SUPPORTED = False
        return False
    try:
        r = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-v", "error",
                # 256x256: NVENC has a minimum supported frame size (~145px);
                # probing at 64x64 falsely reports "no capable devices" on
                # perfectly capable GPUs (e.g. Quadro RTX 4000).
                "-f", "lavfi", "-i", "color=black:s=256x256:d=0.125",
                "-frames:v", "3",
                "-c:v", "h264_nvenc", "-preset", "p1",
                "-rc", "constqp", "-qp", "28",
                "-f", "null", "-",
            ],
            capture_output=True, text=True, timeout=_NVENC_PROBE_TIMEOUT,
        )
        _NVENC_SUPPORTED = (r.returncode == 0)
    except (subprocess.TimeoutExpired, OSError):
        _NVENC_SUPPORTED = False
    return _NVENC_SUPPORTED


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def ffprobe_available() -> bool:
    return shutil.which("ffprobe") is not None


def open_encoder(
    out_path, fps: float, w: int, h: int,
    crf: int = 18, preset: str = "medium",
    use_nvenc: bool = False, nvenc_preset: str = "p1", nvenc_qp: int = 18,
):
    """Spawn ffmpeg with rawvideo on stdin. Returns the Popen or None (no ffmpeg).

    When `use_nvenc=True` AND `h264_nvenc` is present, uses NVIDIA hardware
    encoder with constant-QP rate control (`-rc constqp -qp <nvenc_qp>`).
    Falls back to `libx264` when NVENC is unavailable (warns to stderr once
    per process).
    """
    if not ffmpeg_available():
        return None

    use_hardware = use_nvenc and _detect_nvenc_support()
    if use_nvenc and not use_hardware:
        # warn once per process, then silently fall back to libx264
        global _NVENC_FALLBACK_WARNED
        if not _NVENC_FALLBACK_WARNED:
            print(
                "[ffmpeg] h264_nvenc requested but not available; "
                "falling back to libx264.",
                file=sys.stderr, flush=True,
            )
            _NVENC_FALLBACK_WARNED = True

    if use_hardware:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", f"{fps:.3f}",
            "-i", "pipe:0",
            "-c:v", "h264_nvenc", "-preset", nvenc_preset,
            "-rc", "constqp", "-qp", str(nvenc_qp),
            "-pix_fmt", "yuv420p", str(out_path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        # NVENC session creation is synchronous: if it fails ("No capable
        # devices found" under mid-run conditions, driver hiccup, sessions
        # exhausted), ffmpeg exits on its own before any frame is written.
        # Detect that here and fall back to libx264 instead of crashing the
        # pipeline at the first stdin.write with OSError [Errno 22].
        t0 = time.time()
        dead = False
        while time.time() - t0 < _NVENC_INIT_GRACE_S:
            if proc.poll() is not None:
                print(
                    f"[ffmpeg] h264_nvenc exited during init "
                    f"(code {proc.poll()}); falling back to libx264.",
                    file=sys.stderr, flush=True,
                )
                dead = True
                break
            time.sleep(0.02)
        if not dead:
            return proc
        # fall through to libx264
    cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", f"{fps:.3f}",
            "-i", "pipe:0",
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            # RAM guard: x264 defaults scale thread count with CPU cores and
            # buffer rc-lookahead frames per thread. On many-core machines at
            # 4K that allocates gigabytes of committed RAM and pegs every core.
            # Bounding threads and lookahead keeps encoder memory flat without
            # visibly changing quality (CRF still governs rate).
            "-x264-params", f"rc-lookahead={_X264_RC_LOOKAHEAD}:"
                            f"lookahead-threads={_X264_LOOKAHEAD_THREADS}",
            "-threads", str(_X264_MAX_ENCODER_THREADS),
            "-pix_fmt", "yuv420p", str(out_path),
        ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)



def extract_cut(full_path, cut_path, start_seconds: float, cut_seconds: float) -> bool:
    """Stream-copy a cut window from an already-encoded mp4."""
    if not ffmpeg_available():
        return False
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{max(0.0, start_seconds):.3f}",
        "-i", str(full_path),
        "-t", f"{max(0.0, cut_seconds):.3f}",
        "-c", "copy", str(cut_path),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=60)
    except subprocess.TimeoutExpired:
        return False
    return r.returncode == 0


def probe_duration(path) -> Optional[float]:
    """Return video duration in seconds via ffprobe, or None on failure."""
    if not ffprobe_available():
        return None
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return None
        return float(r.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError):
        return None


def probe_video_meta(path):
    """Probe width / height / fps / duration via ffprobe. Returns dict or None."""
    if not ffprobe_available():
        return None
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_streams", "-select_streams", "v:0", str(path)],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return None
        info = json.loads(r.stdout)
        s = info["streams"][0]
        fps_s = s.get("r_frame_rate", "24/1")
        num, den = (fps_s.split("/") + ["1"])[:2]
        fps = float(num) / float(den) if float(den) else 24.0
        return {
            "width": int(s.get("width", 0)),
            "height": int(s.get("height", 0)),
            "fps": fps,
            "duration": float(s.get("duration", 0.0)),
        }
    except Exception:
        return None