"""ffmpeg helpers: pipe raw frames, stream-copy cut extraction, duration probe.

Supports NVENC (NVIDIA hardware H.264 encoder) via the `use_nvenc` flag.
Falls back to libx264 when NVENC is requested but unavailable.
"""
import json
import math
import os
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

# --- audio remux safety guards --------------------------------------------
# mux_audio must never publish a damaged picture: the muxed temp file is
# probed BEFORE it replaces the encoded video and the call fails closed when
# the result cannot be proven good. The post-mux container/stream length must
# match the encoded picture within max(5% of the picture, 0.20 s): 5% scales
# with long clips where container rounding grows, while 0.20 s covers short
# frame-quantised clips (a 10 fps cut can land one frame short by copy timing).
_MUX_DURATION_TOLERANCE_RATIO = 0.05
_MUX_DURATION_TOLERANCE_MIN_S = 0.20


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


def has_audio_stream(path) -> bool:
    """True when ffprobe reports at least one audio stream."""
    if not ffprobe_available():
        return False
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return r.returncode == 0 and r.stdout.strip() != ""


def _ffprobe_json(path):
    """Run ffprobe -show_format -show_streams and return the parsed JSON (or None)."""
    if not ffprobe_available():
        return None
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if r.returncode != 0:
        return None
    try:
        info = json.loads(r.stdout)
    except ValueError:
        return None
    return info if isinstance(info, dict) else None


def _seconds(value) -> float:
    """Coerce an ffprobe duration field to a positive finite float (else 0.0)."""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return 0.0
    return seconds if seconds > 0 and math.isfinite(seconds) else 0.0


def _stream_duration(stream) -> float:
    """Best-effort duration of one ffprobe stream (0.0 when unknown)."""
    return _seconds((stream or {}).get("duration"))


def _container_duration(info) -> float:
    """Container duration, falling back to the longest stream when absent."""
    duration = _seconds((info.get("format") or {}).get("duration"))
    if duration > 0:
        return duration
    return max((_stream_duration(s) for s in info.get("streams") or []), default=0.0)


def _audio_track_length(path) -> float:
    """Duration in seconds of *path*'s first audio stream (0.0 = none/unknown)."""
    info = _ffprobe_json(path)
    if info is None:
        return 0.0
    audios = [s for s in info.get("streams") or [] if s.get("codec_type") == "audio"]
    return _stream_duration(audios[0]) if audios else 0.0


def _expected_picture_length(video_only) -> float:
    """Length in seconds of the encoded picture to preserve (0.0 = unknowable)."""
    duration = probe_duration(video_only)
    if duration is not None and duration > 0:
        return duration
    meta = probe_video_meta(video_only)
    if meta:
        return _seconds(meta.get("duration"))
    return 0.0


def _mux_output_is_intact(
    tmp, expected_seconds: float, source_has_audio: bool,
    expected_audio_seconds: float = 0.0,
) -> bool:
    """Prove the muxed file carries the full picture (and audio when source had it).

    Fails closed on any doubt: unreadable file, missing/zero-sized video stream,
    missing audio for an audio-bearing source, zero-length audio, a container /
    video-stream length away from the encoded picture beyond tolerance, or an
    audio track whose length does not match the requested cut window (ffmpeg
    silently starts over when a seek runs past the audio end, which would ship
    audio from the wrong position).
    """
    info = _ffprobe_json(tmp)
    if info is None:
        return False
    streams = info.get("streams") or []
    videos = [s for s in streams if s.get("codec_type") == "video"]
    if not videos:
        return False
    video = videos[0]
    try:
        width = int(video.get("width") or 0)
        height = int(video.get("height") or 0)
    except (TypeError, ValueError):
        return False
    if width <= 0 or height <= 0:
        return False
    tolerance = max(_MUX_DURATION_TOLERANCE_RATIO * expected_seconds,
                    _MUX_DURATION_TOLERANCE_MIN_S)
    container = _container_duration(info)
    if container <= 0 or abs(container - expected_seconds) > tolerance:
        return False
    # The picture stream itself must be full length: a container can be padded
    # by a longer audio stream while -shortest truncated the video (F2).
    video_length = _stream_duration(video) or container
    if abs(video_length - expected_seconds) > tolerance:
        return False
    if source_has_audio:
        audios = [s for s in streams if s.get("codec_type") == "audio"]
        if not audios:
            return False
        audio_length = _stream_duration(audios[0])
        if audio_length <= 0:
            return False
        audio_tolerance = max(_MUX_DURATION_TOLERANCE_RATIO * expected_audio_seconds,
                              _MUX_DURATION_TOLERANCE_MIN_S)
        if abs(audio_length - expected_audio_seconds) > audio_tolerance:
            return False
    return True


def mux_audio(video_only, audio_source, dest, *, start_seconds: float = 0.0,
              duration_seconds: float | None = None) -> bool:
    """Remux the source audio into the encoded video without re-encoding.

    Writes a sibling '.muxing' temp file, verifies it, then os.replace()s it
    over dest, so a crash or a bad mux never replaces the encoded video with a
    damaged one. The audio never dictates the length: there is no -shortest and
    the output is capped at the picture length, so short source audio ends
    early (valid) while the picture stays whole.

    Returns True only when the destination is verified to carry both streams
    and the full picture length; a video-only source simply stays video-only
    (that is not an error, so the video is still delivered). Any doubt -
    unreadable output, missing/short video, no audio for an audio-bearing
    source, unexpected length, a cut window with no source audio left in it,
    missing ffprobe/ffmpeg - fails closed: the temp is removed, dest is left
    untouched and False is returned.
    start_seconds/duration_seconds mirror the job's cut window so the sound
    matches exactly the frames that were encoded.
    """
    if not ffmpeg_available():
        return False
    # Callers may pass Path objects or plain str paths (job dictionaries).
    video_only = Path(video_only)
    audio_source = Path(audio_source)
    dest = Path(dest)
    # Fail closed before writing anything when the picture length is unknowable:
    # without it the muxed result cannot be verified.
    expected_seconds = _expected_picture_length(video_only)
    if expected_seconds <= 0:
        return False
    source_has_audio = has_audio_stream(audio_source)
    # Length of audio the requested window can actually provide. A window that
    # starts past the audio end is NOT "zero audio": ffmpeg restarts the track
    # when an input seek runs past EOF (verified: -ss 5 on a 1 s track yields
    # the whole 1.023 s track), which would ship audio from the wrong position.
    # Fail closed instead.
    expected_audio_seconds = 0.0
    if source_has_audio:
        audio_length = _audio_track_length(audio_source)
        if audio_length <= 0:
            return False
        window_start = max(0.0, start_seconds)
        remaining = max(0.0, audio_length - window_start)
        if duration_seconds is not None and duration_seconds > 0:
            window = min(duration_seconds, remaining)
        else:
            window = remaining
        expected_audio_seconds = min(window, expected_seconds)
        if expected_audio_seconds <= 0:
            return False
    tmp = dest.with_name(dest.stem + ".muxing" + dest.suffix)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_only)]
    if start_seconds > 0:
        cmd += ["-ss", f"{start_seconds:.3f}"]
    if duration_seconds is not None and duration_seconds > 0:
        cmd += ["-t", f"{duration_seconds:.3f}"]
    cmd += ["-i", str(audio_source),
            "-map", "0:v:0", "-map", "1:a:0?",
            "-c", "copy",
            # Cap the output at the encoded picture length (output option) so a
            # longer audio stream can never extend or truncate the picture, and
            # omit -shortest so short audio can never cut it.
            "-t", f"{expected_seconds:.3f}", str(tmp)]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=3600)
    except subprocess.TimeoutExpired:
        tmp.unlink(missing_ok=True)
        return False
    if r.returncode != 0 or not tmp.is_file() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        return False
    if not _mux_output_is_intact(tmp, expected_seconds, source_has_audio,
                                 expected_audio_seconds):
        tmp.unlink(missing_ok=True)
        return False
    try:
        os.replace(tmp, dest)
    except OSError:
        tmp.unlink(missing_ok=True)
        return False
    return True
