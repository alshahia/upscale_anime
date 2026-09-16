"""Tests for Q3 (perf/queue-controls-gpu-codec): NVDEC + NVENC.

Pins the Q3 codec contract:
  - _NvDecReader exists and is constructed via PyAV hwaccel_options
  - the decode factory dispatches nvdec / auto / pyav / cv2
  - _detect_nvdec_support caches its result (process-lifetime)
  - open_encoder argv honors use_nvenc / nvenc_preset / nvenc_qp
  - the Decode combobox now offers 'auto' / 'nvdec' as choices
  - nvenc_qp is wired through build_run_job via the panel

The NVDEC functional test only runs when PyAV + a working cuvid decoder
are present; on hosts without NVDEC the test asserts graceful fallback
to PyAV.
"""
import os
from pathlib import Path

import pytest

# Q3 imports
from apps.anime_upscaler_gui.anime_upscaler_gui.decoders import (
    _NvDecReader,
    _HAS_PYAV,
    _detect_nvdec_support,
    _NVDEC_SUPPORTED,
    open_reader,
)
from apps.anime_upscaler_gui.anime_upscaler_gui.widgets.settings_spec import (
    SETTING_SPECS,
    get_spec,
)
from apps.anime_upscaler_gui.anime_upscaler_gui.ffmpeg import (
    ffmpeg_available,
    open_encoder,
    _detect_nvenc_support,
)
from apps.anime_upscaler_gui.anime_upscaler_gui.controllers.job_builder import (
    build_run_job,
)


# ---- _detect_nvdec_support caching ----------------------------------------
def test_detect_nvdec_support_caches():
    """The probe is process-cached: second call returns the cached bool."""
    import apps.anime_upscaler_gui.anime_upscaler_gui.decoders as dec_mod
    # The function uses ``global _NVDEC_SUPPORTED`` so we must clear the
    # module's global, not the local re-export.
    dec_mod._NVDEC_SUPPORTED = None
    a = _detect_nvdec_support()
    b = _detect_nvdec_support()
    assert a == b
    assert dec_mod._NVDEC_SUPPORTED is not None


# ---- open_reader dispatch --------------------------------------------------
def test_open_reader_dispatch_table():
    """open_reader returns the right reader class for each decode string.

    We don't actually open a video file -- we just check that the right
    class is selected for the dispatch arg, by patching the readers.
    """
    from apps.anime_upscaler_gui.anime_upscaler_gui import decoders
    called = []
    # Patch the constructors so we can see which one was chosen.
    orig_nvdec = dec_mod = decoders._NvDecReader
    orig_pyav = decoders._PyAvReader
    orig_cv2 = decoders._Cv2Reader
    try:
        decoders._NvDecReader = lambda p: called.append(("nvdec", p)) or object()
        decoders._PyAvReader = lambda p: called.append(("pyav", p)) or object()
        decoders._Cv2Reader = lambda p: called.append(("cv2", p)) or object()
        open_reader("/fake.mp4", "nvdec")
        open_reader("/fake.mp4", "pyav")
        open_reader("/fake.mp4", "cv2")
        kinds = [c[0] for c in called]
        assert kinds == ["nvdec", "pyav", "cv2"]
    finally:
        decoders._NvDecReader = orig_nvdec
        decoders._PyAvReader = orig_pyav
        decoders._Cv2Reader = orig_cv2


def test_open_reader_auto_prefers_nvdec_when_available():
    """When decode='auto' and NVDEC works, _NvDecReader is chosen."""
    from apps.anime_upscaler_gui.anime_upscaler_gui import decoders
    called = []
    orig_nvdec = decoders._NvDecReader
    orig_pyav = decoders._PyAvReader
    orig_cv2 = decoders._Cv2Reader
    try:
        decoders._detect_nvdec_support = lambda: True
        decoders._NvDecReader = lambda p: called.append("nvdec") or object()
        decoders._PyAvReader = lambda p: called.append("pyav") or object()
        decoders._CvDecReader = decoders._Cv2Reader
        decoders._Cv2Reader = lambda p: called.append("cv2") or object()
        open_reader("/fake.mp4", "auto")
        assert called[0] == "nvdec"
    finally:
        decoders._NvDecReader = orig_nvdec
        decoders._PyAvReader = orig_pyav
        decoders._Cv2Reader = orig_cv2


def test_open_reader_auto_falls_back_to_pyav_then_cv2():
    """When NVDEC is unavailable, auto picks PyAV; if no PyAV either, cv2."""
    from apps.anime_upscaler_gui.anime_upscaler_gui import decoders
    called = []
    orig_nvdec = decoders._NvDecReader
    orig_pyav = decoders._PyAvReader
    orig_cv2 = decoders._Cv2Reader
    try:
        decoders._detect_nvdec_support = lambda: False
        decoders._NvDecReader = lambda p: called.append("nvdec") or object()
        decoders._PyAvReader = lambda p: called.append("pyav") or object()
        decoders._Cv2Reader = lambda p: called.append("cv2") or object()
        # NVDEC off, PyAV on -> pyav
        open_reader("/fake.mp4", "auto")
        assert called[0] == "pyav"
        # NVDEC off, PyAV off -> cv2
        called.clear()
        orig_has = decoders._HAS_PYAV
        decoders._HAS_PYAV = False
        open_reader("/fake.mp4", "auto")
        assert called[0] == "cv2"
        decoders._HAS_PYAV = orig_has
    finally:
        decoders._NvDecReader = orig_nvdec
        decoders._PyAvReader = orig_pyav
        decoders._Cv2Reader = orig_cv2


# ---- _NvDecReader ------------------------------------------------------
def test_nvdec_reader_raises_when_probe_says_unavailable():
    """When _detect_nvdec_support() returns False, constructor raises."""
    from apps.anime_upscaler_gui.anime_upscaler_gui import decoders
    orig = decoders._detect_nvdec_support
    decoders._detect_nvdec_support = lambda: False
    try:
        with pytest.raises(RuntimeError, match="NVDEC unavailable"):
            _NvDecReader("/fake.mp4")
    finally:
        decoders._detect_nvdec_support = orig


def test_nvdec_reader_raises_when_pyav_missing():
    """Without PyAV the reader raises immediately with a clear message."""
    from apps.anime_upscaler_gui.anime_upscaler_gui import decoders
    orig = decoders._HAS_PYAV
    orig_probe = decoders._detect_nvdec_support
    decoders._HAS_PYAV = False
    decoders._detect_nvdec_support = lambda: True
    try:
        with pytest.raises(RuntimeError, match="PyAV not installed"):
            _NvDecReader("/fake.mp4")
    finally:
        decoders._HAS_PYAV = orig
        decoders._detect_nvdec_support = orig_probe


@pytest.mark.skipif(not _HAS_PYAV, reason="PyAV not installed")
def test_nvdec_reader_forwards_hwaccel_kwargs_to_pyav():
    """When NVDEC is up, the constructor forwards the hwaccel kwargs to
    PyAV. PyAV 17 wants ``options={'hwaccel': 'cuda'}`` plus
    ``stream_options=[{'gpu': '0'}]`` (list of single-key dicts, not
    tuples) -- the older recipes that set
    stream.codec_context.options were silently ignored."""
    from apps.anime_upscaler_gui.anime_upscaler_gui import decoders
    orig_probe = decoders._detect_nvdec_support
    decoders._detect_nvdec_support = lambda: True
    try:
        captured = {}
        import av
        orig_av_open = av.open
        def fake_open(path, **kwargs):
            captured.update(kwargs)
            raise SystemExit("stop after capture")
        av.open = fake_open
        try:
            with pytest.raises(SystemExit):
                _NvDecReader("/fake.mp4")
        finally:
            av.open = orig_av_open
        # The crucial line: the hwaccel + gpu kwargs must reach PyAV.
        assert captured.get("options", {}).get("hwaccel") == "cuda"
        # PyAV 17 wants stream_options as a LIST OF SINGLE-KEY DICTS.
        stream_opts = captured.get("stream_options")
        assert isinstance(stream_opts, list)
        # Each element must be a dict; the union of all elements is
        # the effective per-stream option set.
        merged = {}
        for d in stream_opts:
            merged.update(d)
        assert merged.get("gpu") == "0"
    finally:
        decoders._detect_nvdec_support = orig_probe


# ---- settings_spec additions --------------------------------------------
def test_decode_spec_offers_nvdec_and_auto():
    """The Decode combobox now lists auto / nvdec / pyav / cv2."""
    spec = get_spec("decode")
    keys = [c[0] for c in spec.choices]
    assert "auto" in keys
    assert "nvdec" in keys
    # Legacy choices still present (back-compat).
    assert "pyav" in keys
    assert "cv2" in keys


def test_nvenc_qp_spec_exists_with_sensible_range():
    """NVENC QP widget: 16-28, default 18."""
    spec = get_spec("nvenc_qp")
    assert spec.var_name == "nvenc_qp_var"
    lo, hi, step = spec.spin
    assert lo == 16 and hi == 28
    # Default coerces to the defaults dataclass (nvenc_qp=18).
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import Defaults
    assert Defaults().nvenc_qp == 18


# ---- build_run_job: nvenc_qp propagation ---------------------------------
def _fake_job(id=1, is_video=False, **kw):
    class J:
        pass
    j = J()
    j.id = id
    j.input = Path("/tmp/in.mp4")
    j.output = Path("/tmp/out.mp4")
    j.is_video = is_video
    j.cut_start = 0.0
    j.cut_end = 0.0
    j.model_filename = ""
    j.kind = ""
    j.scale = 0
    return j


def test_build_run_job_uses_panel_nvenc_qp_when_present():
    """build_run_job honours value_provider('nvenc_qp') over the dataclass."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import Defaults
    data = Defaults()  # nvenc_qp=18 by default
    out = build_run_job(
        job=_fake_job(is_video=False),
        value_provider=lambda k: 22 if k == "nvenc_qp" else
            ("cuda" if k == "device" else
             True if k == "fp16" else
             4.0 if k == "outscale" else
             1 if k == "batch_size" else
             "cv2" if k == "decode" else
             "sync" if k == "prefetch" else
             0 if k == "downscale_max_edge" else
             "off" if k == "gpu_guard_mode" else
             256 if k == "tile_size" else
             8 if k == "tile_overlap" else
             False if k == "tta" else
             False if k == "use_tensorrt" else
             False if k == "use_nvenc" else
             "p4"),
        data=data,
        resolve_model_path=lambda: "/ckpts/r.pth",
        resolve_kind=lambda: "spandrel",
        on_frame_error=lambda i, m: "skip_frame",
    )
    assert out.nvenc_qp == 22


def test_build_run_job_falls_back_to_data_nvenc_qp():
    """When the panel has no nvenc_qp, the dataclass default wins."""
    from apps.anime_upscaler_gui.anime_upscaler_gui.settings import Defaults
    data = Defaults()  # nvenc_qp=18
    # Value provider without nvenc_qp -- simulates a pre-Q3 caller.
    base = {
        "device": "cuda", "fp16": True, "outscale": 4.0, "batch_size": 1,
        "decode": "cv2", "prefetch": "sync", "downscale_max_edge": 0,
        "gpu_guard_mode": "off", "tile_size": 256, "tile_overlap": 8,
        "tta": False, "use_tensorrt": False, "use_nvenc": False,
        "nvenc_preset": "p4",
    }
    out = build_run_job(
        job=_fake_job(is_video=False),
        value_provider=lambda k: base[k],
        data=data,
        resolve_model_path=lambda: "/ckpts/r.pth",
        resolve_kind=lambda: "spandrel",
        on_frame_error=lambda i, m: "skip_frame",
    )
    assert out.nvenc_qp == 18


# ---- NVENC argv shape (open_encoder) -------------------------------------
def test_open_encoder_libx264_argv_when_nvenc_disabled():
    """When use_nvenc=False, the argv uses libx264 (not h264_nvenc)."""
    if not ffmpeg_available():
        pytest.skip("ffmpeg not on PATH")
    proc = open_encoder(
        "E:/Temp/pytest_q3a/_libx264.mp4",
        fps=24.0, w=128, h=128,
        crf=18, preset="medium",
        use_nvenc=False, nvenc_preset="p1", nvenc_qp=18,
    )
    assert proc is not None
    # libx264 must be in the argv; h264_nvenc must not.
    assert "libx264" in proc.args
    assert "h264_nvenc" not in proc.args
    proc.stdin.close()
    proc.wait(timeout=5)


@pytest.mark.skipif(
    not ffmpeg_available() or not _detect_nvenc_support(),
    reason="ffmpeg / h264_nvenc not available",
)
def test_open_encoder_h264_nvenc_argv_when_enabled_and_supported():
    """When use_nvenc=True AND h264_nvenc works, argv uses h264_nvenc
    and the preset / QP flags are present."""
    if not ffmpeg_available() or not _detect_nvenc_support():
        pytest.skip("NVENC unavailable")
    proc = open_encoder(
        "E:/Temp/pytest_q3a/_nvenc.mp4",
        fps=24.0, w=128, h=128,
        crf=18, preset="medium",
        use_nvenc=True, nvenc_preset="p3", nvenc_qp=21,
    )
    assert proc is not None
    argv = proc.args
    assert "h264_nvenc" in argv
    assert "p3" in argv
    assert "21" in argv  # nvenc_qp
    assert "constqp" in argv
    # Cleanup
    try:
        proc.stdin.close()
    except Exception:
        pass
    proc.wait(timeout=5)
