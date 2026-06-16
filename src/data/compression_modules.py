"""
Prediction-oriented compression modules for anime super-resolution.
Implements APISR-style degradation with WebP, AVIF, and video codec compression.

Reference: APISR (CVPR 2024) - Anime Production Inspired Real-World Anime Super-Resolution
https://github.com/Kiteretsu77/APISR
"""
import io
import logging
import os
import random
import tempfile
import cv2
import numpy as np
from typing import Optional, Tuple, List
from PIL import Image


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend detection (module-level, run once on import)
# ---------------------------------------------------------------------------

_AV_AVAILABLE = False
try:
    import av  # noqa: F401
    try:
        av.logging.set_level(av.logging.FATAL)
        av.logging.set_libav_level(av.logging.FATAL)
    except Exception as e:  # logging setup is best-effort
        logger.debug("av logging level setup failed: %s", e)
    _AV_AVAILABLE = True
except Exception as e:
    logger.debug("av (PyAV) import failed: %s", e)
    av = None  # type: ignore[assignment]

_PILLOW_HEIF_AVAILABLE = False
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    _PILLOW_HEIF_AVAILABLE = True
except Exception as e:
    logger.debug("pillow_heif import failed: %s", e)
    pillow_heif = None  # type: ignore[assignment]

_IMAGEIO_FFMPEG_AVAILABLE = False
try:
    import imageio_ffmpeg  # noqa: F401
    _IMAGEIO_FFMPEG_AVAILABLE = True
except Exception as e:
    logger.debug("imageio_ffmpeg import failed: %s", e)
    imageio_ffmpeg = None  # type: ignore[assignment]


_CODEC_MAP = {
    'h264': 'libx264',
    'h265': 'libx265',
    'mpeg4': 'mpeg4',
    'mpeg2': 'mpeg2video',
}


def _fallback_jpeg(img: np.ndarray, lo: int = 60, hi: int = 85) -> np.ndarray:
    """Last-resort JPEG fallback. Same path used by all codecs when their
    primary backend is unavailable."""
    quality = random.randint(lo, hi)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(encimg, 1)


class WebPCompression:
    """
    WebP compression simulating web/streaming artifacts.
    Uses PIL's WebP encoder with configurable quality.
    """

    def __init__(self, quality_range: Tuple[int, int] = (30, 95)):
        self.quality_range = quality_range
        self._buffer = io.BytesIO()
        self._logged_backend = False

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if not self._logged_backend:
            logger.info("[WebPCompression] backend=pillow (in-process)")
            self._logged_backend = True
        quality = random.randint(self.quality_range[0], self.quality_range[1])
        pil_img = Image.fromarray(img)
        self._buffer.seek(0)
        self._buffer.truncate(0)
        pil_img.save(self._buffer, format='WEBP', quality=quality)
        self._buffer.seek(0)
        compressed = Image.open(self._buffer)
        return np.array(compressed.convert('RGB'))


class AVIFCompression:
    """
    AVIF compression simulating modern codec artifacts.
    Uses pillow_heif (in-process) when available; falls back to JPEG.
    """

    def __init__(self, quality_range: Tuple[int, int] = (30, 90)):
        self.quality_range = quality_range
        self._available = _PILLOW_HEIF_AVAILABLE
        self._buffer = io.BytesIO()
        self._logged_backend = False
        if self._available:
            logger.info(
                "[AVIFCompression] backend=pillow_heif v%s (in-process)",
                getattr(pillow_heif, "__version__", "unknown"),
            )
        else:
            logger.warning(
                "[AVIFCompression] pillow_heif not available; falling back to JPEG."
            )

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if not self._available:
            if not self._logged_backend:
                logger.info("[AVIFCompression] backend=jpeg_fallback")
                self._logged_backend = True
            return _fallback_jpeg(
                img, self.quality_range[0], self.quality_range[1]
            )

        quality = random.randint(self.quality_range[0], self.quality_range[1])
        pil_img = Image.fromarray(img)
        self._buffer.seek(0)
        self._buffer.truncate(0)
        try:
            pil_img.save(self._buffer, format='AVIF', quality=quality)
            self._buffer.seek(0)
            compressed = Image.open(self._buffer)
            return np.array(compressed.convert('RGB'))
        except Exception as e:
            logger.warning(
                "[AVIFCompression] pillow_heif encode failed (%s); using JPEG fallback.",
                e,
            )
            return _fallback_jpeg(
                img, self.quality_range[0], self.quality_range[1]
            )


class PyAVVideoCompression:
    """
    Single-frame video codec compression (H.264/H.265/MPEG) using PyAV.

    PyAV binds libav* in-process -- no subprocess spawn. This is the
    primary path for v7's `compression_stage2: [avif, h264, h265, jpeg]`
    codec list. If `import av` fails at module load time, the class
    transparently falls back to JPEG.

    Reference: APISR uses single-frame video compression to synthesize
    artifacts equivalent to multi-frame video compression.
    """

    def __init__(
        self,
        codec: str = 'h264',
        crf_range: Tuple[int, int] = (18, 40),
    ):
        self.codec = codec
        self.crf_range = crf_range
        self._available = _AV_AVAILABLE
        self._logged_backend = False
        if self._available:
            codec_name = _CODEC_MAP.get(codec, 'libx264')
            logger.info(
                "[PyAVVideoCompression] backend=av v%s codec=%s (in-process)",
                getattr(av, "__version__", "unknown"),
                codec_name,
            )
        else:
            logger.warning(
                "[PyAVVideoCompression] av (PyAV) not available; falling back to JPEG."
            )

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if not self._available:
            if not self._logged_backend:
                logger.info("[PyAVVideoCompression] backend=jpeg_fallback")
                self._logged_backend = True
            return _fallback_jpeg(img)

        crf = random.randint(self.crf_range[0], self.crf_range[1])
        return self._compress_single_frame(img, self.codec, crf)

    def _compress_single_frame(
        self, img: np.ndarray, codec: str, crf: int
    ) -> np.ndarray:
        h, w = img.shape[:2]
        codec_name = _CODEC_MAP.get(codec, 'libx264')

        # Use a fresh in-memory buffer per call to avoid container state
        # pollution between frames. (av.open on a BytesIO requires rewind
        # for reading, but we always close+reopen for the decode side.)
        enc_buf = io.BytesIO()
        try:
            container = av.open(enc_buf, mode='w', format='mp4')
            stream = container.add_stream(codec_name, rate=1)
            stream.width = w
            stream.height = h
            stream.pix_fmt = 'yuv420p'

            if codec_name in ('libx264', 'libx265'):
                opts = {'crf': str(crf), 'preset': 'ultrafast'}
                # libx265 prints ~20 lines of build info to stderr on the
                # first encode unless explicitly told to be quiet. This
                # is purely cosmetic but it floods the trainer logs.
                if codec_name == 'libx265':
                    opts['x265-params'] = 'log-level=error'
                stream.options = opts
            else:
                stream.options = {'qscale': str(crf)}

            frame = av.VideoFrame.from_ndarray(img, format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
            container.close()

            enc_buf.seek(0)
            input_container = av.open(enc_buf)
            decoded = None
            for f in input_container.decode(video=0):
                decoded = f.to_ndarray(format='rgb24')
                break
            input_container.close()
            if decoded is None:
                logger.warning(
                    "[PyAVVideoCompression] decoder produced no frames; using JPEG fallback."
                )
                return _fallback_jpeg(img)
            return decoded
        except Exception as e:
            logger.warning(
                "[PyAVVideoCompression] encode/decode failed (%s); using JPEG fallback.",
                e,
            )
            return _fallback_jpeg(img)


class VideoCodecCompression:
    """
    Single-frame video codec compression (H.264/H.265/MPEG).

    Facade over PyAVVideoCompression (in-process, primary) and an
    `imageio_ffmpeg` subprocess fallback (used only when PyAV is not
    installed). The ffmpeg path is the legacy implementation and is
    exercised only on systems missing the `av` package.

    Reference: APISR uses single-frame video compression to synthesize
    artifacts equivalent to multi-frame video compression.
    """

    def __init__(
        self,
        codec: str = 'h264',
        crf_range: Tuple[int, int] = (18, 40),
    ):
        self.codec = codec
        self.crf_range = crf_range
        self._pyav = PyAVVideoCompression(codec=codec, crf_range=crf_range)
        self._use_pyav = self._pyav._available
        self._ffmpeg_available = _IMAGEIO_FFMPEG_AVAILABLE
        self._logged_backend = False
        if not self._use_pyav and self._ffmpeg_available:
            logger.info(
                "[VideoCodecCompression] backend=imageio_ffmpeg (subprocess) codec=%s",
                codec,
            )
        elif not self._use_pyav and not self._ffmpeg_available:
            logger.warning(
                "[VideoCodecCompression] no backend (PyAV and imageio_ffmpeg both missing); "
                "using JPEG fallback for codec=%s.",
                codec,
            )

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self._use_pyav:
            return self._pyav(img)
        if self._ffmpeg_available:
            if not self._logged_backend:
                logger.info(
                    "[VideoCodecCompression] backend=imageio_ffmpeg (subprocess)"
                )
                self._logged_backend = True
            return self._compress_ffmpeg(img)
        if not self._logged_backend:
            logger.info("[VideoCodecCompression] backend=jpeg_fallback")
            self._logged_backend = True
        return _fallback_jpeg(img)

    def _compress_ffmpeg(self, img: np.ndarray) -> np.ndarray:
        import gc
        import time

        h, w = img.shape[:2]
        codec_name = _CODEC_MAP.get(self.codec, 'libx264')
        crf = random.randint(self.crf_range[0], self.crf_range[1])

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp_path = tmp.name

        result = None
        try:
            pad_h = (16 - h % 16) % 16
            pad_w = (16 - w % 16) % 16
            write_h = h + pad_h
            write_w = w + pad_w

            if pad_h > 0 or pad_w > 0:
                img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
            else:
                img_padded = img

            write_gen = imageio_ffmpeg.write_frames(
                tmp_path,
                size=(write_w, write_h),
                codec=codec_name,
                output_params=['-crf', str(crf), '-preset', 'fast'],
            )
            write_gen.send(None)
            write_gen.send(img_padded)
            write_gen.close()

            read_gen = imageio_ffmpeg.read_frames(tmp_path)
            try:
                for frame_bytes, metadata in read_gen:
                    frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                    frame_array = frame_array.reshape((metadata['height'], metadata['width'], 3))
                    if pad_h > 0 or pad_w > 0:
                        frame_array = frame_array[:h, :w, :]
                    result = frame_array
                    break
            finally:
                read_gen.close()
        except Exception as e:
            logger.warning(
                "[VideoCodecCompression] ffmpeg subprocess failed (%s); using JPEG fallback.",
                e,
            )
            result = _fallback_jpeg(img)
        finally:
            if os.path.exists(tmp_path):
                for attempt in range(5):
                    try:
                        os.remove(tmp_path)
                        break
                    except PermissionError:
                        gc.collect()
                        time.sleep(0.1)

        return result if result is not None else _fallback_jpeg(img)


class CompressionPipeline:
    """
    Combines multiple compression types with configurable probabilities.
    Implements APISR's two-stage compression approach.

    Handles different quality range types:
    - JPEG/WebP/AVIF: use quality ranges (0-100)
    - Video codecs (h264/h265/mpeg): use CRF ranges (18-40)
    """

    _video_codecs = ('h264', 'h265', 'mpeg4', 'mpeg2')

    def __init__(
        self,
        compression_types: List[str] = None,
        probs: List[float] = None,
        quality_ranges: List[Tuple[int, int]] = None,
    ):
        if compression_types is None:
            compression_types = ['jpeg', 'webp']

        self.compression_types = compression_types
        self.probs = probs or [1.0 / len(compression_types)] * len(compression_types)
        self.quality_ranges = quality_ranges or [(60, 95)] * len(compression_types)

        self.compressors = self._build_compressors()
        self._log_backends()

    def _build_compressors(self) -> dict:
        compressors = {}
        for ctype, qrange in zip(self.compression_types, self.quality_ranges):
            if ctype == 'jpeg':
                compressors[ctype] = JPEGCompression(qrange)
            elif ctype == 'webp':
                compressors[ctype] = WebPCompression(qrange)
            elif ctype == 'avif':
                compressors[ctype] = AVIFCompression(qrange)
            elif ctype in self._video_codecs:
                crf_range = self._map_to_crf(qrange)
                compressors[ctype] = VideoCodecCompression(codec=ctype, crf_range=crf_range)
        return compressors

    def _log_backends(self) -> None:
        """Log the detected backend stack at init time so the trainer log
        has a single, grep-able summary of which in-process codecs are
        in use (vs the ffmpeg subprocess fallback)."""
        logger.info(
            "[CompressionPipeline] codecs=%s | backends: av=%s pillow_heif=%s imageio_ffmpeg=%s",
            list(self.compressors.keys()),
            _AV_AVAILABLE,
            _PILLOW_HEIF_AVAILABLE,
            _IMAGEIO_FFMPEG_AVAILABLE,
        )

    def _map_to_crf(self, qrange: Tuple[int, int]) -> Tuple[int, int]:
        """Map JPEG-style quality range to CRF range for video codecs."""
        if qrange[0] > 50:
            return (18, 28)
        elif qrange[0] > 30:
            return (23, 33)
        else:
            return (28, 40)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        choice = random.choices(self.compression_types, weights=self.probs, k=1)[0]
        if choice in self.compressors:
            return self.compressors[choice](img)
        return img


class JPEGCompression:
    """Standard JPEG compression (existing functionality, wrapped for consistency)."""

    def __init__(self, quality_range: Tuple[int, int] = (60, 95)):
        self.quality_range = quality_range

    def __call__(self, img: np.ndarray) -> np.ndarray:
        quality = random.randint(self.quality_range[0], self.quality_range[1])
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', img, encode_param)
        return cv2.imdecode(encimg, 1)
