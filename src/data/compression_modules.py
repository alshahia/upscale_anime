"""
Prediction-oriented compression modules for anime super-resolution.
Implements APISR-style degradation with WebP, AVIF, and video codec compression.

Reference: APISR (CVPR 2024) - Anime Production Inspired Real-World Anime Super-Resolution
https://github.com/Kiteretsu77/APISR
"""
import io
import os
import random
import tempfile
import cv2
import numpy as np
from typing import Optional, Tuple, List
from PIL import Image


class WebPCompression:
    """
    WebP compression simulating web/streaming artifacts.
    Uses PIL's WebP encoder with configurable quality.
    """

    def __init__(self, quality_range: Tuple[int, int] = (30, 95)):
        self.quality_range = quality_range
        self._buffer = io.BytesIO()

    def __call__(self, img: np.ndarray) -> np.ndarray:
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
    Uses pillow_heif if available, falls back to JPEG simulation.
    """

    def __init__(self, quality_range: Tuple[int, int] = (30, 90)):
        self.quality_range = quality_range
        self._available = self._check_availability()
        self._buffer = io.BytesIO()

    @staticmethod
    def _check_availability() -> bool:
        try:
            import pillow_heif
            pillow_heif.register_avif_opener()
            return True
        except Exception:
            return False

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if not self._available:
            return self._fallback_jpeg(img)

        quality = random.randint(self.quality_range[0], self.quality_range[1])
        pil_img = Image.fromarray(img)
        self._buffer.seek(0)
        self._buffer.truncate(0)
        try:
            pil_img.save(self._buffer, format='AVIF', quality=quality)
            self._buffer.seek(0)
            compressed = Image.open(self._buffer)
            return np.array(compressed.convert('RGB'))
        except Exception:
            return self._fallback_jpeg(img)

    def _fallback_jpeg(self, img: np.ndarray) -> np.ndarray:
        quality = random.randint(self.quality_range[0], self.quality_range[1])
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', img, encode_param)
        return cv2.imdecode(encimg, 1)


class PyAVVideoCompression:
    """
    Single-frame video codec compression using PyAV (av library).
    Much faster than imageio-ffmpeg as it uses the av library directly
    without spawning subprocesses.

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
        self._available = self._check_availability()
        self._buffer = io.BytesIO()

    @staticmethod
    def _check_availability() -> bool:
        try:
            import av
            return True
        except Exception:
            return False

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if not self._available:
            return self._fallback_jpeg(img)

        crf = random.randint(self.crf_range[0], self.crf_range[1])
        return self._compress_single_frame(img, self.codec, crf)

    def _compress_single_frame(self, img: np.ndarray, codec: str, crf: int) -> np.ndarray:
        import av

        h, w = img.shape[:2]
        codec_map = {
            'h264': 'libx264',
            'h265': 'libx265',
            'mpeg4': 'mpeg4',
            'mpeg2': 'mpeg2video',
        }
        codec_name = codec_map.get(codec, 'libx264')

        self._buffer.seek(0)
        self._buffer.truncate(0)

        try:
            container = av.open(self._buffer, mode='w', format='mp4')
            stream = container.add_stream(codec_name, rate=1)
            stream.width = w
            stream.height = h
            stream.pix_fmt = 'yuv420p'

            if codec_name in ('libx264', 'libx265'):
                stream.options = {
                    'crf': str(crf),
                    'preset': 'fast',
                }
            else:
                stream.options = {'qscale': str(crf)}

            frame = av.VideoFrame.from_ndarray(img, format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)

            for packet in stream.encode():
                container.mux(packet)

            container.close()

            self._buffer.seek(0)
            input_container = av.open(self._buffer)
            for frame in input_container.decode(video=0):
                result = frame.to_ndarray(format='rgb24')
                input_container.close()
                return result
        except Exception:
            return self._fallback_jpeg(img)

    def _fallback_jpeg(self, img: np.ndarray) -> np.ndarray:
        quality = random.randint(60, 85)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', img, encode_param)
        return cv2.imdecode(encimg, 1)


class VideoCodecCompression:
    """
    Single-frame video codec compression (H.264/H.265/MPEG).
    Uses PyAV if available (fast), falls back to imageio-ffmpeg,
    then to JPEG simulation.

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
        self._ffmpeg_available = self._check_ffmpeg_availability()

    @staticmethod
    def _check_ffmpeg_availability() -> bool:
        try:
            import imageio_ffmpeg
            return True
        except Exception:
            return False

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self._use_pyav:
            return self._pyav(img)
        elif self._ffmpeg_available:
            return self._compress_ffmpeg(img)
        return self._fallback_jpeg(img)

    def _compress_ffmpeg(self, img: np.ndarray) -> np.ndarray:
        import gc
        import time
        import imageio_ffmpeg

        h, w = img.shape[:2]
        codec_map = {
            'h264': 'libx264',
            'h265': 'libx265',
            'mpeg4': 'mpeg4',
            'mpeg2': 'mpeg2video',
        }
        codec_name = codec_map.get(self.codec, 'libx264')
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
        except Exception:
            result = self._fallback_jpeg(img)
        finally:
            if os.path.exists(tmp_path):
                for attempt in range(5):
                    try:
                        os.remove(tmp_path)
                        break
                    except PermissionError:
                        gc.collect()
                        time.sleep(0.1)
                else:
                    pass

        return result if result is not None else self._fallback_jpeg(img)

    def _fallback_jpeg(self, img: np.ndarray) -> np.ndarray:
        quality = random.randint(60, 85)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', img, encode_param)
        return cv2.imdecode(encimg, 1)


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
