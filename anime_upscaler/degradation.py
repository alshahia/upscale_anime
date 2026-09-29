"""APISR-style two-stage degradation for anime4x distillation training.

Applied to HR crops BEFORE the bicubic x1/4 downscale, so the student learns to
recover from realistic encoder artifacts rather than clean bicubic-only LR.

Stage-1 (always): random codec round-trip (JPEG q60/75/90 + WebP q60/75) at
p_codec=1.0. Our HR source is webp-decoded frames, so re-encoding introduces
banding/posterization artifacts typical of compressed video pipelines.

Stage-2 (p_resize=0.5): half the time, an additional down-then-up round-trip
(x0.5 -> codec re-encode -> x2.0) injects mild aliasing on top.

References: APISR (Yu et al., 2024) Real-World Anime SR pipeline section 3.2;
APISR uses a shuffle of (resize, blur, JPEG, sinc) -- we keep the cheap codec
shuffle because anime artifacts are dominated by codec banding.
"""
import io
import random

from PIL import Image

CODECS = {
    "jpeg60": ("JPEG", 60),
    "jpeg75": ("JPEG", 75),
    "jpeg90": ("JPEG", 90),
    "webp60": ("WEBP", 60),
    "webp75": ("WEBP", 75),
}


def _encode_codec(img, codec, quality):
    """Apply one codec round-trip (JPEG or WebP)."""
    if codec == "JPEG" and img.mode != "RGB":
        img = img.convert("RGB")
    if codec in ("JPEG", "WEBP"):
        buf = io.BytesIO()
        img.save(buf, codec, quality=quality)
        buf.seek(0)
        return Image.open(buf).copy()
    return img


def degrade(img, p_codec=1.0, p_resize=0.5):
    """APISR-style two-stage degradation of a PIL HR crop.

    Args:
        img: PIL.Image (RGB). Will be converted to RGB internally if needed.
        p_codec: stage-1 always-on (1.0) or stochastic (e.g. 0.5). We pass 1.0
            because our HR is webp-decoded; skipping codec entirely would
            leave the LR statistically indistinguishable from clean bicubic.
        p_resize: probability of the x0.5 -> x2 round-trip on top of codec.

    Returns:
        PIL.Image of the same size (resolution) -- bicubic downscale to LR
        happens downstream in AnimePairDataset.
    """
    name, q = random.choice(list(CODECS.values()))
    img = _encode_codec(img, name, q)
    if random.random() < p_resize:
        W, H = img.size
        small = img.resize((W // 2, H // 2), Image.BICUBIC)
        name2, q2 = random.choice(list(CODECS.values()))
        small = _encode_codec(small, name2, q2)
        img = small.resize((W, H), Image.BICUBIC)
    return img
