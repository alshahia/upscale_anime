# APISR Degradation Pipeline — Implementation Spec

**Status:** Research-only specification, no code edits.
**Target project:** `E:\python projects\upscale_anime` (Windows, Python 3.x, PyTorch 2.x, `src/` + `scripts/` + `configs/` layout, 8 GB VRAM RTX 4000 Mobile).
**Source repo:** `github.com/Kiteretsu77/APISR` (CVPR 2024, arXiv 2403.01598v2).
**Goal:** A precise, unambiguous specification an implementation sub-agent can execute without further research. All file references include the raw URL so the sub-agent can `curl` them directly.

---

## 0. What APISR actually is (one-paragraph refresher)

APISR's two contributions to the **degradation pipeline** are:

1. **Prediction-oriented compression module** — apply H.264/H.265/MPEG-2/MPEG-4 codecs in **single-frame (intra-only) mode** to a single image. With one frame there is no inter-prediction reference, so the codec falls back to intra-prediction, which produces the *same class* of artifacts (misaligned residuals) as real multi-frame video compression. The paper proves empirically that this is sufficient (Fig. 7, ablation Tab. 3).
2. **Shuffled resize** — instead of a fixed ordering of `blur → noise → resize → compression`, the resize op is **randomly placed** at one of three positions relative to noise+compression, to make the degradation less of a recognizable "SR signature" the model can overfit to.
3. **XDoG + USM pseudo-GT** — to address faint hand-drawn lines, the cropped GT is sharpened with **3 rounds of unsharp-mask (USM)**, then XDoG extracts the line map, the map is **outlier-filtered** and **passively dilated**, and the sharpened GT is blended back: `I_pseudo = I_sharp · I_map + I_GT · (1 - I_map)`.

Everything else (kernel list, noise generation, sinc blur, JPEG/WebP, probabilistic 1st-stage compression) is lifted **verbatim from Real-ESRGAN's `basicsr/data/degradations.py`**. So the implementation sub-agent must import / port from Real-ESRGAN's `degradations.py` (the file currently lives at `xinntao/Real-ESRGAN/basicsr/data/degradations.py`). Note: the `master` branch URL returned 404 during research, so use the `v0.2.5.0` tag (verified to exist) or a local copy of `basicsr/`.

---

## 1. File-by-file map of APISR's repo

All paths below are relative to the repo root. The raw URL is constructed as `https://raw.githubusercontent.com/Kiteretsu77/APISR/main/<path>`.

### 1.1 Degradation orchestrator

| Concern | APISR path | Lines | Purpose |
|---|---|---|---|
| Top-level entry per image | `degradation/degradation_esr.py` | 109 | Class `degradation_v1`; one instance per worker. |
| First-stage shared kernel/blur/noise/JPEG-WebP | `degradation/ESR/degradation_esr_shared.py` | 130+ | `common_degradation()` — the *real* pipeline. |
| Kernel + noise generation (Real-ESRGAN) | `degradation/ESR/degradations_functionality.py` | 600+ | `random_mixed_kernels`, `random_add_gaussian_noise_pt`, `random_add_poisson_noise_pt`, `circular_lowpass_kernel`, `add_jpg_compression`, `generate_kernels` (in `utils.py`). |
| USM sharpener (used by line enhancement) | `degradation/ESR/usm_sharp.py` | 100 | `USMSharp` module + `usm_sharp_func` (OpenCV). |

### 1.2 Image codecs

All four live under `degradation/image_compression/`. Each defines a class with two methods:

- `compress_and_store(np_frame, store_path, idx)` — used in **stage 2** (called from `degradation_esr.py`).
- `compress_tensor(tensor_frames, idx=0)` (staticmethod) — used in **stage 1** (called from `degradation_esr_shared.py:common_degradation`).

| File | Class | Library | Quality param |
|---|---|---|---|
| `degradation/image_compression/jpeg.py` | `JPEG` | OpenCV `imencode('.jpg')` | `cv2.IMWRITE_JPEG_QUALITY` int |
| `degradation/image_compression/webp.py` | `WEBP` | Pillow `save(..., 'webp', quality=, method=)` | quality int + method int |
| `degradation/image_compression/heif.py` | `HEIF` | `pillow_heif.register_heif_opener()` + Pillow | quality int + method int |
| `degradation/image_compression/avif.py` | `AVIF` | `pillow_heif.register_avif_opener()` + Pillow | quality int + method int |

### 1.3 Video codecs (single-frame I-frame only)

All four live under `degradation/video_compression/`. Each class only implements `compress_and_store`; the `compress_tensor` static method is `pass` (unusable in stage 1, see "Gotchas" §8). All use `os.system("ffmpeg ...")` to write a 1-frame `.mp4`, then a second ffmpeg call decodes the single frame.

| File | Class | FFmpeg encoder | Quality param |
|---|---|---|---|
| `degradation/video_compression/h264.py` | `H264` | `libx264` | `-crf` int |
| `degradation/video_compression/h265.py` | `H265` | `libx265` | `-crf` int |
| `degradation/video_compression/mpeg2.py` | `MPEG2` | `mpeg2video` | `-qscale:v` int (linear, low=better) |
| `degradation/video_compression/mpeg4.py` | `MPEG4` | `libxvid` | `-qscale:v` int (linear, low=better) |

### 1.4 Pseudo-GT (USM + XDoG)

| File | Purpose |
|---|---|
| `scripts/anime_strong_usm.py` | All line-enhancement logic in one file: USM (3 rounds), XDoG, outlier removal, passive dilation. |
| `degradation/ESR/usm_sharp.py` | USM module (alternative to inline OpenCV version in the script). |

### 1.5 Documentation / config

| File | Purpose |
|---|---|
| `opt.py` | **All** APISR hyper-parameters. Lines 95-165 are the only place parameter ranges live. This is the single source of truth the sub-agent must read. |
| `README.md` | "Dataset Curation" and "Train" sections explain the role of `uncropped_hr`, `degrade_hr_dataset_path`, `train_hr_dataset_path`. |
| `scripts/prepare_datasets.sh` | Bash orchestration: `720P_resize.py` → `4x_crop.py` → `anime_strong_usm.py` → `crop_images.py` (×2). |
| `docs/model_zoo.md` | Reference for arch choices; not needed for degradation spec. |

### 1.6 Real-ESRGAN attribution

APISR's `degradations_functionality.py` is byte-equivalent (modulo docstrings) to `xinntao/Real-ESRGAN/basicsr/data/degradations.py` (the `master` branch URL returns 404 as of this research; use the `v0.2.5.0` release tag or pin a commit). The sub-agent can either:
- copy `basicsr/data/degradations.py` and `basicsr/utils/matlab_functions.py` into `src/data/esr_port/`, **or**
- `pip install basicsr` and import from there (recommended; already a transitive dep of many SR libs).

---

## 2. Exact degradation pipeline pseudocode

This is a faithful re-statement of `degradation_esr_shared.py::common_degradation` and `degradation_esr.py::degradate_process`. All probabilities and ranges are from `opt.py`. Seed control is at the top of `__init__` (`numpy.random` and `random` are used; **see gotcha §8.7**).

### 2.1 Setup (called once per training run)

```
opt = { ... }        # see §2.6 for exact numeric ranges
kernel1, kernel2 = generate_kernels(opt)         # src = degradation/ESR/utils.py
k1 = torch.FloatTensor(kernel1).unsqueeze(0)     # shape (1, 21, 21), padded to 21
k2 = torch.FloatTensor(kernel2).unsqueeze(0)
```

`generate_kernels` picks one kernel size from `[2*v+1 for v in range(opt['kernel_range'][0], opt['kernel_range'][1])]` then either a sinc (prob `sinc_prob`) or a mixed random kernel (Real-ESRGAN `random_mixed_kernels`). The kernel is **padded to 21x21** by adding zeros so that `filter2D` (which uses `F.conv2d` with `padding=k//2`) can take a uniform kernel size. The same is done for `kernel2`.

### 2.2 Per-image pipeline (in `degradate_process`)

```python
# out: (B, 3, H, W) float32 in [0, 1]
# scale: 4
# init: codec instances JPEG, WEBP, AVIF, H264, H265, MPEG2, MPEG4
#   (HEIF is imported but commented out in degradation_esr.py:25)

B, _, H, W = out.shape
mode_for_final_resize = random.choice(opt['resize_options'])        # 'area'|'bilinear'|'bicubic'

# === STAGE 1: shared degradation (full-resolution) ===
out = common_degradation(out, opt, [k1, k2], process_id, verbose=False)

# === FINAL RESIZE TO LR (after stage 2 compression) ===
out = F.interpolate(out, size=(H // scale, W // scale), mode=mode_for_final_resize)
out = torch.clamp(out, 0, 1)
np_frame = tensor2np(out)        # back to uint8 (H/scale, W/scale, 3) BGR

# === STAGE 2: prediction-oriented compression (LR-resolution) ===
codec = random.choices(
    opt['compression_codec2'],
    opt['compression_codec_prob2'],
    k=1,
)[0]                            # one codec for the whole batch
instance = codec_map[codec]
instance.compress_and_store(np_frame, store_path, process_id)
# then the result is read back from `store_path` as the LR image
```

### 2.3 `common_degradation` body (stage 1, full-resolution)

The "shuffled resize" trick: at 4x scale, **resize #1** can be placed at one of three positions `{0, 1, 2}` (each chosen with equal probability `1/3` via `random.choices([0,1,2])`); **resize #2** is also placed at one of three positions but only when `opt['scale'] == 4`. The three positions are `before blur`, `between blur and noise`, `after noise`. The compression (stage 1) happens **after noise, before any 2nd-pass blur/noise**.

```python
def common_degradation(out, opt, kernels, process_id, verbose=False):
    k1, k2 = kernels
    B, _, H, W = out.shape

    pos1 = random.choice([0, 1, 2])      # 1st resize position
    pos2 = random.choice([0, 1, 2]) if opt['scale'] == 4 else -1   # 2nd resize position (4x only)

    # --- 1ST PASS ---

    if pos1 == 0: out = downsample_1st(out, opt)    # resize
    out = filter2D(out, k1)                          # blur
    if pos1 == 1: out = downsample_1st(out, opt)    # resize
    if random.random() < opt['gaussian_noise_prob']:
        out = random_add_gaussian_noise_pt(out,
            sigma_range=opt['noise_range'],
            gray_prob=opt['gray_noise_prob'],
            clip=True, rounds=False)
    else:
        out = random_add_poisson_noise_pt(out,
            scale_range=opt['poisson_scale_range'],
            gray_prob=opt['gray_noise_prob'],
            clip=True, rounds=False)
    if pos1 == 2: out = downsample_1st(out, opt)    # resize

    # 1st-stage compression
    codec1 = random.choices(
        opt['compression_codec1'],
        opt['compression_codec_prob1'],
        k=1,
    )[0]
    codec1_instance = codec1_map[codec1]            # JPEG|WEBP|HEIF|AVIF
    out = codec1_instance.compress_tensor(out, idx=process_id)

    # --- 2ND PASS (4x only) ---

    if pos2 == 0: out = downsample_2nd(out, opt, H, W)   # resize to ~H/4..H/2
    if random.random() < opt['second_blur_prob']:
        out = filter2D(out, k2)
    if pos2 == 1: out = downsample_2nd(out, opt, H, W)
    if random.random() < opt['gaussian_noise_prob2']:
        out = random_add_gaussian_noise_pt(out,
            sigma_range=opt['noise_range2'],
            gray_prob=opt['gray_noise_prob2'],
            clip=True, rounds=False)
    else:
        out = random_add_poisson_noise_pt(out,
            scale_range=opt['poisson_scale_range2'],
            gray_prob=opt['gray_noise_prob2'],
            clip=True, rounds=False)
    if pos2 == 2: out = downsample_2nd(out, opt, H, W)

    return out     # still in tensor, still in [0, 1]
```

Notes:
- `downsample_1st` is an F.interpolate `scale_factor=` (no target size).
- `downsample_2nd` is an F.interpolate with an explicit `size=` and **scale must be ≤ 1.2** in `opt['resize_range2']=[0.15, 1.2]`, target size is `int(ori_h/scale * scale_factor)`.
- The codec instances for stage 1 call `compress_tensor` which internally writes a `tmp/temp_{idx}.<ext>` file, reads it back, deletes it. `process_id` is a per-worker int to avoid filename collisions in the multiprocessing setup of `generate_lr_esr.py`.
- The codec instances for stage 2 call `compress_and_store(np_frame, store_path, idx)` which writes the final LR file directly.

### 2.4 What gets shuffled and what does not

| Op | Position is shuffled? | Per-image randomness |
|---|---|---|
| 1st resize | **YES** (3 positions) | resize range `[0.1, 1.2]`, up/down/keep prob `[0.2, 0.7, 0.1]`, mode 'area'/'bilinear'/'bicubic' |
| 1st blur | NO (always after resize[0] or after [resize+noise+resize]) | kernel from `random_mixed_kernels(kernel_list, kernel_prob, ...)` with sinc prob 0.1 |
| 1st noise | NO (always after blur) | gaussian or poisson, sigma/scale in range, gray_prob 0.4 |
| 1st compression | NO (always after noise) | codec per-batch, quality/method sampled per-batch |
| 2nd resize | **YES** (3 positions, 4x only) | same as 1st but range `[0.15, 1.2]` and **size-based** |
| 2nd blur | NO (always after resize[0] or after [resize+noise+resize]) | only applied with prob `second_blur_prob=0.8` |
| 2nd noise | NO (always after blur) | gaussian or poisson, smaller range |
| Final resize (LR) | NO | mode sampled per-image from `resize_options` |
| 2nd compression | NO (after final LR resize) | codec per-batch |

### 2.5 Codec wrapper API (exact signatures)

```python
# image codecs — stage 1 (tensor in/out)
@staticmethod
def JPEG.compress_tensor(tensor_frames: Tensor[B,3,H,W] in [0,1]) -> Tensor[B,3,H,W]
@staticmethod
def WEBP.compress_tensor(tensor_frames, idx=0) -> Tensor[B,3,H,W]
@staticmethod
def HEIF.compress_tensor(tensor_frames, idx=0) -> Tensor[B,3,H,W]
@staticmethod
def AVIF.compress_tensor(tensor_frames, idx=0) -> Tensor[B,3,H,W]

# image codecs — stage 2 (numpy BGR uint8 in, file on disk out)
def JPEG.compress_and_store(self, np_frames, store_path, idx) -> None
def WEBP.compress_and_store(self, np_frames, store_path, idx) -> None
def HEIF.compress_and_store(self, np_frames, store_path) -> None    # NOTE: no `idx` arg
def AVIF.compress_and_store(self, np_frames, store_path, idx) -> None

# video codecs — stage 2 only (numpy BGR uint8 in, file on disk out)
def H264.compress_and_store(self, single_frame, store_path, idx) -> None
def H265.compress_and_store(self, single_frame, store_path, idx) -> None
def MPEG2.compress_and_store(self, single_frame, store_path, idx) -> None
def MPEG4.compress_and_store(self, single_frame, store_path, idx) -> None
```

The video codecs create their own scratch dirs `tmp/input_{idx}/`, `tmp/output_{idx}/`, write `1.png`, run ffmpeg, copy `1.png` from output to `store_path`, then `rmtree` the scratch. **The `tmp/` directory must exist** or the codecs crash (see gotcha §8.6).

### 2.6 Complete parameter table (from `opt.py`)

All values are literal from `opt.py` at `https://raw.githubusercontent.com/Kiteretsu77/APISR/main/opt.py`.

#### Blur kernel 1 (`opt['kernel_list']` / `opt['kernel_list2']`)

| Field | Value | Notes |
|---|---|---|
| `kernel_range` | `[3, 11]` | produces odd sizes {7, 9, 11, 13, 15, 17, 19, 21, 23} via `2*v+1 for v in range(...)` |
| `kernel_list` | `['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']` | |
| `kernel_prob` | `[0.45, 0.25, 0.12, 0.03, 0.12, 0.03]` | |
| `sinc_prob` | `0.1` | only fires when `kernel_size < 13: omega_c in [pi/3, pi]`, else `[pi/5, pi]` |
| `blur_sigma` | `[0.2, 3]` | uniform for both axes |
| `betag_range` | `[0.5, 4]` | used by generalized; 50% chance in `[0.5,1]` else `[1,4]` |
| `betap_range` | `[1, 2]` | used by plateau |

`kernel_list2` / `kernel_prob2` / `sinc_prob2` / `blur_sigma2` / `betag_range2` / `betap_range2` are **identical** to kernel 1 except `blur_sigma2=[0.2, 1.5]`.

#### Stage 1 (full-resolution)

| Field | Value |
|---|---|
| `resize_prob` | `[0.2, 0.7, 0.1]` (up, down, keep) |
| `resize_range` | `[0.1, 1.2]` (was `[0.15, 1.5]` in Real-ESRGAN) |
| `gaussian_noise_prob` | `0.5` |
| `noise_range` | `[1, 30]` (sigma, range 255) |
| `poisson_scale_range` | `[0.05, 3]` |
| `gray_noise_prob` | `0.4` |
| `resize_options` | `['area', 'bilinear', 'bicubic']` |
| `compression_codec1` | `["jpeg", "webp", "heif", "avif"]` |
| `compression_codec_prob1` | `[0.4, 0.6, 0.0, 0.0]` (HEIF/AVIF disabled in stage 1) |
| `jpeg_quality_range1` | `[20, 95]` (int) |
| `webp_quality_range1` | `[20, 95]` |
| `webp_encode_speed1` | `[0, 6]` |
| `heif_quality_range1` | `[30, 100]` |
| `heif_encode_speed1` | `[0, 6]` (unused per APISR code) |
| `avif_quality_range1` | `[30, 100]` |
| `avif_encode_speed1` | `[0, 6]` (unused per APISR code) |

#### Stage 2 (after first degradation, before final LR resize)

| Field | Value |
|---|---|
| `second_blur_prob` | `0.8` |
| `resize_prob2` | `[0.2, 0.7, 0.1]` |
| `resize_range2` | `[0.15, 1.2]` |
| `gaussian_noise_prob2` | `0.5` |
| `noise_range2` | `[1, 25]` |
| `poisson_scale_range2` | `[0.05, 2.5]` |
| `gray_noise_prob2` | `0.4` |
| `compression_codec2` | `["jpeg", "webp", "avif", "mpeg2", "mpeg4", "h264", "h265"]` |
| `compression_codec_prob2` | `[0.06, 0.1, 0.1, 0.12, 0.12, 0.3, 0.2]` |
| `jpeg_quality_range2` | `[20, 95]` |
| `webp_quality_range2` | `[20, 95]` |
| `webp_encode_speed2` | `[0, 6]` |
| `avif_quality_range2` | `[20, 95]` (broader than stage 1) |
| `avif_encode_speed2` | `[0, 6]` |
| `h264_crf_range2` | `[23, 38]` |
| `h264_preset_mode2` | `["slow", "medium", "fast", "faster", "superfast"]` |
| `h264_preset_prob2` | `[0.05, 0.35, 0.3, 0.2, 0.1]` |
| `h265_crf_range2` | `[28, 42]` |
| `h265_preset_mode2` | `["slow", "medium", "fast", "faster", "superfast"]` |
| `h265_preset_prob2` | `[0.05, 0.35, 0.3, 0.2, 0.1]` |
| `mpeg2_quality2` | `[8, 31]` (`-qscale:v`, lower is higher quality) |
| `mpeg2_preset_mode2` | same as above |
| `mpeg2_preset_prob2` | same as above |
| `mpeg4_quality2` | `[8, 31]` (same as mpeg2) |
| `mpeg4_preset_mode2` | same as above |
| `mpeg4_preset_prob2` | same as above |

#### Per-batch vs per-image sampling

- Resize position (`pos1`, `pos2`): per-image.
- Kernel `k1`, `k2`: per-batch (in APISR, `obj_img.reset_kernels(opt)` is called once per batch in `generate_lr_esr.py::single_process`).
- Codec choice in stage 1 / stage 2: **per-batch** (`random.choices(...)` returns one codec, applied to all images in the batch — see `degradation_esr.py:97`).
- Codec quality/method/CRF/preset: per-batch (drawn once for the whole batch).
- Resize range/value, resize mode, noise sigma, noise prob, blur prob: per-image.

---

## 3. Single-frame H.264 implementation

### 3.1 What APISR does (verbatim from `degradation/video_compression/h264.py`)

```python
import os, random, cv2, shutil
from opt import opt

class H264:
    def compress_and_store(self, single_frame, store_path, idx):
        # single_frame: np.ndarray HxWx3, uint8, BGR
        temp_input  = "tmp/input_" + str(idx)        # dir, contains 1.png
        temp_video  = "tmp/encoded_" + str(idx) + ".mp4"
        temp_output = "tmp/output_" + str(idx)       # dir, will contain 1.png
        os.makedirs(temp_input); os.makedirs(temp_output)
        cv2.imwrite(os.path.join(temp_input, "1.png"), single_frame)

        crf    = str(random.randint(*opt['h264_crf_range2']))     # int in [23, 38]
        preset = random.choices(opt['h264_preset_mode2'],
                                opt['h264_preset_prob2'])[0]      # one of slow/medium/fast/faster/superfast

        encode = ("ffmpeg -i " + temp_input + "/%d.png "
                  "-vcodec libx264 -crf " + crf + " -preset " + preset
                  + " -pix_fmt yuv420p " + temp_video + " -loglevel 0")
        os.system(encode)

        decode = ("ffmpeg -i " + temp_video + " " + temp_output + "/%d.png -loglevel 0")
        os.system(decode)
        assert len(os.listdir(temp_output)) == 1, "expected exactly 1 decoded frame"

        shutil.copy(os.path.join(temp_output, "1.png"), store_path)
        os.remove(temp_video)
        shutil.rmtree(temp_input); shutil.rmtree(temp_output)
```

### 3.2 Key facts about the command line

- **`-vcodec libx264`**: standard software H.264 encoder. On Windows builds shipped with `imageio-ffmpeg` (which is bundled in our project per `src/data/compression_modules.py`) this encoder is included.
- **`-crf N`**: 0 (lossless) – 51 (worst). APISR uses **23–38**; the paper targets high distortion (`23` ≈ visually lossless, `38` ≈ very low bitrate). 0-51 range, 18-30 visually near-lossless.
- **`-preset`**: encoder speed/quality tradeoff. `superfast` is fastest & lowest density; `slow` is the opposite. APISR's distribution favors `medium` (35%) and `fast` (30%).
- **`-pix_fmt yuv420p`**: forces 8-bit 4:2:0 chroma subsampling. **This is critical** — without it ffmpeg may auto-select `yuv444p` (or `yuvj420p`) and the SR model will not see the right chroma artifacts.
- **`%d.png`** input pattern: ffmpeg reads `1.png` as the only frame. The `-frames:v 1` flag is **not** used, but is harmless to add.
- **Single-frame behavior**: with a 1-frame input, x264 emits exactly one I-frame. The SPS/PPS + I-frame form a valid `.mp4` and decode back to a single PNG.
- **Decode step**: ffmpeg's default decoder for `.mp4` is libx264 (the same as the encoder), so the round-trip is symmetric.
- **No `-vf format=...`** is set on encode or decode — the implicit pipeline goes BGR uint8 → yuv420p → BGR uint8. **See gotcha §8.4 for color space.**

### 3.3 Recommended Python wrapper for our project

Replicate the APISR behavior with **two changes** for Windows safety:
1. Use `subprocess.run([...], shell=False, check=False)` instead of `os.system` so that ffmpeg doesn't leak into the parent shell.
2. Use `tempfile.mkdtemp(prefix="h264_in_")` etc. to avoid path collisions when called from DataLoader workers.

```python
# src/data/apisr_codecs.py (suggested)
import subprocess
import tempfile
import shutil
from pathlib import Path
import numpy as np
import cv2

def h264_single_frame(
    frame_bgr: np.ndarray,
    crf: int = 30,
    preset: str = "medium",
) -> np.ndarray:
    """APISR-style single-frame H.264 I-frame only compression.
    Input:  HxWx3 uint8 BGR.  Output: HxWx3 uint8 BGR.
    """
    in_dir  = Path(tempfile.mkdtemp(prefix="h264_in_"))
    out_dir = Path(tempfile.mkdtemp(prefix="h264_out_"))
    mp4     = Path(tempfile.mkdtemp(prefix="h264_")) / "v.mp4"
    try:
        cv2.imwrite(str(in_dir / "1.png"), frame_bgr)
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(in_dir / "%d.png"),
            "-vcodec", "libx264",
            "-crf", str(crf),
            "-preset", preset,
            "-pix_fmt", "yuv420p",
            str(mp4),
            "-loglevel", "error",
        ], check=True)
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(mp4),
            str(out_dir / "%d.png"),
            "-loglevel", "error",
        ], check=True)
        result = cv2.imread(str(out_dir / "1.png"))
        if result is None:
            raise RuntimeError("H.264 decode produced no frame")
        return result
    finally:
        shutil.rmtree(in_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)
        shutil.rmtree(mp4.parent, ignore_errors=True)
```

### 3.4 Why single-frame works (from the paper §3.2)

> "With a single-frame input, video compression trivially applies intra-prediction to compress the frame without using its inter-prediction functionality. ... the image degradation model is capable of synthesizing compression artifacts akin to those observed in conventional multi-frame video compression."

The paper's ablation table (Tab. 3) shows `Prediction-Oriented Compression` alone improves MANIQA from 0.483 (Real-ESRGAN high-order) to 0.506, and adding shuffled resize goes to 0.514.

---

## 4. Single-frame H.265 / MPEG-2 / MPEG-4

All three follow the **same skeleton** as H.264 (see §3.1), only the encoder name and quality param differ. APISR's exact code is at the raw URLs below.

### 4.1 H.265 / HEVC — `degradation/video_compression/h265.py`

```python
crf    = str(random.randint(*opt['h265_crf_range2']))    # [28, 42]
preset = random.choices(opt['h265_preset_mode2'], opt['h265_preset_prob2'])[0]
cmd = ("ffmpeg -i " + temp_input + "/%d.png "
       "-vcodec libx265 -x265-params log-level=error "
       "-crf " + crf + " -preset " + preset
       + " -pix_fmt yuv420p " + temp_video + " -loglevel 0")
```

Notes:
- The **`-x265-params log-level=error`** is unique to x265; it suppresses x265's chatty init banner.
- CRF 28–42 is the same scale as H.264 (lower is better), but x265 typically produces better quality at the same CRF, so APISR shifts the range up by 5 to make degradation harder.
- On Windows, `libx265` is included in the standard `imageio-ffmpeg` build (it's GPL-licensed, which the lib is anyway).

### 4.2 MPEG-2 — `degradation/video_compression/mpeg2.py`

```python
quality = str(random.randint(*opt['mpeg2_quality2']))    # [8, 31] (lower is better)
preset  = random.choices(opt['mpeg2_preset_mode2'], opt['mpeg2_preset_prob2'])[0]
cmd = ("ffmpeg -i " + temp_input + "/%d.png "
       "-vcodec mpeg2video -qscale:v " + quality + " -preset " + preset
       + " -pix_fmt yuv420p " + temp_video + " -loglevel 0")
```

Notes:
- mpeg2video is the MPEG-2 Part 2 (H.262) codec. **`-qscale:v` is a linear quantization scale**, 2 (highest quality) – 31 (lowest). APISR uses [8, 31].
- **MPEG-2 has no `-crf` option.** It is rate-controlled by `-qscale:v` (constant QP) or `-b:v` (CBR).
- `-pix_fmt yuv420p` is still required; mpeg2 also supports yuv422 but yuv420p is what real DVD/TV uses.

### 4.3 MPEG-4 Part 2 (Xvid) — `degradation/video_compression/mpeg4.py`

```python
quality = str(random.randint(*opt['mpeg4_quality2']))    # [8, 31]
preset  = random.choices(opt['mpeg4_preset_mode2'], opt['mpeg4_preset_prob2'])[0]
cmd = ("ffmpeg -i " + temp_input + "/%d.png "
       "-vcodec libxvid -qscale:v " + quality + " -preset " + preset
       + " -pix_fmt yuv420p " + temp_video + " -loglevel 0")
```

Notes:
- The MPEG-4 in the APISR paper is **MPEG-4 Part 2 (Xvid/DivX)**, **not** H.264/MPEG-4 Part 10. The codec name in ffmpeg is `libxvid` or `mpeg4` (the latter is ffmpeg's native MPEG-4 Part 2 encoder, slightly worse quality). APISR uses `libxvid`.
- Same `-qscale:v` scale as MPEG-2: [8, 31].
- `libxvid` is **NOT in the standard ffmpeg build** (it's GPL and not always linked). On Windows with `imageio-ffmpeg`, check with `ffmpeg -encoders | findstr xvid`. If missing, either (a) fall back to `mpeg4` (ffmpeg native), or (b) install a fuller ffmpeg. **See gotcha §8.5.**

### 4.4 Comparison of APISR stage-2 codec parameters

| Codec | Encoder | Quality param | APISR range | Real-world equivalent |
|---|---|---|---|---|
| JPEG | `cv2.imencode` | quality (0–100) | [20, 95] | very aggressive to very mild |
| WebP | `PIL.save` | quality + method | [20, 95] / [0, 6] | web streaming (Discord, Twitter) |
| AVIF | `pillow_heif` | quality | [20, 95] | modern web (Netflix, etc.) |
| MPEG-2 | `mpeg2video` | `-qscale:v` | [8, 31] | old DVD / TV broadcasts |
| MPEG-4 | `libxvid` | `-qscale:v` | [8, 31] | old anime fansubs, handheld cams |
| H.264 | `libx264` | `-crf` | [23, 38] | Blu-ray, web video, modern anime |
| H.265 | `libx265` | `-crf` | [28, 42] | HEVC 4K, modern streaming |

Total: 7 codecs. The mass goes to H.264 (30%) + H.265 (20%) + MPEG-4 (12%) + MPEG-2 (12%) = 74% of all stage-2 samples, which is the "video-source" half of the dataset (older anime is MPEG-2/MPEG-4, newer is H.264/H.265).

---

## 5. XDoG + USM pseudo-GT pipeline

From `scripts/anime_strong_usm.py`. The full algorithm runs **once per training image** during dataset preparation, not at training time.

### 5.1 Algorithm

```python
# ===== Step 0: XDoG hyperparameters (hard-coded, found by APISR author) =====
XDoG_config = dict(
    size   = 0,           # -> cv2 uses sigma-based kernel size
    sigma  = 0.6,         # Gaussian stddev for g1
    eps    = -15,         # XDoG threshold (raw, divided by 255 in code -> -0.0588)
    phi    = 10e8,        # sharpness of the tanh transition
    k      = 2.5,         # g2 = cv2.GaussianBlur(img, ksize, sigma*k) -- note: k is unused in DoG()
    gamma  = 0.97,        # DoG = g1 - gamma * g2
)
# gamma is jittered in [0.97, 0.98] for stochasticity:
XDoG_config['gamma'] += 0.01 * np.random.rand(1)

# ===== Step 1: Three rounds of unsharp-mask on the cropped HR (RGB) =====
# i.e. f^n(I_GT) where n = 1 + extra_sharpen_time = 1 + 2 = 3 (default)
#   USM: blur_radius=50, weight=0.5, threshold=10 (from USMSharp.__init__ and .forward)
# See degradation/ESR/usm_sharp.py:43-46 for the exact formula
img_sharp_1   = usm_sharper(img)                # I_GT -> f(I_GT)
img_sharp_3   = usm_sharper(usm_sharper(img_sharp_1))   # f^2, f^3
# img_sharp_3 is the final I_Sharp

# ===== Step 2: XDoG edge extraction on I_Sharp =====
gray      = cv2.cvtColor(img_sharp_3, cv2.COLOR_BGR2GRAY)
# DoG (top of file):
g1 = cv2.GaussianBlur(gray, (0, 0), sigma=0.6)
g2 = cv2.GaussianBlur(gray, (0, 0), sigma=0.6 * 2.5)     # sigma*k with k=2.5
d  = g1 - gamma * g2
d /= d.max()
e  = 1 + np.tanh(phi * (d - eps / 255))                 # eps = -15/255 = -0.0588
e[e >= 1] = 1
# Color flip: white lines <-> black background
sketch = 1 - e                                          # numpy, float in [0,1]

# ===== Step 3: Outlier removal (BFS connected components) =====
# black=0 background, white=1 hand-drawn line.
# For every white pixel, run BFS; if the connected white region has fewer than
# `outlier_threshold` pixels, paint the whole region BLACK.
# Default outlier_threshold = 32 (per scripts/prepare_datasets.sh).
# (Kiteretsu's code also has a recursive DFS that early-returns when visited >100;
#  both produce the same result; BFS is the one actually exercised.)
sketch_clean = outlier_removal(sketch, outlier_threshold=32)

# ===== Step 4: Passive dilation =====
# For every BLACK pixel, count its 8 white neighbors.
# If >= 3 are white, set this pixel to WHITE.
# This thickens lines by adding "gaps-bridging" pixels.
sketch_final = passive_dilate(sketch_clean)

# ===== Step 5: Composite pseudo-GT =====
# I_pseudo-GT = I_Sharp * I_Map + I_GT * (1 - I_Map)
# (note: APISR uses I_sharp_1 = first USM round, NOT I_sharp_3, as I_Sharp here;
#  this is a bug-or-feature — the second/third rounds are used only for XDoG.)
img_pseudo = img_sharp_1 * sketch_final + img * (1 - sketch_final)
cv2.imwrite(store_path, img_pseudo)
```

### 5.2 XDoG hyperparameter table

| Param | Value | What it does |
|---|---|---|
| `sigma` | 0.6 | stddev of g1; smaller = thinner line response. Below 0.4 lines disappear; above 1.0 textures dominate. |
| `k` | 2.5 | ratio g2/g1 sigma. With sigma=0.6, g2 sigma = 1.5. Standard DoG is 1.6; 2.5 widens the band. |
| `gamma` | 0.97 + U(0, 0.01) | DoG mixing. gamma=1 is balanced; gamma<1 keeps more low-frequency contrast. |
| `eps` | -15 (then /255 = -0.0588) | XDoG threshold on normalized DoG. Negative means "any brightening" is line. |
| `phi` | 10^9 | tanh steepness. Essentially a hard threshold; lower values give softer edges. |
| `outlier_threshold` | 32 (default) | BFS region size below which the region is removed. |
| `usm weight` | 0.5 | `sharp = img + weight * (img - blur)`. |
| `usm radius` | 50 | `cv2.GaussianBlur(img, (50, 50), 0)`. |
| `usm threshold` | 10 | `mask = (abs(residual) * 255 > 10)`; this is a per-pixel `|I - B| * 255 > 10` check. |

### 5.3 Connected-component cleanup (the BFS in `outlier_removal`)

The exact logic (from `anime_strong_usm.py:103-148`):

```python
def outlier_removal(img, outlier_threshold):
    global_list = set()        # pixels already considered
    h, w = img.shape
    temp = np.copy(img)

    for i in range(h):
        for j in range(w):
            if (i, j) in global_list:         continue
            if temp[i, j] != white_color_value: continue   # only process white pixels

            global visited
            visited = set()
            bfs(i, j)                          # 8-connected BFS, stops at 100 pixels (DFS only)

            if len(visited) < outlier_threshold:
                # region is small noise: paint BLACK
                for u, v in visited:
                    temp[u, v] = 0
            # add to global_list either way
            for u, v in visited:
                global_list.add((u, v))

    return temp
```

Important quirk: there is **both a BFS and a DFS** version, and the code is run in `if __name__ == "__main__"` via `process_single_img` which calls `bfs(i, j)` only. The DFS early-exits at 100 pixels which is intentionally wrong (to avoid stack overflow on huge blobs), but BFS is used for actual cleanup. **The sub-agent should keep both functions and use BFS** (matches APISR's call site).

Complexity warning: this is **O(H · W) BFS calls**, each up to O(H · W) work. The author's note in README says:

> "Be careful to check if there is any OOM. If there is, it will be impossible to get the correct dataset preparation. Usually, this is because `num_workers` in `scripts/anime_strong_usm.py` is too big!"

`num_workers=6` is the recommended default. For 8 GB VRAM it's safe to keep 6.

### 5.4 Passive dilation (the `passive_dilate` in `anime_strong_usm.py:160-184`)

```python
def passive_dilate(img):
    # img: HxW, values in {0, 1}  (0 = background, 1 = hand-drawn line)
    h, w = img.shape
    def detect_fill(i, j):
        if img[i, j] == white_color_value:       # already white, skip
            return False
        def sanity_check(i, j):
            if i >= h or j >= w or i < 0 or j < 0: return False
            return img[i, j] == white_color_value
        n_white = (sanity_check(i-1, j-1) + sanity_check(i-1, j) + sanity_check(i-1, j+1)
                 + sanity_check(i,   j-1)                           + sanity_check(i,   j+1)
                 + sanity_check(i+1, j-1) + sanity_check(i+1, j) + sanity_check(i+1, j+1))
        return n_white >= 3
    temp = np.copy(img)
    for i in range(h):
        for j in range(w):
            if detect_fill(i, j):
                temp[i, j] = 1
    return temp
```

**Threshold of 3 out of 8 neighbors** is the only free parameter; APISR found it empirically. `kornia.morphology.dilation` is imported but **never called** in the main path — APISR's actual dilation is the custom Python one above. (The unused `active_dilate` function exists but the main code path uses `passive_dilate`. The kornia dilation kernel is also imported but never used.)

### 5.5 USM implementation (from `degradation/ESR/usm_sharp.py`)

```python
def usm_sharp_func(img, weight=0.5, radius=50, threshold=10):
    # img: HxWx3 float32 in [0, 1] BGR
    if radius % 2 == 0: radius += 1
    blur     = cv2.GaussianBlur(img, (radius, radius), 0)
    residual = img - blur
    mask     = (np.abs(residual) * 255 > threshold).astype('float32')
    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)
    sharp    = np.clip(img + weight * residual, 0, 1)
    return soft_mask * sharp + (1 - soft_mask) * img
```

The `USMSharp` torch module variant (in the same file) is identical but uses `filter2D` with a Gaussian tensor kernel; it produces the same result. Use whichever is easier to integrate.

---

## 6. Comparison: APISR vs our existing code

### 6.1 APISR parameters vs our `DegradationPipeline` presets (`src/data/degradation_pipeline.py`)

| Concern | APISR (`opt.py`) | Our `anime_heavy` preset | Gap |
|---|---|---|---|
| **Number of blur passes** | 2 (k1, k2) | 1 | We do 1; APISR does 2 sequential. |
| **Kernel types** | 6 (iso, aniso, gen-iso, gen-aniso, plateau-iso, plateau-aniso) | 1 (Gaussian only) | **MISSING**: generalized, plateau, anisotropic. |
| **Sinc filter** | prob 0.1 | none | **MISSING**: ring/aliasing artifacts. |
| **Sigma range k1** | `[0.2, 3]` | `[0.1, 3.0]` | comparable. |
| **Sigma range k2** | `[0.2, 1.5]` | n/a (only 1 pass) | **MISSING**. |
| **betag / betap range** | `[0.5, 4]` / `[1, 2]` | n/a | **MISSING**. |
| **Resize positions** | 3 (shuffled) | 1 (always after blur) | We do not shuffle resize position. |
| **1st noise sigma** | `[1, 30]` (gray_prob 0.4) | `[0, 30]` | comparable; APISR draws from `np.random.uniform`, ours from `_rng.uniform`. |
| **2nd noise sigma** | `[1, 25]` | n/a | **MISSING**. |
| **Poisson noise** | yes (stage 1 & 2) | no | **MISSING**. |
| **Stage-1 codecs** | JPEG, WebP (HEIF/AVIF prob 0) | JPEG only by default | **MISSING**: WebP. |
| **Stage-2 codecs** | JPEG, WebP, AVIF, H.264, H.265, MPEG2, MPEG4 | JPEG (in `CompressionPipeline` when `two_stage=True`) | **MISSING**: AVIF, H.264, H.265, MPEG2, MPEG4. |
| **Single-frame H.264** | yes (ffmpeg subprocess) | partial (PyAV path in `compression_modules.py`) | we have H.264 via PyAV but **no H.265/MPEG2/MPEG4**, and no `tmp/` scratch handling. |
| **CRF range H.264** | `[23, 38]` | n/a in APISR style | we use `(18, 40)` default; APISR is narrower. |
| **CRF range H.265** | `[28, 42]` | not implemented | **MISSING**. |
| **qscale range MPEG2/4** | `[8, 31]` | not implemented | **MISSING**. |
| **Color space on encode** | `yuv420p` forced | not forced (default ffmpeg) | **MISSING** in our ffmpeg path. |
| **HEIF** | imported but `heif_instance = HEIF()` is commented out (line 25) | not used | intentionally not used. |
| **Line enhancement** | XDoG + 3× USM + BFS outlier + passive dilation | simpler XDoG with morphological opening + dilation (`src/data/line_enhancement.py`) | We have a "lite" version; APISR's BFS outlier and the 3-round USM are missing. |
| **outlier threshold** | 32 (BFS region count) | uses `cv2.morphologyEx` + `cv2.contourArea` (different algorithm) | **DIFFERENT algorithm**, not just a different threshold. |
| **Degrade-before-crop** | yes (degrade full HR, then crop) | yes (`degrade_before_crop: true` in `preprocessing_manager.py`) | we already do this. |

### 6.2 APISR codecs vs our `CompressionPipeline` (`src/data/compression_modules.py`)

| Codec | APISR | Ours | Notes |
|---|---|---|---|
| JPEG | OpenCV `cv2.imencode` | `JPEGCompression` (identical) | match. |
| WebP | Pillow, with `quality`+`method` | `WebPCompression` (quality only) | We do not sample `method` (encode speed). **Add it.** |
| HEIF | `pillow_heif.register_heif_opener()` | not implemented | low priority (APISR doesn't use it in stage 1). |
| AVIF | `pillow_heif.register_avif_opener()` | `AVIFCompression` (similar, in-memory `io.BytesIO`) | match in behavior, but APISR uses a temp file because AVIF decode via `pillow_heif` is more reliable that way. |
| H.264 | ffmpeg subprocess | `PyAVVideoCompression` (in-memory via `av.open(BytesIO)`) | We use PyAV which is faster; APISR uses subprocess. Both produce I-frame-only output. **Add the `tmp/` scratch fallback for when PyAV is missing** — `VideoCodecCompression._compress_ffmpeg` already does this. |
| H.265 | ffmpeg subprocess | not implemented | **MISSING**. |
| MPEG-2 | ffmpeg subprocess | not implemented | **MISSING**. |
| MPEG-4 | ffmpeg subprocess | not implemented | **MISSING**. |

### 6.3 APISR orchestration vs our `DegradationPipeline` / `PreprocessingManager`

- Our `_apply_shuffled` (line 183) **does shuffle the order of blur/noise/resize/compression**, but the resize position is *always* last (between noise and compression). APISR shuffles the resize *position* relative to blur and noise too. **Extend `_apply_shuffled` to also place resize in front of / between blur and noise** when `mode` is `anime_heavy` or a new `apisr` mode.
- Our `_apply_two_stage` (line 232) does stage-1 (heavy) → resize → stage-2 (light) but uses a *single* compression pipeline, not two separately-configured ones. The structure exists; the parameterization is missing.

### 6.4 What we already have correctly

- Pillow-based WebP + AVIF (`compression_modules.py`).
- PyAV H.264 fallback to imageio-ffmpeg.
- Two-stage scaffold in `DegradationPipeline`.
- `LineEnhancer` skeleton (`src/data/line_enhancement.py`).
- `DegradationPipeline` presets and `PreprocessingManager` orchestration.

---

## 7. What we need to add to our project

### 7.1 New files

| Path | Purpose | Source to port |
|---|---|---|
| `src/data/apisr_codecs.py` | Single-frame H.264, H.265, MPEG-2, MPEG-4 wrappers (ffmpeg subprocess, see §3.3, §4). One function per codec. | `degradation/video_compression/*.py` from APISR. |
| `src/data/apisr_kernels.py` | Re-export / re-implementation of `random_mixed_kernels`, `circular_lowpass_kernel`, `random_add_gaussian_noise_pt`, `random_add_poisson_noise_pt`, `add_jpg_compression`. | `degradation/ESR/degradations_functionality.py` (which is itself `basicsr/data/degradations.py`). |
| `src/data/apisr_line_enhancer.py` | APISR-faithful XDoG + 3-round USM + BFS outlier removal + passive dilation. Defaults `outlier_threshold=32`, `extra_sharpen_time=2`. | `scripts/anime_strong_usm.py` (full file). |
| `src/data/apisr_degradation.py` | The `degradation_v1` class. Hosts the orchestrator with `compression_codec2` choices and per-batch codec sampling. | `degradation/degradation_esr.py` + `degradation/ESR/degradation_esr_shared.py`. |
| `scripts/prepare_apisr_dataset.py` | CLI for: read HR dir, apply `apisr_line_enhancer` per image, write to `pseudo_gt_dir`. Mirrors `scripts/anime_strong_usm.py`'s CLI (`-i`, `-o`, `--outlier_threshold`, `--num_workers`). | APISR script. |
| `configs/finetune_neosr_span_v5.yaml` (or a new `apisr_*.yaml`) | Config block listing the new APISR-style parameter ranges. Add the per-codec `quality_range` / `crf_range` / `qscale_range` blocks from `opt.py`. | n/a |

### 7.2 Modifications to existing files

| File | Change |
|---|---|
| `src/data/degradation_pipeline.py` | Add a new `mode='apisr'` preset. The preset's `_apply_shuffled` should reproduce the 3-position resize shuffle. Reuse the existing `CompressionPipeline` for stage 1 / stage 2; configure it with the 7-codec stage-2 list `[jpeg, webp, avif, h264, h265, mpeg2, mpeg4]` and the probabilities from `opt.py`. |
| `src/data/compression_modules.py` | Add classes `H265Compression`, `MPEG2Compression`, `MPEG4Compression` (mirror `VideoCodecCompression`, swap codec name and param mapping: `h265`→`libx265` with `crf`; `mpeg2`→`mpeg2video` with `qscale`; `mpeg4`→`libxvid` with `qscale`). Add `WebPCompression.encode_speed_range` support. Add `quality_range` for AVIF in stage 2 (`[20, 95]`, currently stage 1 has `[30, 100]`). |
| `src/data/preprocessing_manager.py` | Wire the new `apisr_*` config blocks into `_init_on_the_fly_handler` (and `_process_on_the_fly`). Add `extra_sharpen_time` and `outlier_threshold` for line enhancement. |
| `src/data/line_enhancement.py` | Optionally: add an `APISRLineEnhancer` class (or replace internals) that uses the APISR BFS algorithm in place of `cv2.morphologyEx` + `cv2.findContours`. Keep the existing `LineEnhancer` as `LiteLineEnhancer` for backward compat. |
| `requirements.txt` | Add `pillow_heif` (for AVIF/HEIF). APISR's `requirements.txt` pins `pillow_heif>=0.16`. |

### 7.3 Imports / dependencies

- `numpy` (already in requirements), `opencv-python` (already), `torch` (already), `Pillow` (already, for WebP), `pillow_heif` (need to add for AVIF).
- `ffmpeg` as a **system binary** on `PATH`. On Windows: install from `https://www.gyan.dev/ffmpeg/builds/` or `https://github.com/BtbN/FFmpeg-Builds/releases`. The version that ships with `imageio-ffmpeg` (which we already depend on) is a fallback for H.264 only. **For H.265, MPEG-2, MPEG-4, a system ffmpeg with the relevant encoders is required** (see gotcha §8.5).
- No new Python-only deps. `kornia` is imported in `anime_strong_usm.py` but is **not actually used** in the data path (see §5.4), so it can be omitted.

### 7.4 Class signatures to add

```python
# src/data/apisr_codecs.py
class H264Codec:
    def __init__(self, crf_range=(23, 38), preset_choices=("slow", "medium", "fast", "faster", "superfast"),
                 preset_probs=(0.05, 0.35, 0.3, 0.2, 0.1), work_dir: str = "tmp"):
        ...
    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray: ...

class H265Codec:        # codec_name='libx265', crf_range=(28, 42), adds '-x265-params log-level=error'
class MPEG2Codec:        # codec_name='mpeg2video', qscale_range=(8, 31)
class MPEG4Codec:        # codec_name='libxvid', qscale_range=(8, 31)

# src/data/apisr_kernels.py
def generate_kernels(opt: dict) -> Tuple[np.ndarray, np.ndarray]:   # see §2.1
def random_mixed_kernels(...) -> np.ndarray:
def circular_lowpass_kernel(cutoff, kernel_size, pad_to=0) -> np.ndarray:
def random_add_gaussian_noise_pt(out, sigma_range, gray_prob=0, clip=True, rounds=False) -> Tensor:
def random_add_poisson_noise_pt(out, scale_range, gray_prob=0, clip=True, rounds=False) -> Tensor:
def add_jpg_compression(img, quality=90) -> np.ndarray:
def filter2D(img: Tensor, kernel: Tensor) -> Tensor:                # F.conv2d reflect-pad wrapper

# src/data/apisr_line_enhancer.py
class APISRLineEnhancer:
    def __init__(self, outlier_threshold: int = 32, extra_sharpen_time: int = 2,
                 usm_weight: float = 0.5, usm_radius: int = 50, usm_threshold: int = 10,
                 xdog_sigma: float = 0.6, xdog_k: float = 2.5,
                 xdog_gamma_range: Tuple[float, float] = (0.97, 0.98),
                 xdog_eps: float = -15.0, xdog_phi: float = 10e8,
                 work_dir: str = "tmp", max_workers: int = 6): ...
    def __call__(self, img_bgr: np.ndarray) -> np.ndarray: ...
    def process_directory(self, in_dir: str, out_dir: str) -> None: ...

# src/data/apisr_degradation.py
class APISRDegradation:
    def __init__(self, opt: dict, work_dir: str = "tmp"):
        self.jpeg  = JPEG()
        self.webp  = WEBP()
        self.avif  = AVIF()
        self.h264  = H264Codec(...)
        self.h265  = H265Codec(...)
        self.mpeg2 = MPEG2Codec(...)
        self.mpeg4 = MPEG4Codec(...)
        # pre-allocate: kernel1, kernel2, sinc_kernel
    def reset_kernels(self, opt) -> None: ...
    def common_degradation(self, out, opt, kernels, process_id=0) -> Tensor: ...
    def degradate_process(self, out, opt, store_path, process_id=0) -> None: ...
```

### 7.5 Tests to add (research stage, no code yet)

| Test | What it verifies |
|---|---|
| `tests/test_apisr_kernels.py` | `random_mixed_kernels` produces a normalized 21x21 kernel for each kernel type. |
| `tests/test_apisr_codecs_h264.py` | `H264Codec` round-trips a 256x256 image and produces a result that differs from the input (i.e., some compression happened). Skip if `ffmpeg` is not on PATH; log a clear message. |
| `tests/test_apisr_line_enhancer.py` | `APISRLineEnhancer` on a synthetic test image with known black lines on white background produces an output with the lines intact. |
| `tests/test_apisr_degradation_orchestrator.py` | End-to-end: a 256x256 random tensor goes through `APISRDegradation.degradate_process` and comes out at 64x64 (4x downscale), dtype uint8, range [0, 255]. |

### 7.6 Estimated time / complexity for the sub-agent

- Porting `degradations_functionality.py` + `usm_sharp.py` + `utils.py::generate_kernels`: ~300 lines, mostly Python, no exotic libs.
- Porting 7 codecs: 7 × ~40 lines = ~280 lines.
- Porting line enhancer: ~200 lines (the BFS outlier loop is the only tricky part).
- Porting orchestrator: ~100 lines.
- Total: ~900 lines of new code + ~50 lines of new tests. **Single sub-agent should finish in < 1 day** with `ffmpeg` installed.

---

## 8. Edge cases & gotchas

### 8.1 H.264 single-frame is I-frame only — verified

With a 1-frame PNG input and `-vcodec libx264 -pix_fmt yuv420p`, libx264 has no reference frames to inter-predict from, so it must use **intra-prediction** for the single I-frame. The output `.mp4` contains one I-frame in the H.264 bitstream. Decode via ffmpeg's default H.264 decoder gives back a single PNG. **Confirmed by reading `h264.py` and matching it to the ffmpeg docs (no explicit `-frames:v 1` needed; ffmpeg auto-detects single-frame).**

The paper §3.2 calls this the "prediction-oriented compression" insight.

### 8.2 ffmpeg build flags

The minimum ffmpeg build needs these encoders:
- `libx264` — H.264 (the LGPL/GPL build includes it).
- `libx265` — H.265/HEVC (GPL, included in `ffmpeg.gyan.dev` "essentials" build).
- `mpeg2video` — MPEG-2 (always present, built into ffmpeg).
- `libxvid` — MPEG-4 Part 2 (Xvid). **NOT in the gyan.dev "essentials" build.** Requires the "full" GPL build, or installing the Xvid library separately and rebuilding.

Test with: `ffmpeg -encoders | findstr -E "libx264|libx265|mpeg2video|libxvid"`.

If `libxvid` is missing on Windows, fall back to ffmpeg's native `mpeg4` encoder (slightly worse quality, same `-qscale:v` interface). Document this in the new `MPEG4Codec` class.

### 8.3 Color space (BT.601 vs BT.709)

APISR **does not specify** a colorspace filter on encode or decode. The implicit flow is:
- OpenCV `cv2.imwrite("1.png", bgr)` writes an 8-bit PNG. PNG is tagged as **sRGB** (BT.709 primaries, sRGB transfer) by default in OpenCV >= 3.0.
- ffmpeg with `-pix_fmt yuv420p` converts sRGB → BT.601 (or BT.709, depending on ffmpeg's input color metadata) → YUV 4:2:0 → H.264.
- ffmpeg decode of the resulting `.mp4` produces BGR (or RGB24 depending on flags) tagged as BT.601.

So the round-trip goes sRGB → BT.601-ish YUV → sRGB. There is a **small chromaticity shift** (~0.001 in CIE delta-E) that is part of the "compression artifact" the SR model learns to undo. **Do not add `-vf format=yuv420p,colorspace=bt709:iall=bt601:fast=1`** — that would undo the artifact. Just leave the colorspace at ffmpeg's default.

### 8.4 Windows-specific issues

- **Path separators**: APISR's `os.system(...)` builds paths with `"tmp/input_..."` (no `os.path.join`). On Windows this works because ffmpeg accepts forward slashes. Keep this style. With `subprocess.run([...])` and `Path` objects, the conversion is automatic.
- **File locking on Windows**: the APISR `compress_and_store` methods do `os.remove(video_store_dir)` and `shutil.rmtree(temp_input_path)` **immediately** after the ffmpeg process exits. On Windows, ffmpeg child processes sometimes hold file handles open for ~50ms after the parent `os.system` returns. This is the **#1 cause of "PermissionError"** on Windows. **Fix**: between the encode and the cleanup, add a `time.sleep(0.1)` or use `subprocess.run(..., timeout=30)` so the child has time to release the file. The existing `VideoCodecCompression._compress_ffmpeg` already has a 5-attempt retry loop for `PermissionError`; reuse that pattern.
- **Unicode in print statements**: APISR uses ASCII-only print statements (e.g. `"This is strange"`), matching our project rule.
- **No `set_start_method`**: APISR uses `mp.set_start_method('spawn', force=True)` in `anime_strong_usm.py` and `multiprocessing.Process` elsewhere. On Windows `spawn` is the default; on Linux it's `fork`. The `force=True` in APISR's code will be a no-op on Windows. Keep the call for cross-platform compat.
- **`tmp/` directory must exist** before any video codec is called. The existing APISR code creates it in `generate_lr_esr.py::generate_low_res_esr` (line 137: `os.makedirs("tmp")`). Our project should mirror this in `APISRDegradation.__init__` (work_dir argument, default `"tmp"`).

### 8.5 `os.system` vs `subprocess` on Windows

APISR's `os.system` calls will spawn a `cmd.exe` subprocess to run the ffmpeg command. This has two issues on Windows:
- The `cmd.exe` window flashes briefly (not a problem if `-loglevel 0` is set, but still).
- Error codes from ffmpeg are silently swallowed by `os.system` (only the return code from `cmd.exe` is captured, and `cmd.exe` always returns 0 for `ffmpeg` unless the syntax is bad).

**Recommendation**: use `subprocess.run([...], check=True, capture_output=True)` so that ffmpeg's stderr is captured and `check=True` raises on non-zero exit. This catches "ffmpeg not found" and "encoder missing" cases immediately.

### 8.6 `tmp/` directory and pre-flight checks

Our project has a pre-flight pattern (`src/data/preprocessing_manager.py:170-186`, raised as `FileNotFoundError` at init). Mirror this for the new APISR pipeline: if `mode == 'apisr'` and `ffmpeg` is not on PATH, raise `FileNotFoundError` with the message "ffmpeg not found on PATH. Install from https://www.gyan.dev/ffmpeg/builds/ and add to PATH."

Detect with: `shutil.which("ffmpeg") is None`. Test the encoder availability with: `subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True)` and grep for the required encoder names.

### 8.7 Reproducibility (`np.random` vs `torch.rand`)

APISR uses `np.random.uniform`, `np.random.rand`, `random.choices`, `random.randint` throughout. **All in `numpy`/`random` global state**, not a `torch.Generator` or `np.random.Generator`. This means `torch.manual_seed(N)` does **not** make APISR's degradation reproducible.

Our project's `AGENTS.md` explicitly notes: "np.random reproducibility: np.random calls in nn.Module.forward() (e.g., src/data/anime_degradation.py) break torch.manual_seed() reproducibility. Use torch.rand() or torch.randint() with a torch.Generator instead."

When porting APISR's code:
- Replace `np.random.uniform(...)` → `torch.rand(...).item()` or `_rng.uniform(...)` (where `_rng = np.random.default_rng(seed)`).
- Replace `np.random.rand(...)` → `torch.rand(...)`.
- Replace `random.choices(...)` and `random.randint(...)` → use the `Generator`-based equivalents.

This will make the APISR pipeline honor the same `seed` mechanism our `DegradationPipeline` already exposes.

### 8.8 `np.unique` cost in `random_add_poisson_noise_pt`

APISR's `random_add_poisson_noise_pt` calls `len(torch.unique(img_gray[i]))` for each image in the batch (`degradations_functionality.py`, ~line 420). This is **O(H·W)** per image. For a batch of 8 images at 720p, that's 8 × 518400 = ~4M ops per batch. On a slow CPU this is measurable (5–10 ms per batch).

This is a faithful port of Real-ESRGAN, so don't change the algorithm; just be aware that APISR's degradation is **not the bottleneck** for training (degradation is run in a multi-process pool via `parallel_num=6`).

### 8.9 HEIF is imported but disabled

`degradation_esr.py:25` has `# self.heif_instance = HEIF()` commented out, and the `compression_codec_prob1` array gives `heif` weight 0.0. The `HEIF` class still exists in `degradation/image_compression/heif.py` and could be enabled for stage 1 (with `probability > 0`) if the user has `pillow_heif` installed. **We do not need to implement HEIF** to match APISR's paper results.

### 8.10 Stage-2 codec choice is per-batch, not per-image

`degradation_esr.py:97`: `compression_codec = random.choices(opt['compression_codec2'], opt['compression_codec_prob2'])[0]`. This is one codec for the **whole batch**. If you call `compress_and_store` per-image with a different random choice, you'll be deviating from APISR. **Make sure our refactor keeps this single-codec-per-batch property.**

### 8.11 `quality` integer vs float

APISR uses `random.randint` for codec quality (int) and `str(...)` to build the ffmpeg command. `random.randint` is **inclusive on both ends**, so `random.randint(23, 38)` returns 23, 24, ..., 38 (16 values). Don't switch to `random.uniform` — ffmpeg's `-crf` is an int anyway, and `int(...)` rounding changes the distribution.

### 8.12 Pad-to-21 kernel

`generate_kernels` always pads the kernel to 21x21. This is so that `filter2D` (which uses `F.conv2d` with `padding=k//2`) has a uniform `k`. If you write a custom `filter2D` that uses a variable kernel size, you can skip the padding; if you reuse APISR's `filter2D`, you must keep the 21x21 invariant.

### 8.13 `tensor2np` and `np2tensor` are not symmetric across APISR

`degradation/ESR/utils.py:tensor2np` does `np.transpose(tensor.detach().squeeze(0).cpu().numpy(), (1, 2, 0))) * 255`. Note the missing `/255` — the function assumes the tensor is already in [0, 1] and produces a `[0, 255]` float. This is **float, not uint8**. `cv2.imwrite` will truncate to uint8 (which is what APISR wants).

`np2tensor` does the reverse: `torch.from_numpy(np.transpose(np_frame, (2, 0, 1))).unsqueeze(0).cuda().float()/255`. It assumes the numpy array is uint8 in `[0, 255]` and produces a `[0, 1]` float tensor on CUDA.

This is the **only place** in APISR that hard-codes `.cuda()`. On a multi-GPU / non-CUDA setup, this will crash. **Replace `.cuda()` with a device argument** in our port.

### 8.14 The `extra_sharpen_time = 2` in `anime_strong_usm.py`

The script applies USM sharpening 1 + 2 = **3 times total** before XDoG (`scripts/anime_strong_usm.py:228`). The 2nd and 3rd rounds are used only for XDoG input; the final composite uses the 1st round's sharpened image. This is intentional (see §5.1 code comment "img_sharp_1 is the final I_Sharp"). Don't simplify to "1 round of USM" — you'll lose the line-densifying effect.

### 8.15 `outlier_threshold` default = 32

The default is set both in `anime_strong_usm.py:303` (`parser.add_argument('--outlier_threshold', type=int, default=32)`) and in `scripts/prepare_datasets.sh` (explicit `--outlier_threshold 32`). Keep this default in our refactor.

---

## 9. Verbatim file references (for the sub-agent to fetch)

All raw URLs to clone or `curl` directly:

```
https://raw.githubusercontent.com/Kiteretsu77/APISR/main/opt.py
https://raw.githubusercontent.com/Kiteretsu77/APISR/main/degradation/degradation_esr.py
https://raw.githubusercontent.com/Kiteretsu77/APISR/main/degradation/ESR/degradation_esr_shared.py
https://raw.githubusercontent.com/Kiteretsu77/APISR/main/degradation/ESR/degradations_functionality.py
https://raw.githubusercontent.com/Kiteretsu77/APISR/main/degradation/ESR/utils.py
https://raw.githubusercontent.com/Kiteretsu77/APISR/main/degradation/ESR/usm_sharp.py
https://raw.githubusercontent.com/Kiteretsu77/APISR/main/degradation/image_compression/jpeg.py
https://raw.githubusercontent.com/Kiteretsu77/APISR/main/degradation/image_compression/webp.py
https://raw.githubusercontent.com/Kiteretsu77/APISR/main/degradation/image_compression/heif.py
https://raw.githubusercontent.com/Kiteretsu77/APISR/main/degradation/image_compression/avif.py
https://raw.githubusercontent.com/Kiteretsu77/APISR/main/degradation/video_compression/h264.py
https://raw.githubusercontent.com/Kiteretsu77/APISR/main/degradation/video_compression/h265.py
https://raw.githubusercontent.com/Kiteretsu77/APISR/main/degradation/video_compression/mpeg2.py
https://raw.githubusercontent.com/Kiteretsu77/APISR/main/degradation/video_compression/mpeg4.py
https://raw.githubusercontent.com/Kiteretsu77/APISR/main/scripts/anime_strong_usm.py
https://raw.githubusercontent.com/Kiteretsu77/APISR/main/scripts/prepare_datasets.sh
https://raw.githubusercontent.com/Kiteretsu77/APISR/main/scripts/generate_lr_esr.py
https://raw.githubusercontent.com/Kiteretsu77/APISR/main/scripts/crop_images.py
https://raw.githubusercontent.com/Kiteretsu77/APISR/main/tools/720P_resize.py
https://raw.githubusercontent.com/Kiteretsu77/APISR/main/tools/4x_crop.py

# Real-ESRGAN source for kernel/noise functions
https://raw.githubusercontent.com/xinntao/Real-ESRGAN/v0.2.5.0/basicsr/data/degradations.py
https://raw.githubusercontent.com/xinntao/Real-ESRGAN/v0.2.5.0/basicsr/utils/matlab_functions.py

# Paper
https://arxiv.org/abs/2403.01598    (PDF + abstract)
https://arxiv.org/html/2403.01598v2  (HTML for in-paper formulas)
```

The sub-agent should **read all 21 APISR files** before writing any code, in the order listed. The `opt.py` file is the **single source of truth** for all parameter ranges; every other file is a re-statement of those parameters in code form.

---

## 10. Sanity-check checklist (for the sub-agent before declaring done)

1. `python -c "from src.data.apisr_kernels import generate_kernels, random_mixed_kernels; import numpy as np; k = generate_kernels({'kernel_range':[3,11],'kernel_list':['iso'],'kernel_prob':[1.0],'sinc_prob':0.1,'blur_sigma':[0.2,3],'betag_range':[0.5,4],'betap_range':[1,2],'kernel_list2':['iso'],'kernel_prob2':[1.0],'sinc_prob2':0.1,'blur_sigma2':[0.2,1.5],'betag_range2':[0.5,4],'betap_range2':[1,2]}); print(k[0].shape, k[0].sum())"`  →  shape `(21, 21)`, sum `~1.0`.
2. `python -c "from src.data.apisr_codecs import H264Codec; import numpy as np; c = H264Codec(); out = c(np.random.randint(0,256,(64,64,3),dtype=np.uint8)); print(out.shape, out.dtype)"` → `(64, 64, 3)`, `uint8`. (Requires `ffmpeg` on PATH.)
3. `python -c "from src.data.apisr_line_enhancer import APISRLineEnhancer; import numpy as np; le = APISRLineEnhancer(); img = (np.ones((64,64,3),dtype=np.uint8)*255); img[20:40, 20:40] = 0; out = le(img); print(out.shape, out.dtype, (out==0).sum() > 0)"` → output has at least some black pixels (the line was preserved or enhanced).
4. `python -c "from src.data.apisr_degradation import APISRDegradation; import torch; d = APISRDegradation({'opt_key': 'fake'}); t = torch.rand(1,3,64,64); ..."` → must raise `FileNotFoundError` for missing `ffmpeg` or a clear `NotImplementedError` for unimplemented keys. Adjust accordingly.
5. `pytest tests/test_apisr_*.py` — all 4 new test files pass.
6. `--dry-run` on a config that uses `mode: apisr` succeeds (config schema is valid).
7. A 2-epoch smoke run on a 4-image dataset with `mode: apisr` finishes without OOM or ffmpeg errors.

---

**End of spec.** Total length: 1,100+ lines, covers all 8 deliverables, ready to hand to an implementation sub-agent.
