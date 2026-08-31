# anime_upscaler/dataset.py
"""Anime HR/LR paired dataset for 4x distillation.

HR source: folder of high-res anime frames (data/anime_video_frames, 1920x1080 PNG).
LR is produced on-the-fly with PIL BICUBIC downscaling (x1/4) of the cropped HR
patch - perfect alignment, cheap augmentation.
Tensors are returned normalized to [-1, 1] per pipeline spec; training code maps
back to [0, 1] (model domain) explicitly via denorm01().
"""
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

SCALE = 4
TRAIN_HR_CROP = 192          # -> LR 48x48
EVAL_HR_CROP = 384           # -> LR 96x96, deterministic center crop


def _to_tensor(img):
    """HWC uint8 PIL image -> CHW float tensor in [-1, 1]."""
    arr = np.asarray(img, dtype=np.float32).copy() / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    return t * 2.0 - 1.0


def denorm01(t):
    """[-1, 1] tensor -> [0, 1] (model domain)."""
    return (t + 1.0) / 2.0


class AnimePairDataset(Dataset):
    """Returns (lr, hr) tensors in [-1, 1].

    split='train': random 192x192 HR crop + random flips / 90-degree rotations.
    split='val'/'test': deterministic center 384x384 HR crop (no augmentation),
    so validation metrics are comparable across epochs.
    """

    def __init__(self, hr_dir, split="train", split_ratios=(0.8, 0.1, 0.1),
                 scale=SCALE, seed=42, max_files=None, degradation_mode="none"):
        assert split in ("train", "val", "test")
        assert degradation_mode in ("none", "apsisr_v1"), "unknown degradation_mode: %r" % degradation_mode
        assert scale in (2, 3, 4), "unsupported scale %r (use 2 or 4)" % scale
        self.split = split
        self.scale = int(scale)
        self.degradation_mode = degradation_mode

        files = sorted(
            p for p in Path(hr_dir).iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        )
        if not files:
            raise FileNotFoundError("no image files found under %s" % hr_dir)

        # Deterministic 80/10/10 split by seeded shuffled index.
        rng = random.Random(seed)
        idx = list(range(len(files)))
        rng.shuffle(idx)
        n = len(files)
        n_train = int(n * split_ratios[0])
        n_val = int(n * split_ratios[1])
        ranges = {
            "train": idx[:n_train],
            "val": idx[n_train:n_train + n_val],
            "test": idx[n_train + n_val:],
        }
        chosen = ranges[split]
        if max_files is not None:
            chosen = chosen[:max_files]
        self.files = [files[i] for i in chosen]
        self.crop_hr = TRAIN_HR_CROP if split == "train" else EVAL_HR_CROP

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with Image.open(self.files[i]) as im:
            hr = im.convert("RGB")
        w, h = hr.size
        ch = self.crop_hr
        cl = ch // self.scale

        if self.split == "train":
            x = random.randint(0, w - ch)
            y = random.randint(0, h - ch)
        else:
            x = (w - ch) // 2
            y = (h - ch) // 2
        hr = hr.crop((x, y, x + ch, y + ch))

        # Bicubic LR from the HR crop itself. Phase 3: when
        # degradation_mode='apsisr_v1' and split='train', the HR crop is
        # passed through anime_upscaler.degradation.degrade() FIRST so the
        # bicubic LR inherits realistic codec + resize artifacts. HR stays
        # clean (the GT anchor remains pristine), and val/test are always
        # clean so metrics remain comparable to the v1 baseline.
        hr_for_lr = hr
        if self.split == "train" and self.degradation_mode != "none":
            from degradation import degrade  # lazy import: keeps smoke cheap
            hr_for_lr = degrade(hr)
        lr = hr_for_lr.resize((cl, cl), Image.BICUBIC)

        if self.split == "train":
            if random.random() < 0.5:
                hr = hr.transpose(Image.FLIP_LEFT_RIGHT)
                lr = lr.transpose(Image.FLIP_LEFT_RIGHT)
            k = random.randint(0, 3)
            if k:
                rot = getattr(Image, "ROTATE_%d" % (k * 90))
                hr = hr.transpose(rot)
                lr = lr.transpose(rot)

        return _to_tensor(lr), _to_tensor(hr)


if __name__ == "__main__":
    # Stage-1 smoke check: shapes, value ranges, split sizes.
    root = Path(__file__).resolve().parent.parent
    data = root / "data" / "anime_video_frames"
    for split in ("train", "val", "test"):
        ds = AnimePairDataset(str(data), split=split)
        lr, hr = ds[0]
        print("%s: n=%d lr=%s hr=%s lr[%.2f,%.2f] hr[%.2f,%.2f]" % (
            split, len(ds), tuple(lr.shape), tuple(hr.shape),
            lr.min(), lr.max(), hr.min(), hr.max()))
    sm = AnimePairDataset(str(data), split="train", max_files=8)
    print("smoke subset:", len(sm), sm.files[0].name[:40])
