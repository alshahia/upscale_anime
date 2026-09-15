"""Preview pane: side-by-side image, video thumbs strip / live scrub."""
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from .decoders import _Cv2Reader, is_image_path, is_video_path, load_image_rgb
from .ui_constants import DISABLED, FONT_HEADING, FONT_MONO, PREVIEW_BG, PREVIEW_LABEL


def _resize_for_canvas(rgb: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    if w <= max_w and h <= max_h:
        return rgb
    s = min(max_w / w, max_h / h)
    new_w, new_h = max(1, int(w * s)), max(1, int(h * s))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class _PreviewPane(ttk.LabelFrame):
    """Two-canvas side-by-side: original | upscaled. Videos show a thumb strip."""

    def __init__(self, parent, on_sr_ready=None):
        super().__init__(parent, text="Preview", padding=6)
        self.on_sr_ready = on_sr_ready  # callable(path) when an SR result lands
        self._tk_orig = None
        self._tk_sr = None
        self._thumbs: List[ImageTk.PhotoImage] = []
        self._thumb_paths: List[int] = []  # frame indices

        self.canvas_orig = tk.Canvas(self, bg=PREVIEW_BG, width=320, height=200, highlightthickness=0)
        self.canvas_orig.pack(side="left", fill="both", expand=True, padx=(0, 4))
        self.canvas_sr = tk.Canvas(self, bg=PREVIEW_BG, width=320, height=200, highlightthickness=0)
        self.canvas_sr.pack(side="left", fill="both", expand=True)

        # Slider for live video scrub (hidden until needed)
        self.scrub_var = tk.DoubleVar(value=0.0)
        self.scrub = ttk.Scale(self, from_=0, to=100, orient="horizontal", variable=self.scrub_var,
                               command=self._on_scrub)
        # Default hidden; .show_video() / .show_image() manage geometry.

    # ---- public API ----
    def show_image(self, input_path: Path, output_path: Optional[Path] = None):
        self.scrub.pack_forget()
        # Original
        rgb = load_image_rgb(input_path) if input_path and input_path.exists() else None
        self._draw_on(self.canvas_orig, rgb, label="original")
        # SR (may not exist yet)
        sr = load_image_rgb(output_path) if output_path and output_path.exists() else None
        self._draw_on(self.canvas_sr, sr, label="upscaled")
        if self.on_sr_ready and output_path and sr is not None:
            self.on_sr_ready(output_path)

    def show_video_thumbs(self, path: Path, n: int = 8):
        """Extract N evenly-spaced frames into a thumb strip; hide SR canvas."""
        if not path.exists():
            return
        self.scrub.pack_forget()
        try:
            reader = _Cv2Reader(path)
        except Exception:
            return
        total = max(1, reader.total)
        indices = [int(round(i * (total - 1) / (n - 1))) for i in range(n)] if n > 1 else [0]
        thumbs = []
        for idx in indices:
            try:
                reader.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, bgr = reader.cap.read()
                if not ok:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb = _resize_for_canvas(rgb, 240, 140)
                pil = Image.fromarray(rgb)
                thumbs.append((idx, ImageTk.PhotoImage(pil)))
            except Exception:
                continue
        reader.release()

        # Hide SR canvas and use its space for a thumb strip
        self.canvas_sr.pack_forget()
        strip = ttk.Frame(self)
        strip.pack(side="left", fill="both", expand=True, padx=(0, 0))
        self._thumb_strip = strip
        for i, (idx, tk_img) in enumerate(thumbs):
            lbl = ttk.Label(strip, image=tk_img)
            lbl.grid(row=0, column=i, padx=2)
            lbl.bind("<Button-1>", lambda e, i=idx: self._seek_video(path, i))
        self._thumbs = [t for _, t in thumbs]
        self._thumb_paths = [i for i, _ in thumbs]

    def _seek_video(self, path: Path, frame_idx: int):
        try:
            reader = _Cv2Reader(path)
            reader.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, bgr = reader.cap.read()
            reader.release()
            if ok:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                self._draw_on(self.canvas_orig, rgb, label=f"frame {frame_idx}")
        except Exception:
            pass

    def clear(self):
        self.canvas_orig.delete("all")
        self.canvas_sr.delete("all")

    # ---- internals ----
    def _draw_on(self, canvas: tk.Canvas, rgb: Optional[np.ndarray], label: str = ""):
        canvas.delete("all")
        if rgb is None:
            canvas.create_text(canvas.winfo_width() // 2, canvas.winfo_height() // 2,
                               text=label or "(none)", fill=DISABLED, font=FONT_HEADING)
            return
        w = max(canvas.winfo_width(), 100)
        h = max(canvas.winfo_height(), 100)
        rgb2 = _resize_for_canvas(rgb, w - 8, h - 8)
        pil = Image.fromarray(rgb2)
        tk_img = ImageTk.PhotoImage(pil)
        if canvas is self.canvas_orig:
            self._tk_orig = tk_img
        else:
            self._tk_sr = tk_img
        canvas.create_image(w // 2, h // 2, image=tk_img)
        if label:
            canvas.create_text(8, 12, text=label, fill=PREVIEW_LABEL, anchor="nw", font=FONT_MONO)

    def _on_scrub(self, value):
        pass  # wired when video live mode is enabled