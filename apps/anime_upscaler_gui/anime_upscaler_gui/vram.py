"""VRAM budget guard: pre-flight estimate + dispatch on mode (warn / auto / off).

Conservative math:
  output_bytes = H * scale * W * scale * 3 * dtype_bytes      # output tensor
  act_bytes    = output_bytes * 4                              # activations
  total        = output + activations + model_size_bytes
  free_vram    = torch.cuda.mem_get_info()[0]
  if total > 0.6 * free_vram: trigger
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

try:
    from .arch_registry import build
except Exception:
    build = None


@dataclass
class VramBudget:
    output_bytes: int
    act_bytes: int
    model_bytes: int
    total_bytes: int
    free_bytes: int
    triggered: bool

    def format_short(self) -> str:
        def _g(b): return f"{b / 1e9:.2f} GB"
        return (f"est output {_g(self.output_bytes)} + activations {_g(self.act_bytes)} "
                f"+ model {_g(self.model_bytes)} = {_g(self.total_bytes)}; "
                f"free VRAM {_g(self.free_bytes)}")


def free_vram_bytes() -> int:
    if not torch.cuda.is_available():
        return 0
    try:
        free, _ = torch.cuda.mem_get_info()
        return int(free)
    except Exception:
        return 0


def dtype_bytes(fp16: bool) -> int:
    return 2 if fp16 else 4


def estimate(height: int, width: int, scale: int, fp16: bool, ckpt_path: Optional[Path] = None,
             kind: Optional[str] = None) -> VramBudget:
    """Compute VRAM budget for a single forward pass.

    ckpt_path/kind optional; when provided, model size is summed from the loaded state_dict.
    When not, model size is estimated as 50 MB (a safe middle for the supported arches).
    """
    db = dtype_bytes(fp16)
    out_h = height * scale
    out_w = width * scale
    output_bytes = out_h * out_w * 3 * db
    act_bytes = output_bytes * 4
    model_bytes = 50 * 1024 * 1024
    if ckpt_path and Path(ckpt_path).exists():
        try:
            model_bytes = Path(ckpt_path).stat().st_size
        except OSError:
            pass
    total = output_bytes + act_bytes + model_bytes
    free = free_vram_bytes()
    triggered = total > 0.6 * free if free > 0 else False
    return VramBudget(output_bytes, act_bytes, model_bytes, total, free, triggered)


def suggest_downscale(height: int, width: int, scale: int, fp16: bool,
                      ckpt_path: Optional[Path] = None) -> tuple:
    """Return (new_h, new_w) that fits in the current VRAM budget."""
    target = estimate(height, width, scale, fp16, ckpt_path)
    if not target.triggered or target.free_bytes <= 0:
        return height, width
    # Want total < 0.6 * free. Approximate by scaling H,W by sqrt(budget_ratio).
    budget_ratio = (0.6 * target.free_bytes) / target.total_bytes
    s = max(0.1, budget_ratio ** 0.5)
    new_h = max(1, int(round(height * s)))
    new_w = max(1, int(round(width * s)))
    return new_h, new_w