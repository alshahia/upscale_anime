# V8+ MambaIRv2 Plan (WSL2 + uv + Tier 1 Bundle)

**Status:** Tier 1 (TTA + Souping) shipped on `perf/v7-tier1-speedups`; MambaIRv2/WSL2 (Phases A+B) deferred to user
**Last updated:** 2026-06-20
**Owner:** user GPU (RTX 4000 Mobile 8GB) + WSL2 Ubuntu-24.04
**Goal:** Verify MambaIRv2 path on Windows + ship TTA + model souping

---

## A. WSL2 Setup with `uv` (one-time, ~15-30 min)

**Tool change:** Use `uv` (https://docs.astral.sh/uv/) instead of `pip`. `uv` is the project's standard package manager per AGENTS.md "Setup" section. `uv pip install` is a drop-in for `pip install` but ~10-100x faster.

### A.1 Install uv in WSL2 (skip if already installed)

```bash
wsl -d Ubuntu-24.04
# uv installs in 1 command; no Python needed
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version  # expect 0.4.x or later
```

### A.2 Bootstrap Python 3.11 in WSL2 via uv

```bash
# Pin Python 3.11 because mamba-ssm pre-built wheels are most reliable for 3.11 + cu124.
# (3.12 may not have a wheel for cu130 yet; 3.11 is the safer target.)
uv python install 3.11
uv python list --only-installed  # confirm 3.11 is installed
```

### A.3 Create `.venv` with uv, point it at the project

```bash
mkdir -p ~/projects/upscale_anime_wsl
cd ~/projects/upscale_anime_wsl

# Symlink to the Windows-side project dir (faster than copying + stays in sync)
ln -sfn "/mnt/e/python projects/upscale_anime" project
cd project

# Create venv with Python 3.11
uv venv --python 3.11 .venv
source .venv/bin/activate
uv --version  # sanity check
python --version  # expect 3.11.x
```

### A.4 Install PyTorch CUDA 12.4 via uv (matches WSL2's CUDA 13.2 driver + mamba-ssm wheel availability)

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

(Alternative `cu126` works too. cu124 chosen because it's the most-tested combo for `mamba-ssm==2.2.x`.)

### A.5 Install mamba-ssm + causal-conv1d

```bash
# Try pre-built wheel first (Linux x86_64, py3.11, cu124).
uv pip install mamba-ssm causal-conv1d

# Verify import
python -c "
import mamba_ssm, causal_conv1d
print('mamba_ssm:', mamba_ssm.__version__)
print('causal_conv1d:', causal_conv1d.__version__)
"
```

If the wheel install fails (likely cu130 driver vs cu124 wheel mismatch):
```bash
# Source build (needs nvcc; install via apt)
sudo apt install -y cuda-nvcc-12-4 cuda-cudart-dev-12-4
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
uv pip install --no-binary mamba-ssm mamba-ssm causal-conv1d
```

If source build fails: fall back to NGC Docker (Path B in survey) — out of scope for this plan.

### A.6 Install the rest of the project deps

```bash
uv pip install -r requirements.txt
uv pip install lpips pyiqa av~=17.0 pillow-heif~=1.4
```

(`av` and `pillow-heif` are codec deps per AGENTS.md "Codec backends" section.)

### A.7 Sanity check

```bash
python -c "
import torch, mamba_ssm, causal_conv1d
print('torch', torch.__version__, 'cuda:', torch.cuda.is_available(), 'device:', torch.cuda.get_device_name(0))
print('mamba_ssm', mamba_ssm.__version__)
print('causal_conv1d', causal_conv1d.__version__)
b, l, d, n = 2, 64, 16, 16
x = torch.randn(b, l, d, device='cuda')
A = torch.randn(d, n, device='cuda')
B = torch.randn(b, n, l, device='cuda')
C = torch.randn(b, n, l, device='cuda')
dt = torch.randn(b, d, l, device='cuda')
y = mamba_ssm.ops.selective_scan_interface.selective_scan_fn(x.permute(0,2,1).contiguous(), dt, -torch.exp(A), B, C)
print('selective_scan_fn forward OK, y.shape:', y.shape)
"
```

Expected: torch CUDA detected, RTX 4000 Mobile, scan forward OK.

---

## B. MambaIRv2 smoke test (1-2 hours)

### B.1 Download MambaIRv2 DF2K cold-start checkpoint

```bash
cd ~/projects/upscale_anime_wsl/project
mkdir -p pretrained

# Try csguoh HuggingFace (verify URL against https://github.com/csguoh/MambaIR README)
# If the URL is wrong, the smoke will report a key mismatch and we'll correct.
uv pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
import os
# Common MambaIRv2 checkpoints (try a few if first fails)
candidates = [
    ('csguoh/MambaIRv2', 'MambaIRv2_SR4.pth'),
    ('csguoh/MambaIR', 'MambaIRv2_SR_x4.pth'),
]
for repo, fname in candidates:
    try:
        path = hf_hub_download(repo_id=repo, filename=fname, local_dir='pretrained')
        print(f'OK: {repo}/{fname} -> {path}')
        break
    except Exception as e:
        print(f'FAIL: {repo}/{fname} -- {e}')
"
```

If neither URL works, use the fallback search:
```bash
# Search HuggingFace for MambaIRv2 anime-relevant checkpoints
python -c "
from huggingface_hub import HfApi
api = HfApi()
models = api.list_models(search='MambaIRv2', limit=10)
for m in models:
    print(m.modelId)
"
```

### B.2 Create smoke config

`configs/finetune_neosr_span_v7_mambair_smoke.yaml`:

```yaml
base: finetune_neosr_span_v7_anime.yaml

output:
  run_name: NEOSR_SPAN_V7_MAMBAIR_SMOKE
  duplicate_handling: auto_increment

model:
  # Cold-start from MambaIRv2 DF2K (NOT span_pix_pretrain — invalid warm-start).
  # The exact filename depends on what B.1 downloaded.
  pretrained_path: pretrained/MambaIRv2_SR4.pth
  sab_type: mamba_v2   # THE WHOLE POINT

training:
  finetune:
    epochs: 2  # smoke only
    batch_size: 4  # conservative
```

### B.3 Run the smoke

```bash
cd ~/projects/upscale_anime_wsl/project
source .venv/bin/activate

# Dry-run first (per AGENTS.md pre-flight)
python scripts/train.py --config configs/finetune_neosr_span_v7_mambair_smoke.yaml --dry-run

# 2-epoch smoke
python scripts/train.py --config configs/finetune_neosr_span_v7_mambair_smoke.yaml --epochs 2
```

Expected: 2 epochs complete, no NaN, loss sane. Wall-clock should be ~225 ms/iter (CUDA path). If 5-8 s/iter, the fallback is in use.

### B.4 Verify CUDA hot-path

```bash
python -c "
import sys; sys.path.insert(0, 'src')
from models.span.mambair_v2 import HAS_MAMBA_SSM, selective_scan_fn
print('HAS_MAMBA_SSM:', HAS_MAMBA_SSM)
print('selective_scan_fn module:', selective_scan_fn.__module__)
"
```

Expected: `True` and `mamba_ssm.ops.selective_scan_interface` (not `mambair_v2`).

---

## C. Tier 1: TTA + Model Souping (1-2 days, runs in parallel on Windows + WSL2)

### C.1 TTA — D4 8x flip/rot ensemble

**File 1**: new `src/inference/tta.py` (~60 LOC). Content as in plan.
**File 2**: edit `scripts/inference.py` to add `--tta` flag.
**File 3**: edit `scripts/compare_checkpoints.py` to add `--tta` flag.
**File 4**: new `tests/test_tta.py` (3 tests).

### C.2 Model Souping

**File 5**: new `scripts/soup_checkpoints.py` (~80 LOC). Content as in plan.
**File 6**: new `tests/test_soup_checkpoints.py` (3 tests).

### C.3 Documentation updates

- Update `AGENTS.md` with new `--tta` flag and `soup_checkpoints.py` command.
- Update `docs/v7_results.md` (or new `docs/v8_results.md`) with Tier 1 results.

---

## D. Execution order

| # | Action | Where | Output |
|---|---|---|---|
| 1 | WSL2 uv setup (A.1-A.7) | WSL2 | `.venv` with mamba-ssm CUDA |
| 2 | Download MambaIRv2 DF2K (B.1) | WSL2 | `pretrained/MambaIRv2_*.pth` |
| 3 | Create smoke config (B.2) | new yaml | `configs/finetune_neosr_span_v7_mambair_smoke.yaml` |
| 4 | Run smoke (B.3) | WSL2 GPU | 2-epoch run, no NaN |
| 5 | Verify CUDA path (B.4) | WSL2 | `HAS_MAMBA_SSM: True` |
| 6 | Sub-agent A: TTA implementation (C.1) | Windows | `src/inference/tta.py`, `--tta` flag, tests |
| 7 | Sub-agent B: Soup script (C.2) | Windows | `scripts/soup_checkpoints.py`, tests |
| 8 | pytest | Windows + WSL2 | all green |
| 9 | `py_compile` on all new files | Windows | clean |
| 10 | `--dry-run` on smoke config | WSL2 | passes |
| 11 | Update AGENTS.md, v7_results.md | edit | docs only |
| 12 | NOT commit (per AGENTS.md convention) | n/a | user reviews diff first |

---

## E. Pre-flight checks (AGENTS.md mandatory)

1. **GPU + VRAM**: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_properties(0).total_memory/1e9)"` in WSL2.
2. **Config paths**: `grep -n "sab_type"` in `src/models/span/neosr_span.py` to confirm.
3. **Batch size vs VRAM**: smoke uses batch=4 (safe for 8 GB at v7 stack).
4. **Loss dict types**: existing v7 guard applies.
5. **Checkpoint I/O**: `--allow-pickle` not needed for MambaIRv2 DF2K (standard state-dict).
6. **2-epoch smoke**: required (Step B.3).