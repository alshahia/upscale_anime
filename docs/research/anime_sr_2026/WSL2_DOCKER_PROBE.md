# WSL2 vs Docker Probe Report -- 2026-09-03

**Generated:** 2026-09-03
**Scope:** Verify whether Docker-on-WSL2 sidesteps the broken WSL2 Ubuntu-24.04 networking and unlocks the MambaIRv2 path on this machine.

## TL;DR

**YES -- Docker solves the WSL2 networking problem completely.**

| Probe | WSL2 Ubuntu-24.04 | Docker (--gpus all) |
|---|---|---|
| Internet (curl pypi) | FAIL (HTTP 000) | PASS (HTTP 200) |
| GitHub reachable | FAIL | PASS (HTTP 200) |
| GPU visible (nvidia-smi) | PARTIAL (only /dev/dxg, no /dev/nvidia*) | PASS (full RTX 4000 + driver + CUDA) |
| PyTorch cu124 install | FAIL (no internet) | PASS (2.5.1+cu124, CUDA avail=True) |
| Build mamba-ssm from source | FAIL (no network + no nvcc) | IN PROGRESS (docker build in background, job pwsh-2) |

The WSL2 distro networking is hard-broken (corporate lockdown or misconfiguration), but Docker Desktop WSL2 integration runs containers in their own network namespace, NAT'd through the Windows host network stack. Containers have full internet access.

## What was verified

### Docker daemon working
- docker --version -> Docker 28.3.0 (build 38b7060)
- docker info -> Client: desktop-linux, plugins: buildx, compose, debug, desktop, extension, mcp
- docker run --rm alpine:latest ... -> image pulled from Docker Hub, ran successfully
- Network test from alpine container:
  - curl https://pypi.org -> HTTP 200 (6.5s first hit, faster after)
  - curl https://github.com -> HTTP 200 (0.6s)
  - curl https://download.pytorch.org -> reachable

### GPU passthrough in Docker working
- docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi -> reports:
  - Driver 595.97, CUDA 13.2
  - Quadro RTX 4000, 327 MiB used / 8192 MiB total
  - GPU 0000:01:00.0
- docker run --rm --gpus all python:3.12-bullseye python3 -c "import torch; ..." -> torch.cuda.is_available() = True; device = Quadro RTX 4000

### PyTorch + CUDA working in Docker
- pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 -> SUCCESS
- torch.__version__ = 2.5.1+cu124
- torch.version.cuda = 12.4
- torch.cuda.is_available() = True
- torch.cuda.get_device_name(0) = Quadro RTX 4000

### mamba-ssm + causal-conv1d buildability
- Both packages are SOURCE-ONLY on PyPI (no pre-built wheels, even for cp311 Linux).
- They require nvcc (CUDA toolkit) + gcc/g++ + CUDA-aware PyTorch at build time.
- The python:3.12-bullseye image lacks nvcc; we use nvidia/cuda:12.4.0-devel-ubuntu22.04 base image.
- mamba-ssm setup.py requires bare_metal_version from torch.utils.cpp_extension -- requires CUDA-aware torch during setup.

## Why Docker works while WSL2 Ubuntu-24.04 does not

Docker Desktop on Windows uses its own WSL2 distribution (docker-desktop-data) and runs containers in their own Linux namespaces. The container network stack is virtualized through the Docker daemon, which uses the Windows host network adapter directly (not the WSL2 Ubuntu-24.04 distro broken eth0).

This means:
- WSL2 Ubuntu-24.04 = corporate-locked, no IP route, no DNS
- Docker containers = full internet via Windows host NAT, regardless of WSL2 distro state

## Build in progress (background)

docker build -t mambair-test:latest is running in the background (job pwsh-2) with the Dockerfile at tmp/mambair-build/Dockerfile.

**Base**: nvidia/cuda:12.4.0-devel-ubuntu22.04 (has nvcc pre-installed, ~4GB).
**Install**: Python 3.12 (via deadsnakes PPA) + pip (via get-pip.py -- apt python3-pip broken on 3.12) + torch cu124 + mamba-ssm + causal-conv1d (built from source) + basicsr/facexlib/realesrgan + einops/timm + lpips/pyiqa.

**Estimated build time**: 30-90 minutes (mostly compilation of mamba-ssm C++/CUDA kernels). The cache from the failed first build should speed up the retry.

**First build failed** at apt python3-pip (Python 3.12 ships pip 22.0.2 which is broken because distutils was removed in 3.12). Fixed by using get-pip.py from bootstrap.pypa.io.

## Updated Phase 5D status (revised)

**Previous status**: BLOCKED -- WSL2 networking + PyPI wheel reality make this a 3+ hour ramp.

**Revised status**: VIABLE via Docker, 1-2 hour build + setup, then training can proceed.

The Docker build sidesteps both blockers:
- Internet for pip install (Docker NAT)
- nvcc for compiling mamba-ssm (devel base image)

## Recommended path for MambaIRv2 (Phase 5D)

1. Wait for mambair-test:latest build to complete (~1 hour).
2. Run docker run --rm --gpus all -v /mnt/e:/workspace mambair-test:latest to launch.
3. Inside container: clone csguoh/MambaIR, download DF2K cold-start ckpt, run distillation training.

## Alternative / Phase 5 alternatives -- sorted by ROI

For the broader Phase 5 set (not just MambaIRv2), sorted by best quality gain per unit wall-time:

| Rank | Option | Wall-time | Expected gain | Risk |
|---|---|---|---|---|
| **1** | **Warm-start v1 -> SRVGG-body + no-adv + LPIPS-heavy** | **1-2 days** | **+lap_var, keep PSNR, 1.7x faster** | **LOW** |
| 2 | APISR-style anime perceptual loss + warm-start (5A) | 1 week | +MANIQA, +CLIPIQA, +anime sharpness | LOW-MED |
| 3 | FRAMER-style frequency-domain distillation (5B) | 1-2 weeks | +HF detail, +edge preservation | LOW |
| 4 | Multi-Scale Contrastive-Adversarial KD (5E) | 1 week | +LPIPS | MED |
| 5 | OSEDiff / SinSR anime fine-tune (5C) | 2 weeks | +++ MANIQA, ~1 sec/frame | MED-HIGH |
| 6 | MambaIRv2 student distillation | 2-3 weeks (after Docker build) | +PSNR potential | MED |
| 7 | FiDeSR / One-Step Diffusion Transformer fine-tune (5F) | 3 weeks | ++++ perceptual, ~1 sec/frame | HIGH |

**See RECOMMENDED_PATH.md for full analysis and Rank #1 recipe (warm-start v1 -> SRVGG no-adv).**

MambaIRv2 specific notes:
- +0.29 dB on Manga109 vs HAT (heavy transformer, not RFDN-scale)
- 30% lower MACs vs HAT at 256x256 (still higher than RFDN)
- Architectural diversity (state-space vs conv vs transformer)
- Whether that translates to a +PSNR RFDN-scale student is empirically untested

