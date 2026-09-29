# MambaIRv2 / anime SR research container (slim hybrid)
# Target GPU: Quadro RTX 4000 (sm_75, Turing, 8 GB)
# Target CUDA: 12.4  Target PyTorch: 2.5.1  Target Python: 3.10
#
# Build-time cost (no cache):
#   base image pull            ~2-3 min (pytorch/pytorch ~3.0 GB; already cached
#                                  if you pulled it once)
#   causal-conv1d from source  ~3-5 min  (single-arch nvcc compile)
#   mamba-ssm from source       ~8-12 min (single-arch nvcc compile)
#   minimal runtime deps       ~30 sec
#   -------------------------------
#   Total                     ~10-15 min first build
#                               ~30 sec rebuild after code-only changes (cache hit)

# ============================================================
# Stage 1 (builder): compile CUDA kernels with nvcc + cuDNN
# ============================================================
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel AS builder

ENV DEBIAN_FRONTEND=noninteractive
# Quadro RTX 4000 = sm_75. Pinning cuts CUDA compile ~5x vs the default
# multi-arch list.  Swap to "7.5;8.6;9.0" if you ever get a different GPU.
ENV TORCH_CUDA_ARCH_LIST="7.5"
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
# CUDA toolkit + ninja already present in base image
ENV MAX_JOBS=8

# Layer A: pip + build tooling (cached unless base image changes)
RUN pip install --upgrade pip setuptools wheel

# Layer B: causal-conv1d from source — small CUDA kernel
# This package is OPTIONAL for mamba-ssm (falls back to PyTorch) but
# recommended; alone it is ~3-5 min to build at sm_75.
RUN pip install causal-conv1d==1.7.0

# Layer C: mamba-ssm from source — selective_scan_cuda kernels.
# ~8-12 min at sm_75 only (vs ~30-60 min at multi-arch default).
# torch + einops + packaging must be present (build isolation off).
RUN pip install --no-build-isolation mamba-ssm==2.3.2.post1

# Layer D: minimal MambaIRv2 runtime deps — only what anime_upscaler/distill.py
# and the MambaIRv2 model code actually import.  Notably ABSENT (verified):
#   - basicsr / facexlib / realesrgan (~1.0 GB; not imported)
#   - timm / accelerate / huggingface_hub (~250 MB; we write MambaIRv2 in plain
#     PyTorch instead of using timm wrappers)
#   - scikit-image / pyiqa / safetensors (~200 MB; not on critical path)
RUN pip install     einops     lpips     opencv-python-headless     pillow     numpy     tqdm     pyyaml     requests

# Layer E: smoke test — fails the build if any dep is broken.
RUN python -c "import torch, mamba_ssm, causal_conv1d, einops, lpips;     print('torch:', torch.__version__, '| cuda:', torch.version.cuda,           '| mamba:', mamba_ssm.__version__, '| cconv:', causal_conv1d.__version__,           '| lpips:', lpips.__version__)"  && nvidia-smi

# ============================================================
# Stage 2 (runtime): slim image, only the compiled site-packages
# ============================================================
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Copy ONLY the compiled site-packages from builder — keeps runtime image small.
# Inspect actual site-packages path on first build; the standard location for
# this base image is:
# Path-agnostic: copy the entire /opt/conda/lib so the runtime finds the
# python3.X/site-packages directory the builder actually created.
# (Base image is python3.11; copying only python3.10/something would
# silently miss the compiled mamba-ssm kernels.)
COPY --from=builder /opt/conda/lib /opt/conda/lib

# Sanity check at runtime (cheap)
RUN python -c "import torch, mamba_ssm, causal_conv1d;     print('runtime image OK:', torch.__version__, mamba_ssm.__version__, causal_conv1d.__version__)"

WORKDIR /workspace
CMD ["/bin/bash"]
