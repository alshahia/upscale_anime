import os
dockerfile = '''# MambaIRv2 / anime SR research container
# Base: NVIDIA CUDA 12.4 + Ubuntu 22.04 with nvcc (devel variant)
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"

# Install Python 3.12 (Ubuntu 22.04 ships 3.10; use deadsnakes PPA)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common ca-certificates wget curl git \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch cu124 (matches our Windows side)
RUN pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Install mamba-ssm + causal-conv1d (built from source on Py3.12; ~30-60 min)
RUN pip install --no-cache-dir --no-build-isolation \
    causal-conv1d==1.7.0 mamba-ssm==2.3.2.post1

# Install our project dependencies + MambaIR dependencies
RUN pip install --no-cache-dir \
    numpy opencv-python pillow lpips pyiqa \
    basicsr facexlib realesrgan \
    einops timm scikit-image \
    matplotlib tqdm pyyaml requests safetensors accelerate

# Sanity check
RUN python3 -c "import torch, mamba_ssm, causal_conv1d; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'mamba_ssm:', mamba_ssm.__version__, 'causal_conv1d:', causal_conv1d.__version__)" \
    && nvidia-smi

WORKDIR /workspace
CMD ["/bin/bash"]
'''
os.makedirs(r"E:\python projects\upscale_anime\tmp\mambair-build", exist_ok=True)
path = r"E:\python projects\upscale_anime\tmp\mambair-build\Dockerfile"
with open(path, "w", encoding="utf-8") as f:
    f.write(dockerfile)
print("Written:", path, "size:", len(dockerfile))
