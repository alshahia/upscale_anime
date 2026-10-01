# Deployment Guide

## Overview

This guide covers deploying the anime super-resolution model in production environments, including Docker containers, API servers, and optimized inference.

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/

# Install package
RUN pip install -e .

# Expose API port
EXPOSE 8000

# Run API server
CMD ["python", "-m", "anime_sr", "serve", "--host", "0.0.0.0", "--port", "8000"]
```

### Build and Run

```bash
# Build image
docker build -t anime-sr:latest .

# Run container
docker run -d \
    --name anime-sr \
    -p 8000:8000 \
    -v $(pwd)/models:/app/models \
    -e DSH_API_KEY=your-secret-key \
    anime-sr:latest
```

### Docker Compose

```yaml
version: '3.8'

services:
  anime-sr:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    environment:
      - DSH_API_KEY=your-secret-key
      - DSH_RATE_LIMIT_PER_MIN=60
      - DSH_MAX_FILE_SIZE_MB=10
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## API Server

### Start Server

```bash
# Basic start
python -m anime_sr serve --host 0.0.0.0 --port 8000

# With config
python -m anime_sr serve --config configs/api.yaml

# Development mode
python -m anime_sr serve --reload

# Production mode
python -m anime_sr serve --workers 4
```

### API Configuration

```yaml
# configs/api.yaml
api:
  host: 0.0.0.0
  port: 8000
  api_key: your-secret-key
  rate_limit: 60  # requests per minute
  max_file_size_mb: 10
  model_path: runs/distill_v1/student_best.pt
  model_type: rfdn
  device: cuda
```

### Environment Variables

```bash
export DSH_API_KEY=your-secret-key
export DSH_RATE_LIMIT_PER_MIN=60
export DSH_MAX_FILE_SIZE_MB=10
export DSH_CUDA_VISIBLE_DEVICES=0
```

### API Endpoints

#### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

#### List Models

```bash
curl http://localhost:8000/models
```

#### Run Inference

```bash
curl -X POST http://localhost:8000/inference \
  -H "Authorization: Bearer your-secret-key" \
  -F "file=@input.png"
```

Response:
```json
{
  "output_url": "/outputs/abc123.png",
  "processing_time_ms": 123.4,
  "input_size": [256, 256],
  "output_size": [1024, 1024]
}
```

---

## ONNX Export

### Export Model

```bash
# Export to ONNX
python -m anime_sr export \
    --checkpoint runs/distill_v1/student_best.pt \
    --format onnx \
    --output exports/student.onnx

# Export to FP16
python -m anime_sr export \
    --checkpoint runs/distill_v1/student_best.pt \
    --format onnx \
    --half \
    --output exports/student_fp16.onnx
```

### ONNX Runtime Inference

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

# Create session
session = ort.InferenceSession(
    "exports/student.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# Prepare input
img = Image.open("input.png").convert("RGB")
lr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0

# Run inference
output = session.run(["sr"], {"lr": lr})[0]

# Save output
sr = (output[0].transpose(1, 2, 0) * 255).astype(np.uint8)
Image.fromarray(sr).save("output.png")
```

---

## TensorRT Optimization

### Convert ONNX to TensorRT

```bash
# Using trtexec
trtexec \
    --onnx=exports/student.onnx \
    --saveEngine=exports/student.trt \
    --fp16 \
    --workspace=4096
```

### TensorRT Inference

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# Load TensorRT engine
with open("exports/student.trt", "rb") as f:
    engine = trt.Runtime(trt.Logger()).deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# Allocate memory
# ... (see TensorRT documentation for full example)
```

---

## Performance Optimization

### Benchmark

```bash
# Benchmark model
python -m anime_sr benchmark \
    --checkpoint runs/distill_v1/student_best.pt \
    --iterations 100
```

### Optimization Tips

1. **Use FP16**: Export with `--half` for 2x speedup on modern GPUs
2. **Use ONNX Runtime**: 1.5-2x faster than PyTorch
3. **Use TensorRT**: 2-3x faster than ONNX Runtime
4. **Batch Processing**: Process multiple images together
5. **Tiled Inference**: For large images, use `--tile` to reduce memory

### Expected Performance

| Model | Device | Resolution | FPS | Latency |
|-------|--------|------------|-----|---------|
| RFDN | RTX 4090 | 1080p | 60+ | <16ms |
| RFDN | RTX 4000 | 1080p | 30 | ~33ms |
| RFDN | CPU | 1080p | 2 | ~500ms |
| TinySRVGG | RTX 4090 | 1080p | 120+ | <8ms |
| TinySRVGG | RTX 4000 | 1080p | 60 | ~16ms |

---

## Monitoring

### Health Checks

```bash
# Check server health
curl http://localhost:8000/health

# Check GPU utilization
nvidia-smi

# Check API stats
curl http://localhost:8000/stats
```

### Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('anime_sr.log'),
        logging.StreamHandler()
    ]
)
```

### Metrics

The API server exposes the following metrics:

- `requests_total`: Total number of requests
- `requests_per_minute`: Current request rate
- `average_latency_ms`: Average inference latency
- `error_rate`: Percentage of failed requests
- `gpu_memory_used_mb`: GPU memory usage

---

## Security

### API Key Authentication

Always set `DSH_API_KEY` in production:

```bash
export DSH_API_KEY=$(openssl rand -hex 32)
```

### Rate Limiting

Configure rate limiting to prevent abuse:

```yaml
api:
  rate_limit: 60  # requests per minute
```

### Input Validation

The API server validates:
- File type (PNG, JPEG, WebP, BMP)
- File size (configurable, default 10MB)
- Image dimensions (max 4096x4096)

### HTTPS

Use a reverse proxy (nginx, Traefik) for HTTPS:

```nginx
server {
    listen 443 ssl;
    server_name api.example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## Scaling

### Horizontal Scaling

Run multiple instances behind a load balancer:

```yaml
# docker-compose.yml
version: '3.8'

services:
  anime-sr:
    build: .
    deploy:
      replicas: 3
    environment:
      - DSH_API_KEY=your-secret-key

  nginx:
    image: nginx:latest
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - anime-sr
```

### GPU Sharing

Use NVIDIA MIG or vGPU for multi-tenant GPU sharing.

---

## Troubleshooting

### Server Won't Start

```bash
# Check port availability
netstat -tlnp | grep 8000

# Check logs
docker logs anime-sr
```

### Out of Memory

```bash
# Reduce batch size
export DSH_INFERENCE__BATCH_SIZE=1

# Use tiled inference
# (in API request, set tile_size parameter)
```

### Slow Inference

```bash
# Check GPU utilization
nvidia-smi

# Use FP16 model
python -m anime_sr export --checkpoint model.pt --format onnx --half

# Use TensorRT
trtexec --onnx=model.onnx --saveEngine=model.trt --fp16
```

---

## See Also

- [Python API Reference](api.md) - Python API documentation
- [CLI Reference](cli.md) - Command-line interface
- [Model Architectures](models.md) - Model architecture details
- [Training Guide](training.md) - Training procedures
- [Development Setup](development.md) - Development environment
