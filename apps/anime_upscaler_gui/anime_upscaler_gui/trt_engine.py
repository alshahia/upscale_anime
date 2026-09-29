"""TensorRT inference engine cache.

Phase 1 of the Real-time 4K plan: wraps NVIDIA TensorRT 11.x and provides
per-shape engine caching with on-disk persistence.

Usage (from pipeline.py):

    from .trt_engine import _TrtBackend, _HAS_TRT
    if _HAS_TRT and job.fp16 and device.type == "cuda":
        backend = _TrtBackend(model, kind, device, fp16, cache_dir)
    else:
        backend = _PyTorchBackend(...)

Falls back silently to PyTorch eager if:
    * TensorRT is not importable
    * torch is on CPU
    * fp16 is False
    * ONNX export fails
    * Engine build fails
"""
import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from .archs import capabilities as _arch_caps

log = logging.getLogger(__name__)

# TRT serialized engines are never smaller than a few KB; a partial write
# from a killed process leaves a stub that makes deserialize() fail with an
# opaque error. Guard with a size floor so we rebuild instead.
_MIN_ENGINE_BYTES = 64 * 1024

try:
    import tensorrt as trt
    _HAS_TRT = True
except Exception as e:
    trt = None
    _HAS_TRT = False
    log.warning("tensorrt not importable: %s", e)


# ---------------------------------------------------------------------------
# Module-level cache (one engine per (model_hash, batch, H, W, fp16))
# ---------------------------------------------------------------------------
class _TrtEngineCache:
    """Builds, caches, and persists TensorRT engines per (batch, H, W)."""

    MAX_CACHE_BYTES = 1 << 30

    def __init__(self, onnx_path, cache_dir, fp16=True, batch_size: int = 1):
        self.onnx_path = Path(onnx_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.fp16 = fp16
        self.batch_size = int(batch_size)
        self._engines = {}
        self._logger = trt.Logger(trt.Logger.WARNING) if _HAS_TRT else None

    def get(self, batch, h, w):
        """Returns (engine, exec_ctx, cuda_stream). Lazily builds + caches.

        The first batch seen locks the engine: subsequent calls must use
        the same batch size (or a new cache is created per backend).
        """
        if batch != self.batch_size:
            raise RuntimeError(
                f"TensorRT backend built for batch={self.batch_size}, got batch={batch}. "
                "Create a new _TrtBackend per batch size."
            )
        key = (batch, h, w)
        cached = self._engines.get(key)
        if cached is not None:
            return cached
        if not _HAS_TRT:
            raise RuntimeError("TensorRT not available")
        engine_path = self._engine_path(key)
        engine_bytes = None
        if engine_path.exists():
            engine_bytes = engine_path.read_bytes()
            if len(engine_bytes) < _MIN_ENGINE_BYTES:
                # Truncated/partial write from a killed process: TRT would
                # fail with an opaque deserialization error. Rebuild instead.
                log.warning("trt: cached engine %s is truncated (%d bytes); rebuilding",
                            engine_path.name, len(engine_bytes))
                engine_bytes = None
            else:
                log.info("trt: loading cached engine %s (%d bytes)", engine_path.name, len(engine_bytes))
        if engine_bytes is None:
            log.info("trt: building engine for batch=%d h=%d w=%d (this takes 30-90 s once)", batch, h, w)
            t0 = time.time()
            engine_bytes = self._build_engine_bytes(batch, h, w)
            log.info("trt: engine built in %.1f s, %d bytes", time.time() - t0, len(engine_bytes))
            self._evict_if_oversized(len(engine_bytes))
            engine_path.write_bytes(engine_bytes)
        runtime = trt.Runtime(self._logger)
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        exec_ctx = engine.create_execution_context()
        stream = torch.cuda.Stream()
        self._engines[key] = (engine, exec_ctx, stream)
        return self._engines[key]

    def _model_tag(self):
        h = hashlib.sha1()
        with open(self.onnx_path, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()[:12]

    def _engine_path(self, key):
        b, h, w = key
        tag = self._model_tag()
        # Note: use chr(123)/chr(125) to avoid brace conflicts in source generators
        precision = "fp16" if self.fp16 else "fp32"
        fname = tag + "_b" + str(b) + "_" + str(h) + "x" + str(w) + "_" + precision + ".engine"
        return self.cache_dir / fname

    def _evict_if_oversized(self, incoming_bytes):
        try:
            files = []
            total = 0
            for ep in self.cache_dir.glob("*.engine"):
                sz = ep.stat().st_size
                total += sz
                files.append((ep.stat().st_mtime, sz, ep))
            if total + incoming_bytes <= self.MAX_CACHE_BYTES:
                return
            files.sort()
            for mtime, sz, ep in files:
                if total + incoming_bytes <= self.MAX_CACHE_BYTES:
                    break
                try:
                    ep.unlink()
                    total -= sz
                    log.info("trt: evicted %s (%d bytes) to stay under %d MB", ep.name, sz, self.MAX_CACHE_BYTES >> 20)
                except OSError:
                    pass
        except Exception as e:
            log.warning("trt: cache eviction skipped: %s", e)

    def _build_engine_bytes(self, batch, h, w):
        builder = trt.Builder(self._logger)
        network = builder.create_network()
        parser = trt.OnnxParser(network, self._logger)
        with open(self.onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    log.error("trt: parse error: %s", parser.get_error(i).desc())
                raise RuntimeError("TensorRT failed to parse ONNX")
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        # TRT 11: there is no BuilderFlag.FP16; precision follows the ONNX graph
        # (fp16 weights -> fp16 kernels). Building for gp32 ONNX produces
        # 200-450 MB engines with ~1 TFLOPS throughput -- always export fp16.
        profile = builder.create_optimization_profile()
        shape = (batch, 3, h, w)
        profile.set_shape("lr", shape, shape, shape)
        config.add_optimization_profile(profile)
        host_mem = builder.build_serialized_network(network, config)
        if host_mem is None:
            raise RuntimeError("TensorRT engine build returned None")
        return bytes(memoryview(host_mem).tobytes())


# ---------------------------------------------------------------------------
# ONNX export (one-time per model)
# ---------------------------------------------------------------------------
def _export_onnx(model, onnx_path, dummy):
    """Export model to ONNX. Returns True on success."""
    onnx_path = Path(onnx_path)
    if onnx_path.exists():
        return True
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model_was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            torch.onnx.export(
                model, dummy, str(onnx_path),
                opset_version=17,
                input_names=["lr"], output_names=["hr"],
                dynamic_axes={"lr": {0: "N", 2: "H", 3: "W"}, "hr": {0: "N", 2: "H", 3: "W"}},
                dynamo=False,
            )
        return True
    except Exception as e:
        log.error("trt: ONNX export failed: %s", e)
        try:
            if onnx_path.exists():
                onnx_path.unlink()
        except OSError:
            pass
        return False
    finally:
        if model_was_training:
            model.train()


# ---------------------------------------------------------------------------
# Backend wrapper (mirrors _PyTorchBackend interface)
# ---------------------------------------------------------------------------
class _TrtBackend:
    """TensorRT-backed callable for inference.

    Pre-conditions:
        * device is CUDA
        * model is already in fp16 (so ONNX export is fp16)
        * kind != "animesr"

    __call__ input/output contract matches _PyTorchBackend:
        x : (B, 3, H, W) on CUDA, dtype = fp16 or fp32
        y : (B, 3, H*4, W*4) on CUDA, same dtype as input
    """

    def __init__(self, model, kind, device, fp16, cache_dir, batch_size: int = 1):
        self.kind = kind
        self.device = device
        self.fp16 = fp16
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise RuntimeError(f"TensorRT batch_size must be >= 1, got {batch_size}")
        # Phase 2 (Real-time 4K): output scale comes from the model so a
        # 2x RFDN student (cascade 2x+2x) and a 4x RFDN student produce
        # differently-sized SR outputs. Defaults to 4 if the model has no
        # explicit scale attr (e.g. SRVGG/SPAN/ERANet always 4x or 2x).
        # Also expose the model itself for the cascade helper, which reads
        # model.scale from the *current* backend (TRT, ONNX, PyTorch all
        # need to behave the same here).
        self.model = model
        self._scale = int(getattr(model, "scale",
                                  getattr(model, "upscale", 4)) or 4)
        self._cache = None
        self._onnx_path = None
        if not _HAS_TRT:
            raise RuntimeError("TensorRT not available")
        if device.type != "cuda":
            raise RuntimeError("TensorRT backend requires CUDA device")
        if not _arch_caps(kind).tensorrt:
            raise RuntimeError(
                f"TensorRT backend unsupported for kind {kind!r}")
        onnx_path = Path(cache_dir) / "_trt_model.onnx"
        # ONNX export: dummy uses batch=1 since dynamic_axes marks batch as
        # dynamic. TRT will get the actual batch via the optimization profile.
        dummy = torch.zeros(1, 3, 16, 16, device=device, dtype=torch.float16 if fp16 else torch.float32)
        if not _export_onnx(model, onnx_path, dummy):
            raise RuntimeError("ONNX export failed")
        self._onnx_path = onnx_path
        self._cache = _TrtEngineCache(onnx_path, cache_dir, fp16=fp16, batch_size=self.batch_size)

    def __call__(self, x):
        assert self._cache is not None
        assert x.is_cuda, "TRT backend expects CUDA tensor"
        b, c, h, w = x.shape
        assert c == 3, "TRT backend expects 3-channel input, got " + str(c)
        engine, exec_ctx, stream = self._cache.get(b, h, w)
        out_h, out_w = h * self._scale, w * self._scale
        y = torch.empty(b, 3, out_h, out_w, device=x.device, dtype=x.dtype)
        exec_ctx.set_input_shape("lr", (b, c, h, w))
        exec_ctx.set_tensor_address("lr", x.data_ptr())
        exec_ctx.set_tensor_address("hr", y.data_ptr())
        # Run the engine on its own stream, then make PyTorch's current
        # stream wait for it. Without this, downstream ops (e.g. y.float())
        # queued on the default stream read stale memory before the engine
        # finishes writing.
        with torch.cuda.stream(stream):
            ok = exec_ctx.execute_async_v3(stream.cuda_stream)
            if not ok:
                stream.synchronize()
                raise RuntimeError("TensorRT execute_async_v3 returned False")
        torch.cuda.current_stream().wait_stream(stream)
        return y
