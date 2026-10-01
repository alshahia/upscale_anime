"""
Production-ready REST API for anime super-resolution inference.
Provides HTTP endpoints for image super-resolution with multiple models.
"""

import os
import sys
import io
import base64
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
import uuid

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
import numpy as np

from anime_sr.models.span.checkpoint_compatible_exact import CheckpointCompatibleSPANExact
from anime_sr.inference.engine import InferenceEngine
from anime_sr.utils.performance_optimizer import PerformanceOptimizer, optimize_model_for_inference
from anime_sr.utils.config import load_config


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
app = FastAPI(
    title="Anime Super-Resolution API",
    description="Production-ready API for anime image super-resolution",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# API key authentication
API_KEY = os.getenv("DSH_API_KEY")

# Rate limiting: 60 requests per minute per IP
RATE_LIMIT = 60
RATE_LIMIT_WINDOW = 60  # seconds
_rate_limit_store: Dict[str, List[float]] = {}


@app.middleware("http")
async def api_key_middleware(request, call_next):
    """Require API key for all endpoints except /health."""
    if request.url.path == "/health":
        return await call_next(request)
    if not API_KEY:
        # No key configured - allow all (development mode)
        return await call_next(request)
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return JSONResponse(
            status_code=401,
            content={"error": "Missing or invalid API key"}
        )
    token = auth_header[7:]
    if token != API_KEY:
        return JSONResponse(
            status_code=401,
            content={"error": "Invalid API key"}
        )
    return await call_next(request)


@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    """Simple in-memory rate limiter: 60 req/min per IP."""
    client_ip = request.client.host if request.client else "unknown"
    now = datetime.now().timestamp()

    # Clean old entries for this IP
    if client_ip in _rate_limit_store:
        _rate_limit_store[client_ip] = [
            t for t in _rate_limit_store[client_ip] if now - t < RATE_LIMIT_WINDOW
        ]
    else:
        _rate_limit_store[client_ip] = []

    if len(_rate_limit_store[client_ip]) >= RATE_LIMIT:
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded"}
        )

    _rate_limit_store[client_ip].append(now)
    return await call_next(request)

# Allowed model names whitelist
ALLOWED_MODELS = {"model_a", "model_b", "ensemble", "checkpoint_compatible"}

# Maximum upload file size (50MB)
MAX_FILE_SIZE = 50 * 1024 * 1024

# Global model cache
models = {}
optimizers = {}
inference_engines = {}


class InferenceRequest(BaseModel):
    """Request model for inference."""
    model_name: str = Field(..., description="Model name to use for inference")
    scale_factor: int = Field(default=4, description="Upscaling factor (2, 3, or 4)")
    enhance_faces: bool = Field(default=False, description="Enable face enhancement")
    tile_size: Optional[int] = Field(default=None, description="Tile size for large images")
    return_base64: bool = Field(default=False, description="Return image as base64 string")


class InferenceResponse(BaseModel):
    """Response model for inference."""
    success: bool
    model_name: str
    scale_factor: int
    processing_time: float
    image_size: Dict[str, int]
    output_size: Dict[str, int]
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    image_data: Optional[str] = None  # Base64 encoded image
    message: Optional[str] = None


class ModelInfo(BaseModel):
    """Model information."""
    name: str
    type: str
    scale_factor: int
    parameters: int
    memory_usage: float
    status: str
    description: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    models_loaded: int
    gpu_available: bool
    memory_usage: Dict[str, float]
    uptime: float


# Lock guarding model loading to prevent concurrent double-loads
_model_load_lock = asyncio.Lock()


# Dependency for model loading
async def get_model(model_name: str) -> InferenceEngine:
    """Get or load a model (double-checked locking against concurrent loads)."""
    if model_name in inference_engines:
        return inference_engines[model_name]
    async with _model_load_lock:
        # Re-check after acquiring the lock: another coroutine may have
        # loaded the model while we were waiting.
        if model_name in inference_engines:
            return inference_engines[model_name]
        await load_model(model_name)
        return inference_engines[model_name]


@app.on_event("startup")
async def startup_event():
    """Initialize API on startup."""
    logger.info("Starting Anime Super-Resolution API...")
    
    # Initialize performance optimizer
    optimizers["default"] = PerformanceOptimizer()
    
    # Load default models
    default_models = ["checkpoint_compatible"]
    for model_name in default_models:
        try:
            await load_model(model_name)
        except Exception as e:
            logger.warning(f"Could not load default model {model_name}: {e}")
    
    logger.info(f"API started successfully. Loaded {len(inference_engines)} models.")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API...")
    
    # Clean up models
    for model_name in list(inference_engines.keys()):
        try:
            del inference_engines[model_name]
        except Exception as e:
            logger.warning(f"Error cleaning up model {model_name}: {e}")
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("API shutdown complete.")


async def load_model(model_name: str):
    """Load a model into memory."""
    logger.info(f"Loading model: {model_name}")
    
    try:
        # Determine model configuration
        if model_name == "checkpoint_compatible":
            model_config = {
                "type": "checkpoint_compatible",
                "scale": 4,
                "channels": 48,
                "hidden_channels": 96,
                "num_blocks": 6,
                "checkpoint_path": "checkpoints/span/spanx4_ch48.pth"
            }
        else:
            # Validate model_name against whitelist to prevent path traversal
            if model_name not in ALLOWED_MODELS:
                raise ValueError(f"Unknown model: {model_name}. Allowed: {ALLOWED_MODELS}")
            # Try to load from config file
            config_path = f"configs/{model_name}.yaml"
            if Path(config_path).exists():
                config = load_config(config_path)
                model_config = config["model"]
            else:
                raise ValueError(f"Unknown model: {model_name}")
        
        # Create model
        if model_config["type"] == "checkpoint_compatible":
            model = CheckpointCompatibleSPANExact(
                scale=model_config["scale"],
                channels=model_config["channels"],
                hidden_channels=model_config["hidden_channels"],
                num_blocks=model_config["num_blocks"]
            )
            
            # Load checkpoint
            checkpoint_path = model_config.get("checkpoint_path", "checkpoints/span/spanx4_ch48.pth")
            if Path(checkpoint_path).exists():
                try:
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                except Exception as e:
                    security_cfg = (model_config or {}).get('security') or {}
                    allow_pickle = security_cfg.get('allow_pickle_checkpoint', False)
                    if not allow_pickle:
                        raise RuntimeError("Checkpoint requires pickle loading. Set 'security.allow_pickle_checkpoint: true' in config. Only use with trusted checkpoints!")
                    import warnings
                    warnings.warn("Loading checkpoint with pickle fallback - only use with trusted sources!", UserWarning, stacklevel=2)
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                
                # Handle different checkpoint formats
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                elif "state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["state_dict"])
                elif "params" in checkpoint:
                    from anime_sr.models.span.checkpoint_compatible_exact import load_checkpoint_compatible_weights_exact
                    success, _ = load_checkpoint_compatible_weights_exact(
                        model, checkpoint_path, device="cpu", verbose=False
                    )
                    if not success:
                        logger.warning(f"Partial checkpoint loading for {model_name}")
                else:
                    model.load_state_dict(checkpoint)
            else:
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
        
        else:
            raise ValueError(f"Unsupported model type: {model_config['type']}")
        
        # Optimize model for inference
        model = optimize_model_for_inference(model)
        
        # Create inference engine
        device = "cuda" if torch.cuda.is_available() else "cpu"
        engine = InferenceEngine(model, device=device)
        
        # Cache model and engine
        models[model_name] = model
        inference_engines[model_name] = engine
        
        logger.info(f"Successfully loaded model: {model_name}")
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Could not load model: {model_name}")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Anime Super-Resolution API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    start_time = getattr(app.state, "start_time", datetime.now())
    uptime = (datetime.now() - start_time).total_seconds()
    
    # Get memory usage
    optimizer = optimizers.get("default", PerformanceOptimizer())
    memory_info = optimizer.get_memory_info()
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        models_loaded=len(inference_engines),
        gpu_available=torch.cuda.is_available(),
        memory_usage=memory_info,
        uptime=uptime
    )


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List available models."""
    model_list = []
    
    for model_name, engine in inference_engines.items():
        # Get model info
        model = engine.model
        total_params = sum(p.numel() for p in model.parameters())
        
        # Get memory usage
        optimizer = optimizers.get("default", PerformanceOptimizer())
        memory_info = optimizer.get_memory_info()
        
        model_info = ModelInfo(
            name=model_name,
            type="CheckpointCompatibleSPANExact",
            scale_factor=model.scale,
            parameters=total_params,
            memory_usage=memory_info.get("system_memory_mb", 0),
            status="loaded",
            description=f"SPAN-F model with {total_params:,} parameters"
        )
        
        model_list.append(model_info)
    
    return model_list


@app.post("/load_model/{model_name}")
async def load_model_endpoint(model_name: str):
    """Load a model."""
    try:
        await load_model(model_name)
        return {"message": f"Model {model_name} loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inference", response_model=InferenceResponse)
async def inference_endpoint(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    model_name: str = Form(..., description="Model name to use for inference"),
    scale_factor: int = Form(4, description="Upscaling factor (2, 3, or 4)"),
    enhance_faces: bool = Form(False, description="Enable face enhancement"),
    tile_size: Optional[int] = Form(None, description="Tile size for large images"),
    return_base64: bool = Form(False, description="Return image as base64 string"),
):
    """Perform inference on uploaded image."""
    
    # Validate scale BEFORE loading any model so bad input returns 400
    # without paying the cost of a model load.
    if scale_factor not in [2, 3, 4]:
        raise HTTPException(status_code=400, detail="Scale factor must be 2, 3, or 4")
    
    # Rebuild the request model from the individual multipart form fields.
    # (A Pydantic model cannot be combined with File in this FastAPI version.)
    request = InferenceRequest(
        model_name=model_name,
        scale_factor=scale_factor,
        enhance_faces=enhance_faces,
        tile_size=tile_size,
        return_base64=return_base64,
    )
    
    # Load the model (validates model_name against the whitelist)
    engine = await get_model(request.model_name)
    
    try:
        # Read image
        image_data = await image.read()

        # Check file size
        if len(image_data) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large (max 50MB)")

        # Validate image
        try:
            img = Image.open(io.BytesIO(image_data))
            img = img.convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")
        
        # Get image info
        input_size = {"width": img.width, "height": img.height}
        
        # Perform inference
        start_time = datetime.now()
        
        # Use secure temp files
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_input:
            img.save(tmp_input.name)
            temp_input_path = tmp_input.name
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_output:
            temp_output_path = tmp_output.name
        
        try:
            img.save(temp_input_path)
            
            # Run inference
            result = engine.process_image(
                temp_input_path,
                temp_output_path,
                scale=request.scale_factor,
                enhance_faces=request.enhance_faces,
                tile_size=request.tile_size
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Get output image info
            output_img = Image.open(temp_output_path)
            output_size = {"width": output_img.width, "height": output_img.height}
            
            # Convert to base64 if requested
            image_data = None
            if request.return_base64:
                buffered = io.BytesIO()
                output_img.save(buffered, format="PNG")
                image_data = base64.b64encode(buffered.getvalue()).decode()
            
            # Create response
            response = InferenceResponse(
                success=True,
                model_name=request.model_name,
                scale_factor=request.scale_factor,
                processing_time=processing_time,
                image_size=input_size,
                output_size=output_size,
                psnr=result.get("psnr"),
                ssim=result.get("ssim"),
                image_data=image_data,
                message="Inference completed successfully"
            )
            
            # Schedule cleanup
            background_tasks.add_task(cleanup_temp_files, temp_input_path, temp_output_path)
            
            return response
            
        except Exception as e:
            # Cleanup on error
            cleanup_temp_files(temp_input_path, temp_output_path)
            raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@app.delete("/models/{model_name}")
async def unload_model(model_name: str):
    """Unload a model from memory."""
    if model_name in inference_engines:
        del inference_engines[model_name]
        if model_name in models:
            del models[model_name]
        return {"message": f"Model {model_name} unloaded successfully"}
    else:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")


@app.post("/optimize_memory")
async def optimize_memory_endpoint():
    """Optimize memory usage."""
    try:
        optimizer = optimizers.get("default", PerformanceOptimizer())
        optimizer.optimize_memory_usage()
        return {"message": "Memory optimization completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    optimizer = optimizers.get("default", PerformanceOptimizer())
    memory_info = optimizer.get_memory_info()
    
    stats = {
        "models": {
            name: {
                "type": "CheckpointCompatibleSPANExact",
                "scale": engine.model.scale,
                "parameters": sum(p.numel() for p in engine.model.parameters()),
                "device": engine.device
            }
            for name, engine in inference_engines.items()
        },
        "memory": memory_info,
        "gpu": {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None
        }
    }
    
    return stats


async def cleanup_temp_files(*file_paths: str):
    """Clean up temporary files."""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.warning(f"Could not clean up file {file_path}: {e}")


# Custom exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )


if __name__ == "__main__":
    # Set startup time
    app.state.start_time = datetime.now()
    
    # Run the API
    uvicorn.run(
        "inference_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
