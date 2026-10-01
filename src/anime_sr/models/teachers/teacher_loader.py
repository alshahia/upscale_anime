"""
Teacher model loader with automatic architecture detection.
Handles EDSR, RCAN, and SwinIR checkpoints.
"""
import logging
import torch
import torch.nn as nn
from pickle import UnpicklingError
from pathlib import Path
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)

from .edsr import EDSR, create_edsr
from .rcan import RCAN, create_rcan
from .swinir import SwinIR, create_swinir


def auto_detect_architecture(checkpoint: Dict) -> Optional[str]:
    """
    Auto-detect model architecture from checkpoint state dict keys.

    Args:
        checkpoint: Loaded checkpoint dict or state_dict

    Returns:
        Architecture name ('edsr', 'rcan', 'swinir') or None if unknown
    """
    # Extract state dict from checkpoint wrapper if needed
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'params' in checkpoint:
            state_dict = checkpoint['params']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            # Assume it's already a state dict
            state_dict = checkpoint
    else:
        return None

    # Get all keys
    keys = list(state_dict.keys())
    if not keys:
        return None

    # Remove 'module.' prefix if present (from DataParallel)
    keys = [k.replace('module.', '') for k in keys]

    # Architecture detection heuristics
    key_str = ' '.join(keys).lower()

    # RCAN: Has 'ca' (channel attention) or 'rcab' keys
    if any(x in key_str for x in ['rcab', 'channelattention', 'conv_du']):
        return 'rcan'

    # SwinIR: Has 'swin', 'attn', 'mlp', 'rstb' keys
    if any(x in key_str for x in ['swin', 'rstb', 'window_attention', 'relative_position']):
        return 'swinir'

    # EDSR: Simple residual blocks with 'body' key, no attention
    if 'body' in key_str and not any(x in key_str for x in ['attention', 'ca.', 'attn']):
        # Further check: EDSR has simple conv-ReLU-conv blocks
        if any(x in key_str for x in ['residualblock', 'res_scale']):
            return 'edsr'

    # Count layers for additional clues
    body_keys = [k for k in keys if 'body' in k]
    if len(body_keys) > 0:
        # If body has ~32-64 conv layers, likely EDSR
        conv_count = len([k for k in body_keys if 'conv' in k and 'weight' in k])
        if 16 <= conv_count <= 128:
            # Default to EDSR if unsure and has residual structure
            return 'edsr'

    return None


def flexible_load_state_dict(model: nn.Module, state_dict: Dict, strict: bool = False) -> Tuple[bool, list]:
    """
    Load state dict with flexible key matching.

    Args:
        model: Model to load into
        state_dict: State dict to load
        strict: If True, require exact match

    Returns:
        (success, missing_keys)
    """
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')
        new_state_dict[new_key] = v

    # Check for size mismatches
    missing_keys = []
    size_mismatches = []

    model_state = model.state_dict()
    for key in list(new_state_dict.keys()):
        if key in model_state:
            if new_state_dict[key].shape != model_state[key].shape:
                size_mismatches.append((key, new_state_dict[key].shape, model_state[key].shape))
                del new_state_dict[key]
        else:
            missing_keys.append(key)
            del new_state_dict[key]

    # Load the filtered state dict
    model.load_state_dict(new_state_dict, strict=False)

    if size_mismatches:
        print(f"  Warning: {len(size_mismatches)} layers had size mismatches and were skipped")
        for key, ckpt_shape, model_shape in size_mismatches[:3]:  # Show first 3
            print(f"    - {key}: checkpoint {ckpt_shape} vs model {model_shape}")

    if missing_keys and strict:
        return False, missing_keys

    return len(new_state_dict) > 0, missing_keys


def create_teacher_model(
    arch: str,
    scale: int = 4,
    large: bool = False,
    checkpoint_path: Optional[str] = None,
) -> nn.Module:
    """
    Create teacher model by architecture name.

    Args:
        arch: Architecture name ('edsr', 'rcan', 'swinir')
        scale: Upsampling scale
        large: If True, use large variant (where applicable)
        checkpoint_path: Optional path to load weights from

    Returns:
        Instantiated model
    """
    arch = arch.lower()

    if arch == 'edsr':
        model = create_edsr(scale, large=large)
    elif arch == 'rcan':
        model = create_rcan(scale)
    elif arch == 'swinir':
        model = create_swinir(scale, small=not large)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    if checkpoint_path:
        load_teacher_weights(model, checkpoint_path)

    return model


def load_teacher_weights(model: nn.Module, checkpoint_path: str) -> bool:
    """
    Load weights into teacher model with validation.

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint

    Returns:
        True if successful
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    except UnpicklingError:
        logger.warning(f"Teacher checkpoint requires pickle deserialization. Only load from trusted sources: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract state dict
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'params' in checkpoint:
            state_dict = checkpoint['params']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Load with flexible matching
    success, missing = flexible_load_state_dict(model, state_dict, strict=False)

    if not success:
        raise RuntimeError(f"Failed to load any weights from {checkpoint_path}")

    if missing:
        print(f"  Note: {len(missing)} keys not loaded (may be expected)")

    return True


def load_teacher_model(
    checkpoint_path: str,
    scale: int = 4,
    arch: Optional[str] = None,
    device: str = 'cuda',
    auto_detect: bool = True,
) -> nn.Module:
    """
    Load teacher model from checkpoint with automatic architecture detection.

    Args:
        checkpoint_path: Path to checkpoint file
        scale: Upsampling scale
        arch: Architecture name (if None, auto-detect)
        device: Device to load model to
        auto_detect: If True, try to auto-detect architecture

    Returns:
        Loaded model in eval mode on specified device
    """
    checkpoint_path = Path(checkpoint_path)
    print(f"Loading teacher model from {checkpoint_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint for inspection
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    except UnpicklingError:
        logger.warning(f"Teacher checkpoint requires pickle deserialization. Only load from trusted sources: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Auto-detect architecture if not specified
    if arch is None and auto_detect:
        detected = auto_detect_architecture(checkpoint)
        if detected:
            arch = detected
            print(f"  Auto-detected architecture: {arch}")
        else:
            print("  Warning: Could not auto-detect architecture, defaulting to EDSR")
            arch = 'edsr'
    elif arch is None:
        arch = 'edsr'

    # Detect model size for SwinIR from checkpoint
    large = False
    if arch == 'swinir':
        # Check embed_dim from first conv layer to determine model size
        if isinstance(checkpoint, dict):
            ckpt_state = checkpoint.get('state_dict', checkpoint.get('params', checkpoint.get('model', checkpoint)))
        else:
            ckpt_state = checkpoint
        
        # SwinIR-M (large) has embed_dim=180, SwinIR-S (small) has embed_dim=60
        conv_first_weight = None
        if 'conv_first.weight' in ckpt_state:
            conv_first_weight = ckpt_state['conv_first.weight']
        elif 'module.conv_first.weight' in ckpt_state:
            conv_first_weight = ckpt_state['module.conv_first.weight']
        
        if conv_first_weight is not None:
            embed_dim = conv_first_weight.shape[0]  # Output channels = embed_dim
            if embed_dim == 180:
                large = True
                print(f"  Detected SwinIR-M (large, embed_dim=180)")
            elif embed_dim == 60:
                large = False
                print(f"  Detected SwinIR-S (small, embed_dim=60)")
    
    # Create model
    model = create_teacher_model(arch, scale, large=large)

    # Load weights
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('state_dict', checkpoint.get('params', checkpoint.get('model', checkpoint)))
    else:
        state_dict = checkpoint

    success, missing = flexible_load_state_dict(model, state_dict, strict=False)

    if not success:
        raise RuntimeError(f"Failed to load weights from {checkpoint_path}")

    # Move to device and set to eval
    model = model.to(device)
    model.eval()

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Validation: test forward pass
    with torch.no_grad():
        test_input = torch.randn(1, 3, 64, 64).to(device)
        test_output = model(test_input)
        expected_size = 64 * scale
        if test_output.shape[2] != expected_size or test_output.shape[3] != expected_size:
            print(f"  Warning: Output size {test_output.shape[2:]} != expected {expected_size}")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  [OK] Loaded {arch.upper()} ({param_count:,} parameters)")

    return model


def load_multiple_teachers(
    checkpoint_paths: list,
    scale: int = 4,
    device: str = 'cuda',
) -> list:
    """
    Load multiple teacher models.

    Args:
        checkpoint_paths: List of checkpoint paths
        scale: Upsampling scale
        device: Device to load to

    Returns:
        List of loaded models
    """
    teachers = []
    for path in checkpoint_paths:
        try:
            teacher = load_teacher_model(path, scale=scale, device=device)
            teachers.append(teacher)
        except Exception as e:
            print(f"  [ERROR] Failed to load {path}: {e}")

    return teachers
