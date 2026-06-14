"""
Checkpoint loading utilities for transfer learning and fine-tuning.
Handles loading pretrained weights from various checkpoint formats.
"""

import torch
from pickle import UnpicklingError
from pathlib import Path
from typing import Dict, Optional, Union, Tuple
import warnings
import logging

logger = logging.getLogger(__name__)


def load_pretrained_weights(
    model: torch.nn.Module,
    checkpoint_path: Union[str, Path],
    strict: bool = False,
    device: str = 'cpu',
    verbose: bool = True
) -> bool:
    """
    Load pretrained weights into a model.
    
    Handles various checkpoint formats:
    - Raw state_dict: {key: tensor, ...}
    - Nested state_dict: {'state_dict': {...}, ...}
    - Model checkpoint: {'model_state_dict': {...}, 'epoch': ..., ...}
    - EMA checkpoint: {'ema_model_state_dict': {...}, ...}
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        strict: If False, allows partial loading (missing/extra keys)
        device: Device to load checkpoint on
        verbose: Print loading information
        
    Returns:
        True if successful, False otherwise
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        warnings.warn(f"Checkpoint not found: {checkpoint_path}")
        return False
    
    try:
        # Load checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except UnpicklingError:
            logger.warning("Checkpoint requires pickle deserialization. Only load from trusted sources.")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Extract state dict
        state_dict = None
        ema_state_dict = None
        
        if 'ema_model_state_dict' in checkpoint:
            ema_state_dict = checkpoint['ema_model_state_dict']
            if verbose:
                print(f"  Found EMA weights in checkpoint")
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'params' in checkpoint:
            state_dict = checkpoint['params']
        elif isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            # Raw state dict
            state_dict = checkpoint
        else:
            warnings.warn(f"Unknown checkpoint format. Keys: {list(checkpoint.keys())}")
            return False
        
        # Try loading EMA weights first (better quality), fallback to regular
        loaded_ema = False
        if ema_state_dict is not None:
            try:
                missing, unexpected = model.load_state_dict(ema_state_dict, strict=strict)
                if len(missing) == 0 or not strict:
                    state_dict = ema_state_dict
                    loaded_ema = True
                    if verbose:
                        print(f"  Loaded EMA weights (better quality)")
            except RuntimeError:
                pass  # Fall back to regular weights
        
        if not loaded_ema:
            # Clean up state dict keys (remove 'module.' prefix if present)
            state_dict = _clean_state_dict_keys(state_dict)
            
            # Load weights
            missing, unexpected = model.load_state_dict(state_dict, strict=strict)
            
            if verbose:
                if missing:
                    print(f"  Missing keys: {len(missing)}")
                    if len(missing) <= 5:
                        for k in missing:
                            print(f"    - {k}")
                if unexpected:
                    print(f"  Unexpected keys: {len(unexpected)}")
                    if len(unexpected) <= 5:
                        for k in unexpected:
                            print(f"    - {k}")
        
        if verbose:
            total_params = sum(p.numel() for p in model.parameters())
            loaded_params = sum(state_dict[k].numel() for k in state_dict.keys() if k in dict(model.named_parameters()))
            print(f"  Loaded {loaded_params:,} / {total_params:,} parameters ({loaded_params/total_params*100:.1f}%)")
            
            if 'epoch' in checkpoint:
                print(f"  From epoch: {checkpoint['epoch']}")
        
        return True
        
    except Exception as e:
        warnings.warn(f"Failed to load checkpoint: {e}")
        return False


def _clean_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Clean up state dict keys by removing common prefixes.
    
    Handles:
    - 'module.' prefix (from DataParallel/DistributedDataParallel)
    - 'model.' prefix
    - '_orig_mod.' prefix (from torch.compile)
    """
    cleaned = {}
    for key, value in state_dict.items():
        # Remove common prefixes
        new_key = key
        for prefix in ['module.', 'model.', '_orig_mod.']:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def get_checkpoint_info(checkpoint_path: Union[str, Path]) -> Optional[Dict]:
    """
    Get information about a checkpoint without loading it fully.
    
    Args:
        checkpoint_path: Path to checkpoint
        
    Returns:
        Dictionary with checkpoint info or None if invalid
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        return None
    
    try:
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        except UnpicklingError:
            logger.warning("Checkpoint requires pickle deserialization. Only load from trusted sources.")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        info = {
            'path': str(checkpoint_path),
            'size_mb': checkpoint_path.stat().st_size / (1024 * 1024),
        }
        
        # Check for various keys
        if 'epoch' in checkpoint:
            info['epoch'] = checkpoint['epoch']
        if 'best_loss' in checkpoint:
            info['best_loss'] = checkpoint['best_loss']
        if 'metrics' in checkpoint:
            info['metrics'] = checkpoint['metrics']
        
        # Detect format
        if 'ema_model_state_dict' in checkpoint:
            info['has_ema'] = True
        if 'state_dict' in checkpoint:
            info['format'] = 'state_dict'
            info['num_params'] = len(checkpoint['state_dict'])
        elif 'model_state_dict' in checkpoint:
            info['format'] = 'model_state_dict'
            info['num_params'] = len(checkpoint['model_state_dict'])
        else:
            info['format'] = 'raw'
            info['num_params'] = len(checkpoint)
        
        return info
        
    except Exception as e:
        return {'error': str(e)}


def convert_spanf_checkpoint(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    verify: bool = True
) -> bool:
    """
    Convert SPAN-F checkpoint from official format to this codebase's format.
    
    The official SPAN-F checkpoint may have different key naming.
    This function remaps keys if necessary.
    
    Args:
        input_path: Path to original checkpoint
        output_path: Path to save converted checkpoint
        verify: Verify the conversion by loading
        
    Returns:
        True if successful
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return False
    
    try:
        try:
            checkpoint = torch.load(input_path, map_location='cpu', weights_only=True)
        except UnpicklingError:
            logger.warning("Checkpoint requires pickle deserialization. Only load from trusted sources.")
            checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
        
        # Extract state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'params' in checkpoint:
            state_dict = checkpoint['params']
        else:
            state_dict = checkpoint
        
        # Clean keys
        state_dict = _clean_state_dict_keys(state_dict)
        
        # Check for BasicSR-style keys and remap if needed
        remapped = {}
        for key, value in state_dict.items():
            new_key = key
            
            # BasicSR often uses 'body.X' for blocks, we use 'blocks.X'
            if key.startswith('body.'):
                parts = key.split('.')
                if len(parts) >= 2 and parts[1].isdigit():
                    block_num = parts[1]
                    rest = '.'.join(parts[2:])
                    new_key = f'blocks.{block_num}.{rest}'
            
            remapped[new_key] = value
        
        # Create new checkpoint
        new_checkpoint = {
            'state_dict': remapped,
            'converted_from': str(input_path),
            'converted': True,
        }
        
        # Preserve other metadata if present
        for key in ['epoch', 'best_loss', 'metrics']:
            if key in checkpoint:
                new_checkpoint[key] = checkpoint[key]
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(new_checkpoint, output_path)
        
        print(f"Converted checkpoint saved to: {output_path}")
        print(f"  Original keys: {len(state_dict)}")
        print(f"  Remapped keys: {len(remapped)}")
        
        if verify:
            # Try loading
            from models.span import SPANF
            model = SPANF(scale=4)
            try:
                model.load_state_dict(remapped, strict=True)
                print("  Verification: OK - All keys loaded successfully")
            except RuntimeError as e:
                print(f"  Verification warning: {e}")
                print("  Some keys may not match - training will still work with partial loading")
        
        return True
        
    except Exception as e:
        print(f"Error converting checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_checkpoint_compatible_span(
    model: torch.nn.Module,
    checkpoint_path: Union[str, Path],
    device: str = 'cpu',
    verbose: bool = True
) -> Tuple[bool, Dict[str, any]]:
    """
    Load checkpoint-compatible SPAN weights with exact parameter mapping.
    
    This function handles the specific checkpoint format with:
    - 'params' and 'params_ema' keys
    - Parameter names like 'conv_1.sk.weight', 'block_1.c1_r.conv.0.weight'
    - 204 total parameters
    
    Args:
        model: CheckpointCompatibleSPAN model instance
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint on
        verbose: Whether to print loading details
        
    Returns:
        Tuple of (success, loading_info)
    """
    loading_info = {
        'total_checkpoint_params': 0,
        'loaded_params': 0,
        'missing_params': [],
        'unexpected_params': [],
        'shape_mismatches': [],
        'load_percentage': 0.0
    }
    
    try:
        if verbose:
            print(f"Loading checkpoint-compatible SPAN from: {checkpoint_path}")
        
        # Load checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except UnpicklingError:
            logger.warning("Checkpoint requires pickle deserialization. Only load from trusted sources.")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Extract parameters
        if 'params' in checkpoint:
            checkpoint_params = checkpoint['params']
            if verbose:
                print(f"  Using 'params' key with {len(checkpoint_params)} parameters")
        elif 'params_ema' in checkpoint:
            checkpoint_params = checkpoint['params_ema']
            if verbose:
                print(f"  Using 'params_ema' key with {len(checkpoint_params)} parameters")
        else:
            checkpoint_params = checkpoint
            if verbose:
                print(f"  Using checkpoint as direct parameter dict with {len(checkpoint_params)} parameters")
        
        loading_info['total_checkpoint_params'] = len(checkpoint_params)
        
        # Get model state dict
        model_state_dict = model.state_dict()
        
        # Create parameter mapping
        mapped_params = {}
        missing_keys = []
        unexpected_keys = []
        shape_mismatches = []
        
        # Map checkpoint parameters to model parameters
        for ckpt_name, ckpt_param in checkpoint_params.items():
            if ckpt_name in model_state_dict:
                # Check shape compatibility
                if ckpt_param.shape == model_state_dict[ckpt_name].shape:
                    mapped_params[ckpt_name] = ckpt_param
                else:
                    shape_mismatches.append({
                        'name': ckpt_name,
                        'ckpt_shape': ckpt_param.shape,
                        'model_shape': model_state_dict[ckpt_name].shape
                    })
                    if verbose:
                        print(f"  Shape mismatch: {ckpt_name}")
                        print(f"    Checkpoint: {ckpt_param.shape}")
                        print(f"    Model: {model_state_dict[ckpt_name].shape}")
            else:
                unexpected_keys.append(ckpt_name)
        
        # Find missing model parameters
        for model_name in model_state_dict.keys():
            if model_name not in checkpoint_params:
                missing_keys.append(model_name)
        
        # Load mapped parameters
        load_result = model.load_state_dict(mapped_params, strict=False)
        
        # Calculate loading statistics
        total_model_params = len(model_state_dict)
        loaded_params = len(mapped_params)
        load_percentage = (loaded_params / total_model_params) * 100
        
        loading_info.update({
            'loaded_params': loaded_params,
            'missing_params': missing_keys,
            'unexpected_params': unexpected_keys,
            'shape_mismatches': shape_mismatches,
            'load_percentage': load_percentage,
            'load_result': load_result
        })
        
        # Print loading report
        if verbose:
            print(f"\n  Loading Report:")
            print(f"    Checkpoint parameters: {len(checkpoint_params)}")
            print(f"    Model parameters: {total_model_params}")
            print(f"    Loaded parameters: {loaded_params}")
            print(f"    Load percentage: {load_percentage:.1f}%")
            
            if missing_keys:
                print(f"    Missing keys: {len(missing_keys)}")
                if verbose and len(missing_keys) <= 10:
                    for key in missing_keys:
                        print(f"      - {key}")
                elif len(missing_keys) > 10:
                    print(f"      (showing first 10)")
                    for key in missing_keys[:10]:
                        print(f"      - {key}")
            
            if unexpected_keys:
                print(f"    Unexpected keys: {len(unexpected_keys)}")
                if verbose and len(unexpected_keys) <= 10:
                    for key in unexpected_keys:
                        print(f"      - {key}")
                elif len(unexpected_keys) > 10:
                    print(f"      (showing first 10)")
                    for key in unexpected_keys[:10]:
                        print(f"      - {key}")
            
            if shape_mismatches:
                print(f"    Shape mismatches: {len(shape_mismatches)}")
                for mismatch in shape_mismatches[:5]:
                    print(f"      - {mismatch['name']}: {mismatch['ckpt_shape']} vs {mismatch['model_shape']}")
            
            # Success assessment
            if load_percentage >= 95:
                print(f"  [OK] Excellent loading success ({load_percentage:.1f}%)")
            elif load_percentage >= 80:
                print(f"  [WARN]  Good loading success ({load_percentage:.1f}%)")
            elif load_percentage >= 50:
                print(f"  [WARN]  Moderate loading success ({load_percentage:.1f}%)")
            else:
                print(f"  [ERROR] Poor loading success ({load_percentage:.1f}%)")
        
        success = load_percentage >= 50  # Consider successful if at least half loaded
        
        return success, loading_info
        
    except Exception as e:
        if verbose:
            print(f"  [ERROR] Error loading checkpoint: {e}")
            import traceback
            traceback.print_exc()
        
        loading_info['error'] = str(e)
        return False, loading_info


def verify_checkpoint_compatibility(
    checkpoint_path: Union[str, Path],
    model_class = None,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Verify checkpoint compatibility without loading into model.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model_class: Model class to test compatibility (optional)
        verbose: Whether to print details
        
    Returns:
        Compatibility analysis dictionary
    """
    analysis = {
        'checkpoint_path': str(checkpoint_path),
        'exists': False,
        'parameter_count': 0,
        'parameter_groups': {},
        'architecture_hints': {},
        'compatible_models': [],
        'issues': []
    }
    
    try:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            analysis['issues'].append(f"Checkpoint file not found: {checkpoint_path}")
            return analysis
        
        analysis['exists'] = True
        
        # Load checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        except UnpicklingError:
            logger.warning("Checkpoint requires pickle deserialization. Only load from trusted sources.")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Extract parameters
        if 'params' in checkpoint:
            params = checkpoint['params']
        elif 'params_ema' in checkpoint:
            params = checkpoint['params_ema']
        else:
            params = checkpoint
        
        analysis['parameter_count'] = len(params)
        
        # Analyze parameter groups
        param_groups = {}
        for name, param in params.items():
            if 'conv_1' in name:
                group = 'conv_1'
            elif 'block_' in name:
                group = 'blocks'
            elif 'upsampler' in name:
                group = 'upsampler'
            else:
                group = 'other'
            
            if group not in param_groups:
                param_groups[group] = []
            param_groups[group].append((name, param.shape))
        
        analysis['parameter_groups'] = param_groups
        
        # Architecture hints
        if 'conv_1' in param_groups:
            analysis['architecture_hints']['has_conv_1'] = True
        
        if 'blocks' in param_groups:
            block_names = set()
            for name, _ in param_groups['blocks']:
                parts = name.split('_')
                if len(parts) >= 2:
                    block_names.add(parts[1])
            analysis['architecture_hints']['block_count'] = len(block_names)
            analysis['architecture_hints']['block_pattern'] = sorted(list(block_names))
        
        if 'upsampler' in param_groups:
            analysis['architecture_hints']['has_upsampler'] = True
        
        # Test model compatibility if provided
        if model_class:
            try:
                model = model_class()
                model_params = model.state_dict()
                
                compatible_params = set(params.keys()) & set(model_params.keys())
                compatibility = len(compatible_params) / len(model_params) * 100
                
                analysis['compatible_models'].append({
                    'model_name': model_class.__name__,
                    'compatibility': compatibility,
                    'compatible_params': len(compatible_params),
                    'total_params': len(model_params)
                })
                
            except Exception as e:
                analysis['issues'].append(f"Error testing {model_class.__name__}: {e}")
        
        # Print analysis
        if verbose:
            print(f"Checkpoint Analysis: {checkpoint_path}")
            print(f"  Parameters: {analysis['parameter_count']}")
            print(f"  Groups: {list(param_groups.keys())}")
            
            for group, params_list in param_groups.items():
                print(f"    {group}: {len(params_list)} parameters")
            
            if analysis['architecture_hints']:
                print(f"  Architecture hints:")
                for hint, value in analysis['architecture_hints'].items():
                    print(f"    {hint}: {value}")
            
            if analysis['compatible_models']:
                print(f"  Model compatibility:")
                for model_info in analysis['compatible_models']:
                    print(f"    {model_info['model_name']}: {model_info['compatibility']:.1f}%")
            
            if analysis['issues']:
                print(f"  Issues:")
                for issue in analysis['issues']:
                    print(f"    - {issue}")
        
        return analysis
        
    except Exception as e:
        analysis['issues'].append(f"Analysis error: {e}")
        if verbose:
            print(f"Error analyzing checkpoint: {e}")
        return analysis
