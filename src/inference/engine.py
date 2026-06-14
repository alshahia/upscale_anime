"""
Inference Engine for Super-Resolution
High-level interface for running inference

Features:
- EMA model support for stable inference
- Test-Time Augmentation (TTA) for quality boost
- Model soup weight averaging
- Batch processing and benchmarking
"""
import logging
import torch
import numpy as np
from pickle import UnpicklingError
from pathlib import Path
from typing import Union, Optional, List, Dict, Callable
import cv2

logger = logging.getLogger(__name__)

# PyTorch AMP compatibility
try:
    from torch.amp import autocast  # PyTorch 2.0+
except ImportError:
    from torch.cuda.amp import autocast as _autocast  # PyTorch 1.x
    
    def autocast(device=None, enabled=True):
        return _autocast(enabled=enabled)

from models.span import create_span_model, SPANModel, create_neosr_span
from models.mamba_pan import create_mamba_pan_model, MambaPANModel
from models.ensemble import EnsembleTeacher, TinyStudent


# TTA augmentation functions
def _flip_h(img: torch.Tensor) -> torch.Tensor:
    """Horizontal flip"""
    return torch.flip(img, dims=[-1])

def _flip_v(img: torch.Tensor) -> torch.Tensor:
    """Vertical flip"""
    return torch.flip(img, dims=[-2])

def _rot90(img: torch.Tensor) -> torch.Tensor:
    """Rotate 90 degrees clockwise"""
    return torch.rot90(img, k=1, dims=[-2, -1])

def _rot180(img: torch.Tensor) -> torch.Tensor:
    """Rotate 180 degrees"""
    return torch.rot90(img, k=2, dims=[-2, -1])

def _rot270(img: torch.Tensor) -> torch.Tensor:
    """Rotate 270 degrees clockwise"""
    return torch.rot90(img, k=3, dims=[-2, -1])

# TTA transforms and their inverses
TTA_TRANSFORMS = {
    'identity': (lambda x: x, lambda x: x),
    'flip_h': (_flip_h, _flip_h),
    'flip_v': (_flip_v, _flip_v),
    'rot90': (_rot90, _rot270),
    'rot180': (_rot180, _rot180),
    'rot270': (_rot270, _rot90),
}


class InferenceEngine:
    """
    High-level inference engine for super-resolution models.
    
    Supports:
    - Single image inference
    - Batch inference
    - Video processing
    - Model switching
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = 'cuda',
        use_amp: bool = True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()
        
        self.use_amp = use_amp and self.device.type == 'cuda'
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        model_type: str = 'span',
        device: str = 'cuda',
        config: Optional[Dict] = None,
        use_ema: bool = True,
    ) -> 'InferenceEngine':
        """
        Create inference engine from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            model_type: 'span', 'mamba_pan', or 'tiny_student'
            device: Device to load model on
            config: Optional model configuration
            use_ema: If True and EMA weights exist, use EMA model for inference
        
        Returns:
            InferenceEngine instance
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        except UnpicklingError as e:
            # Fallback: try with safe_globals for custom Config objects
            try:
                from torch.serialization import safe_globals
                import utils.config
                logger.warning(f"Checkpoint requires safe_globals for Config objects: {e}")
                with safe_globals([utils.config.Config]):
                    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            except (UnpicklingError, AttributeError, ModuleNotFoundError):
                # Final fallback: load without weights_only (legacy behavior)
                logger.warning(f"Loading checkpoint without weights_only - ensure source is trusted: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Determine config
        if config is None:
            if 'config' in checkpoint:
                config = checkpoint['config']
                if hasattr(config, 'to_dict'):
                    config = config.to_dict()
            else:
                config = {}
        
        # Create model
        model_config = config.get('model', {}) if 'model' in config else config
        
        if model_type == 'span':
            model = create_span_model(model_config)
        elif model_type == 'neosr_span':
            model = create_neosr_span(model_config)
        elif model_type == 'mamba_pan':
            model = create_mamba_pan_model(model_config)
        elif model_type == 'tiny_student':
            from models.ensemble import TinyStudent
            model = TinyStudent(scale=model_config.get('scale', 4))
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Determine which weights to load (EMA preferred if available and requested)
        loaded_ema = False
        if use_ema and 'ema_model_state_dict' in checkpoint:
            try:
                model.load_state_dict(checkpoint['ema_model_state_dict'])
                loaded_ema = True
                print(f"Loaded EMA model weights from {checkpoint_path}")
            except Exception as e:
                print(f"Warning: Could not load EMA weights: {e}")
                print("Falling back to regular model weights...")
        
        if not loaded_ema:
            # Load regular weights
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded {model_type} model from {checkpoint_path}")
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return cls(model, device=device)
    
    @classmethod
    def from_ensemble(
        cls,
        model_a_checkpoint: Union[str, Path],
        model_b_checkpoint: Union[str, Path],
        device: str = 'cuda',
        weights: List[float] = [0.6, 0.4],
    ) -> 'InferenceEngine':
        """
        Create inference engine from ensemble.
        
        Args:
            model_a_checkpoint: Path to Model A checkpoint
            model_b_checkpoint: Path to Model B checkpoint
            device: Device to load models on
            weights: Ensemble weights [w_a, w_b]
        
        Returns:
            InferenceEngine instance
        """
        from models.ensemble import create_ensemble_teacher
        
        ensemble = create_ensemble_teacher(
            str(model_a_checkpoint),
            str(model_b_checkpoint),
            device=device,
            weights=weights,
        )
        
        print(f"Created ensemble teacher with weights: {weights}")
        
        return cls(ensemble, device=device)
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess numpy image to tensor.
        
        Args:
            image: Input image [H, W, C] or [H, W], BGR or RGB
        
        Returns:
            Preprocessed tensor [1, C, H, W]
        """
        # Ensure 3 channels
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        tensor = torch.from_numpy(image).float() / 255.0
        
        # HWC to CHW
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        
        return tensor
    
    def postprocess(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Postprocess tensor to numpy image.
        
        Args:
            tensor: Output tensor [1, C, H, W] or [C, H, W]
        
        Returns:
            Numpy image [H, W, C], BGR format
        """
        # Remove batch dimension if present
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # Move to CPU and clamp
        tensor = tensor.detach().cpu().clamp(0, 1)
        
        # CHW to HWC
        image = tensor.permute(1, 2, 0).numpy()
        
        # To uint8
        image = (image * 255).astype(np.uint8)
        
        # RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image
    
    def run(
        self,
        lr_image: Union[np.ndarray, torch.Tensor],
        return_tensor: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Run inference on a single image.
        
        Args:
            lr_image: Low-resolution image (numpy array or tensor)
            return_tensor: If True, return tensor instead of numpy array
        
        Returns:
            Super-resolved image
        """
        # Preprocess if numpy
        if isinstance(lr_image, np.ndarray):
            lr_tensor = self.preprocess(lr_image).to(self.device)
        else:
            lr_tensor = lr_image.to(self.device)
        
        # Add batch dimension if missing
        if lr_tensor.dim() == 3:
            lr_tensor = lr_tensor.unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            with autocast(device='cuda', enabled=self.use_amp):
                sr_tensor = self.model(lr_tensor)
        
        # Postprocess
        if return_tensor:
            return sr_tensor
        
        return self.postprocess(sr_tensor)
    
    def run_tta(
        self,
        lr_image: Union[np.ndarray, torch.Tensor],
        return_tensor: bool = False,
        augmentations: List[str] = None,
        merge_mode: str = 'mean',
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Run inference with Test-Time Augmentation (TTA).
        
        Averages predictions across multiple augmented versions of the input
        for improved quality (+0.2-0.4 dB PSNR typical).
        
        Args:
            lr_image: Low-resolution image (numpy array or tensor)
            return_tensor: If True, return tensor instead of numpy array
            augmentations: List of augmentation names to use. 
                          Default: ['identity', 'flip_h', 'flip_v']
            merge_mode: How to merge predictions ('mean' or 'median')
        
        Returns:
            Super-resolved image (higher quality than run())
        """
        if augmentations is None:
            # Default TTA: horizontal and vertical flips
            augmentations = ['identity', 'flip_h', 'flip_v']
        
        # Preprocess if numpy
        if isinstance(lr_image, np.ndarray):
            lr_tensor = self.preprocess(lr_image).to(self.device)
        else:
            lr_tensor = lr_image.to(self.device)
        
        # Add batch dimension if missing
        if lr_tensor.dim() == 3:
            lr_tensor = lr_tensor.unsqueeze(0)
        
        # Collect predictions from all augmentations
        predictions = []
        
        with torch.no_grad():
            for aug_name in augmentations:
                if aug_name not in TTA_TRANSFORMS:
                    print(f"Warning: Unknown augmentation '{aug_name}', skipping")
                    continue
                
                forward_fn, inverse_fn = TTA_TRANSFORMS[aug_name]
                
                # Apply forward transform
                augmented = forward_fn(lr_tensor)
                
                # Inference
                with autocast(device='cuda', enabled=self.use_amp):
                    pred = self.model(augmented)
                
                # Apply inverse transform to prediction
                pred = inverse_fn(pred)
                
                # Clamp to valid range
                pred = torch.clamp(pred, 0, 1)
                
                predictions.append(pred)
        
        # Merge predictions
        if len(predictions) == 0:
            raise ValueError("No valid predictions from TTA augmentations")
        
        stacked = torch.stack(predictions)
        
        if merge_mode == 'mean':
            sr_tensor = stacked.mean(dim=0)
        elif merge_mode == 'median':
            sr_tensor = stacked.median(dim=0)[0]
        else:
            raise ValueError(f"Unknown merge_mode: {merge_mode}")
        
        # Postprocess
        if return_tensor:
            return sr_tensor
        
        return self.postprocess(sr_tensor)
    
    def run_batch(
        self,
        lr_images: List[Union[np.ndarray, torch.Tensor]],
    ) -> List[np.ndarray]:
        """
        Run inference on a batch of images.
        
        Args:
            lr_images: List of low-resolution images
        
        Returns:
            List of super-resolved images
        """
        results = []
        
        for lr_image in lr_images:
            sr_image = self.run(lr_image)
            results.append(sr_image)
        
        return results
    
    def benchmark(
        self,
        input_size: tuple = (3, 128, 128),
        num_runs: int = 100,
        warmup: int = 10,
    ) -> Dict[str, float]:
        """
        Benchmark inference speed.
        
        Args:
            input_size: Input tensor size (C, H, W)
            num_runs: Number of benchmark runs
            warmup: Number of warmup runs
        
        Returns:
            Dictionary with benchmark results
        """
        import time
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_size).to(self.device)
        
        # Warmup
        print(f"Warming up ({warmup} runs)...")
        for _ in range(warmup):
            with torch.no_grad():
                _ = self.model(dummy_input)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        print(f"Benchmarking ({num_runs} runs)...")
        times = []
        
        for _ in range(num_runs):
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            start = time.time()
            
            with torch.no_grad():
                _ = self.model(dummy_input)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            times.append(time.time() - start)
        
        # Compute statistics
        times = np.array(times)
        
        h, w = input_size[1], input_size[2]
        scale = 4  # Assume 4x
        output_pixels = (h * scale) * (w * scale)
        
        results = {
            'mean_time_ms': np.mean(times) * 1000,
            'std_time_ms': np.std(times) * 1000,
            'min_time_ms': np.min(times) * 1000,
            'max_time_ms': np.max(times) * 1000,
            'fps': 1.0 / np.mean(times),
            'output_pixels': output_pixels,
            'output_resolution': f"{w*scale}x{h*scale}",
        }
        
        print(f"\nBenchmark Results ({num_runs} runs):")
        print(f"  Mean time: {results['mean_time_ms']:.2f} ± {results['std_time_ms']:.2f} ms")
        print(f"  FPS: {results['fps']:.2f}")
        print(f"  Resolution: {results['output_resolution']}")
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'device': str(self.device),
            'use_amp': self.use_amp,
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }


def quick_inference(
    checkpoint_path: str,
    input_path: str,
    output_path: str,
    model_type: str = 'span',
    device: str = 'cuda',
    use_ema: bool = True,
    use_tta: bool = False,
) -> np.ndarray:
    """
    Quick inference function for single image.
    
    Args:
        checkpoint_path: Path to model checkpoint
        input_path: Path to input image
        output_path: Path to save output
        model_type: Model type
        device: Device to use
        use_ema: If True and EMA weights exist, use EMA model
        use_tta: If True, use Test-Time Augmentation (slower but better quality)
    
    Returns:
        Super-resolved image as numpy array
    """
    # Create engine
    engine = InferenceEngine.from_checkpoint(
        checkpoint_path, model_type, device, use_ema=use_ema
    )
    
    # Load image
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Failed to load image: {input_path}")
    
    # Run inference (with or without TTA)
    if use_tta:
        sr_image = engine.run_tta(image)
        print(f"Inference completed with TTA (EMA={use_ema})")
    else:
        sr_image = engine.run(image)
        print(f"Inference completed (EMA={use_ema})")
    
    # Save
    cv2.imwrite(output_path, sr_image)
    print(f"Saved to: {output_path}")
    
    return sr_image


class ModelSoup:
    """
    Model Soup: Averaging weights of multiple checkpoints.
    
    Research shows weight averaging can improve accuracy without increasing
    inference time (Wortsman et al., 2022).
    
    Example:
        >>> soup = ModelSoup.from_checkpoints([
        ...     'checkpoints/epoch_100.pth',
        ...     'checkpoints/epoch_150.pth',
        ...     'checkpoints/best.pth'
        ... ])
        >>> averaged_state = soup.create_soup()
        >>> torch.save(averaged_state, 'checkpoints/soup.pth')
    """
    
    def __init__(self, state_dicts: List[Dict], configs: List[Dict] = None):
        """
        Args:
            state_dicts: List of model state dicts to average
            configs: Optional list of configs (for validation)
        """
        self.state_dicts = state_dicts
        self.configs = configs
        
        if len(state_dicts) < 2:
            raise ValueError("Need at least 2 state dicts for soup")
    
    @classmethod
    def from_checkpoints(
        cls,
        checkpoint_paths: List[Union[str, Path]],
        prefer_ema: bool = True,
    ) -> 'ModelSoup':
        """
        Load model soup from checkpoint files.
        
        Args:
            checkpoint_paths: List of checkpoint file paths
            prefer_ema: If True, use EMA weights when available
        
        Returns:
            ModelSoup instance
        """
        state_dicts = []
        configs = []
        
        for path in checkpoint_paths:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {path}")
            
            # Load checkpoint with fallback for custom objects
            try:
                checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            except UnpicklingError:
                try:
                    from torch.serialization import safe_globals
                    import utils.config
                    with safe_globals([utils.config.Config]):
                        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
                except (UnpicklingError, AttributeError, ModuleNotFoundError):
                    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            
            # Choose weights (EMA preferred)
            if prefer_ema and 'ema_model_state_dict' in checkpoint:
                state_dict = checkpoint['ema_model_state_dict']
                print(f"  {path.name}: Using EMA weights")
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"  {path.name}: Using regular weights")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f"  {path.name}: Using state_dict")
            else:
                state_dict = checkpoint
                print(f"  {path.name}: Using full checkpoint")
            
            state_dicts.append(state_dict)
            
            # Extract config if available
            if 'config' in checkpoint:
                config = checkpoint['config']
                if hasattr(config, 'to_dict'):
                    config = config.to_dict()
                configs.append(config)
        
        return cls(state_dicts, configs if configs else None)
    
    def create_soup(
        self,
        weights: Optional[List[float]] = None,
        exclude_keys: List[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Create averaged (soup) state dict.
        
        Args:
            weights: Optional weights for each checkpoint (default: equal weighting)
            exclude_keys: Keys to exclude from averaging (e.g., 'optimizer_state')
        
        Returns:
            Averaged state dict
        """
        if weights is None:
            weights = [1.0 / len(self.state_dicts)] * len(self.state_dicts)
        
        if len(weights) != len(self.state_dicts):
            raise ValueError("Number of weights must match number of state dicts")
        
        exclude_keys = exclude_keys or []
        
        # Get all keys from first state dict
        reference_dict = self.state_dicts[0]
        averaged = {}
        
        print(f"Creating model soup from {len(self.state_dicts)} checkpoints...")
        
        for key in reference_dict.keys():
            # Skip excluded keys
            if any(excluded in key for excluded in exclude_keys):
                continue
            
            # Collect tensors for this key from all state dicts
            tensors = []
            valid_weights = []
            
            for i, state_dict in enumerate(self.state_dicts):
                if key in state_dict:
                    tensors.append(state_dict[key].float())
                    valid_weights.append(weights[i])
            
            if len(tensors) == 0:
                print(f"  Warning: Key '{key}' not found in any state dict")
                continue
            
            # Normalize weights
            weight_sum = sum(valid_weights)
            normalized_weights = [w / weight_sum for w in valid_weights]
            
            # Weighted average
            if len(tensors) == 1:
                averaged[key] = tensors[0]
            else:
                stacked = torch.stack(tensors)
                weights_tensor = torch.tensor(normalized_weights, dtype=stacked.dtype)
                
                # Reshape weights for broadcasting
                while weights_tensor.dim() < stacked.dim():
                    weights_tensor = weights_tensor.unsqueeze(-1)
                
                averaged[key] = (stacked * weights_tensor).sum(dim=0)
        
        print(f"  Averaged {len(averaged)} parameters")
        return averaged
    
    def save_soup(
        self,
        output_path: Union[str, Path],
        weights: Optional[List[float]] = None,
        reference_config: Optional[Dict] = None,
    ):
        """
        Create and save soup checkpoint.
        
        Args:
            output_path: Where to save the soup checkpoint
            weights: Optional weights for averaging
            reference_config: Config to include in saved checkpoint
        """
        averaged_state = self.create_soup(weights)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': averaged_state,
            'soup_metadata': {
                'num_models': len(self.state_dicts),
                'weights': weights or [1.0 / len(self.state_dicts)] * len(self.state_dicts),
            }
        }
        
        if reference_config:
            checkpoint['config'] = reference_config
        elif self.configs:
            checkpoint['config'] = self.configs[0]
        
        torch.save(checkpoint, output_path)
        print(f"Model soup saved to: {output_path}")
    
    @staticmethod
    def find_checkpoints(
        checkpoint_dir: Union[str, Path],
        pattern: str = 'epoch_*.pth',
        last_n: int = 5,
    ) -> List[Path]:
        """
        Find recent checkpoints in a directory.
        
        Args:
            checkpoint_dir: Directory to search
            pattern: Glob pattern for checkpoint files
            last_n: Number of most recent checkpoints to return
        
        Returns:
            List of checkpoint paths
        """
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            return []
        
        checkpoints = sorted(checkpoint_dir.glob(pattern))
        
        # Return last N checkpoints
        return checkpoints[-last_n:] if len(checkpoints) > last_n else checkpoints
