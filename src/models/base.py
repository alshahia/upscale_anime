"""
Base class for all Super-Resolution models
"""
import logging
import torch
import torch.nn as nn
from pickle import UnpicklingError
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class BaseSRModel(nn.Module, ABC):
    """
    Abstract base class for Super-Resolution models.
    All models (SPAN, Mamba-PAN, Ensemble, TinyStudent) should inherit from this.
    """
    
    def __init__(self, scale: int = 4, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            x: Low-resolution input [B, C, H, W]
        Returns:
            High-resolution output [B, C, H*scale, W*scale]
        """
        pass
    
    def get_feature_maps(self, x: torch.Tensor, layer_indices: Optional[list] = None) -> Dict[int, torch.Tensor]:
        """
        Extract intermediate feature maps for FAKD distillation
        Args:
            x: Input tensor
            layer_indices: Specific layers to extract (None = all registered hooks)
        Returns:
            Dictionary mapping layer index to feature tensor
        """
        # Default implementation - override in subclasses for efficiency
        return {}
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging"""
        return {
            'name': self.__class__.__name__,
            'scale': self.scale,
            'parameters': self.count_parameters(),
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
        }
    
    def load_pretrained(self, checkpoint_path: str, strict: bool = True):
        """Load pretrained weights"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        except UnpicklingError as e:
            logger.warning(f"weights_only=True failed ({e}), using pickle fallback - ensure checkpoint is trusted")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DataParallel)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.load_state_dict(state_dict, strict=strict)
        print(f"Loaded pretrained weights from {checkpoint_path}")
    
    def save_checkpoint(self, path: str, epoch: int, optimizer_state: Optional[dict] = None, **kwargs):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'state_dict': self.state_dict(),
            'model_info': self.get_model_info(),
            **kwargs
        }
        if optimizer_state is not None:
            checkpoint['optimizer'] = optimizer_state
        
        torch.save(checkpoint, path)
