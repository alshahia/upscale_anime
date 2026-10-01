"""
Loss stability utilities for training stability.

Provides loss normalization, NaN detection, and stability monitoring
to prevent training instability and ensure robust convergence.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any
import warnings


class LossStabilizer:
    """
    Loss stabilization utilities for training stability.
    
    Features:
    - Loss value clamping and normalization
    - NaN/Inf detection and handling
    - Loss scaling for stability
    - Stability monitoring and reporting
    """
    
    def __init__(
        self,
        max_loss_value: float = 1000.0,
        min_loss_value: float = 1e-8,
        enable_clamping: bool = True,
        enable_scaling: bool = True,
        verbose: bool = True
    ):
        self.max_loss_value = max_loss_value
        self.min_loss_value = min_loss_value
        self.enable_clamping = enable_clamping
        self.enable_scaling = enable_scaling
        self.verbose = verbose
        
        # Statistics tracking
        self.loss_history = []
        self.nan_count = 0
        self.inf_count = 0
        self.clamp_count = 0
        
    def stabilize_loss(self, loss: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Stabilize loss value to prevent training instability.
        
        Args:
            loss: Raw loss tensor
            
        Returns:
            Tuple of (stabilized_loss, stability_info)
        """
        stability_info = {
            'original_loss': loss.item(),
            'has_nan': False,
            'has_inf': False,
            'was_clamped': False,
            'was_scaled': False,
            'final_loss': None
        }
        
        # Check for NaN/Inf
        if torch.isnan(loss).any():
            stability_info['has_nan'] = True
            self.nan_count += 1
            if self.verbose:
                warnings.warn(f"NaN loss detected! Count: {self.nan_count}")
            # Replace NaN with small positive value
            loss = torch.tensor(self.min_loss_value, device=loss.device, dtype=loss.dtype)
        
        if torch.isinf(loss).any():
            stability_info['has_inf'] = True
            self.inf_count += 1
            if self.verbose:
                warnings.warn(f"Inf loss detected! Count: {self.inf_count}")
            # Replace Inf with max value
            loss = torch.tensor(self.max_loss_value, device=loss.device, dtype=loss.dtype)
        
        # Scale loss if needed
        if self.enable_scaling and loss.item() > 100:
            loss = loss / 100.0
            stability_info['was_scaled'] = True
            if self.verbose:
                warnings.warn(f"Loss scaled down by 100x for stability")
        
        # Clamp loss values
        if self.enable_clamping:
            original_value = loss.item()
            loss = torch.clamp(loss, min=self.min_loss_value, max=self.max_loss_value)
            if loss.item() != original_value:
                stability_info['was_clamped'] = True
                self.clamp_count += 1
                if self.verbose and self.clamp_count <= 5:  # Limit warnings
                    warnings.warn(f"Loss clamped: {original_value:.6f} -> {loss.item():.6f}")
        
        stability_info['final_loss'] = loss.item()
        
        # Track loss history
        self.loss_history.append(loss.item())
        if len(self.loss_history) > 1000:  # Keep only recent history
            self.loss_history.pop(0)
        
        return loss, stability_info
    
    def get_stability_report(self) -> Dict[str, Any]:
        """Get stability statistics report."""
        if not self.loss_history:
            return {'message': 'No loss history available'}
        
        recent_losses = self.loss_history[-100:] if len(self.loss_history) > 100 else self.loss_history
        
        return {
            'total_losses': len(self.loss_history),
            'recent_losses': len(recent_losses),
            'nan_count': self.nan_count,
            'inf_count': self.inf_count,
            'clamp_count': self.clamp_count,
            'recent_mean': sum(recent_losses) / len(recent_losses),
            'recent_min': min(recent_losses),
            'recent_max': max(recent_losses),
            'recent_std': (sum((x - sum(recent_losses)/len(recent_losses))**2 for x in recent_losses) / len(recent_losses))**0.5,
            'stability_score': self._calculate_stability_score(recent_losses)
        }
    
    def _calculate_stability_score(self, losses: list) -> float:
        """Calculate stability score (0-1, higher is better)."""
        if len(losses) < 10:
            return 0.5
        
        mean_loss = sum(losses) / len(losses)
        if mean_loss == 0:
            return 1.0
        
        # Calculate coefficient of variation
        std_loss = (sum((x - mean_loss)**2 for x in losses) / len(losses))**0.5
        cv = std_loss / mean_loss
        
        # Lower CV = higher stability
        stability_score = max(0.0, 1.0 - cv / 2.0)
        return stability_score


class EnhancedLoss(nn.Module):
    """
    Enhanced loss wrapper with stability features.
    
    Wraps any loss function with stability monitoring and NaN handling.
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        stabilizer: Optional[LossStabilizer] = None,
        weight: float = 1.0,
        name: str = "loss"
    ):
        super().__init__()
        self.base_loss = base_loss
        self.stabilizer = stabilizer or LossStabilizer()
        self.weight = weight
        self.name = name
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with stability monitoring.
        
        Args:
            pred: Predicted tensor
            target: Target tensor
            
        Returns:
            Tuple of (loss, loss_info)
        """
        # Compute base loss
        try:
            loss = self.base_loss(pred, target)
        except Exception as e:
            warnings.warn(f"Error computing {self.name}: {e}")
            # Fallback to L1 loss
            loss = torch.nn.functional.l1_loss(pred, target)
        
        # Apply stability measures
        stabilized_loss, stability_info = self.stabilizer.stabilize_loss(loss)
        
        # Apply weight
        final_loss = stabilized_loss * self.weight
        
        # Combine information
        loss_info = {
            'name': self.name,
            'weight': self.weight,
            'base_loss': loss.item(),
            'stabilized_loss': stabilized_loss.item(),
            'final_loss': final_loss.item(),
            **stability_info
        }
        
        return final_loss, loss_info


class StableCombinedLoss(nn.Module):
    """
    Stable combined loss with multiple loss components.
    
    Combines multiple losses with stability monitoring for each component.
    """
    
    def __init__(
        self,
        loss_components: Dict[str, nn.Module],
        stabilizer: Optional[LossStabilizer] = None
    ):
        super().__init__()
        self.loss_components = nn.ModuleDict(loss_components)
        self.stabilizer = stabilizer or LossStabilizer()
        
        # Track component statistics
        self.component_stats = {name: [] for name in loss_components.keys()}
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with component-wise stability monitoring.
        
        Args:
            pred: Predicted tensor
            target: Target tensor
            
        Returns:
            Tuple of (total_loss, loss_info)
        """
        total_loss = 0.0
        loss_info = {
            'components': {},
            'total_base_loss': 0.0,
            'total_stabilized_loss': 0.0,
            'total_final_loss': 0.0
        }
        
        # Compute each component
        for name, loss_fn in self.loss_components.items():
            try:
                if isinstance(loss_fn, EnhancedLoss):
                    component_loss, component_info = loss_fn(pred, target)
                else:
                    # Wrap regular loss
                    raw_loss = loss_fn(pred, target)
                    component_loss, stability_info = self.stabilizer.stabilize_loss(raw_loss)
                    component_info = {
                        'name': name,
                        'base_loss': raw_loss.item(),
                        'stabilized_loss': component_loss.item(),
                        'final_loss': component_loss.item(),
                        **stability_info
                    }
                
                total_loss += component_loss
                loss_info['components'][name] = component_info
                loss_info['total_base_loss'] += component_info.get('base_loss', 0)
                loss_info['total_stabilized_loss'] += component_info.get('stabilized_loss', 0)
                loss_info['total_final_loss'] += component_info.get('final_loss', 0)
                
                # Track component statistics
                self.component_stats[name].append(component_info.get('final_loss', 0))
                if len(self.component_stats[name]) > 1000:
                    self.component_stats[name].pop(0)
                    
            except Exception as e:
                warnings.warn(f"Error in loss component {name}: {e}")
                continue
        
        loss_info['total_final_loss'] = total_loss.item()
        
        return total_loss, loss_info
    
    def get_component_report(self) -> Dict[str, Any]:
        """Get detailed report for each component."""
        report = {}
        
        for name, history in self.component_stats.items():
            if not history:
                report[name] = {'message': 'No history available'}
                continue
            
            recent = history[-100:] if len(history) > 100 else history
            
            report[name] = {
                'total_samples': len(history),
                'recent_samples': len(recent),
                'recent_mean': sum(recent) / len(recent),
                'recent_min': min(recent),
                'recent_max': max(recent),
                'recent_std': (sum((x - sum(recent)/len(recent))**2 for x in recent) / len(recent))**0.5
            }
        
        return report


def create_stable_loss_config() -> Dict[str, Any]:
    """
    Create a stable loss configuration for training.
    
    Returns:
        Dictionary with stable loss settings
    """
    return {
        'loss_stability': {
            'max_loss_value': 1000.0,
            'min_loss_value': 1e-8,
            'enable_clamping': True,
            'enable_scaling': True,
            'verbose': True
        },
        'loss_components': {
            'pixel': {
                'type': 'l1',
                'weight': 1.0,
                'stabilize': True
            },
            'perceptual': {
                'type': 'dists',
                'weight': 0.2,
                'stabilize': True
            },
            'feature_distillation': {
                'type': 'feature_distillation',
                'weight': 0.05,
                'stabilize': True
            }
        }
    }


if __name__ == "__main__":
    # Test the loss stabilizer
    print("TESTING LOSS STABILITY")
    print("=" * 50)
    
    stabilizer = LossStabilizer()
    
    # Test normal loss
    normal_loss = torch.tensor(0.5)
    stabilized, info = stabilizer.stabilize_loss(normal_loss)
    print(f"Normal loss: {normal_loss.item()} -> {stabilized.item()}")
    
    # Test NaN loss
    nan_loss = torch.tensor(float('nan'))
    stabilized, info = stabilizer.stabilize_loss(nan_loss)
    print(f"NaN loss: {info['has_nan']} -> {stabilized.item()}")
    
    # Test Inf loss
    inf_loss = torch.tensor(float('inf'))
    stabilized, info = stabilizer.stabilize_loss(inf_loss)
    print(f"Inf loss: {info['has_inf']} -> {stabilized.item()}")
    
    # Test large loss
    large_loss = torch.tensor(5000.0)
    stabilized, info = stabilizer.stabilize_loss(large_loss)
    print(f"Large loss: {large_loss.item()} -> {stabilized.item()}")
    
    # Get stability report
    report = stabilizer.get_stability_report()
    print(f"\nStability report: {report}")
    
    print("\n[OK] Loss stability test completed")
