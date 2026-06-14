"""
NaN detection and recovery utilities for training stability.

Provides comprehensive NaN detection, automatic recovery mechanisms,
and training state restoration to handle NaN/Inf events during training.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any, List
import warnings
import copy
from pathlib import Path


class NaNHandler:
    """
    Comprehensive NaN detection and recovery system.
    
    Features:
    - Real-time NaN/Inf detection in gradients, outputs, and parameters
    - Automatic parameter reset on NaN detection
    - Training state recovery mechanisms
    - Detailed NaN reporting and statistics
    - Preventive measures to avoid NaN occurrences
    """
    
    def __init__(
        self,
        enabled: bool = True,
        reset_on_nan: bool = True,
        backup_frequency: int = 100,
        max_nan_events: int = 10,
        verbose: bool = True
    ):
        self.enabled = enabled
        self.reset_on_nan = reset_on_nan
        self.backup_frequency = backup_frequency
        self.max_nan_events = max_nan_events
        self.verbose = verbose
        
        # Statistics tracking
        self.nan_events = []
        self.total_nan_count = 0
        self.total_inf_count = 0
        self.reset_count = 0
        self.backup_count = 0
        
        # Recovery state
        self.last_backup = None
        self.backup_step = 0
        self.nan_threshold = 1e-6
        
    def detect_nan_in_tensor(self, tensor: torch.Tensor, name: str = "tensor") -> Dict[str, Any]:
        """
        Detect NaN/Inf values in a tensor.
        
        Args:
            tensor: Tensor to check
            name: Name of the tensor for reporting
            
        Returns:
            Detection results dictionary
        """
        if not self.enabled:
            return {'has_nan': False, 'has_inf': False, 'name': name}
        
        result = {
            'name': name,
            'has_nan': False,
            'has_inf': False,
            'nan_count': 0,
            'inf_count': 0,
            'min_value': None,
            'max_value': None,
            'mean_value': None,
            'std_value': None
        }
        
        # Check for NaN
        nan_mask = torch.isnan(tensor)
        result['nan_count'] = nan_mask.sum().item()
        result['has_nan'] = result['nan_count'] > 0
        
        # Check for Inf
        inf_mask = torch.isinf(tensor)
        result['inf_count'] = inf_mask.sum().item()
        result['has_inf'] = result['inf_count'] > 0
        
        # Statistics (only if no NaN/Inf)
        if not result['has_nan'] and not result['has_inf'] and tensor.numel() > 0:
            result['min_value'] = tensor.min().item()
            result['max_value'] = tensor.max().item()
            result['mean_value'] = tensor.mean().item()
            result['std_value'] = tensor.std().item()
        
        # Update statistics
        if result['has_nan']:
            self.total_nan_count += result['nan_count']
        if result['has_inf']:
            self.total_inf_count += result['inf_count']
        
        return result
    
    def detect_nan_in_model(self, model: nn.Module) -> Dict[str, Any]:
        """
        Detect NaN/Inf values in model parameters.
        
        Args:
            model: Model to check
            
        Returns:
            Detection results dictionary
        """
        results = {
            'model_name': type(model).__name__,
            'total_parameters': 0,
            'nan_parameters': [],
            'inf_parameters': [],
            'healthy_parameters': 0,
            'nan_detected': False,
            'inf_detected': False
        }
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                results['total_parameters'] += param.numel()
                
                detection = self.detect_nan_in_tensor(param.data, name)
                
                if detection['has_nan']:
                    results['nan_parameters'].append({
                        'name': name,
                        'count': detection['nan_count'],
                        'shape': list(param.shape)
                    })
                    results['nan_detected'] = True
                
                if detection['has_inf']:
                    results['inf_parameters'].append({
                        'name': name,
                        'count': detection['inf_count'],
                        'shape': list(param.shape)
                    })
                    results['inf_detected'] = True
                
                if not detection['has_nan'] and not detection['has_inf']:
                    results['healthy_parameters'] += param.numel()
        
        return results
    
    def detect_nan_in_gradients(self, model: nn.Module) -> Dict[str, Any]:
        """
        Detect NaN/Inf values in model gradients.
        
        Args:
            model: Model to check
            
        Returns:
            Detection results dictionary
        """
        results = {
            'model_name': type(model).__name__,
            'total_gradients': 0,
            'nan_gradients': [],
            'inf_gradients': [],
            'healthy_gradients': 0,
            'nan_detected': False,
            'inf_detected': False
        }
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                results['total_gradients'] += param.grad.numel()
                
                detection = self.detect_nan_in_tensor(param.grad.data, f"{name}_grad")
                
                if detection['has_nan']:
                    results['nan_gradients'].append({
                        'name': name,
                        'count': detection['nan_count'],
                        'shape': list(param.grad.shape)
                    })
                    results['nan_detected'] = True
                
                if detection['has_inf']:
                    results['inf_gradients'].append({
                        'name': name,
                        'count': detection['inf_count'],
                        'shape': list(param.grad.shape)
                    })
                    results['inf_detected'] = True
                
                if not detection['has_nan'] and not detection['has_inf']:
                    results['healthy_gradients'] += param.grad.numel()
        
        return results
    
    def create_backup(self, model: nn.Module, optimizer: torch.optim.Optimizer, step: int) -> Dict[str, Any]:
        """
        Create a backup of model and optimizer state.
        
        Args:
            model: Model to backup
            optimizer: Optimizer to backup
            step: Current training step
            
        Returns:
            Backup information dictionary
        """
        if not self.enabled:
            return {'success': False, 'reason': 'NaN handler disabled'}
        
        try:
            backup = {
                'step': step,
                'model_state_dict': copy.deepcopy(model.state_dict()),
                'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                'timestamp': __import__('time').time()
            }
            
            self.last_backup = backup
            self.backup_step = step
            self.backup_count += 1
            
            if self.verbose:
                print(f"Backup created at step {step}")
            
            return {'success': True, 'backup_id': self.backup_count}
            
        except Exception as e:
            if self.verbose:
                print(f"Failed to create backup: {e}")
            return {'success': False, 'reason': str(e)}
    
    def restore_from_backup(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """
        Restore model and optimizer from last backup.
        
        Args:
            model: Model to restore
            optimizer: Optimizer to restore
            
        Returns:
            Restore results dictionary
        """
        if not self.enabled or self.last_backup is None:
            return {'success': False, 'reason': 'No backup available'}
        
        try:
            # Restore model state
            model.load_state_dict(self.last_backup['model_state_dict'])
            
            # Restore optimizer state
            optimizer.load_state_dict(self.last_backup['optimizer_state_dict'])
            
            self.reset_count += 1
            
            if self.verbose:
                print(f"Restored from backup at step {self.last_backup['step']}")
            
            return {
                'success': True,
                'restored_step': self.last_backup['step'],
                'backup_id': self.backup_count
            }
            
        except Exception as e:
            if self.verbose:
                print(f"Failed to restore from backup: {e}")
            return {'success': False, 'reason': str(e)}
    
    def reset_model_parameters(self, model: nn.Module, reset_strategy: str = "xavier") -> Dict[str, Any]:
        """
        Reset model parameters to recover from NaN.
        
        Args:
            model: Model to reset
            reset_strategy: Reset strategy ("xavier", "kaiming", "normal", "zeros")
            
        Returns:
            Reset results dictionary
        """
        if not self.enabled:
            return {'success': False, 'reason': 'NaN handler disabled'}
        
        try:
            reset_count = 0
            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # Check if parameter has NaN/Inf
                    detection = self.detect_nan_in_tensor(param.data, name)
                    
                    if detection['has_nan'] or detection['has_inf']:
                        # Reset parameter based on strategy
                        if reset_strategy == "xavier":
                            nn.init.xavier_uniform_(param.data)
                        elif reset_strategy == "kaiming":
                            nn.init.kaiming_normal_(param.data, mode='fan_out', nonlinearity='relu')
                        elif reset_strategy == "normal":
                            nn.init.normal_(param.data, mean=0.0, std=0.02)
                        elif reset_strategy == "zeros":
                            nn.init.zeros_(param.data)
                        else:
                            nn.init.xavier_uniform_(param.data)  # Default
                        
                        reset_count += 1
                        
                        if self.verbose:
                            print(f"Reset parameter: {name} (strategy: {reset_strategy})")
            
            self.reset_count += 1
            
            return {
                'success': True,
                'reset_parameters': reset_count,
                'strategy': reset_strategy
            }
            
        except Exception as e:
            if self.verbose:
                print(f"Failed to reset parameters: {e}")
            return {'success': False, 'reason': str(e)}
    
    def handle_nan_event(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        loss: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Handle a NaN event with recovery actions.
        
        Args:
            model: Model with NaN issues
            optimizer: Optimizer
            step: Current training step
            loss: Current loss (optional)
            
        Returns:
            Handling results dictionary
        """
        if not self.enabled:
            return {'success': False, 'reason': 'NaN handler disabled'}
        
        # Record NaN event
        event = {
            'step': step,
            'loss': loss.item() if loss is not None else None,
            'timestamp': __import__('time').time()
        }
        
        # Detect NaN locations
        param_detection = self.detect_nan_in_model(model)
        grad_detection = self.detect_nan_in_gradients(model)
        
        event.update({
            'param_nan_detected': param_detection['nan_detected'],
            'param_inf_detected': param_detection['inf_detected'],
            'grad_nan_detected': grad_detection['nan_detected'],
            'grad_inf_detected': grad_detection['inf_detected'],
            'nan_parameters': param_detection['nan_parameters'],
            'inf_parameters': param_detection['inf_parameters'],
            'nan_gradients': grad_detection['nan_gradients'],
            'inf_gradients': grad_detection['inf_gradients']
        })
        
        self.nan_events.append(event)
        
        # Check if we should abort training
        if len(self.nan_events) >= self.max_nan_events:
            return {
                'success': False,
                'reason': f'Too many NaN events ({len(self.nan_events)} >= {self.max_nan_events})',
                'action': 'abort_training'
            }
        
        # Recovery actions
        recovery_results = {}
        
        # 1. Try to restore from backup
        if self.last_backup is not None:
            restore_result = self.restore_from_backup(model, optimizer)
            recovery_results['restore_backup'] = restore_result
            
            if restore_result['success']:
                if self.verbose:
                    print(f"Recovered from NaN event using backup at step {restore_result['restored_step']}")
                return {
                    'success': True,
                    'action': 'restored_from_backup',
                    'event': event,
                    'recovery': recovery_results
                }
        
        # 2. Reset problematic parameters
        if self.reset_on_nan:
            reset_result = self.reset_model_parameters(model, "xavier")
            recovery_results['reset_parameters'] = reset_result
            
            if reset_result['success']:
                if self.verbose:
                    print(f"Recovered from NaN event by resetting {reset_result['reset_parameters']} parameters")
                return {
                    'success': True,
                    'action': 'reset_parameters',
                    'event': event,
                    'recovery': recovery_results
                }
        
        # 3. Last resort - full model reset with kaiming (different strategy)
        full_reset_result = self.reset_model_parameters(model, "kaiming")
        recovery_results['full_reset'] = full_reset_result
        
        return {
            'success': full_reset_result['success'],
            'action': 'full_reset',
            'event': event,
            'recovery': recovery_results
        }
    
    def should_create_backup(self, step: int) -> bool:
        """Check if we should create a backup at this step."""
        return (self.enabled and 
                step % self.backup_frequency == 0 and 
                step > self.backup_step)
    
    def get_nan_report(self) -> Dict[str, Any]:
        """Get comprehensive NaN detection and recovery report."""
        return {
            'enabled': self.enabled,
            'total_nan_events': len(self.nan_events),
            'total_nan_count': self.total_nan_count,
            'total_inf_count': self.total_inf_count,
            'reset_count': self.reset_count,
            'backup_count': self.backup_count,
            'last_backup_step': self.backup_step,
            'max_nan_events': self.max_nan_events,
            'recent_events': self.nan_events[-5:] if self.nan_events else [],
            'nan_events': self.nan_events
        }


class TrainingNaNMonitor:
    """
    Training monitor with integrated NaN detection and recovery.
    
    Monitors training progress and automatically handles NaN events.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        nan_handler: Optional[NaNHandler] = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.nan_handler = nan_handler or NaNHandler()
        self.step = 0
        
    def step_monitor(self, loss: torch.Tensor) -> Dict[str, Any]:
        """
        Monitor a training step and handle any issues.
        
        Args:
            loss: Current loss value
            
        Returns:
            Monitoring results
        """
        self.step += 1
        
        results = {
            'step': self.step,
            'loss': loss.item(),
            'backup_created': False,
            'nan_handled': False,
            'issues_detected': []
        }
        
        # Check for NaN in loss
        loss_detection = self.nan_handler.detect_nan_in_tensor(loss, "loss")
        if loss_detection['has_nan'] or loss_detection['has_inf']:
            results['issues_detected'].append("loss_nan")
        
        # Check for NaN in model parameters
        param_detection = self.nan_handler.detect_nan_in_model(self.model)
        if param_detection['nan_detected'] or param_detection['inf_detected']:
            results['issues_detected'].append("param_nan")
        
        # Check for NaN in gradients (if available)
        grad_detection = self.nan_handler.detect_nan_in_gradients(self.model)
        if grad_detection['nan_detected'] or grad_detection['inf_detected']:
            results['issues_detected'].append("grad_nan")
        
        # Handle NaN events
        if results['issues_detected']:
            handle_result = self.nan_handler.handle_nan_event(
                self.model, self.optimizer, self.step, loss
            )
            results['nan_handled'] = True
            results['nan_recovery'] = handle_result
        else:
            # Create backup if needed
            if self.nan_handler.should_create_backup(self.step):
                backup_result = self.nan_handler.create_backup(self.model, self.optimizer, self.step)
                results['backup_created'] = backup_result['success']
        
        return results


if __name__ == "__main__":
    # Test NaN handler
    print("TESTING NaN HANDLER")
    print("=" * 50)
    
    # Create a simple model
    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Create NaN handler
    nan_handler = NaNHandler(verbose=True)
    
    # Test normal tensor
    normal_tensor = torch.randn(5, 5)
    result = nan_handler.detect_nan_in_tensor(normal_tensor, "normal")
    print(f"Normal tensor: {result}")
    
    # Test NaN tensor
    nan_tensor = torch.full((5, 5), float('nan'))
    result = nan_handler.detect_nan_in_tensor(nan_tensor, "nan")
    print(f"NaN tensor: {result}")
    
    # Test model detection
    model_result = nan_handler.detect_nan_in_model(model)
    print(f"Model detection: {model_result['nan_detected']}")
    
    # Test backup/restore
    backup_result = nan_handler.create_backup(model, optimizer, 100)
    print(f"Backup result: {backup_result}")
    
    # Get report
    report = nan_handler.get_nan_report()
    print(f"NaN report: {report}")
    
    print("\n[OK] NaN handler test completed")
