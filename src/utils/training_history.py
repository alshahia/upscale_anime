"""
Training History Tracker
========================

Tracks and saves training metrics for later visualization.
Integrates with TrainingVisualizer to generate reports.

Usage:
    from utils.training_history import TrainingHistory
    
    history = TrainingHistory()
    
    for epoch in range(epochs):
        train_metrics = train_epoch(...)
        val_metrics = validate(...)
        
        history.log_epoch(epoch, {
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'val_psnr': val_metrics['psnr'],
        })
    
    # Save to file
    history.save('logs/training_history.json')
    
    # Or get data for visualization
    data = history.get_history()
    visualizer.plot_training_curves(data)

Author: Anime SR Training System
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class TrainingHistory:
    """
    Tracks training metrics across epochs.
    
    Provides structured storage, JSON serialization, and
    integration with TrainingVisualizer.
    
    Attributes:
        metrics: Dictionary of metric names to lists of values
        epochs: List of epoch numbers
        start_time: Training start timestamp
        config: Training configuration snapshot
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize training history tracker.
        
        Args:
            config: Training configuration to store with history
        """
        self.metrics = defaultdict(list)
        self.epochs = []
        self.start_time = time.time()
        self.config = config or {}
        
        # Metadata
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'version': '1.0.0',
        }
        
        # Stage tracking for multi-stage training
        self.current_stage = None
        self.stage_boundaries = []
    
    def log_epoch(self, epoch: int, metrics: Dict[str, float], stage: Optional[str] = None):
        """
        Log metrics for an epoch.
        
        Args:
            epoch: Epoch number (0-indexed)
            metrics: Dictionary of metric names and values
            stage: Optional stage name (e.g., 'stage1', 'stage2')
        """
        self.epochs.append(epoch)
        
        # Track stage changes
        if stage and stage != self.current_stage:
            self.stage_boundaries.append({
                'epoch': epoch,
                'stage': stage,
                'timestamp': time.time() - self.start_time,
            })
            self.current_stage = stage
        
        # Store metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.metrics[key].append(float(value))
            else:
                # Handle non-numeric values (e.g., learning rate schedule state)
                pass
    
    def log_batch(self, batch_idx: int, metrics: Dict[str, float]):
        """
        Log metrics for a batch (not stored permanently, for real-time tracking).
        
        Args:
            batch_idx: Batch index within epoch
            metrics: Dictionary of metric names and values
        """
        # Batch metrics are typically not stored in history
        # but can be used for real-time visualization
        pass
    
    def get_history(self) -> Dict[str, List[float]]:
        """
        Get complete training history as dictionary.
        
        Returns:
            Dictionary mapping metric names to lists of values
        """
        return dict(self.metrics)
    
    def get_best_epoch(self, metric: str, mode: str = 'min') -> tuple:
        """
        Find the epoch with best value for a metric.
        
        Args:
            metric: Metric name to analyze
            mode: 'min' for loss, 'max' for accuracy/PSNR
            
        Returns:
            Tuple of (epoch_index, best_value)
        """
        if metric not in self.metrics:
            return (-1, float('inf') if mode == 'min' else float('-inf'))
        
        values = self.metrics[metric]
        if not values:
            return (-1, float('inf') if mode == 'min' else float('-inf'))
        
        if mode == 'min':
            best_idx = min(range(len(values)), key=lambda i: values[i])
        else:
            best_idx = max(range(len(values)), key=lambda i: values[i])
        
        return (best_idx, values[best_idx])
    
    def get_final_metrics(self) -> Dict[str, float]:
        """
        Get final (most recent) values for all metrics.
        
        Returns:
            Dictionary of metric names to final values
        """
        return {k: v[-1] if v else 0.0 for k, v in self.metrics.items()}
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get training summary statistics.
        
        Returns:
            Dictionary with summary information
        """
        summary = {
            'total_epochs': len(self.epochs),
            'duration_seconds': time.time() - self.start_time,
            'stages': self.stage_boundaries,
            'final_metrics': self.get_final_metrics(),
        }
        
        # Add best metrics
        summary['best_metrics'] = {}
        for metric in self.metrics.keys():
            if 'loss' in metric.lower():
                mode = 'min'
            elif 'psnr' in metric.lower() or 'ssim' in metric.lower():
                mode = 'max'
            else:
                mode = 'min'
            
            epoch_idx, best_value = self.get_best_epoch(metric, mode)
            if epoch_idx >= 0:
                summary['best_metrics'][metric] = {
                    'value': best_value,
                    'epoch': self.epochs[epoch_idx] if epoch_idx < len(self.epochs) else epoch_idx,
                }
        
        return summary
    
    def save(self, output_path: str):
        """
        Save training history to JSON file.
        
        Args:
            output_path: Path to save JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'metadata': self.metadata,
            'config': self.config,
            'epochs': self.epochs,
            'metrics': dict(self.metrics),
            'stages': self.stage_boundaries,
            'summary': self.get_summary(),
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, history_path: str) -> 'TrainingHistory':
        """
        Load training history from JSON file.
        
        Args:
            history_path: Path to JSON file
            
        Returns:
            TrainingHistory instance
        """
        with open(history_path, 'r') as f:
            data = json.load(f)
        
        history = cls(config=data.get('config', {}))
        history.metadata = data.get('metadata', {})
        history.epochs = data.get('epochs', [])
        history.metrics = defaultdict(list, data.get('metrics', {}))
        history.stage_boundaries = data.get('stages', [])
        
        return history
    
    def merge(self, other: 'TrainingHistory'):
        """
        Merge another training history into this one.
        Useful for multi-stage training.
        
        Args:
            other: Another TrainingHistory instance
        """
        # Offset epochs
        max_epoch = max(self.epochs) if self.epochs else -1
        
        for epoch in other.epochs:
            self.epochs.append(epoch + max_epoch + 1)
        
        # Merge metrics
        for key, values in other.metrics.items():
            self.metrics[key].extend(values)
        
        # Merge stage boundaries
        for boundary in other.stage_boundaries:
            new_boundary = boundary.copy()
            new_boundary['epoch'] += max_epoch + 1
            self.stage_boundaries.append(new_boundary)


class HistoryCallback:
    """
    Callback for integration with training loops.
    Compatible with the existing callback system.
    """
    
    def __init__(self, history: Optional[TrainingHistory] = None, save_dir: str = 'logs'):
        """
        Initialize history callback.
        
        Args:
            history: Existing TrainingHistory instance or None to create new
            save_dir: Directory to auto-save history
        """
        self.history = history or TrainingHistory()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = 10  # Save every N epochs
    
    def on_epoch_end(self, epoch: int, logs: Dict = None, stage: Optional[str] = None):
        """Called at end of each epoch."""
        if logs:
            self.history.log_epoch(epoch, logs, stage)
        
        # Auto-save periodically
        if epoch % self.save_interval == 0:
            self._auto_save()
    
    def on_train_end(self, logs: Dict = None):
        """Called at end of training."""
        self._auto_save(final=True)
    
    def _auto_save(self, final: bool = False):
        """Auto-save history to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        suffix = '_final' if final else '_checkpoint'
        filename = f'training_history{suffix}_{timestamp}.json'
        
        save_path = self.save_dir / filename
        self.history.save(str(save_path))
    
    def get_history(self) -> TrainingHistory:
        """Get the training history instance."""
        return self.history


# Convenience functions
def create_history_tracker(config: Optional[Dict] = None, save_dir: str = 'logs') -> TrainingHistory:
    """
    Factory function to create a training history tracker.
    
    Args:
        config: Training configuration
        save_dir: Directory for auto-saving
        
    Returns:
        TrainingHistory instance
    """
    return TrainingHistory(config)


def load_history(history_path: str) -> Optional[TrainingHistory]:
    """
    Load training history from file.
    
    Args:
        history_path: Path to history JSON file
        
    Returns:
        TrainingHistory instance or None if file not found
    """
    path = Path(history_path)
    if not path.exists():
        return None
    
    try:
        return TrainingHistory.load(str(path))
    except FileNotFoundError:
        logger.warning(f"Training history file not found: {path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in training history file: {e}")
        return None
    except PermissionError as e:
        logger.error(f"Permission denied reading training history: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading training history: {e}")
        return None


# Example/demo
if __name__ == '__main__':
    print("TrainingHistory Demo\n")
    
    # Create history
    config = {'model': {'type': 'span', 'scale': 4}, 'training': {'epochs': 100}}
    history = TrainingHistory(config)
    
    # Simulate training
    import numpy as np
    
    for epoch in range(50):
        # Simulate metrics
        train_loss = 0.1 * np.exp(-epoch / 20) + np.random.normal(0, 0.005)
        val_loss = train_loss + 0.02
        val_psnr = 20 + 10 * (1 - np.exp(-epoch / 15)) + np.random.normal(0, 0.2)
        
        stage = 'stage1' if epoch < 10 else 'stage2'
        
        history.log_epoch(epoch, {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_psnr': val_psnr,
            'lr': 1e-4 * (0.9 ** (epoch // 10)),
        }, stage=stage)
    
    # Print summary
    summary = history.get_summary()
    print(f"Total epochs: {summary['total_epochs']}")
    print(f"Duration: {summary['duration_seconds']:.2f}s")
    print(f"\nBest validation PSNR:")
    if 'val_psnr' in summary['best_metrics']:
        best = summary['best_metrics']['val_psnr']
        print(f"  Epoch {best['epoch']}: {best['value']:.2f} dB")
    
    print(f"\nBest validation loss:")
    if 'val_loss' in summary['best_metrics']:
        best = summary['best_metrics']['val_loss']
        print(f"  Epoch {best['epoch']}: {best['value']:.6f}")
    
    # Save example
    history.save('demo_history.json')
    print("\nSaved to: demo_history.json")
    
    # Load example
    loaded = TrainingHistory.load('demo_history.json')
    print(f"Loaded: {len(loaded.epochs)} epochs")
