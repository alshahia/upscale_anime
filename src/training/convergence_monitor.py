"""
Convergence monitoring utilities for training control.
Tracks loss trends and detects convergence, plateau, and divergence patterns.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import warnings


class ConvergenceMonitor:
    """
    Monitor training convergence with multiple detection strategies.
    
    Features:
    - Simple patience-based plateau detection
    - Linear regression trend analysis
    - Variance-based stability detection
    - Divergence detection
    
    Args:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to count as improvement
        window_size: Size of moving average window
        divergence_patience: Epochs of increasing loss before divergence alert
    """
    
    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 0.0001,
        window_size: int = 10,
        divergence_patience: int = 5,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.window_size = window_size
        self.divergence_patience = divergence_patience
        
        # History tracking
        self.losses: List[float] = []
        self.metrics: Dict[str, List[float]] = {}
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.counter = 0  # Epochs without improvement
        self.divergence_counter = 0  # Epochs with increasing loss
        
        # Trend analysis
        self.slopes: deque = deque(maxlen=window_size)
        self.variances: deque = deque(maxlen=window_size)
    
    def update(self, loss: float, epoch: int, **metrics) -> Dict[str, any]:
        """
        Update monitor with new epoch data.
        
        Args:
            loss: Current epoch loss value
            epoch: Current epoch number
            **metrics: Additional metrics to track
            
        Returns:
            Dictionary with convergence status and metrics
        """
        self.losses.append(loss)
        
        # Track additional metrics
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
        
        # Check for improvement
        if self._is_better(loss, self.best_loss):
            improvement = abs(self.best_loss - loss)
            if improvement > self.min_delta:
                self.best_loss = loss
                self.best_epoch = epoch
                self.counter = 0
                self.divergence_counter = 0
            else:
                self.counter += 1
        else:
            self.counter += 1
            # Check for divergence (loss increasing)
            if len(self.losses) >= 2 and loss > self.losses[-2]:
                self.divergence_counter += 1
            else:
                self.divergence_counter = 0
        
        # Update trend analysis
        self._update_trends()
        
        return self.get_status()
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current value is better than best (lower is better)."""
        return current < best
    
    def _update_trends(self):
        """Update linear regression and variance trends."""
        if len(self.losses) < 2:
            return
        
        # Moving window for trend analysis
        window = min(self.window_size, len(self.losses))
        recent_losses = self.losses[-window:]
        
        # Linear regression slope
        if len(recent_losses) >= 3:
            x = np.arange(len(recent_losses))
            slope, intercept, r_value, p_value, std_err = self._linear_regression(x, recent_losses)
            self.slopes.append(slope)
        
        # Variance in window
        if len(recent_losses) >= 3:
            variance = np.var(recent_losses)
            self.variances.append(variance)
    
    def _linear_regression(self, x: np.ndarray, y: List[float]) -> Tuple[float, ...]:
        """Simple linear regression."""
        try:
            from scipy import stats
            return stats.linregress(x, y)
        except ImportError:
            # Fallback to numpy polyfit
            coeffs = np.polyfit(x, y, 1)
            slope = coeffs[0]
            intercept = coeffs[1]
            # Approximate r_value
            p = np.poly1d(coeffs)
            y_fit = p(x)
            ss_res = np.sum((np.array(y) - y_fit) ** 2)
            ss_tot = np.sum((np.array(y) - np.mean(y)) ** 2)
            r_value = np.sqrt(1 - ss_res / (ss_tot + 1e-10)) if ss_tot > 0 else 0
            # Return tuple matching scipy.stats.linregress signature:
            # (slope, intercept, rvalue, pvalue, stderr)
            return (slope, intercept, r_value, 0.0, 0.0)
    
    def is_converged(self) -> bool:
        """
        Check if training has converged (no improvement for patience epochs).
        
        Returns:
            True if converged, False otherwise
        """
        return self.counter >= self.patience
    
    def is_plateaued(self, slope_threshold: float = 1e-6) -> bool:
        """
        Check if loss curve has plateaued (slope near zero).
        
        Args:
            slope_threshold: Maximum absolute slope to consider as plateau
            
        Returns:
            True if plateaued, False otherwise
        """
        if len(self.slopes) < 3:
            return False
        
        recent_slopes = list(self.slopes)[-3:]
        avg_slope = np.mean(recent_slopes)
        return abs(avg_slope) < slope_threshold
    
    def is_diverging(self) -> bool:
        """
        Check if loss is consistently increasing (diverging).
        
        Returns:
            True if diverging, False otherwise
        """
        return self.divergence_counter >= self.divergence_patience
    
    def is_stable(self, variance_threshold: float = 0.001) -> bool:
        """
        Check if loss is stable (low variance in recent epochs).
        
        Args:
            variance_threshold: Maximum variance to consider stable
            
        Returns:
            True if stable, False otherwise
        """
        if len(self.variances) < 3:
            return False
        
        recent_variance = np.mean(list(self.variances)[-3:])
        return recent_variance < variance_threshold
    
    def get_status(self) -> Dict[str, any]:
        """
        Get comprehensive convergence status.
        
        Returns:
            Dictionary with all convergence metrics
        """
        status = {
            'converged': self.is_converged(),
            'plateaued': self.is_plateaued(),
            'diverging': self.is_diverging(),
            'stable': self.is_stable(),
            'counter': self.counter,
            'divergence_counter': self.divergence_counter,
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'current_loss': self.losses[-1] if self.losses else None,
            'epochs_without_improvement': self.counter,
        }
        
        # Add trend metrics if available
        if len(self.slopes) > 0:
            status['slope'] = float(self.slopes[-1])
        if len(self.variances) > 0:
            status['variance'] = float(self.variances[-1])
        
        return status
    
    def get_trend_metrics(self) -> Dict[str, Optional[float]]:
        """
        Get detailed trend metrics for analysis.
        
        Returns:
            Dictionary with slope, variance, and R² values
        """
        metrics = {
            'slope': None,
            'avg_slope': None,
            'variance': None,
            'avg_variance': None,
            'r_squared': None,
        }
        
        if len(self.slopes) > 0:
            metrics['slope'] = float(self.slopes[-1])
            metrics['avg_slope'] = float(np.mean(self.slopes))
        
        if len(self.variances) > 0:
            metrics['variance'] = float(self.variances[-1])
            metrics['avg_variance'] = float(np.mean(self.variances))
        
        return metrics
    
    def reset(self):
        """Reset all tracking state."""
        self.losses.clear()
        self.metrics.clear()
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.counter = 0
        self.divergence_counter = 0
        self.slopes.clear()
        self.variances.clear()
    
    def state_dict(self) -> Dict:
        """Get state dictionary for checkpointing."""
        return {
            'losses': self.losses.copy(),
            'metrics': {k: v.copy() for k, v in self.metrics.items()},
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'counter': self.counter,
            'divergence_counter': self.divergence_counter,
            'slopes': list(self.slopes),
            'variances': list(self.variances),
        }
    
    def load_state_dict(self, state: Dict):
        """Load state from checkpoint."""
        self.losses = state.get('losses', [])
        self.metrics = state.get('metrics', {})
        self.best_loss = state.get('best_loss', float('inf'))
        self.best_epoch = state.get('best_epoch', 0)
        self.counter = state.get('counter', 0)
        self.divergence_counter = state.get('divergence_counter', 0)
        self.slopes = deque(state.get('slopes', []), maxlen=self.window_size)
        self.variances = deque(state.get('variances', []), maxlen=self.window_size)


class MultiMetricConvergenceMonitor:
    """
    Monitor convergence across multiple metrics simultaneously.
    Useful for early stopping based on both loss and quality metrics (PSNR).
    """
    
    def __init__(
        self,
        metrics_config: Dict[str, Dict],
        mode: str = "any",  # "any" or "all"
    ):
        """
        Args:
            metrics_config: Dict mapping metric name to config dict
                Each config should have: patience, min_delta, mode ("min" or "max")
            mode: "any" to stop if any metric triggers, "all" to require all
        """
        self.monitors = {}
        self.mode = mode
        
        for metric_name, config in metrics_config.items():
            self.monitors[metric_name] = ConvergenceMonitor(
                patience=config.get('patience', 15),
                min_delta=config.get('min_delta', 0.0001),
                window_size=config.get('window_size', 10),
                divergence_patience=config.get('divergence_patience', 5),
            )
            # Store whether higher is better for this metric
            self.monitors[metric_name].higher_is_better = config.get('mode') == 'max'
    
    def update(self, epoch: int, **metrics) -> Dict[str, Dict]:
        """Update all monitors with epoch data."""
        statuses = {}
        
        for metric_name, monitor in self.monitors.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                # Adjust for higher_is_better metrics
                if monitor.higher_is_better:
                    # Invert for monitoring (monitor expects lower=better)
                    value = -value
                
                status = monitor.update(value, epoch)
                statuses[metric_name] = status
        
        return statuses
    
    def should_stop(self) -> bool:
        """Check if any/all monitors indicate stopping."""
        converged = [m.is_converged() for m in self.monitors.values()]
        
        if self.mode == "any":
            return any(converged)
        else:  # "all"
            return len(converged) > 0 and all(converged)
    
    def get_status(self) -> Dict[str, any]:
        """Get combined status from all monitors."""
        return {
            metric: monitor.get_status()
            for metric, monitor in self.monitors.items()
        }
