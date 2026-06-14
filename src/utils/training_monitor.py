"""
Training monitoring dashboard for comprehensive training stability tracking.

Provides real-time monitoring of training metrics including:
- Gradient norm tracking
- Loss stability indicators
- Teacher model health
- Parameter update statistics
- Training progress visualization
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from collections import defaultdict, deque
import warnings


class TrainingMonitor:
    """
    Comprehensive training monitoring system.
    
    Tracks training stability metrics and provides real-time analysis
    of training health and progress.
    """
    
    def __init__(
        self,
        max_history: int = 1000,
        save_interval: int = 100,
        log_dir: Optional[str] = None,
        verbose: bool = True
    ):
        self.max_history = max_history
        self.save_interval = save_interval
        self.log_dir = Path(log_dir) if log_dir else Path("logs/training_monitor")
        self.verbose = verbose
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # History tracking
        self.step_history = deque(maxlen=max_history)
        self.loss_history = deque(maxlen=max_history)
        self.gradient_norm_history = deque(maxlen=max_history)
        self.learning_rate_history = deque(maxlen=max_history)
        self.parameter_update_history = deque(maxlen=max_history)
        
        # Teacher monitoring
        self.teacher_health = defaultdict(list)
        self.teacher_outputs = defaultdict(list)
        
        # Stability metrics
        self.stability_metrics = {
            'loss_variance': deque(maxlen=100),
            'gradient_variance': deque(maxlen=100),
            'nan_events': deque(maxlen=100),
            'convergence_score': deque(maxlen=100)
        }
        
        # Statistics
        self.total_steps = 0
        self.start_time = time.time()
        self.last_save_time = 0
        
    def log_step(
        self,
        step: int,
        loss: float,
        gradient_norm: float,
        learning_rate: float,
        parameter_updates: Optional[Dict[str, float]] = None,
        teacher_outputs: Optional[Dict[str, float]] = None,
        extra_metrics: Optional[Dict[str, Any]] = None
    ):
        """
        Log training step metrics.
        
        Args:
            step: Training step number
            loss: Current loss value
            gradient_norm: Current gradient norm
            learning_rate: Current learning rate
            parameter_updates: Parameter update statistics
            teacher_outputs: Teacher model outputs
            extra_metrics: Additional metrics
        """
        self.total_steps = max(self.total_steps, step)
        
        # Update histories
        self.step_history.append(step)
        self.loss_history.append(loss)
        self.gradient_norm_history.append(gradient_norm)
        self.learning_rate_history.append(learning_rate)
        
        # Parameter updates
        if parameter_updates:
            param_norm = sum(v**2 for v in parameter_updates.values())**0.5
            self.parameter_update_history.append(param_norm)
        
        # Teacher outputs
        if teacher_outputs:
            for teacher_name, output in teacher_outputs.items():
                self.teacher_outputs[teacher_name].append(output)
                if len(self.teacher_outputs[teacher_name]) > self.max_history:
                    self.teacher_outputs[teacher_name].pop(0)
        
        # Calculate stability metrics
        self._calculate_stability_metrics()
        
        # Save periodically
        if step % self.save_interval == 0:
            self._save_metrics()
        
        # Print summary if verbose
        if self.verbose and step % 50 == 0:
            self._print_summary(step)
    
    def log_teacher_health(self, teacher_name: str, health_metrics: Dict[str, Any]):
        """
        Log teacher model health metrics.
        
        Args:
            teacher_name: Name of the teacher model
            health_metrics: Health metrics dictionary
        """
        self.teacher_health[teacher_name].append({
            'step': self.total_steps,
            'timestamp': time.time(),
            **health_metrics
        })
        
        # Keep only recent history
        if len(self.teacher_health[teacher_name]) > self.max_history:
            self.teacher_health[teacher_name].pop(0)
    
    def _calculate_stability_metrics(self):
        """Calculate stability metrics from recent history."""
        if len(self.loss_history) < 10:
            return
        
        recent_losses = list(self.loss_history)[-100:]
        recent_grads = list(self.gradient_norm_history)[-100:]
        
        # Loss variance (lower is better)
        if len(recent_losses) > 1:
            loss_mean = sum(recent_losses) / len(recent_losses)
            loss_variance = sum((x - loss_mean)**2 for x in recent_losses) / len(recent_losses)
            self.stability_metrics['loss_variance'].append(loss_variance)
        
        # Gradient variance (lower is better)
        if len(recent_grads) > 1:
            grad_mean = sum(recent_grads) / len(recent_grads)
            grad_variance = sum((x - grad_mean)**2 for x in recent_grads) / len(recent_grads)
            self.stability_metrics['gradient_variance'].append(grad_variance)
        
        # Convergence score (higher is better)
        if len(recent_losses) >= 20:
            recent_20 = recent_losses[-20:]
            older_20 = recent_losses[-40:-20] if len(recent_losses) >= 40 else recent_losses[:20]
            
            recent_mean = sum(recent_20) / len(recent_20)
            older_mean = sum(older_20) / len(older_20)
            
            # Positive improvement = higher score
            improvement = (older_mean - recent_mean) / older_mean if older_mean > 0 else 0
            convergence_score = max(0, min(1, improvement * 10))  # Scale to 0-1
            self.stability_metrics['convergence_score'].append(convergence_score)
    
    def get_stability_report(self) -> Dict[str, Any]:
        """Get comprehensive stability report."""
        if not self.loss_history:
            return {'message': 'No training history available'}
        
        recent_losses = list(self.loss_history)[-100:] if len(self.loss_history) > 100 else list(self.loss_history)
        recent_grads = list(self.gradient_norm_history)[-100:] if len(self.gradient_norm_history) > 100 else list(self.gradient_norm_history)
        
        # Basic statistics
        loss_stats = {
            'mean': sum(recent_losses) / len(recent_losses),
            'min': min(recent_losses),
            'max': max(recent_losses),
            'std': (sum((x - sum(recent_losses)/len(recent_losses))**2 for x in recent_losses) / len(recent_losses))**0.5
        }
        
        grad_stats = {
            'mean': sum(recent_grads) / len(recent_grads),
            'min': min(recent_grads),
            'max': max(recent_grads),
            'std': (sum((x - sum(recent_grads)/len(recent_grads))**2 for x in recent_grads) / len(recent_grads))**0.5
        }
        
        # Stability indicators
        stability_indicators = {
            'loss_stability': 'stable' if loss_stats['std'] < loss_stats['mean'] * 0.1 else 'unstable',
            'gradient_stability': 'stable' if grad_stats['std'] < grad_stats['mean'] * 0.5 else 'unstable',
            'convergence_trend': 'improving' if len(self.stability_metrics['convergence_score']) > 0 and self.stability_metrics['convergence_score'][-1] > 0.5 else 'stable',
            'overall_health': self._calculate_overall_health()
        }
        
        # Teacher health
        teacher_health_summary = {}
        for teacher_name, health_list in self.teacher_health.items():
            if health_list:
                recent_health = health_list[-10:] if len(health_list) > 10 else health_list
                teacher_health_summary[teacher_name] = {
                    'recent_health': 'healthy' if all(h.get('healthy', True) for h in recent_health) else 'unhealthy',
                    'total_checks': len(health_list),
                    'last_check': health_list[-1] if health_list else None
                }
        
        return {
            'training_stats': {
                'total_steps': self.total_steps,
                'elapsed_time': time.time() - self.start_time,
                'steps_per_second': self.total_steps / (time.time() - self.start_time) if time.time() - self.start_time > 0 else 0
            },
            'loss_statistics': loss_stats,
            'gradient_statistics': grad_stats,
            'stability_indicators': stability_indicators,
            'teacher_health': teacher_health_summary,
            'recent_trends': {
                'loss_trend': self._calculate_trend(list(self.loss_history)[-20:]) if len(self.loss_history) >= 20 else 'insufficient_data',
                'gradient_trend': self._calculate_trend(list(self.gradient_norm_history)[-20:]) if len(self.gradient_norm_history) >= 20 else 'insufficient_data'
            }
        }
    
    def _calculate_overall_health(self) -> str:
        """Calculate overall training health score."""
        health_score = 0
        max_score = 4
        
        # Loss stability (25%)
        if len(self.stability_metrics['loss_variance']) > 0:
            recent_var = self.stability_metrics['loss_variance'][-1]
            if recent_var < 0.01:
                health_score += 1
            elif recent_var < 0.1:
                health_score += 0.5
        
        # Gradient stability (25%)
        if len(self.stability_metrics['gradient_variance']) > 0:
            recent_var = self.stability_metrics['gradient_variance'][-1]
            if recent_var < 0.1:
                health_score += 1
            elif recent_var < 1.0:
                health_score += 0.5
        
        # Convergence (25%)
        if len(self.stability_metrics['convergence_score']) > 0:
            recent_score = self.stability_metrics['convergence_score'][-1]
            health_score += recent_score
        
        # No NaN events (25%)
        if len(self.stability_metrics['nan_events']) == 0:
            health_score += 1
        elif len(self.stability_metrics['nan_events']) < 5:
            health_score += 0.5
        
        # Convert to health status
        health_ratio = health_score / max_score
        if health_ratio >= 0.8:
            return 'excellent'
        elif health_ratio >= 0.6:
            return 'good'
        elif health_ratio >= 0.4:
            return 'fair'
        else:
            return 'poor'
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend from a list of values."""
        if len(values) < 2:
            return 'insufficient_data'
        
        # Simple linear trend calculation
        n = len(values)
        x = list(range(n))
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum(x[i]**2 for i in range(n)) - sum_x**2)
        
        if abs(slope) < 1e-6:
            return 'stable'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    def _print_summary(self, step: int):
        """Print training summary."""
        if not self.loss_history:
            return
        
        recent_loss = self.loss_history[-1]
        recent_grad = self.gradient_norm_history[-1]
        recent_lr = self.learning_rate_history[-1]
        
        print(f"\nStep {step}: Loss={recent_loss:.6f}, GradNorm={recent_grad:.4f}, LR={recent_lr:.2e}")
        
        # Print stability status
        report = self.get_stability_report()
        stability = report['stability_indicators']
        print(f"  Stability: Loss={stability['loss_stability']}, Grad={stability['gradient_stability']}, Overall={stability['overall_health']}")
    
    def _save_metrics(self):
        """Save metrics to file."""
        try:
            metrics = {
                'timestamp': time.time(),
                'total_steps': self.total_steps,
                'loss_history': list(self.loss_history),
                'gradient_norm_history': list(self.gradient_norm_history),
                'learning_rate_history': list(self.learning_rate_history),
                'stability_metrics': {k: list(v) for k, v in self.stability_metrics.items()},
                'teacher_health': {k: v for k, v in self.teacher_health.items()},
                'report': self.get_stability_report()
            }
            
            # Save to JSON file
            metrics_file = self.log_dir / f"training_metrics_{self.total_steps}.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            
            # Keep only recent files
            self._cleanup_old_files()
            
        except OSError as e:
            if self.verbose:
                logger.warning(f"Failed to save metrics: {e}")
            # Handle specific file system errors
            if e.errno == 28:  # No space left on device
                logger.error("Disk full. Cannot save metrics.")
                raise
        except PermissionError as e:
            if self.verbose:
                logger.error(f"Permission denied saving metrics: {e}")
                raise
        except json.JSONEncodeError as e:
            if self.verbose:
                logger.error(f"Failed to encode metrics to JSON: {e}")
                # Try to save raw metrics
                try:
                    raw_file = self.log_dir / f"training_metrics_raw_{int(time.time())}.txt"
                    with open(raw_file, 'w') as f:
                        f.write(str(self.metrics))
                    logger.info("Saved raw metrics as fallback")
                except Exception as fallback_e:
                    logger.error(f"Failed to save raw metrics: {fallback_e}")
    
    def _cleanup_old_files(self):
        """Clean up old metric files."""
        try:
            metric_files = list(self.log_dir.glob("training_metrics_*.json"))
            metric_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Keep only 10 most recent files
            for old_file in metric_files[10:]:
                old_file.unlink()
                
        except OSError as e:
            if self.verbose:
                logger.warning(f"Failed to cleanup old files: {e}")
            # Handle specific file system errors
            if e.errno == 2:  # No such file or directory
                logger.debug("No old files to cleanup")
            elif e.errno == 13:  # Permission denied
                logger.error("Permission denied during cleanup")
                raise
        except PermissionError as e:
            if self.verbose:
                logger.error(f"Permission denied during cleanup: {e}")
                raise
    
    def export_summary(self, output_path: Optional[str] = None) -> str:
        """Export training summary to file."""
        if output_path is None:
            output_path = self.log_dir / "training_summary.json"
        
        summary = self.get_stability_report()
        summary['export_timestamp'] = time.time()
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return str(output_path)


class TrainingDashboard:
    """
    Training dashboard with visualization capabilities.
    
    Provides a high-level interface for monitoring training progress
    and generating reports.
    """
    
    def __init__(self, monitor: Optional[TrainingMonitor] = None):
        self.monitor = monitor or TrainingMonitor()
        
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data for visualization."""
        report = self.monitor.get_stability_report()
        
        # Prepare data for visualization
        dashboard_data = {
            'current_metrics': {
                'loss': self.monitor.loss_history[-1] if self.monitor.loss_history else 0,
                'gradient_norm': self.monitor.gradient_norm_history[-1] if self.monitor.gradient_norm_history else 0,
                'learning_rate': self.monitor.learning_rate_history[-1] if self.monitor.learning_rate_history else 0,
                'step': self.monitor.total_steps
            },
            'recent_history': {
                'losses': list(self.monitor.loss_history)[-50:],
                'gradient_norms': list(self.monitor.gradient_norm_history)[-50:],
                'learning_rates': list(self.monitor.learning_rate_history)[-50:],
                'steps': list(self.monitor.step_history)[-50:]
            },
            'stability_report': report,
            'alerts': self._generate_alerts(report)
        }
        
        return dashboard_data
    
    def _generate_alerts(self, report: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate alerts based on monitoring data."""
        alerts = []
        
        # Loss alerts
        if report['loss_statistics']['std'] > report['loss_statistics']['mean'] * 0.2:
            alerts.append({
                'type': 'warning',
                'message': 'High loss variance detected',
                'severity': 'medium'
            })
        
        # Gradient alerts
        if report['gradient_statistics']['max'] > 10.0:
            alerts.append({
                'type': 'warning',
                'message': 'Very high gradient norm detected',
                'severity': 'high'
            })
        
        # Stability alerts
        if report['stability_indicators']['overall_health'] == 'poor':
            alerts.append({
                'type': 'error',
                'message': 'Training stability is poor',
                'severity': 'high'
            })
        
        # Teacher alerts
        for teacher_name, health in report['teacher_health'].items():
            if health['recent_health'] == 'unhealthy':
                alerts.append({
                    'type': 'warning',
                    'message': f'Teacher {teacher_name} is unhealthy',
                    'severity': 'medium'
                })
        
        return alerts


if __name__ == "__main__":
    # Test training monitor
    print("TESTING TRAINING MONITOR")
    print("=" * 50)
    
    monitor = TrainingMonitor(verbose=False)
    
    # Simulate some training steps
    for i in range(10):
        loss = 1.0 - i * 0.05 + torch.randn(1).item() * 0.01
        grad_norm = 0.5 + torch.randn(1).item() * 0.1
        lr = 1e-4 * (0.95 ** (i // 10))
        
        monitor.log_step(i, loss, grad_norm, lr)
    
    # Get report
    report = monitor.get_stability_report()
    print("Training report created:", len(report))
    print("Overall health:", report['stability_indicators']['overall_health'])
    
    # Test dashboard
    dashboard = TrainingDashboard(monitor)
    dashboard_data = dashboard.get_dashboard_data()
    print("Dashboard data created:", len(dashboard_data))
    print("Current loss:", dashboard_data['current_metrics']['loss'])
    
    print("\n[OK] Training monitor test completed")
