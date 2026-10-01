"""
Training Visualizer & Documentation Generator
=============================================

Comprehensive visualization and documentation module for the Anime Super-Resolution
training pipeline. Provides:

- Rich console output with formatted tables and progress displays
- Training report generation with visualizations
- Model comparison and analysis
- Architecture documentation
- Real-time training metrics display
- Checkpoint analysis and comparison

Usage:
    from anime_sr.utils.training_visualizer import TrainingVisualizer
    
    # Initialize visualizer
    visualizer = TrainingVisualizer(config, output_dir='training_reports')
    
    # Generate comprehensive training report
    visualizer.generate_training_report(
        model_name='Model A (SPAN)',
        training_history=history,
        checkpoints_dir='checkpoints/model_a'
    )
    
    # Compare multiple models
    visualizer.compare_models([
        {'name': 'Model A', 'history': history_a, 'checkpoints': 'checkpoints/a'},
        {'name': 'Model B', 'history': history_b, 'checkpoints': 'checkpoints/b'},
    ])

Author: Anime SR Training System
Version: 1.0.0
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import json
import time
from collections import defaultdict
import warnings

# Rich library for beautiful console output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.tree import Tree
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.syntax import Syntax
    from rich.markdown import Markdown
    from rich.layout import Layout
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    warnings.warn("Rich library not available. Install with: pip install rich")

# Try to import PIL for image handling
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class TrainingVisualizer:
    """
    Comprehensive training visualization and documentation generator.
    
    Handles all visualization needs for the Anime SR training pipeline including:
    - Console-based progress displays and tables
    - Training curve plots and metric visualizations
    - Model architecture documentation
    - Training pipeline flow diagrams
    - Checkpoint analysis and comparison
    
    Attributes:
        config: Training configuration dictionary
        output_dir: Directory for saving reports and visualizations
        console: Rich Console instance for formatted output
        metrics_history: Accumulated training metrics
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        output_dir: str = "training_reports",
        use_rich: bool = True,
    ):
        """
        Initialize the Training Visualizer.
        
        Args:
            config: Training configuration dictionary
            output_dir: Directory for saving reports and visualizations
            use_rich: Whether to use Rich library for console output
        """
        self.config = config or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Rich console if available
        self.console = Console() if (use_rich and RICH_AVAILABLE) else None
        
        # Metrics storage
        self.metrics_history = defaultdict(list)
        self.training_start_time = None
        self.current_stage = None
        
        # Color scheme for consistent visualization
        self.colors = {
            'model_a': '#3498db',      # Blue
            'model_b': '#e74c3c',      # Red
            'ensemble': '#2ecc71',     # Green
            'teacher': '#9b59b6',      # Purple
            'val': '#f39c12',          # Orange
            'train': '#1abc9c',        # Teal
            'loss': '#e74c3c',         # Red
            'psnr': '#3498db',         # Blue
            'lr': '#9b59b6',           # Purple
        }
        
    def print_header(self, title: str, subtitle: Optional[str] = None):
        """Print a formatted header to console."""
        if self.console:
            self.console.print()
            panel_content = f"[bold cyan]{title}[/bold cyan]"
            if subtitle:
                panel_content += f"\n[dim]{subtitle}[/dim]"
            self.console.print(Panel(
                panel_content,
                box=box.DOUBLE,
                border_style="cyan"
            ))
        else:
            print(f"\n{'='*60}")
            print(f"  {title}")
            if subtitle:
                print(f"  {subtitle}")
            print(f"{'='*60}\n")
    
    def print_config_summary(self, config: Optional[Dict] = None):
        """
        Print a formatted summary of the training configuration.
        
        Displays model settings, training parameters, data configuration,
        and optimization settings in organized tables.
        """
        cfg = config or self.config
        if not cfg:
            return
        
        if self.console:
            # Model Configuration Table
            model_table = Table(title="Model Configuration", box=box.ROUNDED)
            model_table.add_column("Parameter", style="cyan")
            model_table.add_column("Value", style="green")
            
            model_cfg = cfg.get('model', {})
            model_table.add_row("Name", str(model_cfg.get('name', 'N/A')))
            model_table.add_row("Type", str(model_cfg.get('type', 'N/A')))
            model_table.add_row("Scale", str(model_cfg.get('scale', 4)))
            model_table.add_row("Channels", str(model_cfg.get('channels', 'N/A')))
            
            self.console.print(model_table)
            
            # Training Configuration Table
            train_table = Table(title="Training Configuration", box=box.ROUNDED)
            train_table.add_column("Parameter", style="cyan")
            train_table.add_column("Value", style="green")
            
            train_cfg = cfg.get('training', {})
            train_table.add_row("Mode", str(train_cfg.get('mode', 'N/A')))
            train_table.add_row("Batch Size", str(train_cfg.get('batch_size', 8)))
            train_table.add_row("Learning Rate", str(train_cfg.get('lr', 1e-4)))
            train_table.add_row("Optimizer", str(train_cfg.get('optimizer', 'adam')))
            train_table.add_row("Mixed Precision", str(train_cfg.get('mixed_precision', True)))
            train_table.add_row("Device", str(train_cfg.get('device', 'cuda')))
            
            self.console.print(train_table)
            
            # Stage Configuration
            stage1_cfg = train_cfg.get('stage1', {})
            stage2_cfg = train_cfg.get('stage2', {})
            
            if stage1_cfg.get('enabled', False) or stage2_cfg.get('enabled', False):
                stage_table = Table(title="Training Stages", box=box.ROUNDED)
                stage_table.add_column("Stage", style="cyan")
                stage_table.add_column("Epochs", style="green")
                stage_table.add_column("Enabled", style="yellow")
                
                if stage1_cfg:
                    stage_table.add_row(
                        "Stage 1 (Knowledge Aggregation)",
                        str(stage1_cfg.get('epochs', 0)),
                        "✓" if stage1_cfg.get('enabled', False) else "✗"
                    )
                if stage2_cfg:
                    stage_table.add_row(
                        "Stage 2 (Student Distillation)",
                        str(stage2_cfg.get('epochs', 0)),
                        "✓" if stage2_cfg.get('enabled', False) else "✗"
                    )
                
                self.console.print(stage_table)
            
            # Data Configuration
            data_table = Table(title="Data Configuration", box=box.ROUNDED)
            data_table.add_column("Parameter", style="cyan")
            data_table.add_column("Value", style="green")
            
            data_cfg = cfg.get('data', {})
            datasets = data_cfg.get('datasets', [])
            if datasets:
                dataset_names = [d.get('name', 'unknown') for d in datasets if d.get('enabled', True)]
                data_table.add_row("Datasets", ', '.join(dataset_names))
            
            data_table.add_row("Crop Size", str(data_cfg.get('crop_size', 128)))
            data_table.add_row("Augmentation", str(data_cfg.get('augment', True)))
            data_table.add_row("Degradation", str(data_cfg.get('degradation', {}).get('enabled', False)))
            
            video_cfg = data_cfg.get('video', {})
            if video_cfg.get('enabled', False):
                data_table.add_row("Video Processing", "Enabled")
                data_table.add_row("Video Extract Mode", str(video_cfg.get('extract_mode', 'pre')))
            
            self.console.print(data_table)
        else:
            # Fallback to simple print
            print("\n--- Configuration Summary ---")
            print(f"Model: {cfg.get('model', {}).get('name', 'N/A')}")
            print(f"Training Mode: {cfg.get('training', {}).get('mode', 'N/A')}")
            print(f"Batch Size: {cfg.get('training', {}).get('batch_size', 8)}")
            print(f"Learning Rate: {cfg.get('training', {}).get('lr', 1e-4)}")
    
    def create_training_pipeline_diagram(self, output_path: Optional[str] = None) -> str:
        """
        Create a visual diagram of the training pipeline.
        
        Generates a matplotlib figure showing the complete training pipeline
        flow from data input through model stages to output.
        
        Returns:
            Path to saved diagram image
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
        
        # Title
        fig.suptitle('Anime Super-Resolution Training Pipeline', fontsize=16, fontweight='bold')
        
        # Stage 1: Knowledge Aggregation
        ax1 = fig.add_subplot(gs[0, :])
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 3)
        ax1.axis('off')
        ax1.set_title('Stage 1: Knowledge Aggregation', fontsize=12, fontweight='bold', pad=10)
        
        # Draw teacher models
        teacher_models = ['EDSR', 'RCAN', 'SwinIR']
        teacher_colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        for i, (teacher, color) in enumerate(zip(teacher_models, teacher_colors)):
            rect = mpatches.FancyBboxPatch(
                (1 + i*2.5, 1.5), 1.5, 1,
                boxstyle="round,pad=0.1",
                facecolor=color, edgecolor='black', alpha=0.7
            )
            ax1.add_patch(rect)
            ax1.text(1.75 + i*2.5, 2, teacher, ha='center', va='center', fontweight='bold', fontsize=10)
        
        # Draw aggregation network
        agg_rect = mpatches.FancyBboxPatch(
            (4, 0.2), 2, 1,
            boxstyle="round,pad=0.1",
            facecolor='#9b59b6', edgecolor='black', alpha=0.7
        )
        ax1.add_patch(agg_rect)
        ax1.text(5, 0.7, 'Aggregation\nNetwork', ha='center', va='center', fontweight='bold', fontsize=9)
        
        # Draw arrows from teachers to aggregation
        for i in range(3):
            ax1.annotate('', xy=(4.5, 1.2), xytext=(1.75 + i*2.5, 1.5),
                        arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', alpha=0.6))
        
        # Stage 2: Student Training (Model A)
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 3)
        ax2.axis('off')
        ax2.set_title('Stage 2A: SPAN-Tiny Student', fontsize=11, fontweight='bold', pad=10)
        
        # Draw components
        components = [
            ('MTKD Loss\n(Wavelet)', 1, 2, '#3498db'),
            ('FAKD Loss\n(Features)', 4, 2, '#e74c3c'),
            ('SPAN-Tiny\n(~250K params)', 2.5, 0.5, '#2ecc71'),
        ]
        
        for label, x, y, color in components:
            rect = mpatches.FancyBboxPatch(
                (x, y), 2, 1,
                boxstyle="round,pad=0.1",
                facecolor=color, edgecolor='black', alpha=0.7
            )
            ax2.add_patch(rect)
            ax2.text(x+1, y+0.5, label, ha='center', va='center', fontweight='bold', fontsize=9)
        
        # Stage 2: Student Training (Model B)
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.set_xlim(0, 10)
        ax3.set_ylim(0, 3)
        ax3.axis('off')
        ax3.set_title('Stage 2B: Mamba-PAN Student', fontsize=11, fontweight='bold', pad=10)
        
        components_b = [
            ('MTKD Loss\n(Wavelet)', 1, 2, '#3498db'),
            ('FAKD Loss\n(Directional)', 4, 2, '#e74c3c'),
            ('Mamba-PAN\n(4-Direction)', 2.5, 0.5, '#9b59b6'),
        ]
        
        for label, x, y, color in components_b:
            rect = mpatches.FancyBboxPatch(
                (x, y), 2, 1,
                boxstyle="round,pad=0.1",
                facecolor=color, edgecolor='black', alpha=0.7
            )
            ax3.add_patch(rect)
            ax3.text(x+1, y+0.5, label, ha='center', va='center', fontweight='bold', fontsize=9)
        
        # Ensemble Stage
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.set_xlim(0, 10)
        ax4.set_ylim(0, 3)
        ax4.axis('off')
        ax4.set_title('Stage 3: Ensemble', fontsize=11, fontweight='bold', pad=10)
        
        components_ens = [
            ('Model A', 1, 2, '#3498db'),
            ('Model B', 4, 2, '#e74c3c'),
            ('Tiny Student\n(~100K params)', 2.5, 0.5, '#f39c12'),
        ]
        
        for label, x, y, color in components_ens:
            rect = mpatches.FancyBboxPatch(
                (x, y), 2, 1,
                boxstyle="round,pad=0.1",
                facecolor=color, edgecolor='black', alpha=0.7
            )
            ax4.add_patch(rect)
            ax4.text(x+1, y+0.5, label, ha='center', va='center', fontweight='bold', fontsize=9)
        
        # Data Pipeline
        ax5 = fig.add_subplot(gs[2, :])
        ax5.set_xlim(0, 10)
        ax5.set_ylim(0, 3)
        ax5.axis('off')
        ax5.set_title('Data Pipeline', fontsize=12, fontweight='bold', pad=10)
        
        data_components = [
            ('HR Images\n/Video Frames', 0.5, 1.5, '#95a5a6'),
            ('Degradation\n(Blur/Noise/JPEG)', 2.5, 1.5, '#e67e22'),
            ('LR Input\n(Bicubic)', 4.5, 1.5, '#34495e'),
            ('DataLoader\n(Batch)', 6.5, 1.5, '#16a085'),
            ('Training\nLoop', 8.5, 1.5, '#8e44ad'),
        ]
        
        for label, x, y, color in data_components:
            rect = mpatches.FancyBboxPatch(
                (x, y), 1.5, 1,
                boxstyle="round,pad=0.05",
                facecolor=color, edgecolor='black', alpha=0.7
            )
            ax5.add_patch(rect)
            ax5.text(x+0.75, y+0.5, label, ha='center', va='center', fontweight='bold', fontsize=8)
        
        # Draw arrows between data components
        for i in range(len(data_components) - 1):
            ax5.annotate('', xy=(data_components[i+1][1], 2),
                        xytext=(data_components[i][1] + 1.5, 2),
                        arrowprops=dict(arrowstyle='->', lw=2, color='black', alpha=0.5))
        
        plt.tight_layout()
        
        # Save
        if output_path is None:
            output_path = self.output_dir / "training_pipeline_diagram.png"
        else:
            output_path = Path(output_path)
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        if self.console:
            self.console.print(f"[green]Pipeline diagram saved to: {output_path}[/green]")
        
        return str(output_path)
    
    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        output_path: Optional[str] = None,
        title: str = "Training History",
        model_name: str = "Model",
    ) -> str:
        """
        Plot comprehensive training curves.
        
        Creates a multi-panel figure showing loss, PSNR, learning rate,
        and other metrics over training epochs.
        
        Args:
            history: Dictionary with metric histories (e.g., {'train_loss': [...], 'val_psnr': [...]})
            output_path: Path to save the plot
            title: Plot title
            model_name: Name of the model for legend
            
        Returns:
            Path to saved plot
        """
        # Determine available metrics
        metrics = set()
        for key in history.keys():
            if '_' in key:
                metric = key.split('_', 1)[1]
                metrics.add(metric)
        
        metrics = sorted(list(metrics))
        n_metrics = len(metrics)
        
        if n_metrics == 0:
            return ""
        
        # Create subplots
        fig, axes = plt.subplots(
            nrows=(n_metrics + 1) // 2,
            ncols=2,
            figsize=(14, 4 * ((n_metrics + 1) // 2))
        )
        
        if n_metrics == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            train_key = f'train_{metric}'
            val_key = f'val_{metric}'
            
            # Determine available data for this metric
            train_data = history.get(train_key, [])
            val_data = history.get(val_key, [])
            
            # Get max length for epochs calculation
            max_len = max(len(train_data), len(val_data))
            if max_len == 0:
                continue
                
            epochs = range(1, max_len + 1)
            
            # Plot training curve
            if len(train_data) > 0:
                ax.plot(range(1, len(train_data) + 1), train_data,
                       label=f"train_{metric.capitalize()}",
                       color=self.colors.get('train', '#1abc9c'),
                       linewidth=2, alpha=0.8)
            ax.set_title(f"{metric.capitalize()} over time", fontsize=11, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            # Mark best epoch for loss metrics
            if metric in ['loss', 'l1', 'l2'] and val_key in history:
                best_epoch = np.argmin(history[val_key])
                best_value = history[val_key][best_epoch]
                ax.axvline(x=best_epoch + 1, color='red', linestyle=':', alpha=0.5)
                ax.scatter([best_epoch + 1], [best_value], color='red', s=100, zorder=5)
                ax.annotate(f"Best: {best_value:.4f}",
                           xy=(best_epoch + 1, best_value),
                           xytext=(best_epoch + 1, best_value * 1.1 if best_value > 0 else best_value * 0.9),
                           fontsize=8, color='red')
            
            # Mark best epoch for PSNR
            elif metric == 'psnr' and val_key in history:
                best_epoch = np.argmax(history[val_key])
                best_value = history[val_key][best_epoch]
                ax.axvline(x=best_epoch + 1, color='green', linestyle=':', alpha=0.5)
                ax.scatter([best_epoch + 1], [best_value], color='green', s=100, zorder=5)
                ax.annotate(f"Best: {best_value:.2f} dB",
                           xy=(best_epoch + 1, best_value),
                           xytext=(best_epoch + 1, best_value * 0.95),
                           fontsize=8, color='green')
        
        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle(f"{title} - {model_name}", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save
        if output_path is None:
            output_path = self.output_dir / f"{model_name.lower().replace(' ', '_')}_training_curves.png"
        else:
            output_path = Path(output_path)
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(output_path)
    
    def plot_model_comparison(
        self,
        models_data: List[Dict[str, Any]],
        metric: str = 'loss',
        output_path: Optional[str] = None,
    ) -> str:
        """
        Plot comparison of multiple models' training curves.
        
        Args:
            models_data: List of dicts with 'name', 'history' keys
            metric: Metric to compare ('loss', 'psnr', etc.)
            output_path: Path to save the plot
            
        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        colors_list = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
        
        for idx, split in enumerate(['train', 'val']):
            ax = axes[idx]
            
            for i, model_data in enumerate(models_data):
                history = model_data.get('history', {})
                key = f'{split}_{metric}'
                
                if key in history and len(history[key]) > 0:
                    epochs = range(1, len(history[key]) + 1)
                    ax.plot(epochs, history[key],
                           label=model_data.get('name', f'Model {i+1}'),
                           color=colors_list[i % len(colors_list)],
                           linewidth=2, alpha=0.8)
            
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel(metric.capitalize(), fontsize=11)
            ax.set_title(f'{split.capitalize()} {metric.capitalize()}', fontsize=12, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Model Comparison - {metric.capitalize()}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_path is None:
            output_path = self.output_dir / f'model_comparison_{metric}.png'
        else:
            output_path = Path(output_path)
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(output_path)
    
    def analyze_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Analyze a model checkpoint and extract key information.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            Dictionary with checkpoint analysis
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            return {'error': f'Checkpoint not found: {checkpoint_path}'}
        
        try:
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            except Exception:
                # Fallback for checkpoints with custom objects
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except Exception as e:
            return {'error': f'Failed to load checkpoint: {e}'}
        
        analysis = {
            'path': str(checkpoint_path),
            'filename': checkpoint_path.name,
            'size_mb': checkpoint_path.stat().st_size / (1024 * 1024),
        }
        
        # Extract epoch
        if 'epoch' in checkpoint:
            analysis['epoch'] = checkpoint['epoch']
        
        # Extract best loss
        if 'best_loss' in checkpoint:
            analysis['best_loss'] = checkpoint['best_loss']
        
        # Analyze model state
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            
            # Count parameters
            total_params = sum(p.numel() for p in state_dict.values())
            analysis['total_parameters'] = total_params
            analysis['trainable_parameters'] = total_params  # All saved params are trainable
            
            # Model size estimate
            analysis['model_size_mb'] = sum(p.numel() * p.element_size() for p in state_dict.values()) / (1024 * 1024)
            
            # List layer types
            layer_types = defaultdict(int)
            for name in state_dict.keys():
                # Extract layer type from parameter name
                parts = name.split('.')
                if len(parts) >= 2:
                    layer_type = parts[-2] if parts[-1] in ['weight', 'bias'] else parts[-1]
                    layer_types[layer_type] += 1
            
            analysis['layer_types'] = dict(layer_types)
        
        # Extract config if available
        if 'config' in checkpoint:
            analysis['config'] = checkpoint['config']
        
        return analysis
    
    def print_checkpoint_summary(self, checkpoint_path: str):
        """Print a formatted summary of a checkpoint."""
        analysis = self.analyze_checkpoint(checkpoint_path)
        
        if 'error' in analysis:
            if self.console:
                self.console.print(f"[red]{analysis['error']}[/red]")
            else:
                print(f"Error: {analysis['error']}")
            return
        
        if self.console:
            table = Table(title=f"Checkpoint Analysis: {analysis['filename']}", box=box.ROUNDED)
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Path", str(analysis['path']))
            table.add_row("File Size", f"{analysis['size_mb']:.2f} MB")
            
            if 'epoch' in analysis:
                table.add_row("Epoch", str(analysis['epoch']))
            
            if 'best_loss' in analysis:
                table.add_row("Best Loss", f"{analysis['best_loss']:.6f}")
            
            if 'total_parameters' in analysis:
                table.add_row("Total Parameters", f"{analysis['total_parameters']:,}")
                table.add_row("Model Size", f"{analysis['model_size_mb']:.2f} MB")
            
            self.console.print(table)
            
            # Layer types
            if 'layer_types' in analysis:
                layer_table = Table(title="Layer Types", box=box.ROUNDED)
                layer_table.add_column("Layer Type", style="cyan")
                layer_table.add_column("Count", style="yellow")
                
                for layer_type, count in sorted(analysis['layer_types'].items(), key=lambda x: -x[1])[:10]:
                    layer_table.add_row(layer_type, str(count))
                
                self.console.print(layer_table)
        else:
            print(f"\n--- Checkpoint: {analysis['filename']} ---")
            print(f"Size: {analysis['size_mb']:.2f} MB")
            if 'epoch' in analysis:
                print(f"Epoch: {analysis['epoch']}")
            if 'total_parameters' in analysis:
                print(f"Parameters: {analysis['total_parameters']:,}")
    
    def generate_training_report(
        self,
        model_name: str,
        training_history: Dict[str, List[float]],
        checkpoints_dir: str,
        config: Optional[Dict] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate a comprehensive training report with visualizations.
        
        Creates an HTML report with training curves, checkpoint analysis,
        configuration summary, and key metrics.
        
        Args:
            model_name: Name of the model
            training_history: Dictionary with training metrics history
            checkpoints_dir: Directory containing checkpoints
            config: Training configuration
            output_path: Path to save the report
            
        Returns:
            Path to saved HTML report
        """
        cfg = config or self.config
        
        # Create report directory
        report_dir = self.output_dir / model_name.lower().replace(' ', '_')
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate training curves
        curves_path = self.plot_training_curves(
            training_history,
            output_path=report_dir / 'training_curves.png',
            title='Training History',
            model_name=model_name,
        )
        
        # Analyze checkpoints
        checkpoints_path = Path(checkpoints_dir)
        checkpoint_analyses = []
        
        if checkpoints_path.exists():
            for ckpt_file in sorted(checkpoints_path.glob('*.pth'))[:5]:  # Top 5 recent
                analysis = self.analyze_checkpoint(str(ckpt_file))
                if 'error' not in analysis:
                    checkpoint_analyses.append(analysis)
        
        # Generate HTML report
        html_content = self._generate_html_report(
            model_name=model_name,
            config=cfg,
            training_history=training_history,
            checkpoint_analyses=checkpoint_analyses,
            curves_path=curves_path,
        )
        
        # Save report
        if output_path is None:
            output_path = report_dir / 'training_report.html'
        else:
            output_path = Path(output_path)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        if self.console:
            self.console.print(f"[green]Training report saved to: {output_path}[/green]")
        
        return str(output_path)
    
    def _generate_html_report(
        self,
        model_name: str,
        config: Dict,
        training_history: Dict[str, List[float]],
        checkpoint_analyses: List[Dict],
        curves_path: str,
    ) -> str:
        """Generate HTML content for training report."""
        
        # Calculate summary statistics
        final_metrics = {}
        for key, values in training_history.items():
            if values:
                final_metrics[key] = {
                    'final': values[-1],
                    'best': min(values) if 'loss' in key else max(values),
                    'mean': sum(values) / len(values),
                }
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Training Report - {model_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-card h3 {{
            margin: 0 0 10px 0;
            color: #4a5568;
            font-size: 0.9em;
            text-transform: uppercase;
        }}
        .metric-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #2d3748;
        }}
        .metric-card .subvalue {{
            font-size: 0.9em;
            color: #718096;
            margin-top: 5px;
        }}
        .config-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        .config-table th {{
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        .config-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid #e2e8f0;
        }}
        .config-table tr:nth-child(even) {{
            background: #f7fafc;
        }}
        .image-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .image-container img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .checkpoint-list {{
            list-style: none;
            padding: 0;
        }}
        .checkpoint-list li {{
            background: #f7fafc;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .footer {{
            text-align: center;
            color: #718096;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e2e8f0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{model_name}</h1>
        <p>Training Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>Training Summary</h2>
        <div class="metrics-grid">
"""
        
        # Add metric cards
        for key, stats in list(final_metrics.items())[:6]:
            metric_name = key.replace('_', ' ').title()
            html += f"""
            <div class="metric-card">
                <h3>{metric_name}</h3>
                <div class="value">{stats['final']:.4f}</div>
                <div class="subvalue">Best: {stats['best']:.4f}</div>
            </div>
"""
        
        html += """
        </div>
    </div>
"""
        
        # Configuration section
        html += """
    <div class="section">
        <h2>Configuration</h2>
        <table class="config-table">
            <tr><th>Parameter</th><th>Value</th></tr>
"""
        
        train_cfg = config.get('training', {})
        model_cfg = config.get('model', {})
        
        config_items = [
            ('Model Type', model_cfg.get('type', 'N/A')),
            ('Model Name', model_cfg.get('name', 'N/A')),
            ('Scale Factor', model_cfg.get('scale', 4)),
            ('Batch Size', train_cfg.get('batch_size', 8)),
            ('Learning Rate', train_cfg.get('lr', 1e-4)),
            ('Optimizer', train_cfg.get('optimizer', 'adam')),
            ('Mixed Precision', train_cfg.get('mixed_precision', True)),
            ('Device', train_cfg.get('device', 'cuda')),
        ]
        
        for param, value in config_items:
            html += f"<tr><td>{param}</td><td>{value}</td></tr>\n"
        
        html += """
        </table>
    </div>
"""
        
        # Training curves
        html += f"""
    <div class="section">
        <h2>Training Curves</h2>
        <div class="image-container">
            <img src="training_curves.png" alt="Training Curves">
        </div>
    </div>
"""
        
        # Checkpoints section
        if checkpoint_analyses:
            html += """
    <div class="section">
        <h2>Checkpoint Analysis</h2>
        <ul class="checkpoint-list">
"""
            for analysis in checkpoint_analyses:
                html += f"""
            <li>
                <strong>{analysis['filename']}</strong><br>
                Size: {analysis['size_mb']:.2f} MB
                {f"| Epoch: {analysis['epoch']}" if 'epoch' in analysis else ""}
                {f"| Parameters: {analysis['total_parameters']:,}" if 'total_parameters' in analysis else ""}
            </li>
"""
            html += """
        </ul>
    </div>
"""
        
        html += f"""
    <div class="footer">
        <p>Anime Super-Resolution Training System</p>
        <p>Report generated by TrainingVisualizer</p>
    </div>
</body>
</html>
"""
        
        return html
    
    def create_architecture_documentation(self, output_path: Optional[str] = None) -> str:
        """
        Create comprehensive architecture documentation.
        
        Generates an HTML document explaining the complete training architecture,
        including model diagrams, data flow, and loss functions.
        
        Returns:
            Path to saved documentation
        """
        # Create pipeline diagram
        pipeline_path = self.create_training_pipeline_diagram(
            output_path=self.output_dir / 'architecture_pipeline.png'
        )
        
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Anime Super-Resolution - Architecture Documentation</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
            line-height: 1.6;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
        }
        .header p {
            margin: 15px 0 0 0;
            font-size: 1.2em;
            opacity: 0.9;
        }
        .section {
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 25px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .section h2 {
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 15px;
            margin-top: 0;
            font-size: 1.8em;
        }
        .section h3 {
            color: #764ba2;
            margin-top: 25px;
            font-size: 1.4em;
        }
        .model-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 25px;
            border-radius: 10px;
            margin: 20px 0;
            border-left: 5px solid #667eea;
        }
        .model-card h4 {
            margin-top: 0;
            color: #2d3748;
            font-size: 1.3em;
        }
        .feature-list {
            list-style: none;
            padding: 0;
        }
        .feature-list li {
            padding: 8px 0;
            padding-left: 30px;
            position: relative;
        }
        .feature-list li:before {
            content: "✓";
            position: absolute;
            left: 0;
            color: #667eea;
            font-weight: bold;
            font-size: 1.2em;
        }
        .stage-box {
            background: #f7fafc;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
        }
        .stage-box h4 {
            margin-top: 0;
            color: #4a5568;
        }
        .code-block {
            background: #2d3748;
            color: #e2e8f0;
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.9em;
        }
        .image-container {
            text-align: center;
            margin: 30px 0;
        }
        .image-container img {
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .metric-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        .metric-table th {
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
        }
        .metric-table td {
            padding: 12px 15px;
            border-bottom: 1px solid #e2e8f0;
        }
        .metric-table tr:nth-child(even) {
            background: #f7fafc;
        }
        .highlight {
            background: #fef3c7;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: 600;
        }
        .footer {
            text-align: center;
            color: #718096;
            margin-top: 50px;
            padding-top: 30px;
            border-top: 2px solid #e2e8f0;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Anime Super-Resolution</h1>
        <p>Complete Architecture Documentation & Training Pipeline Guide</p>
    </div>

    <div class="section">
        <h2>Overview</h2>
        <p>
            This project implements a <span class="highlight">Progressive Ensemble Distillation</span> system
            for anime super-resolution. It combines state-of-the-art architectures with advanced
            knowledge distillation techniques to achieve high-quality upscaling with efficient inference.
        </p>
        
        <div class="image-container">
            <img src="architecture_pipeline.png" alt="Training Pipeline">
            <p><em>Complete training pipeline from data ingestion to final ensemble model</em></p>
        </div>
    </div>

    <div class="section">
        <h2>Model Architectures</h2>
        
        <div class="model-card">
            <h4>Model A: SPAN-Tiny with MTKD + FAKD</h4>
            <p>NTIRE-winning architecture optimized for super-resolution with parameter-free attention.</p>
            <ul class="feature-list">
                <li><strong>Architecture:</strong> SPAN (Swift Parameter-free Attention Network)</li>
                <li><strong>Parameters:</strong> ~250K (Tiny variant with 26 channels)</li>
                <li><strong>Key Features:</strong> Parameter-free attention, ConvLoRA adaptation</li>
                <li><strong>Training:</strong> 2-stage (Knowledge Aggregation → Student Distillation)</li>
                <li><strong>Losses:</strong> L1 + MTKD (Wavelet) + FAKD (Feature Affinity)</li>
                <li><strong>Inference Speed:</strong> 0.001-0.005s per 720p frame</li>
            </ul>
        </div>
        
        <div class="model-card">
            <h4>Model B: Mamba-PAN with Directional Scanning</h4>
            <p>State Space Model with linear complexity and 4-direction scanning mechanism.</p>
            <ul class="feature-list">
                <li><strong>Architecture:</strong> Mamba-PAN with hierarchical blocks</li>
                <li><strong>Parameters:</strong> Variable based on configuration</li>
                <li><strong>Key Features:</strong> 4-direction scanning (H, V, RH, RV), O(N) complexity</li>
                <li><strong>Training:</strong> Direct student training with teacher guidance</li>
                <li><strong>Losses:</strong> L1 + MTKD + Directional FAKD + Cross-Direction Consistency</li>
                <li><strong>Advantage:</strong> Better long-range dependencies vs O(N²) attention</li>
            </ul>
        </div>
        
        <div class="model-card">
            <h4>Ensemble: Meta-Teacher Distillation</h4>
            <p>Combines Model A and Model B as frozen teachers for ultra-lightweight student.</p>
            <ul class="feature-list">
                <li><strong>Input:</strong> Frozen Model A + Model B checkpoints</li>
                <li><strong>Output:</strong> Tiny Student (~100K parameters)</li>
                <li><strong>Method:</strong> Ensemble predictions as soft targets</li>
                <li><strong>Use Case:</strong> Maximum efficiency deployment</li>
            </ul>
        </div>
    </div>

    <div class="section">
        <h2>Training Pipeline</h2>
        
        <div class="stage-box">
            <h4>Stage 1: Knowledge Aggregation</h4>
            <p>Train a network to fuse outputs from multiple teacher models (EDSR, RCAN, SwinIR).</p>
            <div class="code-block">
# Teachers: EDSR, RCAN, SwinIR (pretrained on DIV2K)
# Aggregation: DCTSwin blocks with frequency-domain attention
# Loss: L1 between aggregated output and ground truth HR

output = AggregationNetwork([edsr(lr), rcan(lr), swinir(lr)])
loss = L1Loss(output, hr)
            </div>
        </div>
        
        <div class="stage-box">
            <h4>Stage 2: Student Distillation (Model A)</h4>
            <p>Train SPAN-Tiny student with multi-teacher knowledge distillation.</p>
            <div class="code-block">
# MTKD: Multi-Teacher Knowledge Distillation (Wavelet-based)
# FAKD: Feature Affinity Knowledge Distillation

loss_l1 = L1Loss(student_output, hr)
loss_mtkd = WaveletDistillation(student_output, teacher_output, hr)
loss_fakd = FeatureAffinity(student_features, teacher_features)

total_loss = loss_l1 + λ_mtkd * loss_mtkd + λ_fakd * loss_fakd
            </div>
        </div>
        
        <div class="stage-box">
            <h4>Stage 2: Student Training (Model B)</h4>
            <p>Train Mamba-PAN with direction-aware distillation.</p>
            <div class="code-block">
# Directional scanning: Horizontal, Vertical, Reverse-H, Reverse-V
# Cross-direction consistency ensures stable predictions

student_pred = MambaPAN(lr, direction=['h', 'v', 'rh', 'rv'])
loss = L1 + MTKD + FAKD + Consistency(direction_outputs)
            </div>
        </div>
        
        <div class="stage-box">
            <h4>Stage 3: Ensemble Training</h4>
            <p>Create ensemble from frozen models and distill to tiny student.</p>
            <div class="code-block">
# Load frozen checkpoints
model_a = load_checkpoint('model_a_best.pth')
model_b = load_checkpoint('model_b_best.pth')

# Ensemble as meta-teacher
ensemble_pred = (model_a(lr) + model_b(lr)) / 2

# Train tiny student with ensemble guidance
student_loss = DistillationLoss(student_pred, ensemble_pred, hr)
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Loss Functions</h2>
        
        <table class="metric-table">
            <tr>
                <th>Loss</th>
                <th>Type</th>
                <th>Purpose</th>
                <th>Weight</th>
            </tr>
            <tr>
                <td>L1/L2</td>
                <td>Pixel</td>
                <td>Direct reconstruction quality</td>
                <td>1.0</td>
            </tr>
            <tr>
                <td>MTKD</td>
                <td>Distillation</td>
                <td>Wavelet-based teacher knowledge</td>
                <td>1.0</td>
            </tr>
            <tr>
                <td>FAKD</td>
                <td>Distillation</td>
                <td>Feature affinity matching</td>
                <td>0.5</td>
            </tr>
            <tr>
                <td>Perceptual (VGG)</td>
                <td>Perceptual</td>
                <td>High-level feature similarity</td>
                <td>0.1</td>
            </tr>
            <tr>
                <td>Adversarial</td>
                <td>GAN</td>
                <td>Realistic texture generation</td>
                <td>0.1</td>
            </tr>
            <tr>
                <td>Consistency</td>
                <td>Regularization</td>
                <td>Cross-direction stability (Mamba)</td>
                <td>0.1</td>
            </tr>
        </table>
    </div>

    <div class="section">
        <h2>Data Pipeline</h2>
        
        <h3>Degradation Model (RealESRGAN-style)</h3>
        <p>Training uses realistic degradation to simulate real-world low-quality inputs:</p>
        <ul class="feature-list">
            <li><strong>Blur:</strong> Gaussian with varying kernel sizes (7-21) and sigma (0.1-3.0)</li>
            <li><strong>Noise:</strong> Gaussian noise with sigma 0-50</li>
            <li><strong>Resize:</strong> Random down/up-sampling (0.15x to 1.5x)</li>
            <li><strong>JPEG:</strong> Compression artifacts (quality 60-100)</li>
        </ul>
        
        <h3>Video Processing</h3>
        <p>Supports training from video sources with automatic frame extraction:</p>
        <ul class="feature-list">
            <li><strong>Pre-extraction:</strong> Extract all frames before training (faster)</li>
            <li><strong>On-demand:</strong> Extract frames during training (less storage)</li>
            <li><strong>Quality filtering:</strong> Automatic blur and duplicate detection</li>
            <li><strong>Formats:</strong> MP4, AVI, MKV, MOV, WebM</li>
        </ul>
    </div>

    <div class="section">
        <h2>Training Modes</h2>
        
        <table class="metric-table">
            <tr>
                <th>Mode</th>
                <th>Description</th>
                <th>VRAM</th>
                <th>Time</th>
            </tr>
            <tr>
                <td>model_a</td>
                <td>Train SPAN with MTKD+FAKD</td>
                <td>~6.5 GB</td>
                <td>8-12 hours</td>
            </tr>
            <tr>
                <td>model_b</td>
                <td>Train Mamba-PAN</td>
                <td>~7.2 GB</td>
                <td>12-18 hours</td>
            </tr>
            <tr>
                <td>both_parallel</td>
                <td>Alternate batches between models</td>
                <td>~8.0 GB</td>
                <td>15-20 hours</td>
            </tr>
            <tr>
                <td>both_sequential</td>
                <td>Train A then B</td>
                <td>~7.0 GB</td>
                <td>20-30 hours</td>
            </tr>
            <tr>
                <td>ensemble</td>
                <td>Train from frozen A+B</td>
                <td>~6.0 GB</td>
                <td>2-4 hours</td>
            </tr>
            <tr>
                <td>full</td>
                <td>Complete pipeline (A → B → Ensemble)</td>
                <td>~8.0 GB</td>
                <td>30-50 hours</td>
            </tr>
        </table>
    </div>

    <div class="section">
        <h2>Quick Start Commands</h2>
        
        <div class="code-block">
# Train Model A (SPAN with MTKD+FAKD)
python scripts/train.py --config configs/model_a_ntire.yaml

# Train Model B (Mamba-PAN)
python scripts/train.py --config configs/model_b_mamba.yaml

# Train both models in parallel
python scripts/train.py --config configs/both_parallel.yaml

# Full pipeline
python scripts/train.py --config configs/full_pipeline.yaml

# With TensorBoard logging
python scripts/train.py --config configs/model_a_ntire.yaml --tensorboard
        </div>
    </div>

    <div class="footer">
        <p>Anime Super-Resolution Training System</p>
        <p>Progressive Ensemble Distillation Architecture</p>
    </div>
</body>
</html>
"""
        
        # Save documentation
        if output_path is None:
            output_path = self.output_dir / 'architecture_documentation.html'
        else:
            output_path = Path(output_path)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        if self.console:
            self.console.print(f"[green]Architecture documentation saved to: {output_path}[/green]")
        
        return str(output_path)
    
    def print_training_status(
        self,
        epoch: int,
        total_epochs: int,
        metrics: Dict[str, float],
        stage: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """
        Print formatted training status to console.
        
        Args:
            epoch: Current epoch
            total_epochs: Total number of epochs
            metrics: Dictionary of current metrics
            stage: Current training stage
            model_name: Name of the model being trained
        """
        if self.console:
            # Create progress display
            progress = f"Epoch {epoch}/{total_epochs}"
            if stage:
                progress += f" | {stage}"
            if model_name:
                progress = f"[{model_name}] {progress}"
            
            # Metrics table
            table = Table(box=box.SIMPLE)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            for key, value in metrics.items():
                if isinstance(value, float):
                    table.add_row(key, f"{value:.6f}")
                else:
                    table.add_row(key, str(value))
            
            self.console.print(f"\n[bold]{progress}[/bold]")
            self.console.print(table)
        else:
            # Simple fallback
            prefix = ""
            if model_name:
                prefix = f"[{model_name}] "
            if stage:
                prefix += f"[{stage}] "
            
            print(f"\n{prefix}Epoch {epoch}/{total_epochs}")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.6f}")
                else:
                    print(f"  {key}: {value}")
    
    def create_progress_bar(self, total: int, description: str = "Processing"):
        """
        Create a Rich progress bar for tracking operations.
        
        Args:
            total: Total number of steps
            description: Description to show
            
        Returns:
            Rich Progress context manager or None
        """
        if self.console and RICH_AVAILABLE:
            return Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console,
            )
        return None


# Convenience functions for standalone usage
def print_welcome_message():
    """Print a welcome message for the training system."""
    if RICH_AVAILABLE:
        console = Console()
        console.print(Panel.fit(
            "[bold cyan]Anime Super-Resolution Training System[/bold cyan]\n"
            "[dim]Progressive Ensemble Distillation with MTKD + FAKD[/dim]",
            border_style="cyan",
            box=box.DOUBLE
        ))
    else:
        print("\nAnime Super-Resolution Training System")
        print("Progressive Ensemble Distillation\n")


def create_visualizer_for_training(config: Dict, output_dir: str = "training_reports") -> TrainingVisualizer:
    """
    Factory function to create a visualizer configured for training.
    
    Args:
        config: Training configuration
        output_dir: Directory for reports
        
    Returns:
        Configured TrainingVisualizer instance
    """
    return TrainingVisualizer(
        config=config,
        output_dir=output_dir,
        use_rich=RICH_AVAILABLE,
    )


if __name__ == "__main__":
    # Demo/test mode
    print_welcome_message()
    
    # Create demo visualizer
    demo_config = {
        'model': {'name': 'Model A', 'type': 'span', 'scale': 4, 'channels': 26},
        'training': {
            'mode': 'model_a',
            'batch_size': 8,
            'lr': 1e-4,
            'optimizer': 'adam',
            'mixed_precision': True,
            'stage1': {'enabled': True, 'epochs': 100},
            'stage2': {'enabled': True, 'epochs': 500},
        },
        'data': {
            'datasets': [{'name': 'div2k', 'enabled': True, 'weight': 1.0}],
            'crop_size': 128,
        }
    }
    
    visualizer = TrainingVisualizer(config=demo_config, output_dir='demo_reports')
    
    # Print config summary
    visualizer.print_header("Demo Mode", "Training Visualizer Test")
    visualizer.print_config_summary()
    
    # Create pipeline diagram
    visualizer.create_training_pipeline_diagram()
    
    # Create architecture documentation
    visualizer.create_architecture_documentation()
    
    if RICH_AVAILABLE:
        Console().print("\n[green]Demo complete! Check 'demo_reports' directory for outputs.[/green]")
    else:
        print("\nDemo complete! Check 'demo_reports' directory for outputs.")
