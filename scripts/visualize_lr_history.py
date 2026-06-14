"""
Visualize learning rate history and analysis.
Plots LR vs epoch, loss vs LR, and reduction events.
"""
import json
import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import matplotlib.pyplot as plt


def load_lr_history(log_dir: Path):
    """Load LR history from JSON file."""
    history_file = log_dir / 'lr_history.json'
    
    if not history_file.exists():
        print(f"Error: No LR history found at {history_file}")
        print("Make sure adaptive_lr is enabled during training.")
        return None
    
    with open(history_file, 'r') as f:
        return json.load(f)


def plot_lr_vs_epoch(history: dict, output_path: Path = None):
    """Plot learning rate over epochs."""
    epochs = history.get('epochs', [])
    lr_values = history.get('lr_values', [])
    
    if not epochs or not lr_values:
        print("No LR data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot LR curve
    ax.plot(epochs, lr_values, linewidth=2, label='Learning Rate')
    
    # Mark reduction events
    reduction_events = history.get('reduction_events', [])
    for event in reduction_events:
        epoch = event.get('epoch', 0)
        old_lr = event.get('old_lr', 0)
        new_lr = event.get('new_lr', 0)
        
        ax.axvline(x=epoch, color='red', linestyle='--', alpha=0.5)
        ax.plot(epoch, old_lr, 'ro', markersize=8)
        ax.plot(epoch, new_lr, 'go', markersize=8)
        ax.annotate(f'{event.get("factor", 0.5):.2f}x', 
                   xy=(epoch, new_lr), xytext=(10, 10),
                   textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved LR plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_loss_vs_lr(history: dict, output_path: Path = None):
    """Plot loss vs learning rate."""
    lr_values = history.get('lr_values', [])
    loss_values = history.get('loss_values', [])
    
    if not lr_values or not loss_values:
        print("No data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot of loss vs LR
    scatter = ax.scatter(lr_values, loss_values, c=range(len(lr_values)), 
                        cmap='viridis', alpha=0.6, s=30)
    
    # Add colorbar for epoch
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Epoch')
    
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Loss')
    ax.set_title('Loss vs Learning Rate')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Annotate best LR
    best_lr = history.get('best_lr')
    best_loss = history.get('best_loss')
    if best_lr and best_loss:
        ax.plot(best_lr, best_loss, 'r*', markersize=15, label='Best LR')
        ax.legend()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved loss vs LR plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def print_summary(history: dict):
    """Print text summary of LR history."""
    stats = history.get('stats', {})
    
    print("\n" + "="*60)
    print("LR History Summary")
    print("="*60)
    
    print(f"Total Epochs: {stats.get('total_epochs', 0)}")
    print(f"LR Reductions: {stats.get('num_reductions', 0)}")
    
    lr_range = stats.get('lr_range', (0, 0))
    print(f"LR Range: {lr_range[0]:.2e} -> {lr_range[1]:.2e}")
    
    final_lr = stats.get('final_lr', 0)
    print(f"Final LR: {final_lr:.2e}")
    
    best_lr = stats.get('best_lr')
    if best_lr:
        print(f"Best LR: {best_lr:.2e} (loss: {stats.get('best_loss', 'N/A')})")
    
    # Reduction events
    reduction_events = history.get('reduction_events', [])
    if reduction_events:
        print("\nLR Reduction Events:")
        for event in reduction_events:
            print(f"  Epoch {event['epoch']}: {event['old_lr']:.2e} -> {event['new_lr']:.2e} "
                  f"(factor: {event['factor']:.2f})")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize LR history')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Directory containing lr_history.json')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save plots')
    parser.add_argument('--show', action='store_true',
                       help='Show plots interactively instead of saving')
    
    args = parser.parse_args()
    
    # Load history
    log_dir = Path(args.log_dir)
    history = load_lr_history(log_dir)
    
    if history is None:
        return 1
    
    # Print summary
    print_summary(history)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    if args.show:
        plot_lr_vs_epoch(history)
        plot_loss_vs_lr(history)
    else:
        plot_lr_vs_epoch(history, output_dir / 'lr_schedule.png')
        plot_loss_vs_lr(history, output_dir / 'loss_vs_lr.png')
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
