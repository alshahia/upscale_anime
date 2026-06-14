# Training Visualizer & Documentation Generator

A comprehensive visualization and documentation module for the Anime Super-Resolution training pipeline. This module provides rich console output, training reports, visualizations, and complete architecture documentation.

## Features

- **Rich Console Output**: Beautiful formatted tables and progress displays (requires `rich` library)
- **Training Reports**: HTML reports with training curves, checkpoint analysis, and metrics
- **Architecture Documentation**: Complete HTML documentation of the training pipeline
- **Visual Diagrams**: Pipeline flowcharts and model comparison charts
- **Checkpoint Analysis**: Detailed analysis of saved model checkpoints
- **Model Comparison**: Side-by-side comparison of multiple training runs

## Quick Start

### Installation

```bash
# Install rich for enhanced console output (optional but recommended)
pip install rich matplotlib
```

### Generate Complete Documentation

```bash
# Generate full documentation package
python scripts/visualize_training.py --mode full

# Output includes:
# - training_reports/architecture_documentation.html
# - training_reports/training_pipeline_diagram.png
```

### Analyze Checkpoints

```bash
# Analyze all checkpoints in directory
python scripts/visualize_training.py --mode analyze --checkpoints-dir checkpoints/

# Analyze specific checkpoint
python scripts/visualize_training.py --mode analyze --checkpoint checkpoints/best.pth
```

### Generate Training Report

```bash
# Generate report from config and history
python scripts/visualize_training.py --mode report \
    --config configs/model_a_ntire.yaml \
    --history logs/history.json \
    --model-name "Model A (SPAN)"
```

### Compare Models

```bash
# Compare two models' training histories
python scripts/visualize_training.py --mode compare \
    --model-a logs/model_a_history.json \
    --model-b logs/model_b_history.json
```

## Python API Usage

### Basic Usage

```python
from utils.training_visualizer import TrainingVisualizer, print_welcome_message

# Print welcome message
print_welcome_message()

# Create visualizer with config
config = {
    'model': {'name': 'Model A', 'type': 'span', 'scale': 4},
    'training': {'batch_size': 8, 'lr': 1e-4, 'mode': 'model_a'},
}

visualizer = TrainingVisualizer(config, output_dir='training_reports')

# Print config summary
visualizer.print_config_summary()

# Create pipeline diagram
visualizer.create_training_pipeline_diagram()

# Generate architecture documentation
visualizer.create_architecture_documentation()
```

### Training Integration

```python
from utils.training_visualizer import create_visualizer_for_training

# In your training script
visualizer = create_visualizer_for_training(config, output_dir='reports')

# Print training status each epoch
for epoch in range(total_epochs):
    metrics = train_epoch(...)
    visualizer.print_training_status(
        epoch=epoch,
        total_epochs=total_epochs,
        metrics=metrics,
        stage='Stage 2',
        model_name='SPAN-Tiny'
    )

# Generate final report
visualizer.generate_training_report(
    model_name='Model A (SPAN)',
    training_history=history,
    checkpoints_dir='checkpoints/',
)
```

### Checkpoint Analysis

```python
# Analyze a checkpoint
analysis = visualizer.analyze_checkpoint('checkpoints/best.pth')
print(f"Parameters: {analysis['total_parameters']:,}")
print(f"Model size: {analysis['model_size_mb']:.2f} MB")
print(f"Epoch: {analysis['epoch']}")

# Print formatted summary
visualizer.print_checkpoint_summary('checkpoints/best.pth')
```

### Training Curve Visualization

```python
# From training history
history = {
    'train_loss': [0.1, 0.08, 0.06, 0.05, 0.04],
    'val_loss': [0.12, 0.09, 0.07, 0.06, 0.055],
    'val_psnr': [25.0, 26.5, 27.8, 28.5, 29.0],
}

# Plot curves
visualizer.plot_training_curves(
    history=history,
    title='Training Progress',
    model_name='Model A',
)

# Compare multiple models
models_data = [
    {'name': 'Model A', 'history': history_a},
    {'name': 'Model B', 'history': history_b},
]
visualizer.plot_model_comparison(
    models_data=models_data,
    metric='loss',
)
```

## Output Files

### Architecture Documentation (`architecture_documentation.html`)

Complete HTML documentation including:
- System overview with pipeline diagram
- Model architectures (Model A, Model B, Ensemble)
- Training pipeline stages
- Loss functions reference table
- Data pipeline with degradation model
- Training modes comparison table
- Quick start commands

### Training Pipeline Diagram (`training_pipeline_diagram.png`)

Visual diagram showing:
- Stage 1: Knowledge Aggregation (EDSR, RCAN, SwinIR → Aggregation Network)
- Stage 2A: SPAN-Tiny Student training
- Stage 2B: Mamba-PAN Student training  
- Stage 3: Ensemble training
- Data pipeline flow

### Training Report (`training_report.html`)

Per-model report with:
- Training summary metrics
- Configuration details
- Training curves visualization
- Checkpoint analysis
- Best epoch markers

### Model Comparison (`model_comparison_*.png`)

Side-by-side plots comparing:
- Training loss curves
- Validation metrics
- PSNR progression

## Command-Line Reference

```
python scripts/visualize_training.py [OPTIONS]

Options:
  --mode {full,docs,analyze,report,diagram,compare}
                        Visualization mode (default: full)
  --config, -c PATH     Path to config YAML file
  --checkpoints-dir PATH
                        Directory containing checkpoints (default: checkpoints)
  --checkpoint PATH     Single checkpoint to analyze
  --history PATH        Path to training history JSON file
  --model-name NAME     Name of the model for reports (default: Model)
  --output-dir, -o PATH Output directory (default: training_reports)
  --model-a PATH        History file for Model A (comparison mode)
  --model-b PATH        History file for Model B (comparison mode)
```

## Modes Explained

### `full` Mode
Generates complete documentation package:
- Architecture documentation HTML
- Training pipeline diagram
- Config summary (if provided)

### `docs` Mode
Generates only the architecture documentation HTML file.

### `diagram` Mode
Generates only the training pipeline diagram PNG.

### `analyze` Mode
Analyzes checkpoint(s):
- File size, epoch, best loss
- Parameter count, model size
- Layer type distribution

### `report` Mode
Generates training report from history:
- Training curves plot
- Summary metrics
- Checkpoint analysis

### `compare` Mode
Compares two models' training histories:
- Side-by-side loss plots
- PSNR comparison (if available)

## Integration with Training

### Option 1: Post-Training Visualization

```bash
# After training completes
python scripts/visualize_training.py --mode report \
    --config configs/model_a_ntire.yaml \
    --history checkpoints/training_history.json
```

### Option 2: Real-Time Integration

Modify your training script:

```python
from utils.training_visualizer import TrainingVisualizer

class MyTrainer:
    def __init__(self, config):
        self.visualizer = TrainingVisualizer(config, output_dir='reports')
        self.visualizer.print_config_summary()
        
    def train(self):
        for epoch in range(self.epochs):
            metrics = self.train_epoch()
            
            # Print status
            self.visualizer.print_training_status(
                epoch, self.epochs, metrics
            )
            
        # Generate final report
        self.visualizer.generate_training_report(...)
```

## Dependencies

### Required
- `torch` - For checkpoint analysis
- `matplotlib` - For plotting
- `numpy` - For numerical operations

### Optional (Enhanced Experience)
- `rich` - For beautiful console output with tables and progress bars
- `PIL` - For image processing in visualizations

## File Structure

```
upscale_anime/
├── src/
│   └── utils/
│       ├── training_visualizer.py    # Main module (800+ lines)
│       ├── visualizer.py              # Existing image visualization
│       └── __init__.py                # Updated with new exports
├── scripts/
│   └── visualize_training.py          # CLI script
├── docs/
│   └── TRAINING_VISUALIZER.md        # This documentation
└── training_reports/                  # Generated output (created on run)
    ├── architecture_documentation.html
    ├── training_pipeline_diagram.png
    └── [model_reports]/
        ├── training_report.html
        └── training_curves.png
```

## Troubleshooting

### Rich library not available
```
Warning: Rich library not available. Install with: pip install rich
```
The visualizer works without Rich, but with plain text output.

### No checkpoint found
```
Error: Checkpoint not found: checkpoints/best.pth
```
Verify the checkpoint path and that training has saved checkpoints.

### Empty history file
```
Warning: History file not found: logs/history.json
```
Ensure your training script saves history as JSON. The module expects format:
```json
{
  "train_loss": [0.1, 0.08, ...],
  "val_loss": [0.12, 0.09, ...],
  "val_psnr": [25.0, 26.5, ...]
}
```
