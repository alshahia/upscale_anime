# Example Training Visualization Outputs

This directory contains example outputs from the `TrainingVisualizer` module demonstrating all visualization capabilities.

## Files

| File | Description | Size |
|------|-------------|------|
| `training_pipeline_diagram.png` | Visual flowchart of complete training pipeline | 114 KB |
| `architecture_pipeline.png` | Pipeline diagram embedded in documentation | 114 KB |
| `architecture_documentation.html` | Complete HTML architecture documentation | 14 KB |
| `demo_training_history.json` | Example training history with 100 epochs | 13 KB |
| `model_a_training_curves.png` | Training curves showing loss and PSNR | 114 KB |

## Quick Preview

### Training Pipeline Diagram
Open `training_pipeline_diagram.png` to see the complete training flow:
- Stage 1: Knowledge Aggregation (EDSR, RCAN, SwinIR)
- Stage 2: SPAN-Tiny and Mamba-PAN student training
- Stage 3: Ensemble training
- Data pipeline flow

### Architecture Documentation
Open `architecture_documentation.html` in a browser for complete documentation:
- System overview
- Model architectures (Model A, Model B, Ensemble)
- Training pipeline stages
- Loss functions reference
- Data pipeline details
- Training modes comparison
- Quick start commands

### Training Curves
Open `model_a_training_curves.png` to see:
- Training loss curve with validation
- PSNR progression
- Best epoch markers
- Learning rate decay

### Training History Data
`demo_training_history.json` contains:
- 100 epochs of simulated training data
- Stage boundaries (stage1: 0-19, stage2: 20-99)
- Metrics: train_loss, val_loss, val_psnr, lr
- Summary statistics with best epochs

## How to Generate Your Own

```bash
# Generate complete documentation
python scripts/visualize_training.py --mode full

# Generate from your training
python scripts/visualize_training.py --mode report \
    --config configs/model_a_ntire.yaml \
    --history logs/history.json \
    --model-name "My Model"
```

## Integration Example

```python
from utils.training_visualizer import TrainingVisualizer
from utils.training_history import TrainingHistory

# Track training
history = TrainingHistory(config)
for epoch in range(epochs):
    metrics = train_epoch(...)
    history.log_epoch(epoch, metrics)

# Generate visualizations
viz = TrainingVisualizer(config, output_dir='reports')
viz.create_training_pipeline_diagram()
viz.plot_training_curves(history.get_history())
viz.generate_training_report('My Model', history.get_history(), 'checkpoints/')
```

## Demo Command

Run the demo to generate these outputs:

```bash
cd e:\python_projects\upscale_anime
python -c "
import sys; sys.path.insert(0, 'src')
from utils.training_visualizer import TrainingVisualizer
from utils.training_history import TrainingHistory
import numpy as np

config = {
    'model': {'name': 'Model A', 'type': 'span', 'scale': 4},
    'training': {'mode': 'model_a', 'batch_size': 8, 'lr': 1e-4}
}

viz = TrainingVisualizer(config, output_dir='example_outputs')
viz.create_training_pipeline_diagram()
viz.create_architecture_documentation()

# Generate demo history
history = TrainingHistory(config)
for epoch in range(100):
    history.log_epoch(epoch, {
        'train_loss': 0.08 * np.exp(-epoch / 30) + 0.02,
        'val_loss': 0.1 * np.exp(-epoch / 30) + 0.03,
        'val_psnr': 26 + 5 * (1 - np.exp(-epoch / 25)),
    })
history.save('example_outputs/demo_training_history.json')

viz.plot_training_curves(history.get_history(), model_name='Model A')
print('Done!')
"
```
