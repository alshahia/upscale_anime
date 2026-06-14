#!/usr/bin/env python
"""
Model export script for production deployment.
Exports to ONNX, TorchScript, and other formats.
"""
import argparse
import torch
import torch.nn as nn
from pathlib import Path
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.span import create_span_model


def export_onnx(model: nn.Module,
                output_path: str,
                input_size: tuple = (1, 3, 64, 64),
                opset_version: int = 11):
    """
    Export model to ONNX format.
    
    Args:
        model: PyTorch model
        output_path: Output ONNX file path
        input_size: Sample input size (batch, channels, height, width)
        opset_version: ONNX opset version
    """
    print(f"Exporting to ONNX: {output_path}")
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_size)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )
    
    print(f"ONNX export complete: {output_path}")
    
    # Verify with onnxruntime if available
    try:
        import onnxruntime as ort
        import numpy as np
        
        ort_session = ort.InferenceSession(output_path)
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        # Compare with PyTorch output
        with torch.no_grad():
            torch_output = model(dummy_input).numpy()
        
        # Check outputs match
        diff = np.abs(torch_output - ort_outputs[0]).max()
        print(f"ONNX verification: max difference = {diff:.6f}")
        
        if diff < 1e-5:
            print("✓ ONNX export verified successfully")
        else:
            print("⚠ Warning: ONNX output differs from PyTorch")
    
    except ImportError:
        print("Note: onnxruntime not available for verification")


def export_torchscript(model: nn.Module, output_path: str, method: str = 'trace'):
    """
    Export model to TorchScript format.
    
    Args:
        model: PyTorch model
        output_path: Output TorchScript file path
        method: 'trace' or 'script'
    """
    print(f"Exporting to TorchScript ({method}): {output_path}")
    
    model.eval()
    
    if method == 'trace':
        # Trace with dummy input
        example_input = torch.randn(1, 3, 64, 64)
        traced_model = torch.jit.trace(model, example_input)
        traced_model.save(output_path)
    elif method == 'script':
        # Script the model
        scripted_model = torch.jit.script(model)
        scripted_model.save(output_path)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"TorchScript export complete: {output_path}")
    
    # Verify
    try:
        loaded = torch.jit.load(output_path)
        test_input = torch.randn(1, 3, 64, 64)
        
        with torch.no_grad():
            orig_out = model(test_input)
            loaded_out = loaded(test_input)
        
        diff = torch.abs(orig_out - loaded_out).max().item()
        print(f"TorchScript verification: max difference = {diff:.6f}")
        
        if diff < 1e-5:
            print("✓ TorchScript export verified successfully")
    except Exception as e:
        print(f"Warning: TorchScript verification failed: {e}")


def export_state_dict(model: nn.Module, output_path: str):
    """Export just the model weights."""
    print(f"Exporting state dict: {output_path}")
    torch.save(model.state_dict(), output_path)
    print("State dict export complete")


def quantize_model(model: nn.Module) -> nn.Module:
    """
    Apply dynamic quantization for faster CPU inference.
    
    Args:
        model: PyTorch model
    
    Returns:
        Quantized model
    """
    print("Applying dynamic quantization...")
    
    # Quantize linear and conv layers
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )
    
    print("Quantization complete")
    return quantized_model


def create_inference_pipeline(checkpoint_path: str,
                              output_dir: str,
                              formats: list = None,
                              device: str = 'cuda'):
    """
    Create complete inference pipeline with multiple formats.
    
    Args:
        checkpoint_path: Path to model checkpoint
        output_dir: Directory to save exported models
        formats: List of formats to export ('onnx', 'torchscript', 'state_dict')
        device: Device to load model on
    """
    formats = formats or ['onnx', 'torchscript', 'state_dict']
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create model
    from omegaconf import OmegaConf
    config = OmegaConf.create({
        'model': {
            'type': 'span',
            'channels': 32,
            'scale': 4
        }
    })
    
    model = create_span_model(config.model)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Get model info
    model_name = Path(checkpoint_path).stem
    
    exported_files = {}
    
    # Export to various formats
    if 'onnx' in formats:
        onnx_path = output_dir / f"{model_name}.onnx"
        export_onnx(model, str(onnx_path))
        exported_files['onnx'] = str(onnx_path)
    
    if 'torchscript' in formats:
        ts_path = output_dir / f"{model_name}.pt"
        export_torchscript(model, str(ts_path), method='trace')
        exported_files['torchscript'] = str(ts_path)
    
    if 'state_dict' in formats:
        sd_path = output_dir / f"{model_name}_weights.pth"
        export_state_dict(model, str(sd_path))
        exported_files['state_dict'] = str(sd_path)
    
    # Export quantized version for CPU
    if 'quantized' in formats:
        print("\nCreating quantized version...")
        quantized = quantize_model(model.cpu())
        q_path = output_dir / f"{model_name}_quantized.pt"
        export_torchscript(quantized, str(q_path), method='trace')
        exported_files['quantized'] = str(q_path)
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'original_checkpoint': checkpoint_path,
        'exported_formats': exported_files,
        'model_config': {
            'type': 'span',
            'channels': 32,
            'scale': 4
        }
    }
    
    metadata_path = output_dir / f"{model_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Export Summary")
    print(f"{'='*60}")
    for fmt, path in exported_files.items():
        print(f"{fmt:<15}: {path}")
    print(f"{'='*60}")
    print(f"Metadata saved: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description='Export SR models for deployment')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--output-dir', type=str, default='exported_models', help='Output directory')
    parser.add_argument('--formats', type=str, default='onnx,torchscript,state_dict',
                       help='Comma-separated list of formats to export')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    formats = args.formats.split(',')
    
    create_inference_pipeline(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        formats=formats,
        device=args.device
    )


if __name__ == '__main__':
    main()
