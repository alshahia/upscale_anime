"""
Test Suite for Model Export Pipeline (Phase 10 of Small Dataset Techniques)

Tests production model export formats including:
- TorchScript export for production deployment
- ONNX export for cross-platform inference
- Quantized model export (INT8)
- Export compatibility verification
- File size validation
- Inference with exported models

Data Strategy:
- Create small mock model checkpoint for testing
- Use data/test_hr/ for inference validation on exported models
"""
import pytest
import sys
import tempfile
import shutil
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

# Add src and utils to path
for _tdm_k in ('utils', 'utils.test_data_manager'):
    sys.modules.pop(_tdm_k, None)
from utils.test_data_manager import ensure_test_data


class MockSRModel(nn.Module):
    """Mock super-resolution model for export testing."""
    
    def __init__(self, scale: int = 4):
        super().__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.upsample = nn.Upsample(scale_factor=scale, mode='bicubic', align_corners=False)
        self.conv3 = nn.Conv2d(16, 3, 3, padding=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.upsample(x)
        x = self.conv3(x)
        return torch.clamp(x, 0, 1)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


@pytest.fixture(scope='module')
def mock_checkpoint():
    """Create a mock model checkpoint for export testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = MockSRModel(scale=4)
        
        # Create checkpoint with model state
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'ema_model_state_dict': model.state_dict(),  # Same for testing
            'epoch': 100,
            'config': {
                'model': {'type': 'mock', 'scale': 4, 'channels': 16}
            }
        }
        
        checkpoint_path = Path(tmpdir) / 'test_model.pth'
        torch.save(checkpoint, checkpoint_path)
        
        yield checkpoint_path


@pytest.fixture(scope='module')
def test_image():
    """Provide test image data."""
    # Create dummy LR image (64x64)
    lr_image = torch.rand(1, 3, 64, 64)
    return lr_image


class TestTorchScriptExport:
    """Test TorchScript export functionality."""
    
    def test_torchscript_export(self, mock_checkpoint):
        """Test exporting model to TorchScript format."""
        try:
            from export import export_torchscript  # Or similar function
            
            with tempfile.TemporaryDirectory() as output_dir:
                output_path = Path(output_dir) / 'model_ts.pt'
                
                # Export model
                export_torchscript(mock_checkpoint, output_path)
                
                # Verify file was created
                assert output_path.exists(), "TorchScript file should be created"
                assert output_path.stat().st_size > 0, "File should not be empty"
                
        except ImportError:
            pytest.skip("TorchScript export module not available")
    
    def test_torchscript_inference(self, mock_checkpoint, test_image):
        """Test inference with exported TorchScript model."""
        try:
            from export import export_torchscript
            
            with tempfile.TemporaryDirectory() as output_dir:
                # Export
                ts_path = Path(output_dir) / 'model_ts.pt'
                export_torchscript(mock_checkpoint, ts_path)
                
                # Load and run inference
                model_ts = torch.jit.load(ts_path)
                model_ts.eval()
                
                with torch.no_grad():
                    output = model_ts(test_image)
                
                # Verify output
                assert output.shape[2] == test_image.shape[2] * 4, "Should be 4x upscaled"
                assert output.shape[3] == test_image.shape[3] * 4, "Should be 4x upscaled"
                assert torch.all(output >= 0) and torch.all(output <= 1), \
                    "Output should be in [0, 1] range"
                
        except ImportError:
            pytest.skip("TorchScript export or loading failed")


class TestONNXExport:
    """Test ONNX export functionality."""
    
    def test_onnx_export(self, mock_checkpoint):
        """Test exporting model to ONNX format."""
        try:
            from export import export_onnx
            
            with tempfile.TemporaryDirectory() as output_dir:
                output_path = Path(output_dir) / 'model.onnx'
                
                # Export model
                export_onnx(mock_checkpoint, output_path, input_size=(1, 3, 64, 64))
                
                # Verify file was created
                assert output_path.exists(), "ONNX file should be created"
                assert output_path.stat().st_size > 0, "File should not be empty"
                
        except ImportError:
            pytest.skip("ONNX export module not available")
    
    def test_onnx_export_compatibility(self, mock_checkpoint):
        """Test that ONNX model can be loaded and validated."""
        try:
            import onnx
            from export import export_onnx
            
            with tempfile.TemporaryDirectory() as output_dir:
                onnx_path = Path(output_dir) / 'model.onnx'
                export_onnx(mock_checkpoint, onnx_path, input_size=(1, 3, 64, 64))
                
                # Load and check ONNX model
                onnx_model = onnx.load(onnx_path)
                onnx.checker.check_model(onnx_model)
                
                # Verify inputs/outputs
                assert len(onnx_model.graph.input) > 0, "Model should have inputs"
                assert len(onnx_model.graph.output) > 0, "Model should have outputs"
                
        except ImportError:
            pytest.skip("ONNX module not available")


class TestQuantizedExport:
    """Test quantized model export."""
    
    def test_quantized_export(self, mock_checkpoint):
        """Test INT8 quantized model export."""
        try:
            from export import export_quantized
            
            with tempfile.TemporaryDirectory() as output_dir:
                output_path = Path(output_dir) / 'model_quantized.pt'
                
                # Export quantized model
                export_quantized(mock_checkpoint, output_path)
                
                # Verify file was created
                assert output_path.exists(), "Quantized model file should be created"
                
        except ImportError:
            pytest.skip("Quantized export module not available")
    
    def test_quantized_model_size_reduction(self, mock_checkpoint):
        """Test that quantized model is smaller than original."""
        try:
            from export import export_quantized
            
            with tempfile.TemporaryDirectory() as output_dir:
                regular_path = Path(output_dir) / 'model_regular.pt'
                quantized_path = Path(output_dir) / 'model_quantized.pt'
                
                # Save regular model
                checkpoint = torch.load(mock_checkpoint, map_location='cpu', weights_only=True)
                torch.save(checkpoint, regular_path)
                
                # Export quantized
                export_quantized(mock_checkpoint, quantized_path)
                
                # Compare sizes
                regular_size = regular_path.stat().st_size
                quantized_size = quantized_path.stat().st_size
                
                assert quantized_size < regular_size, \
                    f"Quantized model ({quantized_size}) should be smaller than regular ({regular_size})"
                
        except ImportError:
            pytest.skip("Quantized export module not available")


class TestExportComparison:
    """Compare different export formats."""
    
    def test_export_file_sizes(self, mock_checkpoint):
        """Compare file sizes of different export formats."""
        formats_to_test = ['torchscript', 'onnx', 'quantized']
        sizes = {}
        
        with tempfile.TemporaryDirectory() as output_dir:
            output_dir = Path(output_dir)
            
            # Test each format
            for fmt in formats_to_test:
                try:
                    if fmt == 'torchscript':
                        from export import export_torchscript
                        path = output_dir / 'model_ts.pt'
                        export_torchscript(mock_checkpoint, path)
                    elif fmt == 'onnx':
                        from export import export_onnx
                        path = output_dir / 'model.onnx'
                        export_onnx(mock_checkpoint, path, input_size=(1, 3, 64, 64))
                    elif fmt == 'quantized':
                        from export import export_quantized
                        path = output_dir / 'model_quant.pt'
                        export_quantized(mock_checkpoint, path)
                    
                    if path.exists():
                        sizes[fmt] = path.stat().st_size / 1024  # KB
                except ImportError:
                    continue
            
            # Print comparison
            print("\nExport Format File Sizes:")
            for fmt, size in sizes.items():
                print(f"  {fmt}: {size:.1f} KB")
            
            # Verify we tested at least one format
            assert len(sizes) > 0, "Should test at least one export format"
    
    def test_inference_with_exported(self, mock_checkpoint, test_image):
        """Test that all exported models produce similar outputs."""
        outputs = {}
        
        with tempfile.TemporaryDirectory() as output_dir:
            output_dir = Path(output_dir)
            
            # Test TorchScript
            try:
                from export import export_torchscript
                ts_path = output_dir / 'model_ts.pt'
                export_torchscript(mock_checkpoint, ts_path)
                
                model_ts = torch.jit.load(ts_path)
                model_ts.eval()
                with torch.no_grad():
                    outputs['torchscript'] = model_ts(test_image)
            except ImportError:
                pass
            
            # Test ONNX Runtime
            try:
                import onnxruntime as ort
                from export import export_onnx
                
                onnx_path = output_dir / 'model.onnx'
                export_onnx(mock_checkpoint, onnx_path, input_size=(1, 3, 64, 64))
                
                session = ort.InferenceSession(str(onnx_path))
                input_name = session.get_inputs()[0].name
                outputs['onnx'] = torch.tensor(
                    session.run(None, {input_name: test_image.numpy()})[0]
                )
            except ImportError:
                pass
            
            # Verify outputs are similar
            if len(outputs) >= 2:
                keys = list(outputs.keys())
                for i in range(len(keys) - 1):
                    diff = torch.abs(outputs[keys[i]] - outputs[keys[i+1]]).mean()
                    assert diff < 0.1, f"Outputs from {keys[i]} and {keys[i+1]} should be similar"


class TestExportScriptIntegration:
    """Integration tests for export script."""
    
    def test_export_script_exists(self):
        """Test that the export script exists."""
        script_path = Path(__file__).parent.parent.parent.parent / 'scripts' / 'export_model.py'
        assert script_path.exists(), f"Export script not found: {script_path}"
    
    def test_export_script_help(self):
        """Test that export script shows help."""
        import subprocess
        
        script_path = Path(__file__).parent.parent.parent.parent / 'scripts' / 'export_model.py'
        result = subprocess.run(
            ['python', str(script_path), '--help'],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, "Help command should succeed"
        assert '--checkpoint' in result.stdout, "Help should mention --checkpoint option"


def test_data_source_report():
    """Report which data source is being used for tests."""
    data_dir = ensure_test_data(min_images=4, verbose=True)
    print(f"\n[ModelExport] Using data: {data_dir}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
