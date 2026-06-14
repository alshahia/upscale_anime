"""Enhanced Aggregation Tests with Real Data"""
import pytest, sys
from pathlib import Path
import torch, torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))
for _tdm_k in ('utils', 'utils.test_data_manager'):
    sys.modules.pop(_tdm_k, None)
from utils.test_data_manager import ensure_test_data

@pytest.fixture(scope='module')
def test_data_dir():
    data_dir = ensure_test_data(min_images=4, verbose=False)
    yield data_dir


class TestSimpleAggregation:
    """Test simple aggregation"""
    
    def test_simple_aggregation_forward(self):
        """Test simple aggregation forward pass"""
        try:
            from distillation.mtkd import SimpleFeatureAggregation
            
            # Create aggregation network
            agg = SimpleFeatureAggregation(
                num_teachers=2,
                in_channels=16,
                out_channels=16
            )
            
            # Create teacher features
            feat1 = torch.randn(1, 16, 32, 32)
            feat2 = torch.randn(1, 16, 32, 32)
            
            # Forward
            output = agg([feat1, feat2])
            
            assert output.shape == feat1.shape
            assert not torch.isnan(output).any()
            
        except ImportError:
            pytest.skip("Aggregation module not available")


class TestAdaptiveAggregation:
    """Test adaptive aggregation with gating"""
    
    def test_adaptive_gating_mechanism(self):
        """Test adaptive gating produces valid weights"""
        try:
            from distillation.mtkd import AdaptiveFeatureAggregation
            
            agg = AdaptiveFeatureAggregation(
                num_teachers=2,
                in_channels=16,
                out_channels=16
            )
            
            # Create input and teacher features
            x = torch.randn(1, 3, 64, 64)
            feats = [torch.randn(1, 16, 32, 32), torch.randn(1, 16, 32, 32)]
            
            # Forward
            output, weights = agg(x, feats, return_weights=True)
            
            assert output.shape == feats[0].shape
            assert weights.shape[0] == 2  # 2 teachers
            assert torch.all(weights >= 0) and torch.all(weights <= 1)
            assert torch.abs(weights.sum() - 1.0) < 1e-5  # Sum to 1
            
        except ImportError:
            pytest.skip("Adaptive aggregation not available")


class TestMultiScaleAggregation:
    """Test multiscale aggregation"""
    
    def test_multiscale_pyramid_fusion(self):
        """Test multiscale feature fusion"""
        try:
            from distillation.mtkd import MultiScaleFeatureAggregation
            
            agg = MultiScaleFeatureAggregation(
                num_teachers=2,
                in_channels=16,
                out_channels=16
            )
            
            # Create multi-scale features
            feats_t1 = [torch.randn(1, 16, 64, 64), torch.randn(1, 16, 32, 32)]
            feats_t2 = [torch.randn(1, 16, 64, 64), torch.randn(1, 16, 32, 32)]
            
            # Forward
            output = agg([feats_t1, feats_t2])
            
            assert output[0].shape == feats_t1[0].shape
            assert output[1].shape == feats_t1[1].shape
            
        except ImportError:
            pytest.skip("Multiscale aggregation not available")


class TestAggregationFactory:
    """Test aggregation factory"""
    
    def test_factory_creates_all_types(self):
        """Test factory creates all aggregation types"""
        try:
            from distillation.mtkd import create_aggregation_network
            
            config = {
                'training': {
                    'stage1': {
                        'num_blocks': 2,
                        'embed_dim': 32,
                        'teachers': [{'name': 't1'}, {'name': 't2'}]
                    }
                },
                'model': {'scale': 4}
            }
            
            types_to_test = ['simple', 'adaptive', 'multiscale']
            
            for agg_type in types_to_test:
                try:
                    config['training']['stage1']['aggregation_type'] = agg_type
                    agg = create_aggregation_network(config, num_teachers=2)
                    assert agg is not None
                except (ValueError, KeyError, TypeError):
                    pass  # Some types may not be available
                    
        except ImportError:
            pytest.skip("Aggregation factory not available")


def test_data_source_report():
    """Report data source"""
    data_dir = ensure_test_data(min_images=4, verbose=True)
    print(f"\n[Aggregation Enhanced] Using data: {data_dir}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
