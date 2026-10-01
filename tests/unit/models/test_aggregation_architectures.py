"""
Unit tests for knowledge aggregation architectures.
Tests all aggregation types: simple, adaptive, multiscale, feature-based
"""
import sys
import os

import torch
import pytest
from anime_sr.distillation.mtkd import (
    SimpleKnowledgeAggregation,
    AdaptiveTeacherAggregation,
    MultiScaleKnowledgeAggregation,
    create_aggregation_network,
    get_available_aggregation_types,
)
from anime_sr.distillation.mtkd.feature_aggregation import (
    FeatureAlignmentModule,
    CrossTeacherAttention,
    FeatureKnowledgeAggregation,
)

class TestSimpleKnowledgeAggregation:
    """Test SimpleKnowledgeAggregation."""
    
    def test_init(self):
        """Test initialization."""
        model = SimpleKnowledgeAggregation(num_teachers=3, embed_dim=64, num_blocks=4, scale=4)
        assert model.num_teachers == 3
        assert model.scale == 4
        assert len(model.blocks) == 4
    
    def test_forward(self):
        """Test forward pass."""
        model = SimpleKnowledgeAggregation(num_teachers=3, embed_dim=64, num_blocks=4, scale=4)
        model.eval()
        
        # Create dummy teacher outputs (4x upscaling)
        batch_size = 2
        hr_size = 64
        lr_size = hr_size // 4
        
        teacher_outputs = [
            torch.randn(batch_size, 3, hr_size, hr_size)
            for _ in range(3)
        ]
        
        with torch.no_grad():
            output = model(teacher_outputs)
        
        assert output.shape == (batch_size, 3, hr_size, hr_size)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_gradient_checkpointing(self):
        """Test gradient checkpointing enable."""
        model = SimpleKnowledgeAggregation(num_teachers=2, embed_dim=32, num_blocks=2, scale=2)
        model.gradient_checkpointing_enable()
        assert model.use_gradient_checkpointing

class TestAdaptiveTeacherAggregation:
    """Test AdaptiveTeacherAggregation."""
    
    def test_init(self):
        """Test initialization."""
        model = AdaptiveTeacherAggregation(num_teachers=3, embed_dim=64, num_blocks=4, scale=4)
        assert model.num_teachers == 3
        assert hasattr(model, 'teacher_gating')
        assert hasattr(model, 'teacher_branches')
    
    def test_forward(self):
        """Test forward pass."""
        model = AdaptiveTeacherAggregation(num_teachers=3, embed_dim=64, num_blocks=4, scale=4)
        model.eval()
        
        batch_size = 2
        hr_size = 64
        
        teacher_outputs = [
            torch.randn(batch_size, 3, hr_size, hr_size)
            for _ in range(3)
        ]
        
        with torch.no_grad():
            output = model(teacher_outputs)
        
        assert output.shape == (batch_size, 3, hr_size, hr_size)
        assert not torch.isnan(output).any()
    
    def test_teacher_weights(self):
        """Test teacher weight extraction."""
        model = AdaptiveTeacherAggregation(num_teachers=3, embed_dim=64, num_blocks=2, scale=4)
        model.eval()
        
        batch_size = 2
        hr_size = 64
        
        teacher_outputs = [
            torch.randn(batch_size, 3, hr_size, hr_size)
            for _ in range(3)
        ]
        
        with torch.no_grad():
            weights = model.get_teacher_weights(teacher_outputs)
        
        assert weights.shape == (batch_size, 3)
        # Check that weights sum to approximately 1 (softmax)
        assert torch.allclose(weights.sum(dim=1), torch.ones(batch_size), atol=1e-5)

class TestMultiScaleKnowledgeAggregation:
    """Test MultiScaleKnowledgeAggregation."""
    
    def test_init(self):
        """Test initialization."""
        model = MultiScaleKnowledgeAggregation(
            num_teachers=3, embed_dim=64, scale=4, scale_factors=[1, 2]
        )
        assert len(model.scale_blocks) == 2
    
    def test_forward(self):
        """Test forward pass."""
        model = MultiScaleKnowledgeAggregation(
            num_teachers=3, embed_dim=64, scale=4, scale_factors=[1, 2]
        )
        model.eval()
        
        batch_size = 2
        hr_size = 64
        
        teacher_outputs = [
            torch.randn(batch_size, 3, hr_size, hr_size)
            for _ in range(3)
        ]
        
        with torch.no_grad():
            output = model(teacher_outputs)
        
        assert output.shape == (batch_size, 3, hr_size, hr_size)
        assert not torch.isnan(output).any()

class TestFeatureAggregation:
    """Test feature-based aggregation components."""
    
    def test_feature_alignment(self):
        """Test FeatureAlignmentModule."""
        align = FeatureAlignmentModule([64, 64, 180], target_dim=64)
        
        features = [
            torch.randn(2, 64, 32, 32),
            torch.randn(2, 64, 32, 32),
            torch.randn(2, 180, 32, 32),
        ]
        
        output = align(features)
        assert output.shape == (2, 64 * 3, 32, 32)  # 3 teachers * 64 channels
    
    def test_cross_teacher_attention(self):
        """Test CrossTeacherAttention."""
        attention = CrossTeacherAttention(dim=64, num_teachers=3, num_heads=4)
        
        features = [
            torch.randn(2, 64, 32, 32)
            for _ in range(3)
        ]
        
        output = attention(features)
        assert len(output) == 3
        assert output[0].shape == (2, 64, 32, 32)
    
    def test_feature_knowledge_aggregation(self):
        """Test FeatureKnowledgeAggregation."""
        model = FeatureKnowledgeAggregation(
            num_teachers=3,
            feature_dims=[64, 64, 180],
            aligned_dim=64,
            num_blocks=2,
            scale=4,
        )
        model.eval()
        
        batch_size = 2
        feature_size = 16  # LR size
        
        teacher_features = [
            torch.randn(batch_size, 64, feature_size, feature_size),
            torch.randn(batch_size, 64, feature_size, feature_size),
            torch.randn(batch_size, 180, feature_size, feature_size),
        ]
        
        with torch.no_grad():
            output = model(teacher_features)
        
        assert output.shape[0] == batch_size
        assert output.shape[1] == 3  # RGB
        assert not torch.isnan(output).any()

class TestAggregationFactory:
    """Test aggregation factory function."""
    
    def test_available_types(self):
        """Test get_available_aggregation_types."""
        types = get_available_aggregation_types()
        assert 'simple' in types
        assert 'adaptive' in types
        assert 'multiscale' in types
    
    def test_create_simple(self):
        """Test creating simple aggregation."""
        config = {
            'training': {
                'stage1': {
                    'aggregation_type': 'simple',
                    'num_blocks': 4,
                    'embed_dim': 64,
                    'teachers': [{'name': 't1'}, {'name': 't2'}, {'name': 't3'}],
                }
            },
            'model': {'scale': 4}
        }
        
        model = create_aggregation_network(config)
        assert isinstance(model, SimpleKnowledgeAggregation)
    
    def test_create_adaptive(self):
        """Test creating adaptive aggregation."""
        config = {
            'training': {
                'stage1': {
                    'aggregation_type': 'adaptive',
                    'num_blocks': 4,
                    'embed_dim': 64,
                    'teachers': [{'name': 't1'}, {'name': 't2'}],
                }
            },
            'model': {'scale': 4}
        }
        
        model = create_aggregation_network(config)
        assert isinstance(model, AdaptiveTeacherAggregation)
    
    def test_create_multiscale(self):
        """Test creating multiscale aggregation."""
        config = {
            'training': {
                'stage1': {
                    'aggregation_type': 'multiscale',
                    'embed_dim': 64,
                    'teachers': [{'name': 't1'}, {'name': 't2'}],
                    'scale_factors': [1, 2],
                }
            },
            'model': {'scale': 4}
        }
        
        model = create_aggregation_network(config)
        assert isinstance(model, MultiScaleKnowledgeAggregation)

def test_no_nan_gradients():
    """Test that no NaN gradients are produced during training."""
    model = SimpleKnowledgeAggregation(num_teachers=2, embed_dim=32, num_blocks=2, scale=2)
    model.train()
    
    teacher_outputs = [
        torch.randn(1, 3, 32, 32, requires_grad=False)
        for _ in range(2)
    ]
    
    target = torch.randn(1, 3, 32, 32)
    
    output = model(teacher_outputs)
    loss = torch.nn.functional.l1_loss(output, target)
    loss.backward()
    
    # Check no NaN in gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN in gradient for {name}"
            assert not torch.isinf(param.grad).any(), f"Inf in gradient for {name}"

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
