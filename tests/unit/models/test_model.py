#!/usr/bin/env python3
"""
Unit tests for model architectures
"""
import sys
from pathlib import Path
import torch
import pytest
from anime_sr.models.span import SPANModel, SPANTiny, SPANF, create_span_model
from anime_sr.distillation.mtkd import KnowledgeAggregationNetwork
from anime_sr.losses import L1Loss, CharbonnierLoss, WaveletLoss
from anime_sr.distillation.fakd import FeatureAffinityLoss


def test_span_model_creation():
    """Test SPAN model can be created"""
    model = SPANModel(scale=4, num_channels=26, num_blocks=12)
    assert model is not None
    assert model.scale == 4
    print(f"SPAN parameters: {model.count_parameters():,}")


def test_span_tiny():
    """Test SPAN-Tiny variant"""
    model = SPANTiny(scale=4)
    assert model.num_channels == 26
    assert model.num_blocks == 12
    print(f"SPAN-Tiny parameters: {model.count_parameters():,}")


def test_span_forward():
    """Test SPAN forward pass"""
    model = SPANModel(scale=4, num_channels=26, num_blocks=4)
    model.eval()
    
    # Create dummy input
    x = torch.randn(1, 3, 64, 64)
    
    with torch.no_grad():
        out = model(x)
    
    assert out.shape == (1, 3, 256, 256)  # 4x upsampling
    assert out.min() >= 0 and out.max() <= 1  # Clamped to [0,1]


def test_span_feature_extraction():
    """Test feature extraction for FAKD"""
    model = SPANModel(scale=4, num_channels=26, num_blocks=4)
    model.eval()
    
    x = torch.randn(1, 3, 64, 64)
    
    with torch.no_grad():
        out, features = model.forward_with_features(x)
    
    assert out.shape == (1, 3, 256, 256)
    assert len(features) > 0
    print(f"Extracted features from {len(features)} layers")


def test_knowledge_aggregation():
    """Test Knowledge Aggregation Network"""
    model = KnowledgeAggregationNetwork(
        num_teachers=3,
        embed_dim=64,
        num_blocks=2,
        scale=4
    )
    
    # Create dummy teacher outputs
    teacher_outputs = [torch.randn(1, 3, 256, 256) for _ in range(3)]
    
    out = model(teacher_outputs)
    assert out.shape == (1, 3, 256, 256)


def test_l1_loss():
    """Test L1 loss"""
    loss_fn = L1Loss()
    pred = torch.randn(1, 3, 64, 64)
    target = torch.randn(1, 3, 64, 64)
    
    loss = loss_fn(pred, target)
    assert loss.item() >= 0


def test_wavelet_loss():
    """Test wavelet loss"""
    loss_fn = WaveletLoss(levels=2)
    pred = torch.randn(1, 3, 64, 64)
    target = torch.randn(1, 3, 64, 64)
    
    loss = loss_fn(pred, target)
    assert loss.item() >= 0


def test_fakd_loss():
    """Test FAKD loss"""
    loss_fn = FeatureAffinityLoss(layers=[2, 4])
    
    # Create dummy features
    student_features = {
        2: torch.randn(1, 32, 32, 32),
        4: torch.randn(1, 32, 32, 32),
    }
    teacher_features = {
        2: torch.randn(1, 32, 32, 32),
        4: torch.randn(1, 32, 32, 32),
    }
    
    loss = loss_fn(student_features, teacher_features)
    assert loss.item() >= 0


def test_model_config_factory():
    """Test model creation from config"""
    config = {
        'type': 'span',
        'scale': 4,
        'channels': 26,
        'num_blocks': 8,
        'use_lora': True,
        'lora_rank': 4,
    }
    
    model = create_span_model(config)
    assert model is not None
    assert model.num_channels == 26
    assert model.num_blocks == 8


def run_all_tests():
    """Run all tests"""
    tests = [
        test_span_model_creation,
        test_span_tiny,
        test_span_forward,
        test_span_feature_extraction,
        test_knowledge_aggregation,
        test_l1_loss,
        test_wavelet_loss,
        test_fakd_loss,
        test_model_config_factory,
    ]
    
    passed = 0
    failed = 0
    
    print("\n" + "="*50)
    print("Running Model Tests")
    print("="*50 + "\n")
    
    for test in tests:
        try:
            test()
            print(f"✅ {test.__name__} - PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} - FAILED: {e}")
            failed += 1
    
    print("\n" + "="*50)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*50 + "\n")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
