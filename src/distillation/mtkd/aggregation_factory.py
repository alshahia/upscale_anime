"""
Factory for creating knowledge aggregation networks.
Supports multiple architectures: simple, adaptive, dctswin, multiscale
"""
import torch.nn as nn
from typing import Dict
import warnings


def create_aggregation_network(config: Dict) -> nn.Module:
    """
    Factory function to create knowledge aggregation network based on config.
    
    Args:
        config: Configuration dictionary with training.stage1 settings
        
    Returns:
        Knowledge aggregation network instance
    """
    stage1_cfg = config.get('training', {}).get('stage1', {})
    
    # Get aggregation type
    aggregation_type = stage1_cfg.get('aggregation_type', 'simple').lower()
    
    # Get architecture parameters
    num_teachers = len(stage1_cfg.get('teachers', []))
    if num_teachers == 0:
        raise ValueError("No teachers configured for Stage 1")
    
    scale = config.get('model', {}).get('scale', 4)
    num_blocks = stage1_cfg.get('num_blocks', 6)
    embed_dim = stage1_cfg.get('embed_dim', 96)
    
    # Create appropriate aggregation network
    if aggregation_type == 'simple':
        from distillation.mtkd.simple_aggregation import SimpleKnowledgeAggregation
        print(f"Creating SimpleKnowledgeAggregation (num_teachers={num_teachers}, embed_dim={embed_dim}, num_blocks={num_blocks})")
        model = SimpleKnowledgeAggregation(
            num_teachers=num_teachers,
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            scale=scale,
        )
        
    elif aggregation_type == 'adaptive':
        from distillation.mtkd.adaptive_aggregation import AdaptiveTeacherAggregation
        print(f"Creating AdaptiveTeacherAggregation (num_teachers={num_teachers}, embed_dim={embed_dim}, num_blocks={num_blocks})")
        model = AdaptiveTeacherAggregation(
            num_teachers=num_teachers,
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            scale=scale,
        )
        
    elif aggregation_type == 'dctswin':
        from distillation.mtkd.aggregation import KnowledgeAggregationNetwork
        warnings.warn(
            "DCT-Swin aggregation is deprecated due to non-functional DCT layers. "
            "Consider using 'simple' or 'adaptive' for better stability.",
            DeprecationWarning
        )
        print(f"Creating KnowledgeAggregationNetwork (DCT-Swin) (num_teachers={num_teachers}, embed_dim={embed_dim}, num_blocks={num_blocks})")
        model = KnowledgeAggregationNetwork(
            num_teachers=num_teachers,
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            scale=scale,
        )
        
    elif aggregation_type == 'multiscale':
        from distillation.mtkd.multiscale_aggregation import MultiScaleKnowledgeAggregation
        print(f"Creating MultiScaleKnowledgeAggregation (num_teachers={num_teachers}, embed_dim={embed_dim})")
        scale_factors = stage1_cfg.get('scale_factors', [1, 2, 4])
        model = MultiScaleKnowledgeAggregation(
            num_teachers=num_teachers,
            embed_dim=embed_dim,
            scale=scale,
            scale_factors=scale_factors,
        )
        
    else:
        raise ValueError(f"Unknown aggregation_type: {aggregation_type}. "
                        f"Choose from: simple, adaptive, dctswin, multiscale")
    
    return model


def get_available_aggregation_types() -> list:
    """Return list of available aggregation types."""
    return ['simple', 'adaptive', 'dctswin', 'multiscale']
