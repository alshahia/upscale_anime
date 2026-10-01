from anime_sr.distillation.mtkd.aggregation import KnowledgeAggregationNetwork
from anime_sr.distillation.mtkd.simple_aggregation import SimpleKnowledgeAggregation
from anime_sr.distillation.mtkd.adaptive_aggregation import AdaptiveTeacherAggregation
from anime_sr.distillation.mtkd.multiscale_aggregation import MultiScaleKnowledgeAggregation
from anime_sr.distillation.mtkd.aggregation_factory import create_aggregation_network, get_available_aggregation_types
from anime_sr.distillation.mtkd.distillation import WaveletDistillationLoss, FullMTKDLoss

__all__ = [
    'KnowledgeAggregationNetwork',
    'SimpleKnowledgeAggregation',
    'AdaptiveTeacherAggregation',
    'MultiScaleKnowledgeAggregation',
    'create_aggregation_network',
    'get_available_aggregation_types',
    'WaveletDistillationLoss',
    'FullMTKDLoss',
]
