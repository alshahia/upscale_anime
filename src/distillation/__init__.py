from distillation.fakd.affinity_loss import FeatureAffinityLoss, DirectionalFeatureAffinityLoss, CrossDirectionConsistencyLoss
from distillation.mtkd.aggregation import KnowledgeAggregationNetwork
from distillation.mtkd.distillation import WaveletDistillationLoss

__all__ = [
    'FeatureAffinityLoss',
    'DirectionalFeatureAffinityLoss',
    'CrossDirectionConsistencyLoss',
    'KnowledgeAggregationNetwork',
    'WaveletDistillationLoss',
]
