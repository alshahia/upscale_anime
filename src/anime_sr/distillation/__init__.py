from anime_sr.distillation.fakd.affinity_loss import FeatureAffinityLoss, DirectionalFeatureAffinityLoss, CrossDirectionConsistencyLoss
from anime_sr.distillation.mtkd.aggregation import KnowledgeAggregationNetwork
from anime_sr.distillation.mtkd.distillation import WaveletDistillationLoss

__all__ = [
    'FeatureAffinityLoss',
    'DirectionalFeatureAffinityLoss',
    'CrossDirectionConsistencyLoss',
    'KnowledgeAggregationNetwork',
    'WaveletDistillationLoss',
]
