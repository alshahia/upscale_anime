from anime_sr.losses.pixel_loss import L1Loss, L2Loss, CharbonnierLoss, LossFactory
from anime_sr.losses.wavelet_loss import WaveletLoss, DirectionalWaveletLoss
from anime_sr.losses.perceptual_loss import VGGPerceptualLoss, ResNetPerceptualLoss, DISTSLoss, PerceptualLossFactory
from anime_sr.losses.dual_perceptual_loss import DualPerceptualLoss
from anime_sr.losses.twin_perceptual_loss import TwinPerceptualLoss
from anime_sr.losses.adversarial_loss import (
    AdversarialLoss,
    VanillaGANLoss,
    RelativisticGANLoss,
    HingeGANLoss,
    SRDiscriminator,
    GANTrainer,
    create_adversarial_loss,
    create_discriminator,
)
from anime_sr.losses.gradient_loss import GradientLoss, LaplacianLoss, CombinedGradientLoss
from anime_sr.losses.diversity_loss import TeacherDiversityLoss, TeacherDisagreementLoss, TeacherConsistencyLoss
from anime_sr.losses.combined_loss import Stage1CombinedLoss, AdaptiveCombinedLoss
from anime_sr.losses.anime_losses import (
    LineArtPreservationLoss,
    ColorConsistencyLoss,
    FlatRegionPreservationLoss,
    AnimeCombinedLoss,
)
from anime_sr.losses.temporal_loss import TemporalConsistencyLoss, FlowGuidedTemporalLoss, VideoAwareLoss
from anime_sr.losses.frequency_aware_loss import FrequencyAwareLoss, CombinedFrequencyLoss
from anime_sr.losses.fdl_loss import FDLLoss
from anime_sr.losses.wavelet_guided_loss import WaveletGuidedLoss

__all__ = [
    # Pixel losses
    'L1Loss',
    'L2Loss',
    'CharbonnierLoss',
    'LossFactory',
    # Wavelet losses
    'WaveletLoss',
    'DirectionalWaveletLoss',
    # Perceptual losses
    'VGGPerceptualLoss',
    'ResNetPerceptualLoss',
    'DISTSLoss',
    'DualPerceptualLoss',
    'TwinPerceptualLoss',
    'PerceptualLossFactory',
    # Adversarial losses
    'AdversarialLoss',
    'VanillaGANLoss',
    'RelativisticGANLoss',
    'HingeGANLoss',
    'SRDiscriminator',
    'GANTrainer',
    'create_adversarial_loss',
    'create_discriminator',
    # Gradient losses
    'GradientLoss',
    'LaplacianLoss',
    'CombinedGradientLoss',
    # Diversity losses
    'TeacherDiversityLoss',
    'TeacherDisagreementLoss',
    'TeacherConsistencyLoss',
    # Combined losses
    'Stage1CombinedLoss',
    'AdaptiveCombinedLoss',
    # Anime-specific losses
    'LineArtPreservationLoss',
    'ColorConsistencyLoss',
    'FlatRegionPreservationLoss',
    'AnimeCombinedLoss',
    # Temporal losses
    'TemporalConsistencyLoss',
    'FlowGuidedTemporalLoss',
    'VideoAwareLoss',
    # Frequency-aware losses (ESPAN)
    'FrequencyAwareLoss',
    'CombinedFrequencyLoss',
    # FDL loss (DINOv2)
    'FDLLoss',
    # Wavelet-guided GAN
    'WaveletGuidedLoss',
]
