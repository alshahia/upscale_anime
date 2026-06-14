from losses.pixel_loss import L1Loss, L2Loss, CharbonnierLoss, LossFactory
from losses.wavelet_loss import WaveletLoss, DirectionalWaveletLoss
from losses.perceptual_loss import VGGPerceptualLoss, ResNetPerceptualLoss, DISTSLoss, PerceptualLossFactory
from losses.dual_perceptual_loss import DualPerceptualLoss
from losses.twin_perceptual_loss import TwinPerceptualLoss
from losses.adversarial_loss import (
    AdversarialLoss,
    VanillaGANLoss,
    RelativisticGANLoss,
    HingeGANLoss,
    SRDiscriminator,
    GANTrainer,
    create_adversarial_loss,
    create_discriminator,
)
from losses.gradient_loss import GradientLoss, LaplacianLoss, CombinedGradientLoss
from losses.diversity_loss import TeacherDiversityLoss, TeacherDisagreementLoss, TeacherConsistencyLoss
from losses.combined_loss import Stage1CombinedLoss, AdaptiveCombinedLoss
from losses.anime_losses import (
    LineArtPreservationLoss,
    ColorConsistencyLoss,
    FlatRegionPreservationLoss,
    AnimeCombinedLoss,
)
from losses.temporal_loss import TemporalConsistencyLoss, FlowGuidedTemporalLoss, VideoAwareLoss
from losses.frequency_aware_loss import FrequencyAwareLoss, CombinedFrequencyLoss
from losses.fdl_loss import FDLLoss
from losses.wavelet_guided_loss import WaveletGuidedLoss

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
