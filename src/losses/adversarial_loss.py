"""
Adversarial Loss for GAN-based Super-Resolution
Includes discriminator and various GAN loss formulations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from typing import Tuple, Optional, Dict


class SRDiscriminator(nn.Module):
    """
    Discriminator for Super-Resolution GAN.
    PatchGAN-style architecture for local realism.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_features: int = 64,
        num_layers: int = 3,
    ):
        super().__init__()
        
        layers = []
        
        # First layer
        layers.append(nn.Conv2d(in_channels, num_features, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Middle layers
        nf = num_features
        for i in range(num_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if i == num_layers - 1 else 2
            layers.append(nn.Conv2d(nf_prev, nf, kernel_size=4, stride=stride, padding=1))
            layers.append(nn.BatchNorm2d(nf))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Final layer - output single value per patch
        layers.append(nn.Conv2d(nf, 1, kernel_size=4, stride=1, padding=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input image [B, 3, H, W]
        
        Returns:
            Discriminator output [B, 1, H', W']
        """
        return self.model(x)


class VanillaGANLoss(nn.Module):
    """Standard GAN loss (binary cross entropy)"""
    
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        disc_output: torch.Tensor,
        is_real: bool,
    ) -> torch.Tensor:
        """
        Compute GAN loss.
        
        Args:
            disc_output: Discriminator output
            is_real: True for real samples, False for fake
        
        Returns:
            GAN loss
        """
        target = torch.ones_like(disc_output) if is_real else torch.zeros_like(disc_output)
        return self.loss_fn(disc_output, target)


class RelativisticGANLoss(nn.Module):
    """
    Relativistic-average GAN loss (RA-GAN, ESRGAN-style).

    PR-3 fix (Issue #3): the previous `forward(real, fake)` was symmetric in D/G
    and returned a constant when the cross-terms cancelled. The two new methods
    below are asymmetric and match the standard ESRGAN formulation:

        D wants:  D(real) - D(fake).mean()  ->  +inf   (push real up vs fake)
                  D(fake) - D(real).mean()  ->  -inf   (push fake down vs real)
        G wants:  D(real) - D(fake).mean()  ->  0      (fool D: make them equal)
                  D(fake) - D(real).mean()  ->  +inf   (push fake up)

    The old `forward(real, fake)` is kept for backward compatibility but emits
    a DeprecationWarning on every call. New code should use the new methods.
    """

    def __init__(self):
        super().__init__()
        self._deprecation_warned = False

    def _cross_term(
        self,
        anchor: torch.Tensor,
        opponent: torch.Tensor,
        target_is_real: bool,
    ) -> torch.Tensor:
        """BCEWithLogits between (anchor - opponent.mean()) and a target tensor.

        For discriminator training, the opponent should NOT contribute gradients
        (we call .detach() on the opponent.mean() at the call sites). For the
        generator, the opponent contributes gradients as usual.
        """
        target = torch.ones_like(anchor) if target_is_real else torch.zeros_like(anchor)
        return F.binary_cross_entropy_with_logits(anchor - opponent.mean(), target)

    def discriminator_loss(
        self,
        real_output: torch.Tensor,
        fake_output: torch.Tensor,
    ) -> torch.Tensor:
        """Loss for the discriminator (pushes D(real) up and D(fake) down).

        fake_output's mean is detached so gradients flow only into the
        discriminator via real_output.
        """
        # D wants real high, fake low (relative to each other)
        loss_real = self._cross_term(
            real_output,
            fake_output.detach(),
            target_is_real=True,
        )
        loss_fake = self._cross_term(
            fake_output,
            real_output.detach(),
            target_is_real=False,
        )
        return (loss_real + loss_fake) * 0.5

    def generator_loss(
        self,
        real_output: torch.Tensor,
        fake_output: torch.Tensor,
    ) -> torch.Tensor:
        """Loss for the generator (fools the discriminator).

        Both real_output and fake_output receive gradients.
        """
        # G wants D(fake) close to D(real); from D's perspective these are the
        # opposite targets (G inverts D's targets).
        loss_real = self._cross_term(
            real_output,
            fake_output,
            target_is_real=False,
        )
        loss_fake = self._cross_term(
            fake_output,
            real_output,
            target_is_real=True,
        )
        return (loss_real + loss_fake) * 0.5

    def forward(
        self,
        real_output: torch.Tensor,
        fake_output: torch.Tensor,
    ) -> torch.Tensor:
        """DEPRECATED. Use `discriminator_loss` or `generator_loss` instead.

        Kept for backward compatibility; returns the discriminator loss.
        """
        if not self._deprecation_warned:
            warnings.warn(
                "RelativisticGANLoss.forward(real, fake) is deprecated. "
                "Use .discriminator_loss() for the D-step and "
                ".generator_loss() for the G-step. "
                "The old API returned a symmetric, near-constant value.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._deprecation_warned = True
        return self.discriminator_loss(real_output, fake_output)


class HingeGANLoss(nn.Module):
    """Hinge GAN loss (used in SAGAN, BigGAN)"""
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        disc_output: torch.Tensor,
        is_real: bool,
    ) -> torch.Tensor:
        """
        Compute hinge GAN loss.
        
        Args:
            disc_output: Discriminator output
            is_real: True for real, False for fake
        
        Returns:
            Hinge loss
        """
        if is_real:
            return torch.mean(F.relu(1.0 - disc_output))
        else:
            return torch.mean(F.relu(1.0 + disc_output))


class AdversarialLoss(nn.Module):
    """
    Adversarial loss wrapper for generator training.
    """

    def __init__(
        self,
        loss_type: str = "vanilla",  # vanilla, relativistic, hinge
    ):
        super().__init__()
        self.loss_type = loss_type

        if loss_type == "vanilla":
            self.loss_fn = VanillaGANLoss()
        elif loss_type == "relativistic":
            self.loss_fn = RelativisticGANLoss()
        elif loss_type == "hinge":
            self.loss_fn = HingeGANLoss()
        else:
            raise ValueError(f"Unknown GAN loss type: {loss_type}")

    def discriminator_loss(
        self,
        disc_output_real: torch.Tensor,
        disc_output_fake: torch.Tensor,
    ) -> torch.Tensor:
        """Loss for the discriminator (one step)."""
        if self.loss_type == "relativistic":
            return self.loss_fn.discriminator_loss(disc_output_real, disc_output_fake)
        # For vanilla and hinge
        loss_real = self.loss_fn(disc_output_real, is_real=True)
        loss_fake = self.loss_fn(disc_output_fake, is_real=False)
        return (loss_real + loss_fake) * 0.5

    def generator_loss(
        self,
        disc_output_real: torch.Tensor,
        disc_output_fake: torch.Tensor,
    ) -> torch.Tensor:
        """Loss for the generator (fools the discriminator)."""
        if self.loss_type == "relativistic":
            return self.loss_fn.generator_loss(disc_output_real, disc_output_fake)
        # For vanilla and hinge, generator wants discriminator to output 'real'
        # for fake samples. Both losses use the same path here.
        return self.loss_fn(disc_output_fake, is_real=True)

    def forward(
        self,
        disc_output_fake: torch.Tensor,
        disc_output_real: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        DEPRECATED wrapper that returns the GENERATOR loss.

        New code should use `.generator_loss(real, fake)` or
        `.discriminator_loss(real, fake)` explicitly. Argument order is the
        historical `(fake, real)` to keep existing call sites working, but the
        D/G ambiguity that caused Issue #3 cannot be expressed via forward().
        """
        if disc_output_real is None:
            return self.loss_fn(disc_output_fake, is_real=True)
        return self.generator_loss(disc_output_real, disc_output_fake)


class GANTrainer:
    """
    Helper class for training GAN components.
    Not a loss function, but manages discriminator training.
    """
    
    def __init__(
        self,
        discriminator: nn.Module,
        loss_type: str = "vanilla",
        lr: float = 1e-4,
    ):
        self.discriminator = discriminator
        self.adversarial_loss = AdversarialLoss(loss_type)
        self.optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.9, 0.99))
    
    def train_step(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Train discriminator one step.
        
        Args:
            real_images: Real HR images
            fake_images: Generated SR images
        
        Returns:
            Dict with discriminator loss
        """
        self.optimizer.zero_grad()
        
        # Discriminator on real
        disc_real = self.discriminator(real_images)
        
        # Discriminator on fake
        disc_fake = self.discriminator(fake_images.detach())
        
        # Compute loss
        if isinstance(self.adversarial_loss.loss_fn, RelativisticGANLoss):
            d_loss = self.adversarial_loss.loss_fn.discriminator_loss(disc_real, disc_fake)
        else:
            loss_real = self.adversarial_loss.loss_fn(disc_real, is_real=True)
            loss_fake = self.adversarial_loss.loss_fn(disc_fake, is_real=False)
            d_loss = (loss_real + loss_fake) / 2
        
        d_loss.backward()
        self.optimizer.step()
        
        return {
            'd_loss': d_loss.item(),
            'disc_real': disc_real.mean().item(),
            'disc_fake': disc_fake.mean().item(),
        }


# Convenience function
def create_adversarial_loss(loss_type: str = "vanilla") -> AdversarialLoss:
    """Create adversarial loss"""
    return AdversarialLoss(loss_type)


def create_discriminator(
    in_channels: int = 3,
    num_features: int = 64,
) -> SRDiscriminator:
    """Create discriminator"""
    return SRDiscriminator(in_channels, num_features)
