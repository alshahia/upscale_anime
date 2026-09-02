# anime_upscaler/adv_losses
"""Loss package for adversarial student distillation (Phase 3).

Houses the GAN discriminator + adversarial loss (PatchGAN70 + HingeGANLoss)
and the Sobel-based edge loss. Both are imported by anime_upscaler/distill.py
when --teacher is one of the RealESR-based checkpoints (animevideov3, lsdir).

Named `adv_losses` (not `losses`) because src/losses/ already exists at the
repo top level (BasicSR-style losses for the training pipeline) and would
shadow this package when both src/ and anime_upscaler/ are on sys.path.

Phase 3 handoff: docs/plans/student_adversarial_handoff_2026_08.md (section A.1).
"""
__all__ = []
