"""Loss package for adversarial student distillation (Phase 3).

Houses the GAN discriminator + adversarial loss (PatchGAN70 + HingeGANLoss)
and the Sobel-based edge loss. Both are imported by
anime_sr.training.distillation.distill when --teacher is one of the RealESR-based
checkpoints (animevideov3, lsdir).

Phase 3 handoff: docs/plans/student_adversarial_handoff_2026_08.md (section A.1).
"""
__all__ = []
