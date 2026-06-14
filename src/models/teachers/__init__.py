"""
Teacher models for knowledge distillation.
Supports EDSR, RCAN, and SwinIR architectures.
"""
from .edsr import EDSR, create_edsr
from .rcan import RCAN, create_rcan
from .swinir import SwinIR, create_swinir
from .teacher_loader import load_teacher_model, create_teacher_model, auto_detect_architecture

__all__ = [
    'EDSR',
    'RCAN',
    'SwinIR',
    'create_edsr',
    'create_rcan',
    'create_swinir',
    'load_teacher_model',
    'create_teacher_model',
    'auto_detect_architecture',
]
