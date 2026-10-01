"""
Frequency Distribution Loss (FDL) with DINOv2 backend.
Perceptual loss that matches frequency distributions of features.

Reference: neosr wiki - fdl_opt/fdl_loss
Uses DINOv2 features with sliced Wasserstein distance for perceptual quality.
Recommended to increase num_proj at end of finetuning for better results.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FDLLoss(nn.Module):
    """
    Frequency Distribution Loss using DINOv2 features.

    Matches the frequency distribution of features between SR and HR images
    using sliced Wasserstein distance on DINOv2 feature projections.

    Reference: neosr framework - fdl_opt
    https://github.com/neosr-project/neosr/wiki/Losses

    Args:
        num_proj: Number of random projections for sliced Wasserstein (default 24)
                  Increase to 64-128 for better quality at end of training
        dino_variant: DINOv2 variant ('small', 'base', 'large')
        weight: Loss weight multiplier
    """

    def __init__(
        self,
        num_proj: int = 24,
        dino_variant: str = 'small',
        weight: float = 1.0,
    ):
        super().__init__()
        self.num_proj = num_proj
        self.weight = weight

        model_map = {
            'small': 'dinov2_vits14',
            'base': 'dinov2_vitb14',
            'large': 'dinov2_vitl14',
        }
        model_name = model_map.get(dino_variant, 'dinov2_vits14')

        self.feature_extractor = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.patch_size = 14

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.max() > 1.0:
            x = x / 255.0
        if x.min() < 0:
            x = (x + 1) / 2
        return (x - self.mean) / self.std

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        h_pad = (self.patch_size - h % self.patch_size) % self.patch_size
        w_pad = (self.patch_size - w % self.patch_size) % self.patch_size
        if h_pad > 0 or w_pad > 0:
            x = F.pad(x, (0, w_pad, 0, h_pad), mode='reflect')

        features = self.feature_extractor.get_intermediate_layers(x, n=4)

        features = torch.cat([f.mean(dim=1) for f in features], dim=-1)
        return features

    def sliced_wasserstein_distance(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        num_proj: int,
    ) -> torch.Tensor:
        if pred.shape != target.shape:
            pred = F.interpolate(pred, size=target.shape[1:], mode='bilinear', align_corners=False)

        pred_flat = pred.view(pred.shape[0], -1)
        target_flat = target.view(target.shape[0], -1)

        device = pred_flat.device
        dim = pred_flat.shape[1]

        projections = torch.randn(dim, num_proj, device=device)
        projections = F.normalize(projections, dim=0)

        pred_proj = pred_flat @ projections
        target_proj = target_flat @ projections

        pred_proj = pred_proj.sort(dim=0)[0]
        target_proj = target_proj.sort(dim=0)[0]

        swd = ((pred_proj - target_proj) ** 2).mean()
        return torch.sqrt(swd + 1e-8)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_norm = self.normalize_input(pred)
        target_norm = self.normalize_input(target)

        # DINOv2 backbone is frozen (requires_grad=False on all params) and the
        # target-side features are constant w.r.t. the generator. Skip the
        # autograd graph on the target forward to halve FDL memory + compute.
        with torch.no_grad():
            target_feat = self.extract_features(target_norm)

        pred_feat = self.extract_features(pred_norm)

        loss = self.sliced_wasserstein_distance(pred_feat, target_feat, self.num_proj)

        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=pred.device, requires_grad=False)

        return self.weight * loss
