"""
Balanced Twin Perceptual Loss for Anime Super-Resolution.
Combines VGG19 (ImageNet) for photorealistic features with ResNet50 for anime-specific features.

Reference: APISR (CVPR 2024) - Balanced Twin Perceptual Loss
L_per = L_ResNet + delta * L_VGG

This addresses the issue of unwanted color artifacts when using only ImageNet-trained
perceptual loss on anime content, which differs significantly from photorealistic features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List, Optional
import os


class TwinPerceptualLoss(nn.Module):
    """
    Balanced Twin Perceptual Loss combining VGG19 (ImageNet) and ResNet50 features.

    VGG19 captures photorealistic features (texture, general structure).
    ResNet50 captures anime-specific features (when Danbooru-pretrained).

    Formula: L_per = L_ResNet + delta * L_VGG
    where delta balances the two contributions (default 0.1-0.3).

    Reference: APISR (CVPR 2024), Section 3.4
    https://arxiv.org/abs/2403.01598
    """

    def __init__(
        self,
        vgg_layers: List[str] = None,
        resnet_layers: List[str] = None,
        loss_type: str = "l1",
        delta: float = 0.1,
        danbooru_weight: Optional[float] = None,
        vgg_weight: Optional[float] = None,
        use_danbooru_resnet: bool = True,
        danbooru_resnet_path: Optional[str] = None,
    ):
        super().__init__()

        if vgg_layers is None:
            vgg_layers = ['relu2_2', 'relu3_3', 'relu4_3']

        if resnet_layers is None:
            resnet_layers = ['layer2', 'layer3']

        self.vgg_layers = vgg_layers
        self.resnet_layers = resnet_layers
        self.loss_type = loss_type

        if danbooru_weight is not None and vgg_weight is not None:
            self.danbooru_weight = float(danbooru_weight)
            self.vgg_weight = float(vgg_weight)
            self.delta = 0.0
        else:
            self.danbooru_weight = 1.0
            self.vgg_weight = float(delta)
            self.delta = float(delta)

        # Build VGG19 feature extractor (ImageNet pretrained)
        self.vgg_extractor = self._build_vgg19_extractor(vgg_layers)

        # Build ResNet50 feature extractor
        self.resnet_extractor = self._build_resnet50_extractor(
            resnet_layers,
            use_danbooru=use_danbooru_resnet,
            danbooru_path=danbooru_resnet_path,
        )

        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Disable in-place operations for AMP compatibility
        for module in self.vgg_extractor.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

    def _build_vgg19_extractor(self, layers: List[str]) -> nn.ModuleDict:
        """Build VGG19 feature extractor."""
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = False

        # VGG19 layer mapping (features indices)
        vgg_layer_map = {
            'relu1_1': 2, 'relu1_2': 5,
            'relu2_1': 7, 'relu2_2': 10,
            'relu3_1': 12, 'relu3_2': 15, 'relu3_3': 18, 'relu3_4': 21,
            'relu4_1': 23, 'relu4_2': 26, 'relu4_3': 29, 'relu4_4': 32,
            'relu5_1': 34, 'relu5_2': 37, 'relu5_3': 40, 'relu5_4': 43,
        }

        extractor = nn.ModuleDict()
        layer_indices = [(name, vgg_layer_map[name]) for name in layers if name in vgg_layer_map]
        layer_indices.sort(key=lambda x: x[1])

        prev_idx = 0
        for name, idx in layer_indices:
            extractor[name] = nn.Sequential(*list(vgg[prev_idx:idx+1]))
            prev_idx = idx + 1

        return extractor

    def _build_resnet50_extractor(
        self,
        layers: List[str],
        use_danbooru: bool = True,
        danbooru_path: Optional[str] = None,
    ) -> nn.ModuleDict:
        """Build ResNet50 feature extractor, optionally with Danbooru pretrained weights."""
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Try to load Danbooru-pretrained weights if available
        if use_danbooru:
            danbooru_weights = self._find_danbooru_weights(danbooru_path)
            if danbooru_weights is not None:
                try:
                    state_dict = torch.load(danbooru_weights, map_location='cpu', weights_only=True)
                    if 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                    # Filter and load
                    resnet_state = resnet.state_dict()
                    filtered = {k: v for k, v in state_dict.items() if k in resnet_state and v.shape == resnet_state[k].shape}
                    if filtered:
                        resnet.load_state_dict(filtered, strict=False)
                        print(f"[TwinPerceptualLoss] Loaded Danbooru ResNet50 weights: {len(filtered)} keys")
                    else:
                        print(f"[TwinPerceptualLoss] No matching Danbooru weights found, using ImageNet")
                except Exception as e:
                    print(f"[TwinPerceptualLoss] Failed to load Danbooru weights: {e}, using ImageNet")
            else:
                print(f"[TwinPerceptualLoss] Danbooru weights not found, using ImageNet ResNet50")

        resnet.eval()
        for param in resnet.parameters():
            param.requires_grad = False

        # Disable in-place ReLU
        for module in resnet.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

        extractor = nn.ModuleDict()
        extractor['conv1'] = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        extractor['layer1'] = resnet.layer1

        available = ['layer1', 'layer2', 'layer3', 'layer4']
        for layer_name in layers:
            if layer_name in available:
                extractor[layer_name] = getattr(resnet, layer_name)

        return extractor

    def _find_danbooru_weights(self, danbooru_path: Optional[str] = None) -> Optional[str]:
        """Find Danbooru-pretrained ResNet50 weights."""
        candidates = []

        if danbooru_path and os.path.exists(danbooru_path):
            candidates.append(danbooru_path)

        # Check common locations
        search_paths = [
            'pretrained/danbooru_resnet50.pth',
            'pretrained/danbooru_resnet50.pt',
            'pretrained/resnet50_danbooru.pth',
            os.path.expanduser('~/.cache/torch/hub/checkpoints/danbooru_resnet50.pth'),
        ]
        candidates.extend(search_paths)

        for path in candidates:
            if os.path.exists(path):
                return path

        return None

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input to ImageNet statistics."""
        if x.max() > 1.0:
            x = x / 255.0
        if x.min() < 0:
            x = (x + 1) / 2
        return (x - self.mean) / self.std

    def _compute_feature_loss(self, pred_feat: torch.Tensor, target_feat: torch.Tensor) -> torch.Tensor:
        """Compute loss between feature maps."""
        if pred_feat.shape != target_feat.shape:
            pred_feat = F.interpolate(pred_feat, size=target_feat.shape[2:], mode='bilinear', align_corners=False)

        if self.loss_type == "l1":
            return F.l1_loss(pred_feat, target_feat)
        elif self.loss_type == "l2":
            return F.mse_loss(pred_feat, target_feat)
        else:
            return F.l1_loss(pred_feat, target_feat)

    def compute_vgg_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute VGG19-based perceptual loss."""
        loss = 0.0
        prev_pred = pred
        prev_target = target

        for layer_name in self.vgg_layers:
            if layer_name not in self.vgg_extractor:
                continue

            pred_feat = self.vgg_extractor[layer_name](prev_pred)
            target_feat = self.vgg_extractor[layer_name](prev_target)

            loss += self._compute_feature_loss(pred_feat, target_feat)
            prev_pred = pred_feat
            prev_target = target_feat

        return loss / max(len(self.vgg_layers), 1)

    def compute_resnet_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute ResNet50-based perceptual loss."""
        x_pred = self.resnet_extractor['conv1'](pred)
        x_target = self.resnet_extractor['conv1'](target)

        layer_order = ['layer1', 'layer2', 'layer3', 'layer4']
        loss = 0.0
        loss_count = 0

        for layer_name in layer_order:
            if layer_name not in self.resnet_extractor:
                continue

            x_pred = self.resnet_extractor[layer_name](x_pred)
            x_target = self.resnet_extractor[layer_name](x_target)

            if layer_name in self.resnet_layers:
                loss += self._compute_feature_loss(x_pred, x_target)
                loss_count += 1

        return loss / max(loss_count, 1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute balanced twin perceptual loss.

        L_per = L_ResNet + delta * L_VGG

        Args:
            pred: Predicted SR image [B, 3, H, W] in [0, 1]
            target: Ground truth HR image [B, 3, H, W] in [0, 1]

        Returns:
            Combined perceptual loss
        """
        pred_norm = self.normalize_input(pred)
        target_norm = self.normalize_input(target)

        resnet_loss = self.compute_resnet_loss(pred_norm, target_norm)
        vgg_loss = self.compute_vgg_loss(pred_norm, target_norm)

        total_loss = self.danbooru_weight * resnet_loss + self.vgg_weight * vgg_loss

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return torch.tensor(0.0, device=pred.device, requires_grad=False)

        return total_loss
