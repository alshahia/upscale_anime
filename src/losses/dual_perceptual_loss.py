"""
Dual Perceptual Loss combining VGG and ResNet features.
Research: "Dual Perceptual Loss for Single Image Super-Resolution Using ESRGAN" (arXiv:2201.06383)

VGG features capture texture/pattern information.
ResNet features capture structural information.
Dynamic weighting balances both contributions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List


class DualPerceptualLoss(nn.Module):
    """
    Dual Perceptual Loss combining VGG16 and ResNet18 features.
    
    VGG16 extracts texture/pattern features (shallow layers).
    ResNet18 extracts structural features (residual blocks).
    Dynamic weighting prevents one loss from dominating.
    
    Reference: "Dual Perceptual Loss for Single Image Super-Resolution Using ESRGAN"
    https://ar5iv.labs.arxiv.org/html/2201.06383
    
    Example:
        >>> loss_fn = DualPerceptualLoss()
        >>> loss = loss_fn(sr, hr)  # sr=super-resolved, hr=high-resolution target
    """
    
    def __init__(
        self,
        vgg_layers: List[str] = None,
        resnet_layers: List[str] = None,
        loss_type: str = "l1",
        dynamic_weighting: bool = True,
    ):
        super().__init__()
        
        # Default VGG layers for texture (early layers capture fine textures)
        if vgg_layers is None:
            vgg_layers = ['relu2_2', 'relu3_3']
        
        # Default ResNet layers for structure
        if resnet_layers is None:
            resnet_layers = ['layer2', 'layer3']
        
        self.vgg_layers = vgg_layers
        self.resnet_layers = resnet_layers
        self.loss_type = loss_type
        self.dynamic_weighting = dynamic_weighting
        
        # Load pretrained models
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        vgg.eval()
        resnet.eval()
        
        for param in vgg.parameters():
            param.requires_grad = False
        for param in resnet.parameters():
            param.requires_grad = False
        
        # Disable in-place ReLU for AMP compatibility
        for module in vgg.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
        
        # VGG16 layer mapping
        vgg_layer_map = {
            'relu1_1': 3, 'relu1_2': 6,
            'relu2_1': 8, 'relu2_2': 11,
            'relu3_1': 13, 'relu3_2': 16, 'relu3_3': 19,
            'relu4_1': 22, 'relu4_2': 25, 'relu4_3': 28,
            'relu5_1': 31, 'relu5_2': 34, 'relu5_3': 37,
        }
        
        # Build VGG feature extractor
        self.vgg_extractor = self._build_vgg_extractor(vgg, vgg_layers, vgg_layer_map)
        
        # Build ResNet feature extractor
        self.resnet_extractor = self._build_resnet_extractor(resnet, resnet_layers)
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def _build_vgg_extractor(self, vgg: nn.Module, layers: List[str], layer_map: dict) -> nn.ModuleDict:
        """Build VGG feature extractor for specified layers."""
        extractor = nn.ModuleDict()
        
        layer_indices = [(name, layer_map[name]) for name in layers]
        layer_indices.sort(key=lambda x: x[1])
        
        prev_idx = 0
        for name, idx in layer_indices:
            extractor[name] = nn.Sequential(*list(vgg[prev_idx:idx+1]))
            prev_idx = idx + 1
        
        return extractor
    
    def _build_resnet_extractor(self, resnet: nn.Module, layers: List[str]) -> nn.ModuleDict:
        """Build ResNet feature extractor for specified layers."""
        extractor = nn.ModuleDict()
        extractor['conv1'] = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        extractor['layer1'] = resnet.layer1
        
        available_layers = ['layer1', 'layer2', 'layer3', 'layer4']
        for layer_name in layers:
            if layer_name in available_layers:
                if layer_name == 'layer1':
                    extractor[layer_name] = resnet.layer1
                elif layer_name == 'layer2':
                    extractor[layer_name] = resnet.layer2
                elif layer_name == 'layer3':
                    extractor[layer_name] = resnet.layer3
                elif layer_name == 'layer4':
                    extractor[layer_name] = resnet.layer4
        
        return extractor
    
    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input to ImageNet statistics."""
        if x.max() > 1.0:
            x = x / 255.0
        if x.min() < 0:
            x = (x + 1) / 2
        
        x = (x - self.mean) / self.std
        return x
    
    def compute_vgg_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute VGG-based perceptual loss for texture."""
        loss = 0.0
        
        prev_pred = pred
        prev_target = target
        
        for layer_name in self.vgg_layers:
            pred_feat = self.vgg_extractor[layer_name](prev_pred)
            target_feat = self.vgg_extractor[layer_name](prev_target)
            
            if pred_feat.shape != target_feat.shape:
                pred_feat = F.interpolate(pred_feat, size=target_feat.shape[2:], 
                                        mode='bilinear', align_corners=False)
                target_feat = F.interpolate(target_feat, size=target_feat.shape[2:], 
                                          mode='bilinear', align_corners=False)
            
            if self.loss_type == "l1":
                layer_loss = F.l1_loss(pred_feat, target_feat)
            elif self.loss_type == "l2":
                layer_loss = F.mse_loss(pred_feat, target_feat)
            else:
                layer_loss = F.l1_loss(pred_feat, target_feat)
            
            loss += layer_loss
            prev_pred = pred_feat
            prev_target = target_feat
        
        return loss / len(self.vgg_layers)
    
    def compute_resnet_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute ResNet-based perceptual loss for structure."""
        x_pred = self.resnet_extractor['conv1'](pred)
        x_target = self.resnet_extractor['conv1'](target)
        
        loss = 0.0
        
        for layer_name in self.resnet_layers:
            if layer_name not in self.resnet_extractor:
                continue
            
            x_pred = self.resnet_extractor[layer_name](x_pred)
            x_target = self.resnet_extractor[layer_name](x_target)
            
            if x_pred.shape != x_target.shape:
                x_pred = F.interpolate(x_pred, size=x_target.shape[2:], 
                                     mode='bilinear', align_corners=False)
                x_target = F.interpolate(x_target, size=x_target.shape[2:], 
                                       mode='bilinear', align_corners=False)
            
            if self.loss_type == "l1":
                layer_loss = F.l1_loss(x_pred, x_target)
            elif self.loss_type == "l2":
                layer_loss = F.mse_loss(x_pred, x_target)
            else:
                layer_loss = F.l1_loss(x_pred, x_target)
            
            loss += layer_loss
        
        return loss / len(self.resnet_layers)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute dual perceptual loss.
        
        Args:
            pred: Predicted/upscaled image [B, 3, H, W]
            target: Ground truth high-resolution image [B, 3, H, W]
        
        Returns:
            Combined perceptual loss (lower is better)
        """
        # Normalize inputs
        pred_norm = self.normalize_input(pred)
        target_norm = self.normalize_input(target)
        
        # Compute VGG loss (texture)
        vgg_loss = self.compute_vgg_loss(pred_norm, target_norm)
        
        # Compute ResNet loss (structure)
        resnet_loss = self.compute_resnet_loss(pred_norm, target_norm)
        
        # Dynamic weighting: balance VGG and ResNet contributions
        if self.dynamic_weighting:
            # Weight resnet loss to match vgg loss magnitude
            with torch.no_grad():
                if vgg_loss.item() > 1e-8:
                    weight = vgg_loss.detach() / (resnet_loss.detach() + 1e-8)
                else:
                    weight = torch.tensor(1.0, device=pred.device)
            
            total_loss = vgg_loss + weight * resnet_loss
        else:
            total_loss = vgg_loss + resnet_loss
        
        # Safety check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return torch.tensor(0.0, device=pred.device, requires_grad=False)
        
        return total_loss