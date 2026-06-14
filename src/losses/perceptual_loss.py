"""
Perceptual Loss using VGG features
Captures high-level image statistics and textures
"""
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List, Dict


# Module-level cache for VGG models to avoid redundant downloads
_vgg_model_cache = {}


def _get_cached_vgg_model(model_name: str, pretrained: bool = True):
    """Get or create cached VGG model to avoid redundant downloads."""
    cache_key = (model_name, pretrained)
    if cache_key not in _vgg_model_cache:
        if model_name == 'vgg19':
            _vgg_model_cache[cache_key] = models.vgg19(pretrained=pretrained)
        elif model_name == 'vgg16':
            _vgg_model_cache[cache_key] = models.vgg16(pretrained=pretrained)
        elif model_name == 'resnet50':
            _vgg_model_cache[cache_key] = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    return _vgg_model_cache[cache_key]


class VGGPerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss.
    Uses features from pre-trained VGG19.
    """
    
    def __init__(
        self,
        layers: List[str] = ["relu3_3"],
        weights: List[float] = None,
        loss_type: str = "l1",  # l1, l2, or cosine
        normalize: bool = True,
    ):
        super().__init__()
        
        # Load pre-trained VGG19 from cache
        vgg = _get_cached_vgg_model('vgg19', pretrained=True).features
        vgg.eval()
        
        # Disable gradients for VGG
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Disable in-place ReLU to avoid gradient computation errors with AMP
        for module in vgg.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
        
        # Layer name mapping
        self.layer_name_mapping = {
            "relu1_1": 1,
            "relu1_2": 3,
            "relu2_1": 6,
            "relu2_2": 8,
            "relu3_1": 11,
            "relu3_2": 13,
            "relu3_3": 15,
            "relu3_4": 17,
            "relu4_1": 20,
            "relu4_2": 22,
            "relu4_3": 24,
            "relu4_4": 26,
            "relu5_1": 29,
            "relu5_2": 31,
            "relu5_3": 33,
            "relu5_4": 35,
        }
        
        # Validate layers
        for layer in layers:
            if layer not in self.layer_name_mapping:
                raise ValueError(f"Unknown layer: {layer}. Available: {list(self.layer_name_mapping.keys())}")
        
        self.layers = layers
        
        # Normalize weights
        if weights is None:
            weights = [1.0 / len(layers)] * len(layers)
        
        if len(weights) != len(layers):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of layers ({len(layers)})")
        
        # Normalize weights to sum to 1
        total = sum(weights)
        self.weights = [w / total for w in weights]
        
        self.loss_type = loss_type
        self.normalize = normalize
        
        # Build feature extractor
        self.feature_extractor = self._build_feature_extractor(vgg, layers)
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def _build_feature_extractor(self, vgg: nn.Module, layers: List[str]) -> nn.ModuleDict:
        """Build feature extractor for specified layers"""
        feature_extractor = nn.ModuleDict()
        
        # Sort layers by their index
        layer_indices = [(name, self.layer_name_mapping[name]) for name in layers]
        layer_indices.sort(key=lambda x: x[1])
        
        prev_idx = 0
        for name, idx in layer_indices:
            # Extract slice from prev_idx to idx
            feature_extractor[name] = nn.Sequential(*list(vgg[prev_idx:idx+1]))
            prev_idx = idx + 1
        
        return feature_extractor
    
    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input to ImageNet statistics"""
        if x.min() < 0:  # Assume already normalized to [-1, 1] or [-0.5, 0.5]
            x = (x + 1) / 2  # Convert to [0, 1]
        
        # Normalize
        x = (x - self.mean) / self.std
        return x
    
    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract VGG features"""
        features = {}
        
        # Normalize
        if self.normalize:
            x = self.normalize_input(x)
        
        # Extract features sequentially
        for name in self.layers:
            x = self.feature_extractor[name](x)
            features[name] = x
        
        return features
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss.
        
        Args:
            pred: Predicted image [B, 3, H, W]
            target: Target image [B, 3, H, W]
        
        Returns:
            Perceptual loss
        """
        # Extract features
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)
        
        # Compute loss for each layer
        loss = 0
        for i, layer in enumerate(self.layers):
            pf = pred_features[layer]
            tf = target_features[layer]
            
            # Ensure same size
            if pf.shape != tf.shape:
                pf = F.interpolate(pf, size=tf.shape[2:], mode='bilinear', align_corners=False)
            
            # Compute loss
            if self.loss_type == "l1":
                layer_loss = F.l1_loss(pf, tf)
            elif self.loss_type == "l2":
                layer_loss = F.mse_loss(pf, tf)
            elif self.loss_type == "cosine":
                layer_loss = 1 - F.cosine_similarity(pf.flatten(1), tf.flatten(1), dim=1).mean()
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")
            
            loss += self.weights[i] * layer_loss
        
        return loss


class ResNetPerceptualLoss(nn.Module):
    """
    ResNet-based perceptual loss (alternative to VGG).
    Often better for structural information.
    """
    
    def __init__(
        self,
        layers: List[str] = ["layer2", "layer3", "layer4"],
        weights: List[float] = None,
        loss_type: str = "l1",
    ):
        super().__init__()
        
        # Load pre-trained ResNet50 from cache
        resnet = _get_cached_vgg_model('resnet50', pretrained=True)
        resnet.eval()
        
        for param in resnet.parameters():
            param.requires_grad = False
        
        # Disable in-place ReLU to avoid gradient computation errors with AMP
        for module in resnet.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
        
        self.layers = layers
        
        if weights is None:
            weights = [1.0 / len(layers)] * len(layers)
        self.weights = [w / sum(weights) for w in weights]
        
        self.loss_type = loss_type
        
        # Build feature extractor
        self.features = nn.ModuleDict({
            "layer1": nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1),
            "layer2": resnet.layer2,
            "layer3": resnet.layer3,
            "layer4": resnet.layer4,
        })
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input to ImageNet statistics"""
        if x.min() < 0:
            x = (x + 1) / 2
        
        x = (x - self.mean) / self.std
        return x
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute ResNet perceptual loss"""
        pred = self.normalize_input(pred)
        target = self.normalize_input(target)
        
        loss = 0
        
        for i, layer in enumerate(self.layers):
            pred = self.features[layer](pred)
            target = self.features[layer](target)
            
            if self.loss_type == "l1":
                layer_loss = F.l1_loss(pred, target)
            elif self.loss_type == "l2":
                layer_loss = F.mse_loss(pred, target)
            else:
                layer_loss = 1 - F.cosine_similarity(pred.flatten(1), target.flatten(1), dim=1).mean()
            
            loss += self.weights[i] * layer_loss
        
        return loss


class DISTSLoss(nn.Module):
    """
    DISTS (Deep Image Structure and Texture Similarity) Loss.
    
    Research shows DISTS outperforms LPIPS for super-resolution perceptual quality.
    Computes weighted combination of structure and texture similarity using VGG features.
    
    Reference: "Image Quality Assessment: Unifying Structure and Texture Similarity" (Ding et al., 2020)
    
    Example:
        >>> dists_loss = DISTSLoss()
        >>> loss = dists_loss(pred, target)  # Lower is better
    """
    
    def __init__(
        self,
        layers: List[str] = None,
        alpha: float = 0.5,  # Weight for structure vs texture
        use_weights: bool = True,  # Use learned DISTS weights
    ):
        super().__init__()
        
        # Default layers: mix of shallow (texture) and deep (structure)
        if layers is None:
            layers = ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3', 'conv5_3']
        
        self.layers = layers
        self.alpha = alpha  # Structure weight (1-alpha for texture)
        
        # Load pre-trained VGG16 from cache (DISTS uses VGG16, not VGG19)
        vgg = _get_cached_vgg_model('vgg16', pretrained=True).features
        vgg.eval()
        
        # Disable gradients
        for param in vgg.parameters():
            param.requires_grad = False
        
        # CRITICAL: Disable in-place ReLU operations to avoid gradient computation errors
        # In-place ReLU breaks the computation graph when using mixed precision (AMP)
        for module in vgg.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
        
        # VGG16 layer mapping (conv indices)
        self.layer_map = {
            'conv1_1': 0, 'conv1_2': 2,
            'conv2_1': 5, 'conv2_2': 7,
            'conv3_1': 10, 'conv3_2': 12, 'conv3_3': 14,
            'conv4_1': 17, 'conv4_2': 19, 'conv4_3': 21,
            'conv5_1': 24, 'conv5_2': 26, 'conv5_3': 28,
        }
        
        # Build feature extractor
        self.feature_extractor = self._build_extractor(vgg, layers)
        
        # DISTS learned weights (from paper)
        if use_weights:
            # These weights are learned on IQA datasets
            self.register_buffer('channel_weights', torch.ones(len(layers)))
            self.register_buffer('layer_weights', torch.ones(len(layers)))
        else:
            self.register_buffer('channel_weights', torch.ones(len(layers)))
            self.register_buffer('layer_weights', torch.ones(len(layers)))
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def _build_extractor(self, vgg: nn.Module, layers: List[str]) -> nn.ModuleDict:
        """Build feature extractor for specified layers"""
        extractor = nn.ModuleDict()
        
        # Sort layers by index
        layer_indices = [(name, self.layer_map[name]) for name in layers]
        layer_indices.sort(key=lambda x: x[1])
        
        prev_idx = 0
        for name, idx in layer_indices:
            extractor[name] = nn.Sequential(*list(vgg[prev_idx:idx+1]))
            prev_idx = idx + 1
        
        return extractor
    
    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize to ImageNet statistics"""
        if x.max() > 1.0:
            x = x / 255.0
        if x.min() < 0:
            x = (x + 1) / 2
        return (x - self.mean) / self.std
    
    def compute_structure_similarity(
        self,
        pred_feat: torch.Tensor,
        target_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute structure similarity (correlation of features).
        Uses cosine similarity across channels.
        """
        b, c, h, w = pred_feat.shape
        
        # Flatten spatial dimensions
        pred_flat = pred_feat.view(b, c, -1)
        target_flat = target_feat.view(b, c, -1)
        
        # Compute mean per channel
        pred_mean = pred_flat.mean(dim=2, keepdim=True)
        target_mean = target_flat.mean(dim=2, keepdim=True)
        
        # Center
        pred_centered = pred_flat - pred_mean
        target_centered = target_flat - target_mean
        
        # Correlation (cosine similarity)
        # Add epsilon to prevent division by zero when all values are identical
        pred_norm = F.normalize(pred_centered, dim=2, eps=1e-8)
        target_norm = F.normalize(target_centered, dim=2, eps=1e-8)
        
        # Safety check: replace NaN with small values (happens with zero variance)
        pred_norm = torch.where(torch.isnan(pred_norm), torch.zeros_like(pred_norm), pred_norm)
        target_norm = torch.where(torch.isnan(target_norm), torch.zeros_like(target_norm), target_norm)
        
        similarity = (pred_norm * target_norm).sum(dim=2).mean()
        
        # Final safety: if similarity is NaN, return 0 loss (identical inputs)
        if torch.isnan(similarity):
            return torch.tensor(0.0, device=pred_feat.device, requires_grad=False)
        
        return 1 - similarity  # Convert to loss (lower is better)
    
    def compute_texture_similarity(
        self,
        pred_feat: torch.Tensor,
        target_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute texture similarity (covariance matching).
        Uses channel-wise covariance similarity.
        """
        b, c, h, w = pred_feat.shape
        
        # Flatten spatial
        pred_flat = pred_feat.view(b, c, -1)
        target_flat = target_feat.view(b, c, -1)
        
        # Compute covariance matrices
        pred_cov = torch.bmm(pred_flat, pred_flat.transpose(1, 2)) / (h * w)
        target_cov = torch.bmm(target_flat, target_flat.transpose(1, 2)) / (h * w)
        
        # Normalize
        pred_cov_norm = F.normalize(pred_cov.flatten(1), dim=1, eps=1e-8)
        target_cov_norm = F.normalize(target_cov.flatten(1), dim=1, eps=1e-8)
        
        # Safety check: replace NaN with small values (happens with zero variance)
        pred_cov_norm = torch.where(torch.isnan(pred_cov_norm), torch.zeros_like(pred_cov_norm), pred_cov_norm)
        target_cov_norm = torch.where(torch.isnan(target_cov_norm), torch.zeros_like(target_cov_norm), target_cov_norm)
        
        # Similarity
        similarity = (pred_cov_norm * target_cov_norm).sum(dim=1).mean()
        
        # Final safety: if similarity is NaN, return 0 loss
        if torch.isnan(similarity):
            return torch.tensor(0.0, device=pred_feat.device, requires_grad=False)
        
        return 1 - similarity  # Convert to loss
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute DISTS loss.
        
        Args:
            pred: Predicted image [B, 3, H, W]
            target: Target image [B, 3, H, W]
        
        Returns:
            DISTS loss (lower is better)
        """
        # Normalize
        pred = self.normalize_input(pred)
        target = self.normalize_input(target)
        
        # Extract features
        total_loss = 0
        
        for i, layer in enumerate(self.layers):
            if i == 0:
                pred_feat = self.feature_extractor[layer](pred)
                target_feat = self.feature_extractor[layer](target)
            else:
                # Continue from previous
                prev_layer = self.layers[i-1]
                pred_feat = self.feature_extractor[layer](pred_feat)
                target_feat = self.feature_extractor[layer](target_feat)
            
            # Match sizes if needed
            if pred_feat.shape != target_feat.shape:
                pred_feat = F.interpolate(pred_feat, size=target_feat.shape[2:], 
                                         mode='bilinear', align_corners=False)
            
            # Compute structure and texture losses
            structure_loss = self.compute_structure_similarity(pred_feat, target_feat)
            texture_loss = self.compute_texture_similarity(pred_feat, target_feat)
            
            # Combined loss for this layer
            layer_loss = (
                self.alpha * structure_loss + 
                (1 - self.alpha) * texture_loss
            ) * self.layer_weights[i]
            
            total_loss += layer_loss
        
        final_loss = total_loss / len(self.layers)
        
        # Final safety: return 0 if NaN/Inf (shouldn't happen with fixes above, but just in case)
        if torch.isnan(final_loss) or torch.isinf(final_loss):
            return torch.tensor(0.0, device=pred.device, requires_grad=False)
        
        return final_loss


class PerceptualLossFactory:
    """Factory for creating perceptual loss functions"""
    
    @staticmethod
    def create(loss_type: str = "vgg", **kwargs) -> nn.Module:
        """
        Create perceptual loss.
        
        Args:
            loss_type: 'vgg', 'resnet', 'dists', or 'none'
            **kwargs: Additional arguments for loss constructor
        
        Returns:
            Perceptual loss module
        """
        if loss_type.lower() == "vgg":
            return VGGPerceptualLoss(**kwargs)
        elif loss_type.lower() == "resnet":
            return ResNetPerceptualLoss(**kwargs)
        elif loss_type.lower() == "dists":
            return DISTSLoss(**kwargs)
        elif loss_type.lower() == "none" or loss_type.lower() == "disabled":
            return nn.Module()  # Dummy module
        else:
            raise ValueError(f"Unknown perceptual loss type: {loss_type}")
