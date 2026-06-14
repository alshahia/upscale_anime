"""
Temporal consistency losses for video super-resolution.
Ensures smooth transitions between consecutive frames.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConsistencyLoss(nn.Module):
    """
    Loss that enforces temporal consistency between consecutive frames.
    
    The idea: consecutive frames should have similar changes in
    prediction as they do in the input (or ground truth).
    """
    
    def __init__(self, flow_weight: float = 1.0, consistency_weight: float = 1.0):
        super().__init__()
        self.flow_weight = flow_weight
        self.consistency_weight = consistency_weight
    
    def forward(
        self,
        pred_curr: torch.Tensor,
        pred_prev: torch.Tensor,
        hr_curr: torch.Tensor,
        hr_prev: torch.Tensor,
    ) -> dict:
        """
        Compute temporal consistency loss.
        
        Args:
            pred_curr: Current frame prediction [B, C, H, W]
            pred_prev: Previous frame prediction [B, C, H, W]
            hr_curr: Current frame ground truth [B, C, H, W]
            hr_prev: Previous frame ground truth [B, C, H, W]
            
        Returns:
            Dict with losses
        """
        # Compute differences
        pred_diff = pred_curr - pred_prev
        hr_diff = hr_curr - hr_prev
        
        # Temporal consistency: prediction difference should match GT difference
        consistency_loss = F.l1_loss(pred_diff, hr_diff)
        
        # Also ensure overall smoothness
        pred_var = torch.var(pred_diff, dim=[2, 3]).mean()
        
        total = self.consistency_weight * consistency_loss + 0.1 * pred_var
        
        return {
            'temporal': total,
            'consistency': consistency_loss.detach(),
            'pred_variability': pred_var.detach(),
        }


class FlowGuidedTemporalLoss(nn.Module):
    """
    Temporal loss guided by optical flow.
    Warps previous frame using optical flow and compares.
    """
    
    def __init__(self, flow_weight: float = 1.0):
        super().__init__()
        self.flow_weight = flow_weight
    
    def warp(self, x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Warp image using optical flow.
        
        Args:
            x: Image to warp [B, C, H, W]
            flow: Optical flow [B, 2, H, W]
            
        Returns:
            Warped image [B, C, H, W]
        """
        b, c, h, w = x.shape
        
        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=x.device),
            torch.arange(w, device=x.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).float()  # [2, H, W]
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1)  # [B, 2, H, W]
        
        # Add flow to grid
        grid = grid + flow
        
        # Normalize to [-1, 1]
        grid[:, 0] = 2.0 * grid[:, 0] / (w - 1) - 1.0
        grid[:, 1] = 2.0 * grid[:, 1] / (h - 1) - 1.0
        
        # Permute for grid_sample [B, H, W, 2]
        grid = grid.permute(0, 2, 3, 1)
        
        # Warp
        warped = F.grid_sample(
            x,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )
        
        return warped
    
    def compute_flow_simple(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute simple optical flow between two frames.
        This is a placeholder - in practice, use a real flow network.
        
        Args:
            x1: First frame [B, C, H, W]
            x2: Second frame [B, C, H, W]
            
        Returns:
            Flow [B, 2, H, W]
        """
        # Simplified: assume small motion, just return zeros
        # In practice, use RAFT or similar
        b, c, h, w = x1.shape
        return torch.zeros(b, 2, h, w, device=x1.device)
    
    def forward(
        self,
        pred_curr: torch.Tensor,
        pred_prev: torch.Tensor,
        hr_curr: torch.Tensor,
        hr_prev: torch.Tensor,
        flow: torch.Tensor = None,
    ) -> dict:
        """
        Compute flow-guided temporal loss.
        
        Args:
            pred_curr: Current frame prediction
            pred_prev: Previous frame prediction
            hr_curr: Current frame ground truth
            hr_prev: Previous frame ground truth
            flow: Pre-computed optical flow (optional)
            
        Returns:
            Dict with losses
        """
        b, c, h, w = pred_curr.shape
        
        # Compute flow if not provided
        if flow is None:
            flow = self.compute_flow_simple(hr_prev, hr_curr)
        
        # Warp previous prediction
        warped_pred_prev = self.warp(pred_prev, flow)
        
        # Warp previous ground truth
        warped_hr_prev = self.warp(hr_prev, flow)
        
        # Consistency: warped previous should match current
        pred_loss = F.l1_loss(pred_curr, warped_pred_prev)
        
        # Also check ground truth consistency (sanity check)
        gt_loss = F.l1_loss(hr_curr, warped_hr_prev)
        
        total = self.flow_weight * pred_loss
        
        return {
            'flow_temporal': total,
            'pred_warp_error': pred_loss.detach(),
            'gt_warp_error': gt_loss.detach(),
        }


class SceneChangeDetector:
    """
    Detects scene changes in video sequences.
    Used to avoid applying temporal loss across scene boundaries.
    """
    
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
    
    def detect(self, frame1: torch.Tensor, frame2: torch.Tensor) -> bool:
        """
        Detect if there's a scene change between two frames.
        
        Args:
            frame1: First frame [B, C, H, W]
            frame2: Second frame [B, C, H, W]
            
        Returns:
            True if scene change detected
        """
        # Compute frame difference
        diff = F.l1_loss(frame1, frame2)
        
        # Threshold for scene change
        return diff > self.threshold
    
    def find_boundaries(self, frames: list) -> list:
        """
        Find all scene boundaries in a sequence.
        
        Args:
            frames: List of frames
            
        Returns:
            List of boundary indices
        """
        boundaries = []
        
        for i in range(len(frames) - 1):
            if self.detect(frames[i], frames[i + 1]):
                boundaries.append(i)
        
        return boundaries


class VideoAwareLoss(nn.Module):
    """
    Wrapper that applies temporal loss only when appropriate.
    Skips temporal loss across scene boundaries.
    """
    
    def __init__(
        self,
        temporal_loss: nn.Module,
        scene_detector: SceneChangeDetector = None,
    ):
        super().__init__()
        self.temporal_loss = temporal_loss
        self.scene_detector = scene_detector or SceneChangeDetector()
        self.prev_frame = None
        self.prev_hr = None
    
    def forward(
        self,
        pred: torch.Tensor,
        hr: torch.Tensor,
        reset: bool = False,
    ) -> dict:
        """
        Compute video-aware loss.
        
        Args:
            pred: Current prediction
            hr: Current ground truth
            reset: If True, treat as new scene start
            
        Returns:
            Dict with losses
        """
        losses = {'temporal': torch.tensor(0.0, device=pred.device)}
        
        # Check if we have previous frame
        if self.prev_frame is not None and not reset:
            # Check for scene change
            is_scene_change = self.scene_detector.detect(self.prev_hr, hr)
            
            if not is_scene_change:
                # Apply temporal loss
                temporal_dict = self.temporal_loss(pred, self.prev_frame, hr, self.prev_hr)
                losses.update(temporal_dict)
        
        # Store current frame for next iteration
        self.prev_frame = pred.detach()
        self.prev_hr = hr.detach()
        
        return losses
    
    def reset(self):
        """Reset stored frames."""
        self.prev_frame = None
        self.prev_hr = None
