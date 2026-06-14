"""
Advanced training metrics monitoring.
Tracks teacher disagreement, frequency analysis, and per-teacher contributions.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


class TeacherDisagreementMonitor:
    """
    Monitor teacher output disagreement and correlations.
    Helps understand which teachers are most/least reliable.
    """
    
    def __init__(self, num_teachers: int, log_interval: int = 100):
        self.num_teachers = num_teachers
        self.log_interval = log_interval
        self.step_count = 0
        
        # Running statistics
        self.teacher_means = defaultdict(list)
        self.teacher_stds = defaultdict(list)
        self.pairwise_correlations = defaultdict(list)
        self.disagreement_map = []
    
    def compute_disagreement(
        self,
        teacher_outputs: List[torch.Tensor],
    ) -> Dict[str, float]:
        """
        Compute disagreement metrics between teachers.
        
        Args:
            teacher_outputs: List of [B, C, H, W] teacher predictions
            
        Returns:
            Dictionary of disagreement metrics
        """
        metrics = {}
        
        # Stack outputs: [N, B, C, H, W]
        stack = torch.stack(teacher_outputs)
        
        # Overall variance (disagreement)
        variance = torch.var(stack, dim=0).mean().item()
        metrics['overall_variance'] = variance
        
        # Per-teacher statistics
        for i, output in enumerate(teacher_outputs):
            self.teacher_means[i].append(output.mean().item())
            self.teacher_stds[i].append(output.std().item())
        
        # Pairwise correlations
        correlations = []
        for i in range(self.num_teachers):
            for j in range(i + 1, self.num_teachers):
                # Flatten for correlation
                flat_i = teacher_outputs[i].flatten()
                flat_j = teacher_outputs[j].flatten()
                
                # Pearson correlation
                corr = torch.corrcoef(torch.stack([flat_i, flat_j]))[0, 1].item()
                correlations.append(corr)
                self.pairwise_correlations[f"{i}-{j}"].append(corr)
        
        metrics['mean_correlation'] = np.mean(correlations)
        metrics['min_correlation'] = np.min(correlations)
        
        # Identify most/least agreed regions
        spatial_variance = torch.var(stack, dim=0).mean(dim=1)  # [B, H, W]
        metrics['max_disagreement'] = spatial_variance.max().item()
        metrics['disagreement_90th'] = torch.quantile(
            spatial_variance.flatten(), 0.9
        ).item()
        
        return metrics
    
    def get_teacher_rankings(self) -> Dict[int, float]:
        """
        Rank teachers by consistency (lower variance = more consistent).
        
        Returns:
            Dict mapping teacher index to consistency score
        """
        rankings = {}
        
        for i in range(self.num_teachers):
            if len(self.teacher_stds[i]) > 0:
                # Lower std = more consistent
                avg_std = np.mean(self.teacher_stds[i])
                rankings[i] = 1.0 / (avg_std + 1e-8)
        
        return rankings
    
    def reset(self):
        """Reset all statistics."""
        self.teacher_means.clear()
        self.teacher_stds.clear()
        self.pairwise_correlations.clear()
        self.disagreement_map.clear()
        self.step_count = 0
    
    def should_log(self) -> bool:
        """Check if should log this step."""
        self.step_count += 1
        return self.step_count % self.log_interval == 0


class FrequencyAnalysisMonitor:
    """
    Monitor frequency-domain training metrics.
    Tracks LL, LH, HL, HH wavelet components separately.
    """
    
    def __init__(self, wavelet_levels: int = 3, log_interval: int = 100):
        self.wavelet_levels = wavelet_levels
        self.log_interval = log_interval
        self.step_count = 0
        
        # Running statistics per band
        self.ll_losses = []
        self.lh_losses = []
        self.hl_losses = []
        self.hh_losses = []
        self.psnr_per_band = defaultdict(list)
    
    def dwt_haar(self, x: torch.Tensor, levels: int = 1):
        """Simple Haar wavelet transform."""
        coeffs = []
        current = x
        
        for _ in range(levels):
            b, c, h, w = current.shape
            
            # Pad if odd
            if h % 2 == 1:
                current = F.pad(current, (0, 0, 0, 1))
            if w % 2 == 1:
                current = F.pad(current, (0, 1, 0, 0))
            
            # Unfold
            h2, w2 = current.shape[-2] // 2, current.shape[-1] // 2
            x_unfold = current.view(b, c, h2, 2, w2, 2).permute(0, 1, 2, 4, 3, 5)
            x_unfold = x_unfold.reshape(b, c, h2, w2, 4)
            
            # Haar coefficients
            a = x_unfold[..., 0]
            b = x_unfold[..., 1]
            c = x_unfold[..., 2]
            d = x_unfold[..., 3]
            
            ll = (a + b + c + d) / 2.0
            lh = (a + b - c - d) / 2.0  # Horizontal
            hl = (a - b + c - d) / 2.0  # Vertical
            hh = (a - b - c + d) / 2.0  # Diagonal
            
            coeffs.append((ll, (lh, hl, hh)))
            current = ll
        
        return coeffs
    
    def compute_frequency_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute frequency-domain metrics.
        
        Args:
            pred: Predictions [B, C, H, W]
            target: Targets [B, C, H, W]
            
        Returns:
            Dict of frequency metrics
        """
        metrics = {}
        
        # Wavelet decomposition
        pred_coeffs = self.dwt_haar(pred, self.wavelet_levels)
        target_coeffs = self.dwt_haar(target, self.wavelet_levels)
        
        # Compute loss per band
        for level_idx, (pred_level, target_level) in enumerate(
            zip(pred_coeffs, target_coeffs)
        ):
            pred_ll, (pred_lh, pred_hl, pred_hh) = pred_level
            target_ll, (target_lh, target_hl, target_hh) = target_level
            
            # LL (low frequency)
            ll_loss = F.l1_loss(pred_ll, target_ll).item()
            metrics[f'll_loss_l{level_idx}'] = ll_loss
            self.ll_losses.append(ll_loss)
            
            # LH (horizontal edges)
            lh_loss = F.l1_loss(pred_lh, target_lh).item()
            metrics[f'lh_loss_l{level_idx}'] = lh_loss
            self.lh_losses.append(lh_loss)
            
            # HL (vertical edges)
            hl_loss = F.l1_loss(pred_hl, target_hl).item()
            metrics[f'hl_loss_l{level_idx}'] = hl_loss
            self.hl_losses.append(hl_loss)
            
            # HH (diagonal edges)
            hh_loss = F.l1_loss(pred_hh, target_hh).item()
            metrics[f'hh_loss_l{level_idx}'] = hh_loss
            self.hh_losses.append(hh_loss)
            
            # PSNR per band
            for name, p, t in [
                ('ll', pred_ll, target_ll),
                ('lh', pred_lh, target_lh),
                ('hl', pred_hl, target_hl),
                ('hh', pred_hh, target_hh),
            ]:
                mse = F.mse_loss(p, t)
                psnr = -10 * torch.log10(mse + 1e-10)
                self.psnr_per_band[f"{name}_l{level_idx}"].append(psnr.item())
        
        # Summary stats
        metrics['mean_low_freq_loss'] = np.mean(self.ll_losses[-10:])
        metrics['mean_high_freq_loss'] = np.mean(
            self.lh_losses[-10:] + self.hl_losses[-10:] + self.hh_losses[-10:]
        )
        
        return metrics
    
    def get_frequency_summary(self) -> Dict[str, float]:
        """Get summary of frequency performance."""
        summary = {}
        
        if self.ll_losses:
            summary['avg_ll_loss'] = np.mean(self.ll_losses)
            summary['avg_lh_loss'] = np.mean(self.lh_losses)
            summary['avg_hl_loss'] = np.mean(self.hl_losses)
            summary['avg_hh_loss'] = np.mean(self.hh_losses)
            
            # Which band needs most attention
            losses = [
                ('ll', summary['avg_ll_loss']),
                ('lh', summary['avg_lh_loss']),
                ('hl', summary['avg_hl_loss']),
                ('hh', summary['avg_hh_loss']),
            ]
            worst_band = max(losses, key=lambda x: x[1])
            summary['weakest_band'] = worst_band[0]
        
        return summary
    
    def reset(self):
        """Reset all statistics."""
        self.ll_losses.clear()
        self.lh_losses.clear()
        self.hl_losses.clear()
        self.hh_losses.clear()
        self.psnr_per_band.clear()
        self.step_count = 0


class TeacherContributionMonitor:
    """
    Monitor how much each teacher contributes to final output.
    For adaptive aggregation with gating.
    """
    
    def __init__(self, num_teachers: int, log_interval: int = 100):
        self.num_teachers = num_teachers
        self.log_interval = log_interval
        self.step_count = 0
        
        # Running average of gate values
        self.gate_means = defaultdict(list)
        self.gate_maxs = defaultdict(list)
        self.dominant_teacher_counts = defaultdict(int)
    
    def log_gate_values(self, gate_values: torch.Tensor):
        """
        Log teacher gate values from adaptive aggregation.
        
        Args:
            gate_values: [B, num_teachers] softmax weights
        """
        # Per-teacher statistics
        for i in range(self.num_teachers):
            self.gate_means[i].append(gate_values[:, i].mean().item())
            self.gate_maxs[i].append(gate_values[:, i].max().item())
        
        # Count dominant teacher per sample
        dominant = gate_values.argmax(dim=1)
        for d in dominant.cpu().numpy():
            self.dominant_teacher_counts[int(d)] += 1
    
    def get_contribution_summary(self) -> Dict:
        """Get summary of teacher contributions."""
        summary = {}
        
        # Average gate values
        avg_gates = {}
        for i in range(self.num_teachers):
            if self.gate_means[i]:
                avg_gates[f'teacher_{i}'] = np.mean(self.gate_means[i])
        
        summary['average_gate_values'] = avg_gates
        
        # Dominant teacher distribution
        total = sum(self.dominant_teacher_counts.values())
        if total > 0:
            dominant_pct = {
                f'teacher_{k}': v / total * 100
                for k, v in self.dominant_teacher_counts.items()
            }
            summary['dominant_percentage'] = dominant_pct
        
        # Most/least used teachers
        if avg_gates:
            sorted_teachers = sorted(avg_gates.items(), key=lambda x: x[1], reverse=True)
            summary['most_influential'] = sorted_teachers[0][0]
            summary['least_influential'] = sorted_teachers[-1][0]
        
        return summary
    
    def reset(self):
        """Reset all statistics."""
        self.gate_means.clear()
        self.gate_maxs.clear()
        self.dominant_teacher_counts.clear()
        self.step_count = 0


class TrainingMetricsCollector:
    """
    Collect all advanced metrics in one place.
    Integrates with TensorBoard logging.
    """
    
    def __init__(
        self,
        num_teachers: int,
        wavelet_levels: int = 3,
        log_interval: int = 100,
    ):
        self.disagreement_monitor = TeacherDisagreementMonitor(
            num_teachers, log_interval
        )
        self.frequency_monitor = FrequencyAnalysisMonitor(
            wavelet_levels, log_interval
        )
        self.contribution_monitor = TeacherContributionMonitor(
            num_teachers, log_interval
        )
        
        self.log_interval = log_interval
        self.step = 0
    
    def log_batch(
        self,
        teacher_outputs: List[torch.Tensor],
        student_output: torch.Tensor,
        target: torch.Tensor,
        gate_values: torch.Tensor = None,
    ) -> Dict[str, float]:
        """
        Log metrics for a training batch.
        
        Returns:
            Dictionary of metrics to log
        """
        self.step += 1
        metrics = {}
        
        if self.step % self.log_interval == 0:
            # Teacher disagreement
            disagreement = self.disagreement_monitor.compute_disagreement(
                teacher_outputs
            )
            metrics.update({f'disagreement/{k}': v for k, v in disagreement.items()})
            
            # Frequency analysis
            frequency = self.frequency_monitor.compute_frequency_metrics(
                student_output, target
            )
            metrics.update({f'frequency/{k}': v for k, v in frequency.items()})
            
            # Teacher contributions
            if gate_values is not None:
                self.contribution_monitor.log_gate_values(gate_values)
                contribution = self.contribution_monitor.get_contribution_summary()
                if 'average_gate_values' in contribution:
                    for k, v in contribution['average_gate_values'].items():
                        metrics[f'contribution/{k}'] = v
        
        return metrics
    
    def get_summary(self) -> Dict:
        """Get summary of all monitored metrics."""
        return {
            'disagreement': self.disagreement_monitor.get_teacher_rankings(),
            'frequency': self.frequency_monitor.get_frequency_summary(),
            'contribution': self.contribution_monitor.get_contribution_summary(),
        }
    
    def reset(self):
        """Reset all monitors."""
        self.disagreement_monitor.reset()
        self.frequency_monitor.reset()
        self.contribution_monitor.reset()
        self.step = 0
