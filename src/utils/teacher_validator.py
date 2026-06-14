"""
Teacher Model Validation and Health Monitoring

Validates teacher models before training to ensure they produce stable outputs.
Filters out problematic teachers that produce NaN, Inf, or abnormal outputs.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import warnings
from pathlib import Path


class TeacherValidator:
    """
    Validates teacher models for knowledge distillation.
    
    Checks for:
    - NaN/Inf outputs
    - Zero/near-zero outputs  
    - Abnormal output ranges
    - Memory usage issues
    - Forward pass stability
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.validation_results = {}
    
    def validate_teacher(
        self, 
        teacher_model: nn.Module, 
        teacher_name: str,
        input_shape: Tuple[int, int, int, int] = (1, 3, 64, 64),
        num_trials: int = 3
    ) -> Dict[str, any]:
        """
        Validate a single teacher model.
        
        Args:
            teacher_model: Teacher model to validate
            teacher_name: Name of the teacher
            input_shape: Input tensor shape (B, C, H, W)
            num_trials: Number of validation trials
            
        Returns:
            Validation results dictionary
        """
        print(f"Validating teacher: {teacher_name}")
        
        results = {
            'teacher_name': teacher_name,
            'is_healthy': True,
            'issues': [],
            'warnings': [],
            'output_stats': {},
            'memory_usage': {},
            'recommendations': []
        }
        
        try:
            # Move model to device
            teacher_model = teacher_model.to(self.device)
            teacher_model.eval()
            
            # Test with different inputs
            for trial in range(num_trials):
                print(f"  Trial {trial + 1}/{num_trials}")
                
                # Create test input
                test_input = torch.randn(*input_shape).to(self.device)
                
                # Measure memory before
                if torch.cuda.is_available() and self.device.startswith('cuda'):
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    mem_before = torch.cuda.memory_allocated()
                
                # Forward pass
                with torch.no_grad():
                    try:
                        output = teacher_model(test_input)
                    except Exception as e:
                        results['is_healthy'] = False
                        results['issues'].append(f"Forward pass failed: {e}")
                        print(f"    [ERROR] Forward pass failed: {e}")
                        continue
                
                # Check output properties
                output_issues = self._check_output_health(output, trial)
                results['issues'].extend(output_issues['issues'])
                results['warnings'].extend(output_issues['warnings'])
                
                # Store output statistics
                if trial == 0:  # Store stats from first trial
                    results['output_stats'] = {
                        'shape': list(output.shape),
                        'min': output.min().item(),
                        'max': output.max().item(),
                        'mean': output.mean().item(),
                        'std': output.std().item(),
                        'has_nan': torch.isnan(output).any().item(),
                        'has_inf': torch.isinf(output).any().item(),
                        'zero_ratio': (output == 0).float().mean().item(),
                    }
                
                # Check memory usage
                if torch.cuda.is_available() and self.device.startswith('cuda'):
                    mem_after = torch.cuda.memory_allocated()
                    mem_peak = torch.cuda.max_memory_allocated()
                    
                    results['memory_usage'] = {
                        'allocated_mb': mem_after / 1024**2,
                        'peak_mb': mem_peak / 1024**2,
                        'increase_mb': (mem_after - mem_before) / 1024**2,
                    }
                    
                    # Check for excessive memory usage
                    if mem_peak > 2 * 1024**3:  # 2GB limit
                        results['warnings'].append(f"High memory usage: {mem_peak / 1024**3:.1f} GB")
                
                print(f"    Output range: [{output.min():.3f}, {output.max():.3f}]")
                print(f"    Mean: {output.mean():.3f}, Std: {output.std():.3f}")
                
                if output_issues['issues']:
                    print(f"    [ERROR] Issues: {output_issues['issues']}")
                if output_issues['warnings']:
                    print(f"    [WARN]  Warnings: {output_issues['warnings']}")
            
            # Overall health assessment
            if results['issues']:
                results['is_healthy'] = False
                print(f"  [ERROR] Teacher {teacher_name} has {len(results['issues'])} issues")
            else:
                print(f"  [OK] Teacher {teacher_name} is healthy")
            
            # Generate recommendations
            results['recommendations'] = self._generate_recommendations(results)
            
        except Exception as e:
            results['is_healthy'] = False
            results['issues'].append(f"Validation failed: {e}")
            print(f"  [ERROR] Validation failed: {e}")
        
        # Store results
        self.validation_results[teacher_name] = results
        
        return results
    
    def _check_output_health(self, output: torch.Tensor, trial: int) -> Dict[str, List[str]]:
        """Check output tensor for health issues."""
        issues = []
        warnings = []
        
        # Check for NaN/Inf
        if torch.isnan(output).any():
            issues.append("NaN values detected in output")
        
        if torch.isinf(output).any():
            issues.append("Inf values detected in output")
        
        # Check for zero/near-zero outputs
        zero_ratio = (output == 0).float().mean()
        if zero_ratio > 0.9:
            warnings.append(f"High zero ratio: {zero_ratio:.1%}")
        elif zero_ratio > 0.99:
            issues.append(f"Extremely high zero ratio: {zero_ratio:.1%}")
        
        # Check output range
        output_min, output_max = output.min().item(), output.max().item()
        
        if output_max - output_min < 1e-6:
            issues.append("Output range is too small (near-constant)")
        elif output_max - output_min < 1e-3:
            warnings.append(f"Small output range: {output_max - output_min:.6f}")
        
        # Check for extreme values
        if abs(output_min) > 1000 or abs(output_max) > 1000:
            warnings.append(f"Extreme output values: [{output_min:.3f}, {output_max:.3f}]")
        elif abs(output_min) > 100 or abs(output_max) > 100:
            issues.append(f"Very extreme output values: [{output_min:.3f}, {output_max:.3f}]")
        
        # Check for negative values (should be positive for images)
        negative_ratio = (output < 0).float().mean()
        if negative_ratio > 0.1:
            warnings.append(f"High negative ratio: {negative_ratio:.1%}")
        
        # Check for values > 1 (should be in [0,1] range for images)
        over_one_ratio = (output > 1).float().mean()
        if over_one_ratio > 0.1:
            warnings.append(f"High >1 ratio: {over_one_ratio:.1%}")
        
        return {'issues': issues, 'warnings': warnings}
    
    def _generate_recommendations(self, results: Dict[str, any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if not results['is_healthy']:
            recommendations.append("Do not use this teacher for training")
            
            if any("NaN" in issue for issue in results['issues']):
                recommendations.append("Check teacher model initialization and weights")
            
            if any("Inf" in issue for issue in results['issues']):
                recommendations.append("Check for gradient explosion in teacher model")
            
            if any("zero ratio" in issue for issue in results['issues']):
                recommendations.append("Teacher may be producing dead outputs")
        else:
            if results['warnings']:
                recommendations.append("Use with caution - monitor during training")
            
            if any("memory" in warning for warning in results['warnings']):
                recommendations.append("Consider reducing batch size when using this teacher")
            
            if any("range" in warning for warning in results['warnings']):
                recommendations.append("Monitor teacher output quality during training")
        
        return recommendations
    
    def filter_healthy_teachers(
        self, 
        teachers: List[Tuple[str, nn.Module]], 
        input_shape: Tuple[int, int, int, int] = (1, 3, 64, 64)
    ) -> List[Tuple[str, nn.Module]]:
        """
        Filter out unhealthy teachers from a list.
        
        Args:
            teachers: List of (name, model) tuples
            input_shape: Input tensor shape for validation
            
        Returns:
            List of healthy (name, model) tuples
        """
        print(f"\nFiltering {len(teachers)} teachers...")
        
        healthy_teachers = []
        unhealthy_teachers = []
        
        for name, model in teachers:
            results = self.validate_teacher(model, name, input_shape)
            
            if results['is_healthy']:
                healthy_teachers.append((name, model))
            else:
                unhealthy_teachers.append((name, model))
        
        print(f"\nTeacher Filtering Results:")
        print(f"  Healthy teachers: {len(healthy_teachers)}")
        for name, _ in healthy_teachers:
            print(f"    [OK] {name}")
        
        print(f"  Unhealthy teachers: {len(unhealthy_teachers)}")
        for name, _ in unhealthy_teachers:
            results = self.validation_results[name]
            print(f"    [ERROR] {name}: {', '.join(results['issues'][:2])}")
        
        return healthy_teachers
    
    def get_validation_report(self) -> str:
        """Get a formatted validation report."""
        if not self.validation_results:
            return "No validation results available."
        
        report = ["Teacher Validation Report", "=" * 50]
        
        for name, results in self.validation_results.items():
            status = "[OK] HEALTHY" if results['is_healthy'] else "[ERROR] UNHEALTHY"
            report.append(f"\n{name}: {status}")
            
            if results['output_stats']:
                stats = results['output_stats']
                report.append(f"  Output: {stats['shape']}, range: [{stats['min']:.3f}, {stats['max']:.3f}]")
                report.append(f"  Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
                report.append(f"  Zero ratio: {stats['zero_ratio']:.1%}")
            
            if results['issues']:
                report.append("  Issues:")
                for issue in results['issues']:
                    report.append(f"    - {issue}")
            
            if results['warnings']:
                report.append("  Warnings:")
                for warning in results['warnings']:
                    report.append(f"    - {warning}")
            
            if results['recommendations']:
                report.append("  Recommendations:")
                for rec in results['recommendations']:
                    report.append(f"    - {rec}")
        
        return "\n".join(report)


def validate_teacher_config(config: Dict, device: str = 'cpu') -> Dict[str, any]:
    """
    Validate teachers from training configuration.
    
    Args:
        config: Training configuration
        device: Device for validation
        
    Returns:
        Updated config with filtered teachers
    """
    print("VALIDATING TEACHER MODELS")
    print("=" * 50)
    
    validator = TeacherValidator(device)
    
    # Extract teacher configurations
    stage1_config = config.get('training', {}).get('stage1', {})
    teacher_configs = stage1_config.get('teachers', [])
    
    if not teacher_configs:
        print("No teachers configured for validation")
        return config
    
    # Load teacher models
    teachers = []
    from utils.pretrained_models import get_teacher_model_path, download_pretrained_model
    
    for teacher_config in teacher_configs:
        teacher_name = teacher_config.get('name')
        if not teacher_name:
            continue
        
        try:
            print(f"Loading teacher: {teacher_name}")
            
            # Get teacher model path
            model_path = get_teacher_model_path(teacher_name)
            if not model_path.exists():
                print(f"  Downloading teacher model...")
                download_pretrained_model(teacher_name, scale=4)
                model_path = get_teacher_model_path(teacher_name)
            
            # Load teacher model
            if teacher_name == 'swinir':
                from models.swinir import SwinIR
                teacher_model = SwinIR(scale=4)
            elif teacher_name == 'edsr':
                from models.edsr import EDSR
                teacher_model = EDSR(scale=4)
            elif teacher_name == 'rcan':
                from models.rcan import RCAN
                teacher_model = RCAN(scale=4)
            else:
                print(f"  [WARN]  Unknown teacher: {teacher_name}")
                continue
            
            # Load pretrained weights
            if model_path.exists():
                try:
                    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
                except Exception:
                    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                if 'state_dict' in checkpoint:
                    teacher_model.load_state_dict(checkpoint['state_dict'])
                else:
                    teacher_model.load_state_dict(checkpoint)
                print(f"  [OK] Loaded pretrained weights")
            else:
                print(f"  [WARN]  No pretrained weights found")
            
            teachers.append((teacher_name, teacher_model))
            
        except Exception as e:
            print(f"  [ERROR] Failed to load teacher {teacher_name}: {e}")
            continue
    
    if not teachers:
        print("No teachers loaded successfully")
        return config
    
    # Filter healthy teachers
    healthy_teachers = validator.filter_healthy_teachers(teachers)
    
    # Update configuration with healthy teachers only
    if healthy_teachers:
        healthy_teacher_configs = []
        for name, _ in healthy_teachers:
            # Find original config for this teacher
            for teacher_config in teacher_configs:
                if teacher_config.get('name') == name:
                    healthy_teacher_configs.append(teacher_config)
                    break
        
        config['training']['stage1']['teachers'] = healthy_teacher_configs
        print(f"[OK] Updated config with {len(healthy_teacher_configs)} healthy teachers")
    else:
        # Disable knowledge distillation if no healthy teachers
        config['training']['stage1']['teachers'] = []
        config['training']['stage1']['distillation_weight'] = 0.0
        config['training']['stage1']['feature_distillation']['enabled'] = False
        print("[ERROR] No healthy teachers found - disabled knowledge distillation")
    
    # Print validation report
    print("\n" + validator.get_validation_report())
    
    return config


if __name__ == "__main__":
    # Test the validator
    import yaml
    
    # Load config
    with open("configs/finetune_stable.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate teachers
    updated_config = validate_teacher_config(config)
    
    # Save updated config
    with open("configs/finetune_stable_validated.yaml", 'w') as f:
        yaml.dump(updated_config, f, default_flow_style=False)
    
    print(f"\n[OK] Saved validated config to: configs/finetune_stable_validated.yaml")
