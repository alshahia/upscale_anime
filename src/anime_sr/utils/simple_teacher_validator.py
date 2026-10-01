"""
Simple Teacher Model Validation

Basic validation for available teacher models without requiring external downloads.
Focuses on detecting NaN/Inf outputs and basic health checks.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import warnings


class SimpleTeacherValidator:
    """
    Simple validation for teacher models that are already available.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.validation_results = {}
    
    def validate_available_teacher(
        self, 
        teacher_model: nn.Module, 
        teacher_name: str,
        input_shape: Tuple[int, int, int, int] = (1, 3, 64, 64)
    ) -> Dict[str, any]:
        """
        Validate a teacher model that's already loaded.
        
        Args:
            teacher_model: Teacher model to validate
            teacher_name: Name of the teacher
            input_shape: Input tensor shape (B, C, H, W)
            
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
            'recommendations': []
        }
        
        try:
            # Move model to device
            teacher_model = teacher_model.to(self.device)
            teacher_model.eval()
            
            # Create test input
            test_input = torch.randn(*input_shape).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                try:
                    output = teacher_model(test_input)
                except Exception as e:
                    results['is_healthy'] = False
                    results['issues'].append(f"Forward pass failed: {e}")
                    print(f"    [ERROR] Forward pass failed: {e}")
                    return results
            
            # Check output properties
            if torch.isnan(output).any():
                results['is_healthy'] = False
                results['issues'].append("NaN values detected in output")
                print(f"    [ERROR] NaN values detected")
            
            if torch.isinf(output).any():
                results['is_healthy'] = False
                results['issues'].append("Inf values detected in output")
                print(f"    [ERROR] Inf values detected")
            
            # Check output statistics
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
            
            # Check for problematic outputs
            zero_ratio = results['output_stats']['zero_ratio']
            if zero_ratio > 0.9:
                results['warnings'].append(f"High zero ratio: {zero_ratio:.1%}")
                if zero_ratio > 0.99:
                    results['issues'].append(f"Extremely high zero ratio: {zero_ratio:.1%}")
            
            # Check output range
            output_range = results['output_stats']['max'] - results['output_stats']['min']
            if output_range < 1e-6:
                results['issues'].append("Output range is too small (near-constant)")
            elif output_range < 1e-3:
                results['warnings'].append(f"Small output range: {output_range:.6f}")
            
            # Generate recommendations
            if results['is_healthy']:
                if results['warnings']:
                    results['recommendations'].append("Use with caution - monitor during training")
                else:
                    results['recommendations'].append("Teacher is healthy and ready for use")
            else:
                results['recommendations'].append("Do not use this teacher for training")
            
            # Print results
            print(f"    Output range: [{results['output_stats']['min']:.3f}, {results['output_stats']['max']:.3f}]")
            print(f"    Mean: {results['output_stats']['mean']:.3f}, Std: {results['output_stats']['std']:.3f}")
            print(f"    Zero ratio: {zero_ratio:.1%}")
            
            if results['is_healthy']:
                print(f"    [OK] Teacher {teacher_name} is healthy")
            else:
                print(f"    [ERROR] Teacher {teacher_name} has issues:")
                for issue in results['issues']:
                    print(f"      - {issue}")
            
            if results['warnings']:
                print(f"    [WARN]  Warnings:")
                for warning in results['warnings']:
                    print(f"      - {warning}")
            
        except Exception as e:
            results['is_healthy'] = False
            results['issues'].append(f"Validation failed: {e}")
            print(f"    [ERROR] Validation failed: {e}")
        
        # Store results
        self.validation_results[teacher_name] = results
        
        return results
    
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
            results = self.validate_available_teacher(model, name, input_shape)
            
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


def create_safe_teacher_config(config: Dict) -> Dict:
    """
    Create a safe teacher configuration by filtering out problematic teachers.
    
    Args:
        config: Training configuration
        
    Returns:
        Updated config with safe teacher settings
    """
    print("CREATING SAFE TEACHER CONFIGURATION")
    print("=" * 50)
    
    # Extract teacher configurations
    stage1_config = config.get('training', {}).get('stage1', {})
    teacher_configs = stage1_config.get('teachers', [])
    
    if not teacher_configs:
        print("No teachers configured")
        return config
    
    # Create a conservative teacher list (only SwinIR which is known to work)
    safe_teachers = []
    
    for teacher_config in teacher_configs:
        teacher_name = teacher_config.get('name')
        
        # Only include SwinIR as it's known to work
        if teacher_name == 'swinir':
            safe_teachers.append(teacher_config)
            print(f"[OK] Including safe teacher: {teacher_name}")
        else:
            print(f"[ERROR] Excluding risky teacher: {teacher_name} (known to have issues)")
    
    # Update configuration
    if safe_teachers:
        config['training']['stage1']['teachers'] = safe_teachers
        print(f"[OK] Updated config with {len(safe_teachers)} safe teachers")
    else:
        # Disable knowledge distillation if no safe teachers
        config['training']['stage1']['teachers'] = []
        config['training']['stage1']['distillation_weight'] = 0.0
        
        # Check if feature_distillation exists before disabling
        if 'feature_distillation' in config['training']['stage1']:
            config['training']['stage1']['feature_distillation']['enabled'] = False
        
        print("[ERROR] No safe teachers found - disabled knowledge distillation")
    
    # Reduce distillation weights for stability
    if safe_teachers:
        config['training']['stage1']['distillation_weight'] = 0.1  # Reduced
        
        # Check if feature_distillation exists
        if 'feature_distillation' in config['training']['stage1']:
            config['training']['stage1']['feature_distillation']['weight'] = 0.05  # Reduced
        
        print("[OK] Reduced distillation weights for stability")
    
    return config


if __name__ == "__main__":
    # Test the safe config creation
    import yaml
    
    # Load config
    with open("configs/finetune_stable.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Create safe teacher config
    safe_config = create_safe_teacher_config(config)
    
    # Save safe config
    with open("configs/finetune_stable_safe.yaml", 'w') as f:
        yaml.dump(safe_config, f, default_flow_style=False)
    
    print(f"\n[OK] Saved safe config to: configs/finetune_stable_safe.yaml")
