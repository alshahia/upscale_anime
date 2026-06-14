#!/usr/bin/env python
"""
Master pipeline script for small dataset super-resolution training.
Orchestrates the entire workflow: validation → pre-training → fine-tuning → benchmarking → export.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional
import time


class PipelineRunner:
    """Manages the complete training pipeline."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.results = {}
        self.start_time = None
    
    def run_command(self, cmd: List[str], description: str) -> bool:
        """Run a command and log output."""
        print(f"\n{'='*70}")
        print(f"[PIN] {description}")
        print(f"{'='*70}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'-'*70}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            return True
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            return False
    
    def step1_validate_dataset(self) -> bool:
        """Step 1: Validate and clean dataset."""
        data_dir = self.config.get('data_dir', 'data/anime_hr')
        
        cmd = [
            'python', 'scripts/validate_dataset.py',
            '--data-dir', data_dir,
            '--num-workers', str(self.config.get('num_workers', 4))
        ]
        
        if self.config.get('remove_duplicates', True):
            cmd.append('--remove-duplicates')
        if self.config.get('remove_invalid', True):
            cmd.append('--remove-invalid')
        
        success = self.run_command(cmd, "Step 1/5: Dataset Validation")
        
        if success:
            # Check validation report
            report_path = Path(data_dir) / 'validation_report.json'
            if report_path.exists():
                with open(report_path) as f:
                    report = json.load(f)
                    self.results['validation'] = report.get('summary', {})
                    print(f"[OK] Valid images: {report.get('summary', {}).get('valid', 0)}")
        
        return success
    
    def step2_pretrain(self) -> bool:
        """Step 2: Self-supervised pre-training."""
        if not self.config.get('run_pretrain', True):
            print("\n[SKIP] Skipping pre-training (disabled in config)")
            return True
        
        config_file = self.config.get('pretrain_config', 'configs/pretrain_selfsupervised.yaml')
        name = self.config.get('pretrain_name', 'pretrain_anime')
        epochs = self.config.get('pretrain_epochs', 100)
        
        cmd = [
            'python', 'scripts/train.py',
            '--config', config_file,
            '--name', name,
            '--epochs', str(epochs)
        ]
        
        success = self.run_command(cmd, "Step 2/5: Self-Supervised Pre-Training")
        
        if success:
            self.results['pretrain'] = {
                'config': config_file,
                'name': name,
                'epochs': epochs,
                'checkpoint': f'checkpoints/{name}/best_model.pth'
            }
        
        return success
    
    def step3_finetune(self) -> bool:
        """Step 3: Transfer learning fine-tuning."""
        if not self.config.get('run_finetune', True):
            print("\n[SKIP] Skipping fine-tuning (disabled in config)")
            return True
        
        config_file = self.config.get('finetune_config', 'configs/finetune_transfer.yaml')
        name = self.config.get('finetune_name', 'finetune_anime')
        epochs = self.config.get('finetune_epochs', 200)
        
        cmd = [
            'python', 'scripts/train.py',
            '--config', config_file,
            '--name', name,
            '--epochs', str(epochs)
        ]
        
        # Use pre-trained checkpoint if available
        if 'pretrain' in self.results:
            pretrained = self.results['pretrain']['checkpoint']
            if Path(pretrained).exists():
                cmd.extend(['--checkpoint', pretrained])
        
        success = self.run_command(cmd, "Step 3/5: Transfer Learning Fine-Tuning")
        
        if success:
            self.results['finetune'] = {
                'config': config_file,
                'name': name,
                'epochs': epochs,
                'checkpoint': f'checkpoints/{name}/best_model.pth'
            }
        
        return success
    
    def step4_benchmark(self) -> bool:
        """Step 4: Benchmark the model."""
        if not self.config.get('run_benchmark', True):
            print("\n[SKIP] Skipping benchmark (disabled in config)")
            return True
        
        # Determine which checkpoint to benchmark
        checkpoint = None
        if 'finetune' in self.results:
            checkpoint = self.results['finetune']['checkpoint']
        elif 'pretrain' in self.results:
            checkpoint = self.results['pretrain']['checkpoint']
        else:
            checkpoint = self.config.get('benchmark_checkpoint', 'checkpoints/best_model.pth')
        
        if not Path(checkpoint).exists():
            print(f"[ERROR] Checkpoint not found: {checkpoint}")
            return False
        
        lr_dir = self.config.get('test_lr_dir', 'data/test_lr')
        hr_dir = self.config.get('test_hr_dir', 'data/test_hr')
        output = self.config.get('benchmark_output', 'results/benchmark.json')
        
        cmd = [
            'python', 'scripts/benchmark_model.py',
            '--checkpoint', checkpoint,
            '--lr-dir', lr_dir,
            '--hr-dir', hr_dir,
            '--output', output
        ]
        
        success = self.run_command(cmd, "Step 4/5: Benchmarking")
        
        if success and Path(output).exists():
            with open(output) as f:
                benchmark = json.load(f)
                self.results['benchmark'] = benchmark
                if 'average' in benchmark:
                    avg = benchmark['average']
                    print(f"\n[RESULTS]:")
                    print(f"   PSNR:  {avg.get('psnr', 0):.2f} dB")
                    print(f"   SSIM:  {avg.get('ssim', 0):.4f}")
                    print(f"   LPIPS: {avg.get('lpips', 0):.4f}")
        
        return success
    
    def step5_export(self) -> bool:
        """Step 5: Export for production."""
        if not self.config.get('run_export', True):
            print("\n[SKIP] Skipping export (disabled in config)")
            return True
        
        # Determine which checkpoint to export
        checkpoint = None
        if 'finetune' in self.results:
            checkpoint = self.results['finetune']['checkpoint']
        elif 'pretrain' in self.results:
            checkpoint = self.results['pretrain']['checkpoint']
        else:
            checkpoint = self.config.get('export_checkpoint', 'checkpoints/best_model.pth')
        
        if not Path(checkpoint).exists():
            print(f"[ERROR] Checkpoint not found: {checkpoint}")
            return False
        
        output_dir = self.config.get('export_dir', 'production_models')
        formats = self.config.get('export_formats', 'onnx,torchscript,state_dict')
        
        cmd = [
            'python', 'scripts/export_model.py',
            '--checkpoint', checkpoint,
            '--output-dir', output_dir,
            '--formats', formats
        ]
        
        success = self.run_command(cmd, "Step 5/5: Production Export")
        
        if success:
            self.results['export'] = {
                'checkpoint': checkpoint,
                'output_dir': output_dir,
                'formats': formats
            }
        
        return success
    
    def run_pipeline(self) -> bool:
        """Run the complete pipeline."""
        print(f"\n{'='*70}")
        print(f"STARTING FULL TRAINING PIPELINE")
        print(f"{'='*70}\n")
        
        self.start_time = time.time()
        
        steps = [
            ("Dataset Validation", self.step1_validate_dataset),
            ("Pre-Training", self.step2_pretrain),
            ("Fine-Tuning", self.step3_finetune),
            ("Benchmarking", self.step4_benchmark),
            ("Export", self.step5_export),
        ]
        
        completed = 0
        failed = 0
        
        for name, step_func in steps:
            try:
                if step_func():
                    completed += 1
                else:
                    failed += 1
                    if self.config.get('stop_on_error', True):
                        print(f"\n[ERROR] Pipeline stopped due to error in: {name}")
                        break
            except Exception as e:
                print(f"\n[ERROR] Exception in {name}: {e}")
                failed += 1
                if self.config.get('stop_on_error', True):
                    break
        
        # Summary
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        
        print(f"\n{'='*70}")
        print(f"PIPELINE SUMMARY")
        print(f"{'='*70}")
        print(f"[OK] Completed: {completed}/{len(steps)} steps")
        print(f"[ERROR] Failed: {failed}/{len(steps)} steps")
        print(f"[TIME] Total time: {hours}h {minutes}m")
        print(f"{'='*70}")
        
        # Save results
        results_file = self.config.get('results_file', 'results/pipeline_results.json')
        Path(results_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump({
                'config': self.config,
                'results': self.results,
                'completed_steps': completed,
                'failed_steps': failed,
                'elapsed_time': elapsed
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        return failed == 0


def load_pipeline_config(config_path: str) -> Dict:
    """Load pipeline configuration from JSON or use defaults."""
    defaults = {
        'data_dir': 'data/anime_hr',
        'test_lr_dir': 'data/test_lr',
        'test_hr_dir': 'data/test_hr',
        'run_pretrain': True,
        'run_finetune': True,
        'run_benchmark': True,
        'run_export': True,
        'remove_duplicates': True,
        'remove_invalid': True,
        'num_workers': 4,
        'pretrain_epochs': 100,
        'finetune_epochs': 200,
        'stop_on_error': True,
        'export_formats': 'onnx,torchscript,state_dict'
    }
    
    if Path(config_path).exists():
        with open(config_path) as f:
            user_config = json.load(f)
            defaults.update(user_config)
    
    return defaults


def main():
    parser = argparse.ArgumentParser(
        description='Run complete training pipeline for anime SR',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with defaults
  python scripts/run_full_pipeline.py
  
  # Run with custom config
  python scripts/run_full_pipeline.py --config my_pipeline_config.json
  
  # Skip pre-training (use existing checkpoint)
  python scripts/run_full_pipeline.py --no-pretrain
  
  # Skip export
  python scripts/run_full_pipeline.py --no-export

Pipeline Config JSON format:
  {
    "data_dir": "data/anime_hr",
    "test_lr_dir": "data/test_lr",
    "test_hr_dir": "data/test_hr",
    "pretrain_epochs": 100,
    "finetune_epochs": 200,
    "export_formats": "onnx,torchscript"
  }
        """
    )
    
    parser.add_argument('--config', type=str, default='pipeline_config.json',
                       help='Pipeline configuration JSON file')
    parser.add_argument('--data-dir', type=str, help='Training data directory')
    parser.add_argument('--no-pretrain', action='store_true', help='Skip pre-training')
    parser.add_argument('--no-finetune', action='store_true', help='Skip fine-tuning')
    parser.add_argument('--no-benchmark', action='store_true', help='Skip benchmarking')
    parser.add_argument('--no-export', action='store_true', help='Skip export')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_pipeline_config(args.config)
    
    # Override with command line args
    if args.data_dir:
        config['data_dir'] = args.data_dir
    if args.no_pretrain:
        config['run_pretrain'] = False
    if args.no_finetune:
        config['run_finetune'] = False
    if args.no_benchmark:
        config['run_benchmark'] = False
    if args.no_export:
        config['run_export'] = False
    
    # Run pipeline
    runner = PipelineRunner(config)
    success = runner.run_pipeline()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
