#!/usr/bin/env python3
"""
Main training script for Anime Super-Resolution
Supports: Model A (NTIRE+MTKD+FAKD), Model B (Mamba-PAN), Ensemble training
"""
import sys
import argparse
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.config import Config
from data.dataloader import build_dataloader, build_separate_loaders
from training import TrainingOrchestrator, create_orchestrator


def parse_config_override(value: str):
    """Parse config override value to appropriate type"""
    # Handle None/null explicitly
    if value.lower() in ('none', 'null'):
        return None
    # Try bool first
    if value.lower() in ('true', 'yes', 'on'):
        return True
    if value.lower() in ('false', 'no', 'off'):
        return False
    # Try int
    try:
        return int(value)
    except ValueError:
        pass
    # Try float
    try:
        return float(value)
    except ValueError:
        pass
    # Return as string
    return value


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Anime Super-Resolution Model')
    
    # Config
    parser.add_argument('--config', '-c', type=str, required=True,
                      help='Path to config YAML file')
    parser.add_argument('--mode', '-m', type=str,
                      choices=['model_a', 'model_b', 'both_parallel', 'both_sequential', 'ensemble', 'full'],
                      help='Training mode override')
    
    # Training control
    parser.add_argument('--resume', '-r', type=str,
                      help='Resume from checkpoint path')
    parser.add_argument('--allow-pickle', action='store_true',
                      help='Allow pickle deserialization when loading checkpoints '
                           '(sets security.allow_pickle_checkpoint=true). Use only '
                           'with trusted checkpoints.')
    parser.add_argument('--run-name', type=str, default=None,
                      help='Custom run name (optional, auto-generated if not provided)')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device (cuda/cpu)')
    parser.add_argument('--num-workers', type=int,
                      help='Number of dataloader workers')
    parser.add_argument('--batch-size', type=int,
                      help='Override batch size (sets training.batch_size, '
                           'data.batch_size, and training.finetune.batch_size)')
    parser.add_argument('--crop-size', type=int,
                      help='Override crop size (sets data.crop_size and '
                           'data.preprocessing.storage.crop_size)')
    parser.add_argument('--grad-accum', type=int,
                      help='Override gradient accumulation steps '
                           '(sets training.finetune.gradient_accumulation_steps)')
    parser.add_argument('--no-fdl', action='store_true',
                      help='Disable FDL (DINOv2) loss; sets loss.fdl.enabled=false')
    parser.add_argument('--no-wavelet', action='store_true',
                      help='Disable wavelet-guided loss; sets '
                           'loss.wavelet_guided.enabled=false')
    parser.add_argument('--epochs', type=int,
                      help='Override number of epochs')
    parser.add_argument('--lr', type=float,
                      help='Override learning rate')
    
    # Logging
    parser.add_argument('--dry-run', action='store_true',
                      help='Validate config and exit')
    parser.add_argument('--tensorboard', action='store_true',
                      help='Enable TensorBoard logging')
    parser.add_argument('--separate-loaders', action='store_true',
                      help='Use separate train/val loaders (recommended for different dataset sizes)')
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='Verbose output')
    
    # Parse known args first
    args, unknown = parser.parse_known_args()
    
    # Process unknown args as config overrides (--key.subkey value)
    args.config_overrides = {}
    i = 0
    while i < len(unknown):
        arg = unknown[i]
        if arg.startswith('--'):
            key = arg[2:]  # Remove '--'
            # Get value if next arg exists and isn't another flag
            if i + 1 < len(unknown) and not unknown[i + 1].startswith('--'):
                value = parse_config_override(unknown[i + 1])
                args.config_overrides[key] = value
                i += 2
            else:
                # Boolean flag without value
                args.config_overrides[key] = True
                i += 1
        else:
            i += 1
    
    return args


def validate_config(config: Config) -> bool:
    """Validate configuration"""
    print("Validating configuration...")
    
    try:
        config.validate()
        
        # Check data paths
        data_configs = config.get('data.datasets', [])
        for dataset_cfg in data_configs:
            if dataset_cfg.get('enabled', True):
                hr_dir = dataset_cfg.get('hr_dir')
                if hr_dir:
                    hr_path = Path(hr_dir)
                    if not hr_path.exists():
                        print(f"Warning: HR directory does not exist: {hr_dir}")
        
        # Check stage configuration consistency
        mode = config.get('training.mode', 'model_a')
        stage1_cfg = config.get('training.stage1', {})
        stage2_cfg = config.get('training.stage2', {})
        stage_automation = config.get('training.stage_automation', {})
        
        # Validate auto_stage mode requirements
        if mode == 'auto_stage':
            if not stage1_cfg.get('enabled', False):
                print("ERROR: mode='auto_stage' requires training.stage1.enabled=true")
                print("  Stage 1 must be enabled for auto-stage training.")
                return False
            
            if not stage2_cfg.get('enabled', False):
                print("WARNING: mode='auto_stage' but training.stage2.enabled=false")
                print("  Only Stage 1 will run. Set stage2.enabled=true for full pipeline.")
            
            if not stage_automation.get('enabled', False):
                print("WARNING: mode='auto_stage' but training.stage_automation.enabled=false")
                print("  Auto-transition disabled - training will stop after Stage 1.")
        
        # Validate teacher configuration for Stage 1
        if stage1_cfg.get('enabled', False):
            teachers = stage1_cfg.get('teachers', [])
            if not teachers:
                print("WARNING: Stage 1 enabled but no teachers configured.")
                print("  Add teachers to training.stage1.teachers for knowledge aggregation.")
            
            # Check for teacher path validity
            missing_teacher_paths = []
            for teacher in teachers:
                teacher_path = teacher.get('path')
                if teacher_path and not Path(teacher_path).exists():
                    auto_download = stage1_cfg.get('auto_download_teachers', True)
                    if not auto_download:
                        missing_teacher_paths.append(teacher.get('name', 'unknown'))
            
            if missing_teacher_paths:
                print(f"WARNING: Teacher models not found: {missing_teacher_paths}")
                print("  Enable auto_download_teachers or provide valid paths.")
        
        # Check CUDA availability
        device = config.get('training.device', 'cuda')
        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA not available, switching to CPU")
            config.set('training.device', 'cpu')
        
        # Estimate VRAM usage
        # Resolve batch_size with proper precedence (matching dataloader logic)
        batch_size = (
            config.get('training.batch_size') or
            config.get('data.batch_size') or
            8  # base.yaml default from training.batch_size
        )
        
        # Display effective batch_size and source
        training_batch = config.get('training.batch_size')
        data_batch = config.get('data.batch_size')
        
        if training_batch:
            batch_source = 'training.batch_size'
        elif data_batch:
            batch_source = 'data.batch_size'
        else:
            batch_source = 'default'
        
        print(f"Batch size: {batch_size} (from {batch_source})")
        print(f"  Override with: --batch-size <value> or set 'training.batch_size' in config")
        model_type = config.get('model.type', 'span')
        
        if model_type == 'mamba_pan':
            est_vram = batch_size * 0.9  # ~0.9GB per sample for Mamba
        else:
            est_vram = batch_size * 0.4  # ~0.4GB per sample for SPAN
        
        print(f"Estimated VRAM usage: {est_vram:.1f}GB (batch_size={batch_size})")
        
        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"Available VRAM: {total_vram:.1f}GB")
            
            if est_vram > total_vram * 0.9:
                print(f"Warning: Estimated VRAM ({est_vram:.1f}GB) may exceed available VRAM ({total_vram:.1f}GB)")
                print("Consider reducing batch_size or enabling gradient_checkpointing")
        
        print("Configuration validation passed!")
        return True
        
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False


def estimate_training_time(config: Config) -> str:
    """Estimate training time"""
    mode = config.get('training.mode', 'model_a')
    
    if mode == 'finetune' or config.get('training.finetune.enabled', False):
        total_epochs = config.get('training.finetune.epochs', 50)
    else:
        stage1_epochs = config.get('training.stage1.epochs', 0)
        stage2_epochs = config.get('training.stage2.epochs', 0)
        total_epochs = stage1_epochs + stage2_epochs
    
    # Rough estimate: ~2-5 minutes per epoch on RTX 4000
    min_time = total_epochs * 2
    max_time = total_epochs * 5
    
    return f"{min_time}-{max_time} minutes ({min_time/60:.1f}-{max_time/60:.1f} hours)"


def main():
    """Main entry point"""
    args = parse_args()
    
    print(f"\n{'='*60}")
    print(f"Anime Super-Resolution Training")
    print(f"{'='*60}\n")
    
    # Load config
    print(f"Loading config: {args.config}")
    
    # Validate config path exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    try:
        config = Config.load(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    # Apply config overrides from CLI (e.g., --data.video.enabled true)
    if args.config_overrides:
        print("Applying config overrides:")
        for key, value in args.config_overrides.items():
            print(f"  {key} = {value}")
            config.set(key, value)
    
    # Merge CLI arguments
    config.merge_args(args)
    
    # Set training mode from CLI if provided
    if args.mode:
        config.set('training.mode', args.mode)

    def _log_override(key: str, old: object, new: object) -> None:
        """Print a [CLI override] line. Paths verified to match the trainer
        (see src/training/neosr_finetuner.py)."""
        print(f"[CLI override] {key}: {old} -> {new}")

    # Handle batch_size override - set in both training and data sections
    if args.batch_size is not None:
        old_tbs = config.get('training.batch_size')
        old_dbs = config.get('data.batch_size')
        _log_override('training.batch_size', old_tbs, args.batch_size)
        _log_override('data.batch_size', old_dbs, args.batch_size)
        config.set('training.batch_size', args.batch_size)
        config.set('data.batch_size', args.batch_size)
        # Also override finetune.batch_size so the trainer picks it up
        # (see NeosrSPANFinetuner self.finetune_cfg.get('batch_size', 4))
        old_fbs = config.get('training.finetune.batch_size')
        if old_fbs is not None:
            _log_override('training.finetune.batch_size', old_fbs, args.batch_size)
            config.set('training.finetune.batch_size', args.batch_size)

    # Handle crop_size override - data.crop_size and preprocessing.storage.crop_size
    if args.crop_size is not None:
        old_dcs = config.get('data.crop_size')
        _log_override('data.crop_size', old_dcs, args.crop_size)
        config.set('data.crop_size', args.crop_size)
        # Mirror to preprocessing.storage.crop_size (used by PreprocessingManager)
        old_pcs = config.get('data.preprocessing.storage.crop_size')
        if old_pcs is not None:
            _log_override(
                'data.preprocessing.storage.crop_size', old_pcs, args.crop_size,
            )
            config.set('data.preprocessing.storage.crop_size', args.crop_size)

    # Handle gradient accumulation override
    if args.grad_accum is not None:
        old_ga = config.get('training.finetune.gradient_accumulation_steps')
        _log_override(
            'training.finetune.gradient_accumulation_steps',
            old_ga, args.grad_accum,
        )
        config.set('training.finetune.gradient_accumulation_steps', args.grad_accum)

    # Handle --no-fdl: disable FDL (DINOv2) loss
    if args.no_fdl:
        old_fdl = config.get('training.finetune.loss.fdl.enabled')
        _log_override('training.finetune.loss.fdl.enabled', old_fdl, False)
        config.set('training.finetune.loss.fdl.enabled', False)

    # Handle --no-wavelet: disable wavelet-guided loss
    if args.no_wavelet:
        old_wg = config.get('training.finetune.loss.wavelet_guided.enabled')
        _log_override(
            'training.finetune.loss.wavelet_guided.enabled', old_wg, False,
        )
        config.set('training.finetune.loss.wavelet_guided.enabled', False)

    # For finetune mode, use finetune-specific batch_size if available
    # This must happen BEFORE validation so the correct batch_size is used
    mode = config.get('training.mode', 'model_a')
    if mode == 'finetune' or config.get('training.finetune.enabled', False):
        finetune_batch = config.get('training.finetune.batch_size')
        if finetune_batch is not None:
            config.set('training.batch_size', finetune_batch)
            print(f"[Finetune] Using training.finetune.batch_size={finetune_batch}")
    
    # Handle epochs override - map to both stage1 and stage2
    if args.epochs is not None:
        stage1_cfg = config.get('training.stage1', {})
        stage2_cfg = config.get('training.stage2', {})
        if stage1_cfg.get('enabled', False):
            config.set('training.stage1.epochs', args.epochs)
        if stage2_cfg.get('enabled', False):
            config.set('training.stage2.epochs', args.epochs)

    # Handle --allow-pickle flag
    if args.allow_pickle:
        # The trainer reads `self.config.get('security')` (top-level), not
        # `training.security` — keep the path consistent with what
        # `load_checkpoint` actually checks.
        config.set('security.allow_pickle_checkpoint', True)
        print("[Security] Pickle deserialization ENABLED (--allow-pickle). "
              "Use only with trusted checkpoints.")

    # Dry run: validate and exit
    if args.dry_run:
        print("\nDry run mode - validating configuration...")
        if validate_config(config):
            print("\nConfig is valid! Estimated training time:", estimate_training_time(config))
            print("\nTo start training, run without --dry-run")
        else:
            print("\nConfig validation failed!")
            sys.exit(1)
        return
    
    # Validate config
    if not validate_config(config):
        print("\nFix configuration errors before training.")
        sys.exit(1)
    
    # Print training info
    mode = config.get('training.mode', 'model_a')
    
    # Resolve and display effective batch_size
    # The dataloader uses this precedence: training.batch_size > data.batch_size > default
    effective_batch_size = (
        config.get('training.batch_size') or
        config.get('data.batch_size') or
        8  # base.yaml default
    )
    
    # Show which config key is being used
    if config.get('training.batch_size'):
        batch_source = 'training.batch_size'
    elif config.get('data.batch_size'):
        batch_source = 'data.batch_size'
    else:
        batch_source = 'base.yaml default (8)'
        print(f"[INFO] Using default batch_size=8 (from {batch_source})")
        print(f"  To customize: Set 'training.batch_size' or 'data.batch_size' in config")

    print(f"\nTraining mode: {mode}")
    print(f"Model: {config.get('model.name', 'unknown')}")
    print(f"Batch size: {effective_batch_size} (from {batch_source})")

    if mode == 'finetune' or config.get('training.finetune.enabled', False):
        finetune_epochs = config.get('training.finetune.epochs', 50)
        prog_enabled = config.get('training.finetune.progressive_crop.enabled', False)
        if prog_enabled:
            print(f"Epochs: {finetune_epochs} (progressive crop)")
        else:
            print(f"Epochs: {finetune_epochs}")
    else:
        print(f"Epochs: Stage1={config.get('training.stage1.epochs', 'N/A')}, Stage2={config.get('training.stage2.epochs', 'N/A')}")

    print(f"Estimated time: {estimate_training_time(config)}")
    print(f"Device: {config.get('training.device', 'cuda')}")
    print()
    
    # Build dataloaders
    print("Building dataloaders...")
    try:
        if args.separate_loaders:
            print("  Using separate train/val loaders (recommended for different dataset sizes)")
            train_loader, val_loader = build_separate_loaders(config)
        else:
            train_loader = build_dataloader(config, mode='train')
            val_loader = build_dataloader(config, mode='val')
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nTo fix this, you can:")
        print("1. Create test data:")
        print(f"   python scripts/create_test_data.py --output data/test_hr --num-images 10")
        print("\n2. Update config to point to your data:")
        print(f"   python scripts/train.py --config {args.config} --data.datasets.0.hr_dir 'path/to/your/data'")
        print("\n3. Prepare your dataset:")
        print(f"   python scripts/prepare_data.py --input path/to/videos --output data/processed --mode full")
        sys.exit(1)
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] building dataloaders: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create orchestrator or finetuner
    print(f"\nCreating training orchestrator for mode: {mode}...")
    
    # Check for finetune mode
    if mode == 'finetune' or config.get('training.finetune.enabled', False):
        print("\n[Finetune Mode] Using neosr SPAN finetuner")
        try:
            from training.neosr_finetuner import create_finetuner
            from utils.checkpoint_manager import CheckpointManager

            device = config.get('training.device', 'cuda' if torch.cuda.is_available() else 'cpu')

            checkpoint_manager = CheckpointManager(config, cli_run_name=args.run_name, resume_from=args.resume)
            checkpoint_manager.print_run_info()

            finetuner = create_finetuner(config, device, checkpoint_manager=checkpoint_manager)
            finetuner.train(train_loader, val_loader, resume_from=args.resume)
            print("\nFine-tuning complete!")
            return
        except Exception as e:
            print(f"\n[ERROR] in finetune mode: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    try:
        orchestrator = create_orchestrator(config)
        orchestrator.setup()
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except NotImplementedError as e:
        print(f"\n[WARNING] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] setting up orchestrator: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Start training
    print(f"\n{'='*60}")
    print(f"Starting training")
    print(f"{'='*60}\n")
    
    try:
        orchestrator.train(train_loader, val_loader)
        orchestrator.save_metadata()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Attempting to save checkpoint...")
        try:
            orchestrator.save_metadata()
            print("Metadata saved. Exiting.")
        except Exception:
            print("Could not save metadata. Exiting.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
