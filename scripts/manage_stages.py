"""
CLI tool for managing multi-stage training.
Commands: status, advance, reset, resume
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from anime_sr.training.stage_state import StageStateManager
from anime_sr.training.stage_controller import StageTransitionController
from anime_sr.training.auto_stage_trainer import AutoStageTrainer
import yaml


def load_config(config_path: Path) -> dict:
    """Load training configuration."""
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return None
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def cmd_status(args):
    """Show current stage status."""
    checkpoint_dir = Path(args.checkpoint_dir)
    manager = StageStateManager(state_dir=checkpoint_dir, training_id=args.id)
    
    if not manager.exists():
        print(f"No stage state found in {checkpoint_dir}")
        print("Training has not started or state was cleared.")
        return 0
    
    state = manager.load()
    print(state.get_summary())
    
    # Show checkpoints
    checkpoints = manager.get_all_checkpoints()
    if checkpoints:
        print("\nCheckpoints:")
        for stage, path in checkpoints.items():
            status = "✓" if path else "✗"
            print(f"  Stage {stage}: {status} {path or 'N/A'}")
    
    return 0


def cmd_advance(args):
    """Manually trigger stage transition."""
    checkpoint_dir = Path(args.checkpoint_dir)
    manager = StageStateManager(state_dir=checkpoint_dir, training_id=args.id)
    
    if not manager.exists():
        print(f"Error: No stage state found. Cannot advance.")
        return 1
    
    state = manager.load()
    
    if state.current_stage == 1:
        if not state.stage1_completed:
            print("Warning: Stage 1 not marked as complete. Advancing anyway.")
        
        # Execute transition
        config = load_config(Path(args.config)) if args.config else {}
        controller = StageTransitionController(config, manager)
        
        success = controller.execute_transition(
            from_stage=1,
            to_stage=2,
            checkpoint_path=args.checkpoint or state.stage1_best_checkpoint,
            metrics={'val_loss': args.metric} if args.metric else {}
        )
        
        if success:
            print("✓ Stage transition completed: 1 -> 2")
            print(f"Checkpoint: {args.checkpoint or state.stage1_best_checkpoint}")
        else:
            print("✗ Stage transition failed")
            return 1
    else:
        print(f"Already at Stage {state.current_stage}")
    
    return 0


def cmd_reset(args):
    """Reset stage state."""
    checkpoint_dir = Path(args.checkpoint_dir)
    manager = StageStateManager(state_dir=checkpoint_dir, training_id=args.id)
    
    if not manager.exists():
        print("No stage state to reset.")
        return 0
    
    if not args.force:
        response = input("Are you sure you want to reset stage state? [y/N]: ")
        if response.lower() != 'y':
            print("Reset cancelled.")
            return 0
    
    manager.clear()
    print("✓ Stage state reset. Next training will start from Stage 1.")
    return 0


def cmd_resume(args):
    """Resume interrupted multi-stage training."""
    if not args.config:
        print("Error: --config required for resume")
        return 1
    
    config = load_config(Path(args.config))
    if config is None:
        return 1
    
    checkpoint_dir = Path(args.checkpoint_dir)
    
    # Check if state exists
    manager = StageStateManager(state_dir=checkpoint_dir, training_id=args.id)
    if not manager.exists():
        print("No previous stage state found. Starting fresh training.")
    else:
        state = manager.load()
        print(state.get_summary())
        
        if state.current_stage == 2 and not state.stage2_completed:
            print("\nResuming from Stage 2...")
        elif state.current_stage == 1 and not state.stage1_completed:
            print("\nResuming from Stage 1...")
        else:
            print("\nTraining already complete.")
            return 0
    
    print("\nTo resume training, run:")
    print(f"  python train.py --config {args.config}")
    print(f"\nThe trainer will automatically detect and load the previous state.")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Manage multi-stage training pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check current stage status
  %(prog)s status
  
  # Advance from Stage 1 to Stage 2 manually
  %(prog)s advance --checkpoint checkpoints/stage1_best.pth
  
  # Reset stage state (start over)
  %(prog)s reset --force
  
  # Check resume status
  %(prog)s resume --config configs/my_config.yaml
        """
    )
    
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory containing checkpoints and stage state')
    parser.add_argument('--id', type=str, default='auto_stage',
                       help='Training ID for state management')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show current stage status')
    status_parser.set_defaults(func=cmd_status)
    
    # Advance command
    advance_parser = subparsers.add_parser('advance', help='Manually advance to next stage')
    advance_parser.add_argument('--checkpoint', type=str, help='Checkpoint path to use')
    advance_parser.add_argument('--metric', type=float, help='Best metric value')
    advance_parser.add_argument('--config', type=str, help='Config file path')
    advance_parser.set_defaults(func=cmd_advance)
    
    # Reset command
    reset_parser = subparsers.add_parser('reset', help='Reset stage state')
    reset_parser.add_argument('--force', action='store_true',
                             help='Skip confirmation prompt')
    reset_parser.set_defaults(func=cmd_reset)
    
    # Resume command
    resume_parser = subparsers.add_parser('resume', help='Check resume status')
    resume_parser.add_argument('--config', type=str, help='Config file path')
    resume_parser.set_defaults(func=cmd_resume)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
