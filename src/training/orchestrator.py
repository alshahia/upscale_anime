"""
Training Orchestrator - Master trainer for all modes
Supports: model_a, model_b, both_parallel, both_sequential, ensemble, full
"""
import torch
import torch.nn as nn
from typing import Dict, Optional
from enum import Enum
from pathlib import Path

from training.model_a_trainer import ModelATrainer
from training.model_b_trainer import ModelBTrainer


class TrainingMode(Enum):
    """Supported training modes"""
    MODEL_A = "model_a"
    MODEL_B = "model_b"
    BOTH_PARALLEL = "both_parallel"
    BOTH_SEQUENTIAL = "both_sequential"
    ENSEMBLE = "ensemble"
    FULL = "full"
    AUTO_STAGE = "auto_stage"  # Automated multi-stage training


class TrainingOrchestrator:
    """
    Master orchestrator for training.
    
    Handles:
    - Mode selection and validation
    - Trainer instantiation
    - Multi-model training coordination
    - Checkpoint management across modes
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.mode = self._get_training_mode()
        
        # Device
        self.device = torch.device(
            config.get('training', {}).get('device', 'cuda') 
            if torch.cuda.is_available() else 'cpu'
        )
        
        # Checkpoints
        self.checkpoint_dir = Path(config.get('paths', {}).get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Active trainers
        self.trainer_a: Optional[ModelATrainer] = None
        self.trainer_b: Optional[ModelBTrainer] = None
        
        print(f"Training Orchestrator initialized")
        print(f"Mode: {self.mode.value}")
        print(f"Device: {self.device}")
    
    def _get_training_mode(self) -> TrainingMode:
        """Determine training mode from config"""
        mode_str = self.config.get('training', {}).get('mode', 'model_a')
        
        try:
            return TrainingMode(mode_str)
        except ValueError:
            raise ValueError(
                f"Unknown training mode: {mode_str}\n"
                f"Available modes: {[m.value for m in TrainingMode]}"
            )
    
    def setup(self):
        """Setup trainers based on mode"""
        
        if self.mode == TrainingMode.MODEL_A:
            print("\nSetting up Model A trainer...")
            self.trainer_a = ModelATrainer(self.config)
        
        elif self.mode == TrainingMode.MODEL_B:
            print("\nSetting up Model B trainer...")
            self.trainer_b = ModelBTrainer(self.config)
        
        elif self.mode == TrainingMode.BOTH_PARALLEL:
            print("\nSetting up both models for parallel training...")
            self.trainer_a = ModelATrainer(self.config)
            self.trainer_b = ModelBTrainer(self.config)
            print("Models will be trained in parallel (alternating batches)")
        
        elif self.mode == TrainingMode.BOTH_SEQUENTIAL:
            print("\nSetting up both models for sequential training...")
            self.trainer_a = ModelATrainer(self.config)
            self.trainer_b = ModelBTrainer(self.config)
            print("Model A will be trained first, then Model B")
        
        elif self.mode == TrainingMode.ENSEMBLE:
            print("\nSetting up Ensemble trainer...")
            print("Loading pre-trained Model A and Model B...")
            # Ensemble trainer is created in _train_ensemble after checkpoint paths are validated
        
        elif self.mode == TrainingMode.FULL:
            print("\nSetting up full pipeline: A -> B -> Ensemble...")
            self.trainer_a = ModelATrainer(self.config)
            self.trainer_b = ModelBTrainer(self.config)
            print("Complete pipeline will be executed")
        
        elif self.mode == TrainingMode.AUTO_STAGE:
            print("\nSetting up automated multi-stage training...")
            from training.auto_stage_trainer import AutoStageTrainer
            self.auto_trainer = AutoStageTrainer(self.config)
            print("Stage 1 -> Stage 2 automation enabled")
    
    def train(self, train_loader, val_loader=None) -> None:
        """
        Execute training based on mode.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
        """
        if self.mode == TrainingMode.MODEL_A:
            return self._train_model_a(train_loader, val_loader)
        
        elif self.mode == TrainingMode.MODEL_B:
            return self._train_model_b(train_loader, val_loader)
        
        elif self.mode == TrainingMode.BOTH_PARALLEL:
            return self._train_both_parallel(train_loader, val_loader)
        
        elif self.mode == TrainingMode.BOTH_SEQUENTIAL:
            return self._train_both_sequential(train_loader, val_loader)
        
        elif self.mode == TrainingMode.ENSEMBLE:
            return self._train_ensemble(train_loader, val_loader)
        
        elif self.mode == TrainingMode.FULL:
            return self._train_full_pipeline(train_loader, val_loader)
        
        elif self.mode == TrainingMode.AUTO_STAGE:
            return self._train_auto_stage(train_loader, val_loader)
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _train_model_a(self, train_loader, val_loader) -> None:
        """Train Model A only"""
        print("\n" + "="*60)
        print("Training Model A (NTIRE + MTKD + FAKD)")
        print("="*60)
        
        self.trainer_a.train(train_loader, val_loader)
        
        print("\nModel A training complete!")
        print(f"Best checkpoint: {self.checkpoint_dir / 'best.pth'}")
    
    def _train_model_b(self, train_loader, val_loader) -> None:
        """Train Model B only"""
        print("\n" + "="*60)
        print("Training Model B (Mamba-PAN + MTKD + FAKD)")
        print("="*60)
        
        self.trainer_b.train(train_loader, val_loader)
        
        print("\nModel B training complete!")
        print(f"Best checkpoint: {self.checkpoint_dir / 'best.pth'}")
    
    def _train_both_parallel(self, train_loader, val_loader) -> None:
        """
        Train both models in parallel (alternating batches).
        More memory efficient than running simultaneously.
        """
        print("\n" + "="*60)
        print("Training Both Models in Parallel Mode")
        print("="*60)
        print("Training will alternate between Model A and Model B")

        # Get epochs from config
        epochs_a = self.config.get('training', {}).get('model_a_epochs', 100)
        epochs_b = self.config.get('training', {}).get('model_b_epochs', 100)
        max_epochs = max(epochs_a, epochs_b)

        # Initialize optimizers and schedulers once (CRITICAL FIX: don't recreate every epoch)
        optimizer_a = None
        optimizer_b = None
        scheduler_a = None
        scheduler_b = None

        if epochs_a > 0:
            print("\nInitializing Model A optimizer and scheduler...")
            optimizer_a = self.trainer_a._create_optimizer()
            scheduler_a = self.trainer_a._create_scheduler(optimizer_a)

        if epochs_b > 0:
            print("Initializing Model B optimizer and scheduler...")
            optimizer_b = self.trainer_b._create_optimizer()
            scheduler_b = self.trainer_b._create_scheduler(optimizer_b)

        for epoch in range(max_epochs):
            print(f"\n--- Epoch {epoch + 1}/{max_epochs} ---")

            # Train Model A
            if epoch < epochs_a:
                print("Training Model A...")
                self.trainer_a.current_epoch = epoch
                metrics_a = self.trainer_a.train_epoch(train_loader, optimizer_a)
                print(f"  Model A - Loss: {metrics_a['loss']:.4f}")

            # Train Model B
            if epoch < epochs_b:
                print("Training Model B...")
                self.trainer_b.current_epoch = epoch
                metrics_b = self.trainer_b.train_epoch(train_loader, optimizer_b)
                print(f"  Model B - Loss: {metrics_b['loss']:.4f}")

            # Step schedulers after each epoch (CRITICAL FIX: was never called before)
            if scheduler_a is not None and epoch < epochs_a:
                scheduler_a.step()
                if epoch % 10 == 0:  # Log LR occasionally
                    current_lr = optimizer_a.param_groups[0]['lr']
                    print(f"  Model A LR: {current_lr:.6f}")

            if scheduler_b is not None and epoch < epochs_b:
                scheduler_b.step()
                if epoch % 10 == 0:
                    current_lr = optimizer_b.param_groups[0]['lr']
                    print(f"  Model B LR: {current_lr:.6f}")

            # Validation (every N epochs)
            val_interval = self.config.get('training', {}).get('val_interval', 10)
            if val_loader is not None and epoch % val_interval == 0:
                if epoch < epochs_a:
                    val_a = self.trainer_a.validate(val_loader)
                    print(f"  Model A Val - Loss: {val_a['loss']:.4f}, PSNR: {val_a['psnr']:.2f}")

                if epoch < epochs_b:
                    val_b = self.trainer_b.validate(val_loader)
                    print(f"  Model B Val - Loss: {val_b['loss']:.4f}, PSNR: {val_b['psnr']:.2f}")

                # Save best checkpoints based on validation loss
                if epoch < epochs_a:
                    if val_a['loss'] < self.trainer_a.best_loss:
                        self.trainer_a.best_loss = val_a['loss']
                        self.trainer_a.save_checkpoint(epoch, optimizer_a, is_best=True)

                if epoch < epochs_b:
                    if val_b['loss'] < self.trainer_b.best_loss:
                        self.trainer_b.best_loss = val_b['loss']
                        self.trainer_b.save_checkpoint(epoch, optimizer_b, is_best=True)
        
        print("\nParallel training complete!")
    
    def _train_both_sequential(self, train_loader, val_loader) -> None:
        """
        Train Model A first, then Model B.
        """
        print("\n" + "="*60)
        print("Training Both Models Sequentially")
        print("="*60)
        
        # Stage 1: Train Model A
        print("\n>>> Stage 1: Training Model A")
        self._train_model_a(train_loader, val_loader)
        
        # Stage 2: Train Model B
        print("\n>>> Stage 2: Training Model B")
        self._train_model_b(train_loader, val_loader)
        
        print("\nSequential training complete!")
        print(f"Model A: {self.checkpoint_dir / 'model_a_best.pth'}")
        print(f"Model B: {self.checkpoint_dir / 'model_b_best.pth'}")
    
    def _train_ensemble(self, train_loader, val_loader) -> None:
        """
        Train ensemble from frozen Model A and B.
        """
        print("\n" + "="*60)
        print("Training Ensemble")
        print("="*60)
        
        # Get checkpoint paths from config
        checkpoint_a = self.config.get('paths', {}).get('model_a_checkpoint')
        checkpoint_b = self.config.get('paths', {}).get('model_b_checkpoint')
        
        if not checkpoint_a or not checkpoint_b:
            raise ValueError(
                "Ensemble mode requires model_a_checkpoint and model_b_checkpoint in config.\n"
                "Add to config:\n"
                "  paths:\n"
                "    model_a_checkpoint: 'checkpoints/model_a_best.pth'\n"
                "    model_b_checkpoint: 'checkpoints/model_b_best.pth'"
            )
        
        # Import ensemble trainer
        from training.ensemble_trainer import EnsembleTrainer
        from models.ensemble import create_ensemble_teacher
        
        # Create ensemble teacher from frozen checkpoints
        print(f"\nLoading Model A from: {checkpoint_a}")
        print(f"Loading Model B from: {checkpoint_b}")
        
        try:
            ensemble_teacher = create_ensemble_teacher(
                checkpoint_a,
                checkpoint_b,
                device=self.device,
            )
            print("[OK] Ensemble teacher created successfully")
        except Exception as e:
            print(f"[ERROR] Failed to create ensemble teacher: {e}")
            raise
        
        # Create ensemble trainer
        self.trainer_ensemble = EnsembleTrainer(self.config, ensemble_teacher)
        
        # Train tiny student
        print("\nTraining Tiny Student with ensemble distillation...")
        self.trainer_ensemble.train(train_loader, val_loader)
        
        print("\nEnsemble training complete!")
        print(f"Tiny student saved to: {self.checkpoint_dir / 'ensemble_best.pth'}")
    
    def _train_full_pipeline(self, train_loader, val_loader) -> None:
        """
        Full pipeline: A → B → Ensemble
        """
        print("\n" + "="*60)
        print("Full Training Pipeline")
        print("="*60)
        
        # Phase 1: Train Model A
        print("\n[Phase 1/3] Training Model A...")
        self._train_model_a(train_loader, val_loader)
        
        # Save Model A checkpoint to model-specific path
        best_a = self.checkpoint_dir / 'best.pth'
        model_a_checkpoint = self.checkpoint_dir / 'model_a_best.pth'
        if best_a.exists():
            import shutil
            shutil.copy2(best_a, model_a_checkpoint)
            print(f"Saved Model A checkpoint to: {model_a_checkpoint}")
        
        # Phase 2: Train Model B
        print("\n[Phase 2/3] Training Model B...")
        self._train_model_b(train_loader, val_loader)
        
        # Save Model B checkpoint to model-specific path
        best_b = self.checkpoint_dir / 'best.pth'
        model_b_checkpoint = self.checkpoint_dir / 'model_b_best.pth'
        if best_b.exists():
            import shutil
            shutil.copy2(best_b, model_b_checkpoint)
            print(f"Saved Model B checkpoint to: {model_b_checkpoint}")
        
        # Phase 3: Train Ensemble
        print("\n[Phase 3/3] Training Ensemble...")
        
        # Update config with checkpoint paths for ensemble
        if 'paths' not in self.config:
            self.config['paths'] = {}
        self.config['paths']['model_a_checkpoint'] = str(model_a_checkpoint)
        self.config['paths']['model_b_checkpoint'] = str(model_b_checkpoint)
        
        self._train_ensemble(train_loader, val_loader)
        
        print("\n" + "="*60)
        print("Full pipeline complete!")
        print("="*60)
    
    def _train_auto_stage(self, train_loader, val_loader) -> None:
        """
        Automated multi-stage training (Stage 1 -> Stage 2).
        Uses AutoStageTrainer for full pipeline automation.
        """
        print("\n" + "="*60)
        print("Automated Multi-Stage Training")
        print("="*60)
        print("Stage 1: Knowledge Aggregation -> Stage 2: Student Distillation")
        print("="*60)
        
        results = self.auto_trainer.train(train_loader, val_loader)
        
        print("\n" + "="*60)
        print("Automated training complete!")
        print("="*60)
        print(f"Stage 1: {'[OK]' if results['stage1_completed'] else '[FAIL]'}")
        print(f"  Best checkpoint: {results.get('stage1_best_checkpoint', 'N/A')}")
        print(f"Stage 2: {'[OK]' if results['stage2_completed'] else '[FAIL]'}")
        print(f"  Best checkpoint: {results.get('stage2_best_checkpoint', 'N/A')}")
        print("="*60)
    
    def save_metadata(self) -> None:
        """Save training metadata"""
        import json
        
        # Convert config to dict if it's a Config object
        config_dict = self.config.to_dict() if hasattr(self.config, 'to_dict') else self.config
        
        # Build checkpoint paths
        checkpoint_a = None
        checkpoint_b = None
        
        if self.trainer_a is not None:
            checkpoint_a = str(self.checkpoint_dir / 'model_a_best.pth')
        
        if self.trainer_b is not None:
            checkpoint_b = str(self.checkpoint_dir / 'model_b_best.pth')
        
        metadata = {
            'mode': self.mode.value,
            'config': config_dict,
            'checkpoints': {
                'model_a': checkpoint_a,
                'model_b': checkpoint_b,
            }
        }
        
        metadata_path = self.checkpoint_dir / 'training_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to: {metadata_path}")


def create_orchestrator(config: Dict) -> TrainingOrchestrator:
    """
    Factory function to create training orchestrator.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        TrainingOrchestrator instance
    """
    return TrainingOrchestrator(config)
