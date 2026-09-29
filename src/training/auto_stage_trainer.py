"""
Automated multi-stage training orchestrator.
Handles Stage 1 -> Stage 2 transitions automatically.
"""
import torch
from typing import Dict, Optional, Callable
from pathlib import Path

from training.model_a_trainer import ModelATrainer
from training.stage_controller import StageTransitionController
from training.stage_state import StageStateManager
from training.callbacks import StageMonitorCallback


class AutoStageTrainer:
    """
    Automated multi-stage training trainer.
    
    Orchestrates the full pipeline:
    - Stage 1: Train Knowledge Aggregation until convergence
    - Auto-transition to Stage 2 with best checkpoint
    - Stage 2: Train SPAN Student until convergence
    
    Args:
        config: Training configuration
        state_manager: Optional StageStateManager (created if None)
        stage_controller: Optional StageTransitionController (created if None)
    """
    
    def __init__(
        self,
        config: Dict,
        state_manager: Optional[StageStateManager] = None,
        stage_controller: Optional[StageTransitionController] = None,
    ):
        self.config = config
        
        # Initialize state management
        checkpoint_dir = config.get('paths', {}).get('checkpoint_dir', 'checkpoints')
        self.state_manager = state_manager or StageStateManager(
            state_dir=checkpoint_dir,
            training_id="auto_stage"
        )
        
        # Initialize stage controller
        self.stage_controller = stage_controller or StageTransitionController(
            config=config,
            state_manager=self.state_manager,
        )
        
        # Create underlying trainer
        self.trainer = ModelATrainer(config)
        
        # Track stage progress
        self.current_stage = 1
        self.stage_completed = False
    
    def train(self, train_loader, val_loader=None) -> Dict[str, any]:
        """
        Execute full automated multi-stage training.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            
        Returns:
            Dictionary with training results and stage info
        """
        results = {
            'stage1_completed': False,
            'stage1_best_checkpoint': None,
            'stage2_completed': False,
            'stage2_best_checkpoint': None,
            'stages_advanced': False,
        }
        
        # Check if starting from Stage 2 (resumed training)
        if self.stage_controller.state.stage2_started:
            print("\n*** Resuming from Stage 2 ***\n")
            self.current_stage = 2
        
        # Stage 1: Knowledge Aggregation
        if self.current_stage == 1 and not self.stage_controller.state.stage1_completed:
            print("\n" + "="*60)
            print("Stage 1: Training Knowledge Aggregation (Auto-Stage Mode)")
            print("="*60)
            
            stage1_result = self._run_stage1(train_loader, val_loader)
            results['stage1_completed'] = stage1_result['completed']
            results['stage1_best_checkpoint'] = stage1_result['best_checkpoint']
            results['stage1_best_metric'] = stage1_result['best_metric']
            
            if stage1_result['completed'] and self.stage_controller.state.stage2_enabled:
                # Execute transition to Stage 2
                transition_success = self.stage_controller.execute_transition(
                    from_stage=1,
                    to_stage=2,
                    checkpoint_path=stage1_result['best_checkpoint'],
                    metrics={'val_loss': stage1_result['best_metric']}
                )
                
                if transition_success:
                    # Transition the underlying trainer to Stage 2
                    self.trainer.transition_to_stage2()
                    results['stages_advanced'] = True
                    self.current_stage = 2
        
        # Stage 2: Student Distillation
        if (self.current_stage == 2 and 
            self.stage_controller.state.stage2_enabled and 
            self.stage_controller.state.stage1_completed):
            print("\n" + "="*60)
            print("Stage 2: Training SPAN Student (Auto-Stage Mode)")
            print("="*60)
            
            stage2_result = self._run_stage2(train_loader, val_loader)
            results['stage2_completed'] = stage2_result['completed']
            results['stage2_best_checkpoint'] = stage2_result['best_checkpoint']
            results['stage2_best_metric'] = stage2_result['best_metric']
        elif self.current_stage == 2 and self.stage_controller.state.stage2_enabled and not self.stage_controller.state.stage1_completed:
            print("\n[WARNING] Stage 2 enabled but Stage 1 not completed. Skipping Stage 2.")
        
        # Final summary
        print("\n" + "="*60)
        print("Auto-Stage Training Complete")
        print("="*60)
        self.stage_controller.print_status()
        
        return results
    
    def _run_stage1(self, train_loader, val_loader) -> Dict:
        """Run Stage 1 training."""
        result = {
            'completed': False,
            'best_checkpoint': None,
            'best_metric': float('inf'),
        }
        
        # Get stage 1 config
        stage1_cfg = self.config.get('training', {}).get('stage1', {})
        max_epochs = stage1_cfg.get('epochs', 100)
        
        # Setup trainer for Stage 1
        self.trainer.stage = 1
        self.trainer.stage1_epochs = max_epochs
        self.trainer.model = self.trainer._create_model(self.config)
        self.trainer.model = self.trainer.model.to(self.trainer.device)
        
        # Load best_loss from checkpoint if resuming
        if self.state_manager.load() is not None:
            saved_state = self.state_manager.load()
            if saved_state.stage1_best_metric is not None and saved_state.stage1_best_metric < float('inf'):
                self.trainer.best_loss = saved_state.stage1_best_metric
        else:
            self.trainer.best_loss = float('inf')
        
        # Update trainer LR for stage 1 (convert to float to ensure correct type)
        stage1_lr = stage1_cfg.get('lr', 1e-5)
        if isinstance(stage1_lr, str):
            stage1_lr = float(stage1_lr)
        self.trainer.lr = stage1_lr
        
        # Update other stage-specific training params
        stage1_weight_decay = stage1_cfg.get('weight_decay', 0.001)
        if isinstance(stage1_weight_decay, str):
            stage1_weight_decay = float(stage1_weight_decay)
        self.trainer.weight_decay = stage1_weight_decay
        
        # Create optimizer and scheduler
        optimizer = self.trainer._create_optimizer()
        self.trainer.set_scheduler(self.trainer._create_scheduler(optimizer, stage='stage1'))
        optimizer._step_count = 0
        
        # Setup stage monitoring
        stage_monitor = StageMonitorCallback(
            stage_controller=self.stage_controller,
            current_stage=1,
        )
        
        # Add to callbacks if not already present
        if self.trainer.callbacks is None:
            from training.callbacks import CallbackList
            self.trainer.callbacks = CallbackList([stage_monitor])
        else:
            self.trainer.callbacks.add(stage_monitor)
        
        # Training loop with auto-stop
        start_epoch = self.state_manager.load().stage1_epochs_completed if self.state_manager.load() else 0
        for epoch in range(start_epoch, max_epochs):
            self.trainer.current_epoch = epoch
            
            # Train
            train_metrics = self.trainer.train_stage1_epoch(train_loader, optimizer)
            print(f"Epoch {epoch}: loss={train_metrics['loss']:.4f}")
            
            # Validate
            val_metrics = {}
            if val_loader is not None and epoch % self.trainer.val_interval == 0:
                val_metrics = self.trainer.validate(val_loader)
                print(f"Validation: loss={val_metrics['loss']:.4f}, PSNR={val_metrics['psnr']:.2f}")
                
                if val_metrics['loss'] < self.trainer.best_loss:
                    self.trainer.best_loss = val_metrics['loss']
                    self.trainer.save_checkpoint(epoch, optimizer, is_best=True)
                    result['best_checkpoint'] = str(self.trainer.checkpoint_dir / 'best.pth')
                    result['best_metric'] = val_metrics['loss']
            
            # Save checkpoint
            if epoch % self.trainer.save_interval == 0:
                self.trainer.save_checkpoint(epoch, optimizer)
            
            # Step scheduler
            if self.trainer.scheduler is not None:
                self.trainer.scheduler.step()
            
            # Callback: epoch end
            combined_metrics = {**train_metrics, **val_metrics}
            self.trainer.callbacks.on_epoch_end(epoch, combined_metrics)
            
            # Check if stage complete
            if stage_monitor.is_stage_complete():
                print(f"\nStage 1 converged at epoch {epoch}")
                result['completed'] = True
                break
            
            # Check early stopping if enabled
            if self.trainer.callbacks.should_stop():
                print(f"\nEarly stopping triggered at epoch {epoch}")
                result['completed'] = True
                break
        
        # If not converged, mark as complete anyway (reached max epochs)
        if not result['completed']:
            result['completed'] = True
            result['best_checkpoint'] = str(self.trainer.checkpoint_dir / 'best.pth')
        
        return result
    
    def _run_stage2(self, train_loader, val_loader) -> Dict:
        """Run Stage 2 training."""
        result = {
            'completed': False,
            'best_checkpoint': None,
            'best_metric': float('inf'),
        }
        
        # Get stage 2 config
        stage2_cfg = self.config.get('training', {}).get('stage2', {})
        max_epochs = stage2_cfg.get('epochs', 200)
        
        # Recreate model as SPAN student
        self.trainer.stage = 2
        self.trainer.model = self.trainer._create_model(self.config)
        self.trainer.model = self.trainer.model.to(self.trainer.device)
        
        # Load best_loss from checkpoint if resuming
        saved_state = self.state_manager.load()
        if saved_state is not None and saved_state.stage2_best_metric is not None and saved_state.stage2_best_metric < float('inf'):
            self.trainer.best_loss = saved_state.stage2_best_metric
        else:
            self.trainer.best_loss = float('inf')
        
        # Reinitialize EMA model
        if self.trainer.use_ema:
            self.trainer.ema_model = self.trainer._create_ema_model()
        
        # Update trainer LR for stage 2 (convert to float to ensure correct type)
        stage2_lr = stage2_cfg.get('lr', 5e-6)
        if isinstance(stage2_lr, str):
            stage2_lr = float(stage2_lr)
        self.trainer.lr = stage2_lr
        
        # Update other stage-specific training params
        stage2_weight_decay = stage2_cfg.get('weight_decay', 0.001)
        if isinstance(stage2_weight_decay, str):
            stage2_weight_decay = float(stage2_weight_decay)
        self.trainer.weight_decay = stage2_weight_decay
        
        # Create optimizer and scheduler
        optimizer = self.trainer._create_optimizer()
        self.trainer.set_scheduler(self.trainer._create_scheduler(optimizer, stage='stage2'))
        optimizer._step_count = 0
        
        # Re-initialize callbacks for Stage 2
        self.trainer._setup_callbacks()
        
        # Setup stage monitoring
        stage_monitor = StageMonitorCallback(
            stage_controller=self.stage_controller,
            current_stage=2,
        )
        if self.trainer.callbacks is None:
            from training.callbacks import CallbackList
            self.trainer.callbacks = CallbackList([stage_monitor])
        else:
            self.trainer.callbacks.add(stage_monitor)
        
        # Training loop
        start_epoch = self.state_manager.load().stage2_epochs_completed if self.state_manager.load() else 0
        for epoch in range(start_epoch, max_epochs):
            self.trainer.current_epoch = epoch
            
            # Train
            train_metrics = self.trainer.train_stage2_epoch(train_loader, optimizer)
            
            # Verbose epoch summary
            metric_str = f"Epoch {epoch}: loss={train_metrics['loss']:.4f}, l1={train_metrics['l1']:.4f}"
            if 'perceptual' in train_metrics:
                metric_str += f", perc={train_metrics['perceptual']:.4f}"
            if 'wavelet' in train_metrics:
                metric_str += f", wav={train_metrics['wavelet']:.4f}"
            if 'fakd' in train_metrics:
                metric_str += f", fakd={train_metrics['fakd']:.4f}"
            print(metric_str)
            
            # Validate
            val_metrics = {}
            if val_loader is not None and epoch % self.trainer.val_interval == 0:
                val_metrics = self.trainer.validate(val_loader)
                print(f"Validation: loss={val_metrics['loss']:.4f}, PSNR={val_metrics['psnr']:.2f}")
                
                if val_metrics['loss'] < self.trainer.best_loss:
                    self.trainer.best_loss = val_metrics['loss']
                    self.trainer.save_checkpoint(epoch, optimizer, is_best=True)
                    result['best_checkpoint'] = str(self.trainer.checkpoint_dir / 'best.pth')
                    result['best_metric'] = val_metrics['loss']
            
            # Save checkpoint
            if epoch % self.trainer.save_interval == 0:
                self.trainer.save_checkpoint(epoch, optimizer)
            
            # Step scheduler
            if self.trainer.scheduler is not None:
                self.trainer.scheduler.step()
            
            # Callback: epoch end
            combined_metrics = {**train_metrics, **val_metrics}
            self.trainer.callbacks.on_epoch_end(epoch, combined_metrics)
            
            # Check early stopping if enabled
            if self.trainer.callbacks.should_stop():
                print(f"\nEarly stopping triggered at epoch {epoch}")
                result['completed'] = True
                break
        
        # If not converged, mark as complete anyway
        if not result['completed']:
            result['completed'] = True
            result['best_checkpoint'] = str(self.trainer.checkpoint_dir / 'best.pth')
        
        return result
    
    def reset(self):
        """Reset the trainer for fresh training."""
        self.stage_controller.reset()
        self.current_stage = 1
        self.stage_completed = False
        print("AutoStageTrainer reset. Ready for fresh training.")
    
    def get_status(self) -> Dict:
        """Get current training status."""
        return {
            'current_stage': self.current_stage,
            'stage_controller_status': self.stage_controller.get_stage_status(),
        }
