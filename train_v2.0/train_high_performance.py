#!/usr/bin/env python3
"""
High-Performance 3D CLIP Training Script
Optimized for maximum performance on medical CT volumes
"""

import os
import sys
import argparse
import math
import json
from datetime import datetime
from pathlib import Path
import wandb
from typing import Optional, Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import autocast, GradScaler

# Import from parent directory
from train import make, preprocess_text, setup_validation
from eval import evaluate
from .advanced_data_loader import create_advanced_data_loaders


def parse_args():
    parser = argparse.ArgumentParser(description="High-Performance 3D CLIP Training")
    
    # Data paths
    parser.add_argument('--ct_filepath', type=str, required=True, help='Path to CT volumes HDF5')
    parser.add_argument('--txt_filepath', type=str, required=True, help='Path to text reports CSV')
    parser.add_argument('--val_ct_filepath', type=str, help='Validation CT volumes')
    parser.add_argument('--val_label_path', type=str, help='Validation labels')
    
    # Model architecture
    parser.add_argument('--dinov2_model_name', type=str, default='dinov2_vitl14', 
                       choices=['dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'],
                       help='DinoV2 backbone size (larger = better performance)')
    parser.add_argument('--embed_dim', type=int, default=768, help='Embedding dimension')
    parser.add_argument('--context_length', type=int, default=77, help='Text context length')
    parser.add_argument('--transformer_depth', type=int, default=3, help='Slice fusion transformer depth')
    parser.add_argument('--transformer_heads', type=int, default=12, help='Attention heads')
    parser.add_argument('--freeze_dinov2', action='store_true', help='Freeze DinoV2 backbone')
    
    # Training hyperparameters (optimized for performance)
    parser.add_argument('--batch_size', type=int, default=8, help='Per-GPU batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='Peak learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio of total steps')
    parser.add_argument('--grad_accum_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping norm')
    
    # Advanced training techniques
    parser.add_argument('--use_ema', action='store_true', help='Use Exponential Moving Average')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='EMA decay rate')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing')
    parser.add_argument('--mixup_alpha', type=float, default=0.0, help='Mixup alpha (0 = disabled)')
    parser.add_argument('--temperature_init', type=float, default=0.07, help='Initial temperature')
    parser.add_argument('--temperature_learnable', action='store_true', help='Learn temperature')
    
    # Data augmentation
    parser.add_argument('--use_augmentation', action='store_true', help='Enable data augmentation')
    parser.add_argument('--noise_std', type=float, default=0.01, help='Gaussian noise std')
    parser.add_argument('--brightness_delta', type=float, default=0.1, help='Brightness variation')
    
    # Performance optimization
    parser.add_argument('--compile_model', action='store_true', help='Use torch.compile (PyTorch 2.0+)')
    parser.add_argument('--channels_last', action='store_true', help='Use channels_last memory format')
    parser.add_argument('--dataloader_workers', type=int, default=8, help='DataLoader workers per GPU')
    parser.add_argument('--pin_memory', action='store_true', default=True, help='Pin memory')
    parser.add_argument('--persistent_workers', action='store_true', default=True, help='Persistent workers')
    
    # Compatibility modes
    parser.add_argument('--use_advanced_loader', action='store_true', help='Use advanced data loader with augmentation')
    parser.add_argument('--use_advanced_optimizer', action='store_true', help='Use advanced optimizer with parameter groups')
    
    # Distributed training
    parser.add_argument('--use_ddp', action='store_true', help='Use DistributedDataParallel')
    parser.add_argument('--backend', type=str, default='nccl', help='DDP backend')
    
    # Validation and logging
    parser.add_argument('--do_validate', action='store_true', help='Run validation')
    parser.add_argument('--valid_interval', type=int, default=500, help='Validation interval (steps)')
    parser.add_argument('--log_interval', type=int, default=50, help='Logging interval (steps)')
    parser.add_argument('--save_interval', type=int, default=2000, help='Checkpoint save interval (steps)')
    
    # Paths and metadata
    parser.add_argument('--save_dir', type=str, default='checkpoints_hp/', help='Save directory')
    parser.add_argument('--model_name', type=str, default='clip_3d_hp', help='Model name')
    parser.add_argument('--wandb_project', type=str, default='clip_3d_ct', help='W&B project name')
    parser.add_argument('--resume_path', type=str, help='Resume training from checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Compatibility
    parser.add_argument('--column', type=str, default='Impressions_EN', help='Text column name')
    parser.add_argument('--pretrained', type=bool, default=False, help='Compatibility arg')
    parser.add_argument('--val_batch_size', type=int, default=4, help='Validation batch size')
    
    return parser.parse_args()


class EMAModel:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class HighPerformanceTrainer:
    def __init__(self, config: argparse.Namespace):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rank = 0
        self.world_size = 1
        
        # Set up distributed training
        if config.use_ddp and torch.cuda.is_available():
            self._setup_ddp()
        
        # Set random seeds
        self._set_seeds()
        
        # Initialize model and data
        self._init_model_and_data()
        
        # Initialize training components
        self._init_training_components()
        
        # Set up logging
        if self.rank == 0:
            self._setup_logging()
    
    def _setup_ddp(self):
        """Initialize distributed training"""
        dist.init_process_group(backend=self.config.backend)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        torch.cuda.set_device(self.rank)
        print(f"DDP initialized: rank {self.rank}/{self.world_size}")
    
    def _set_seeds(self):
        """Set random seeds for reproducibility"""
        torch.manual_seed(self.config.seed + self.rank)
        torch.cuda.manual_seed_all(self.config.seed + self.rank)
    
    def _init_model_and_data(self):
        """Initialize model and data loaders"""
        # Create model and data using original make function for compatibility
        self.model, self.train_loader, _, self.criterion, self.optimizer = make(
            self.config, self.config.ct_filepath, self.config.txt_filepath
        )
        
        # Optionally replace with advanced data loader if requested
        if hasattr(self.config, 'use_advanced_loader') and self.config.use_advanced_loader:
            self.train_loader, self.val_loader = create_advanced_data_loaders(
                train_ct_path=self.config.ct_filepath,
                train_txt_path=self.config.txt_filepath,
                val_ct_path=self.config.val_ct_filepath,
                val_txt_path=self.config.val_label_path,
                batch_size=self.config.batch_size,
                num_workers=self.config.dataloader_workers,
                pin_memory=self.config.pin_memory,
                persistent_workers=self.config.persistent_workers,
                use_ddp=self.config.use_ddp
            )
        else:
            # Use original data loader - setup validation if needed
            self.val_loader = None
            if self.config.do_validate and self.config.val_ct_filepath and self.config.val_label_path:
                self.val_data = setup_validation(self.config)
        
        # Move to device and set memory format
        self.model = self.model.to(self.device)
        if self.config.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        
        # Wrap with DDP
        if self.config.use_ddp:
            self.model = DDP(self.model, device_ids=[self.rank], 
                           output_device=self.rank, find_unused_parameters=True)
        
        # Compile model for performance (PyTorch 2.0+)
        if self.config.compile_model and hasattr(torch, 'compile'):
            print("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)
    
    def _init_training_components(self):
        """Initialize optimizer, scheduler, and other training components"""
        # Calculate training steps
        self.total_steps = self.config.epochs * len(self.train_loader) // self.config.grad_accum_steps
        self.warmup_steps = int(self.total_steps * self.config.warmup_ratio)
        
        # Use original optimizer from make() function, but update learning rate and weight decay
        if hasattr(self.config, 'use_advanced_optimizer') and self.config.use_advanced_optimizer:
            # Advanced optimizer with parameter groups
            param_groups = [
                {'params': [], 'lr': self.config.lr, 'weight_decay': self.config.weight_decay},  # backbone
                {'params': [], 'lr': self.config.lr * 10, 'weight_decay': 0.0}  # projection layers
            ]
            
            model_for_params = self.model.module if hasattr(self.model, 'module') else self.model
            for name, param in model_for_params.named_parameters():
                if 'visual.projection' in name or 'text_projection' in name:
                    param_groups[1]['params'].append(param)
                else:
                    param_groups[0]['params'].append(param)
            
            self.optimizer = optim.AdamW(param_groups, eps=1e-6, betas=(0.9, 0.98))
        else:
            # Use original optimizer but update parameters if needed
            if hasattr(self.config, 'lr') and self.config.lr != 1e-4:
                # Update learning rate to match config
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.config.lr
            if hasattr(self.config, 'weight_decay') and self.config.weight_decay != 0:
                # Update weight decay
                for param_group in self.optimizer.param_groups:
                    param_group['weight_decay'] = self.config.weight_decay
        
        # Cosine annealing with warmup
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / self.warmup_steps
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # Mixed precision scaler
        self.scaler = GradScaler()
        
        # EMA model
        if self.config.use_ema:
            self.ema = EMAModel(self.model, self.config.ema_decay)
        
        # Initialize counters
        self.global_step = 0
        self.epoch = 0
        self.best_val_score = 0.0
    
    def _setup_logging(self):
        """Set up W&B logging"""
        run_name = f"{self.config.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        wandb.init(
            project=self.config.wandb_project,
            name=run_name,
            config=vars(self.config),
            tags=["3d_clip", "medical_ct", "high_performance"]
        )
        
        # Create save directory
        self.save_dir = Path(self.config.save_dir) / run_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.save_dir / 'config.json', 'w') as f:
            json.dump(vars(self.config), f, indent=2)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        # Set epoch for distributed sampler
        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(self.epoch)
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, data in enumerate(self.train_loader):
            images = data['img'].to(self.device, non_blocking=True)  # (B, 1, D, H, W)
            texts = data['txt']
            
            # Convert to channels_last if needed
            if self.config.channels_last:
                images = images.to(memory_format=torch.channels_last)
            
            # Preprocess text - EXACTLY like original
            model_for_text = self.model.module if hasattr(self.model, 'module') else self.model
            texts = preprocess_text(texts, model_for_text).to(self.device, non_blocking=True)
            
            # Forward pass with autocast - EXACTLY like original
            with autocast('cuda'):
                logits_per_image, logits_per_text = self.model(images, texts)
                # Use original loss computation
                labels = torch.arange(images.size(0), device=self.device)
                loss_img = self.criterion(logits_per_image, labels)
                loss_txt = self.criterion(logits_per_text, labels)
                loss = (loss_img + loss_txt) / 2
            
            # Backward pass
            self.scaler.scale(loss / self.config.grad_accum_steps).backward()
            
            # Update weights
            if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                # Update EMA
                if self.config.use_ema:
                    self.ema.update(self.model)
                
                # Scheduler step
                self.scheduler.step()
                self.global_step += 1
                
                # Logging
                if self.rank == 0 and self.global_step % self.config.log_interval == 0:
                    self._log_metrics(loss.item(), self.scheduler.get_last_lr()[0])
                
                # Validation
                if (self.config.do_validate and self.global_step % self.config.valid_interval == 0):
                    self._run_validation()
                
                # Save checkpoint
                if self.global_step % self.config.save_interval == 0:
                    self._save_checkpoint()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _log_metrics(self, loss: float, lr: float):
        """Log training metrics"""
        metrics = {
            'train/loss': loss,
            'train/lr': lr,
            'train/epoch': self.epoch,
            'train/step': self.global_step,
        }
        
        if self.config.use_ema:
            metrics['train/ema_decay'] = self.ema.decay
        
        wandb.log(metrics, step=self.global_step)
        print(f"Step {self.global_step}: Loss={loss:.4f}, LR={lr:.2e}")
    
    def _run_validation(self):
        """Run validation evaluation"""
        if not hasattr(self, 'val_data') or self.val_data is None:
            return
        
        print("Running validation...")
        
        # Use EMA model for validation if available
        if self.config.use_ema:
            self.ema.apply_shadow(self.model)
        
        try:
            # Use original validation logic from setup_validation
            val_loader, val_y, label_names, templates, input_resolution = self.val_data
            
            if val_loader is not None:
                # Run evaluation using original evaluate function
                results = evaluate(
                    model=self.model,
                    data_loader=val_loader,
                    device=self.device,
                    amp=True
                )
                
                # Log validation metrics
                val_metrics = {
                    'val/auc_mean': results.get('auc_mean', 0.0),
                    'val/acc_mean': results.get('acc_mean', 0.0),
                    'step': self.global_step
                }
                
                wandb.log(val_metrics, step=self.global_step)
                
                # Save best model
                current_score = results.get('auc_mean', 0.0)
                if current_score > self.best_val_score:
                    self.best_val_score = current_score
                    self._save_checkpoint(is_best=True)
                    print(f"New best validation AUC: {current_score:.4f}")
            else:
                print("Validation data loader not available")
        
        except Exception as e:
            print(f"Validation failed: {e}")
        
        finally:
            # Restore original model weights
            if self.config.use_ema:
                self.ema.restore(self.model)
            self.model.train()
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        if self.rank != 0:
            return
        
        model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        
        checkpoint = {
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_val_score': self.best_val_score,
            'config': vars(self.config)
        }
        
        if self.config.use_ema:
            checkpoint['ema_state_dict'] = self.ema.shadow
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f'checkpoint_step_{self.global_step}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.save_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
        
        # Keep only recent checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self, keep_last: int = 3):
        """Remove old checkpoints to save disk space"""
        checkpoints = list(self.save_dir.glob('checkpoint_step_*.pt'))
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
        
        for checkpoint in checkpoints[:-keep_last]:
            checkpoint.unlink()
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config.epochs} epochs...")
        print(f"Total steps: {self.total_steps}, Warmup steps: {self.warmup_steps}")
        
        for epoch in range(self.config.epochs):
            self.epoch = epoch
            
            # Train one epoch
            avg_loss = self.train_epoch()
            
            if self.rank == 0:
                print(f"Epoch {epoch+1}/{self.config.epochs} completed. Avg loss: {avg_loss:.4f}")
                
                # Log epoch metrics
                wandb.log({
                    'epoch': epoch + 1,
                    'train/epoch_loss': avg_loss,
                }, step=self.global_step)
        
        # Final validation and save
        if self.config.do_validate:
            self._run_validation()
        
        self._save_checkpoint()
        
        if self.rank == 0:
            print("Training completed!")
            print(f"Best validation score: {self.best_val_score:.4f}")
            print(f"Models saved to: {self.save_dir}")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.config.use_ddp:
            dist.destroy_process_group()
        
        if self.rank == 0 and wandb.run is not None:
            wandb.finish()


def main():
    config = parse_args()
    
    # Validate configuration
    assert torch.cuda.is_available(), "CUDA is required for high-performance training"
    
    # Initialize trainer
    trainer = HighPerformanceTrainer(config)
    
    try:
        # Start training
        trainer.train()
    finally:
        # Cleanup
        trainer.cleanup()


if __name__ == '__main__':
    main()