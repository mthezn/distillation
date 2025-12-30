"""
Unified Training Script for Surgical Tool Segmentation via Knowledge Distillation

This script supports three training modes:
1. Encoder Distillation (Stage 1): Align encoder features with teacher
2. Decoder Distillation (Stage 2): Train decoder to match teacher masks
3. Fine-tuning (Stage 3): End-to-end supervised training

Usage:
    # Stage 1: Encoder distillation
    python train.py --mode encoder --config configs/encoder_distillation.yaml
    
    # Stage 2: Decoder distillation
    python train.py --mode decoder --config configs/decoder_distillation.yaml --checkpoint checkpoints/encoder.pth
    
    # Stage 3: Fine-tuning
    python train.py --mode finetune --config configs/finetune.yaml --checkpoint checkpoints/decoder.pth
"""

import os
import argparse
import yaml
import copy
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, ConcatDataset
from timm.optim import create_optimizer_v2
from timm.utils import NativeScaler
from datasets import load_dataset
import wandb

from modeling.build_sam import sam_model_registry
from Dataset import ImageMaskDataset, CholecDataset
from engine import (
    train_one_epoch,           # Encoder distillation
    validate_one_epoch,        # Encoder validation
    train_one_epoch_auto,      # Decoder distillation
    validate_one_epoch_auto,   # Decoder validation
    train_one_epoch_fine,      # Fine-tuning
    validate_one_epoch_fine    # Fine-tuning validation
)
from losses import dice_loss
from utility import generate_random_name, contains_instrument


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration class to store all training parameters"""
    
    def __init__(self, config_path=None):
        # Default values
        self.mode = 'encoder'  # 'encoder', 'decoder', 'finetune'
        self.seed = 42
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Model
        self.teacher_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
        self.student_checkpoint = None
        self.model_type = "autoSamUnet"
        
        # Training
        self.epochs = 50
        self.batch_size = 2
        self.lr = 1e-4
        self.weight_decay = 1e-4
        self.optimizer = 'adamw'
        self.scheduler_type = 'cosine'  # 'cosine' or 'plateau'
        self.patience = 5
        
        # Data
        self.use_miccai = True
        self.use_cholec = False
        self.increase_data = True
        self.image_size = 1024
        
        # Augmentation
        self.use_augmentation = True
        self.horizontal_flip = 0.5
        self.vertical_flip = 0.5
        self.rotation_limit = 45
        
        # Logging
        self.wandb_project = "surgical-tool-segmentation"
        self.wandb_key = None  # Set from environment or config file
        self.checkpoint_dir = "checkpoints"
        self.experiment_name = None  # Auto-generated if None
        
        # Dataset paths
        self.miccai_train_images = []
        self.miccai_train_masks = []
        self.miccai_val_images = []
        self.miccai_val_masks = []
        
        # Load from file if provided
        if config_path and os.path.exists(config_path):
            self.load_from_yaml(config_path)
    
    def load_from_yaml(self, path):
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def save_to_yaml(self, path):
        """Save configuration to YAML file"""
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if not k.startswith('_')}
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


# ============================================================================
# DATA LOADING
# ============================================================================

def setup_transforms(config):
    """Setup data augmentation transforms"""
    
    train_transform = A.Compose([
        A.Resize(config.image_size, config.image_size),
        A.HorizontalFlip(p=config.horizontal_flip if config.use_augmentation else 0),
        A.VerticalFlip(p=config.vertical_flip if config.use_augmentation else 0),
        A.Rotate(limit=config.rotation_limit, p=0.5 if config.use_augmentation else 0),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(config.image_size, config.image_size),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
    
    return train_transform, val_transform


def setup_dataloaders(config, train_transform, val_transform):
    """Setup train and validation dataloaders"""
    
    datasets = []
    
    # MICCAI Dataset
    if config.use_miccai:
        miccai_train = ImageMaskDataset(
            image_dirs=config.miccai_train_images,
            mask_dirs=config.miccai_train_masks,
            transform=train_transform,
            increase=config.increase_data
        )
        datasets.append(miccai_train)
        
        miccai_val = ImageMaskDataset(
            image_dirs=config.miccai_val_images,
            mask_dirs=config.miccai_val_masks,
            transform=val_transform,
            increase=False
        )
    
    # CholecSeg8k Dataset
    if config.use_cholec:
        dataset_cholec = load_dataset("minwoosun/CholecSeg8k", trust_remote_code=True)
        filtered_ds = dataset_cholec["train"].filter(contains_instrument)
        cholec_train = CholecDataset(filtered_ds, transform=train_transform)
        datasets.append(cholec_train)
    
    # Combine datasets
    if len(datasets) > 1:
        train_dataset = ConcatDataset(datasets)
    else:
        train_dataset = datasets[0]
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        miccai_val,
        batch_size=config.batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader


# ============================================================================
# MODEL SETUP
# ============================================================================

def setup_teacher(config):
    """Load and freeze teacher model (SAM)"""
    
    sam = sam_model_registry["vit_h"](checkpoint=config.teacher_checkpoint)
    sam.to(device=config.device)
    sam.eval()
    
    # Freeze all parameters
    for param in sam.parameters():
        param.requires_grad = False
    
    return sam


def setup_student(config, mode):
    """Setup student model based on training mode"""
    
    model = sam_model_registry[config.model_type]()
    
    # Load checkpoint if provided
    if config.student_checkpoint and os.path.exists(config.student_checkpoint):
        print(f"Loading checkpoint from {config.student_checkpoint}")
        state_dict = torch.load(config.student_checkpoint, map_location=config.device)
        model.load_state_dict(state_dict, strict=False)
    
    model.to(device=config.device)
    model.train()
    
    # Configure trainable parameters based on mode
    if mode == 'encoder':
        # Stage 1: Only train encoder
        for param in model.parameters():
            param.requires_grad = False
        for param in model.image_encoder.parameters():
            param.requires_grad = True
    
    elif mode == 'decoder':
        # Stage 2: Only train decoder
        for param in model.parameters():
            param.requires_grad = False
        for param in model.mask_decoder.parameters():
            param.requires_grad = True
    
    elif mode == 'finetune':
        # Stage 3: Train entire model
        for param in model.parameters():
            param.requires_grad = True
    
    return model


# ============================================================================
# TRAINING SETUP
# ============================================================================

def setup_optimizer(model, config):
    """Setup optimizer"""
    
    optimizer_cfg = {
        'opt': config.optimizer,
        'lr': config.lr,
        'weight_decay': config.weight_decay,
    }
    
    optimizer = create_optimizer_v2(model, **optimizer_cfg)
    return optimizer


def setup_scheduler(optimizer, config):
    """Setup learning rate scheduler"""
    
    if config.scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
            eta_min=1e-6
        )
    elif config.scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=3,
            threshold=1e-6
        )
    else:
        raise ValueError(f"Unknown scheduler type: {config.scheduler_type}")
    
    return scheduler


def setup_criterion(mode):
    """Setup loss function based on training mode"""
    
    if mode == 'encoder':
        return nn.MSELoss()
    elif mode == 'decoder':
        return nn.BCEWithLogitsLoss()
    elif mode == 'finetune':
        return dice_loss
    else:
        raise ValueError(f"Unknown mode: {mode}")


def setup_wandb(config, mode):
    """Initialize Weights & Biases logging"""
    
    if config.wandb_key:
        wandb.login(key=config.wandb_key)
    
    # Generate experiment name if not provided
    if config.experiment_name is None:
        prefix = {'encoder': 'enc', 'decoder': 'dec', 'finetune': 'ft'}[mode]
        config.experiment_name = f"{prefix}_{generate_random_name(5)}"
    
    run = wandb.init(
        project=config.wandb_project,
        name=config.experiment_name,
        config={
            "mode": mode,
            "learning_rate": config.lr,
            "architecture": config.model_type,
            "dataset": "MICCAI" + (" + CholecSeg8k" if config.use_cholec else ""),
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "optimizer": config.optimizer,
            "weight_decay": config.weight_decay,
        }
    )
    
    return run


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(config, mode):
    """Main training function"""
    
    # Set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Setup environment
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Setup data
    train_transform, val_transform = setup_transforms(config)
    train_loader, val_loader = setup_dataloaders(config, train_transform, val_transform)
    
    # Setup models
    if mode in ['encoder', 'decoder']:
        teacher = setup_teacher(config)
    else:
        teacher = None
    
    student = setup_student(config, mode)
    
    # Setup training
    optimizer = setup_optimizer(student, config)
    scheduler = setup_scheduler(optimizer, config)
    criterion = setup_criterion(mode)
    loss_scaler = NativeScaler()
    
    # Setup logging
    run = setup_wandb(config, mode)
    
    # Training variables
    best_val_loss = float('inf')
    epochs_no_improve = 0
    checkpoint_path = os.path.join(config.checkpoint_dir, f"{config.experiment_name}.pth")
    
    # Save configuration
    config.save_to_yaml(os.path.join(config.checkpoint_dir, f"{config.experiment_name}_config.yaml"))
    
    print(f"\n{'='*80}")
    print(f"Starting {mode.upper()} training")
    print(f"Experiment: {config.experiment_name}")
    print(f"Device: {config.device}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.lr}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*80}\n")
    
    # Training loop
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        print("-" * 80)
        
        # Training
        if mode == 'encoder':
            train_stats = train_one_epoch(
                student.image_encoder,
                teacher.image_encoder,
                epoch,
                criterion,
                train_loader,
                optimizer,
                config.device,
                run
            )
        
        elif mode == 'decoder':
            train_stats = train_one_epoch_auto(
                teacher,
                student,
                train_loader,
                optimizer,
                config.device,
                run,
                epoch,
                criterion
            )
        
        elif mode == 'finetune':
            train_stats = train_one_epoch_fine(
                student,
                train_loader,
                optimizer,
                config.device,
                run,
                epoch,
                criterion
            )
        
        # Validation
        if mode == 'encoder':
            val_loss = validate_one_epoch(
                student.image_encoder,
                teacher.image_encoder,
                val_loader,
                criterion,
                config.device,
                epoch,
                run
            )
        
        elif mode == 'decoder':
            val_loss = validate_one_epoch_auto(
                teacher,
                student,
                val_loader,
                criterion,
                config.device,
                epoch,
                run
            )
        
        elif mode == 'finetune':
            val_loss = validate_one_epoch_fine(
                student,
                val_loader,
                config.device,
                run,
                epoch,
                criterion
            )
        
        # Update scheduler
        if config.scheduler_type == 'cosine':
            scheduler.step()
        else:
            scheduler.step(val_loss)
        
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            print(f"✓ Validation loss improved: {best_val_loss:.4f} → {val_loss:.4f}")
            best_val_loss = val_loss
            epochs_no_improve = 0
            
            torch.save(student.state_dict(), checkpoint_path)
            print(f"✓ Model saved to {checkpoint_path}")
        else:
            epochs_no_improve += 1
            print(f"✗ No improvement for {epochs_no_improve} epoch(s)")
        
        # Early stopping
        if epochs_no_improve >= config.patience:
            print(f"\n⚠ Early stopping triggered after {epoch + 1} epochs")
            break
        
        # Clear cache
        torch.cuda.empty_cache()
    
    print(f"\n{'='*80}")
    print(f"Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {checkpoint_path}")
    print(f"{'='*80}\n")
    
    wandb.finish()
    return checkpoint_path


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    """Parse command line arguments"""
    
    parser = argparse.ArgumentParser(
        description="Unified training script for surgical tool segmentation"
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['encoder', 'decoder', 'finetune'],
        help='Training mode: encoder (stage 1), decoder (stage 2), or finetune (stage 3)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to load'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for training'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (overrides config)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )
    
    parser.add_argument(
        '--wandb_key',
        type=str,
        default=None,
        help='Weights & Biases API key'
    )
    
    parser.add_argument(
        '--experiment_name',
        type=str,
        default=None,
        help='Experiment name for logging'
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    
    args = parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Override with command line arguments
    if args.checkpoint:
        config.student_checkpoint = args.checkpoint
    if args.device:
        config.device = args.device
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.lr = args.lr
    if args.wandb_key:
        config.wandb_key = args.wandb_key
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    
    # Start training
    checkpoint_path = train(config, args.mode)
    
    print(f"\n✓ Training complete! Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
