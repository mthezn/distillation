#!/usr/bin/python3

import os
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import json
from tqdm import tqdm
import argparse

# Import your existing modules
from modeling.build_sam import sam_model_registry
from DatasetMMI import MMIDataset
from losses import *
from engine import train_one_epoch_fine, validate_one_epoch_fine
import albumentations as A
from albumentations.pytorch import ToTensorV2
from losses import *

class KFoldCrossValidator:
    def __init__(self, dataset, n_splits=5, shuffle=True, random_state=42, stratified=False):
        """
        Initialize K-Fold Cross Validation
        
        Args:
            dataset: Your MMIDataset instance
            n_splits: Number of folds (default: 5)
            shuffle: Whether to shuffle data before splitting
            random_state: Random seed for reproducibility
            stratified: Whether to use stratified k-fold (useful for imbalanced classes)
        """
        self.dataset = dataset
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.stratified = stratified
        
        # Create the splitter
        if stratified:
            # For stratified k-fold, you need class labels
            # This assumes you can extract class information from your dataset
            self.kfold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            self.labels = self._extract_labels()
        else:
            self.kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            self.labels = None
    
    def _extract_labels(self):
        """
        Extract labels for stratified k-fold
        For segmentation, you might want to stratify based on:
        - Presence/absence of instruments
        - Number of instruments
        - Image complexity, etc.
        """
        labels = []
        for i in range(len(self.dataset)):
            # Example: stratify based on whether image contains instruments
            try:
                _, mask = self.dataset[i]
                # Simple binary classification: has instrument (1) or not (0)
                has_instrument = 1 if mask.sum() > 0 else 0
                labels.append(has_instrument)
            except:
                # Fallback to random labels if extraction fails
                labels.append(0)
        return np.array(labels)
    
    def get_fold_datasets(self, fold_idx):
        """
        Get train and validation datasets for a specific fold
        
        Args:
            fold_idx: Index of the current fold
            
        Returns:
            train_dataset, val_dataset: Subset datasets for training and validation
        """
        indices = np.arange(len(self.dataset))
        
        if self.stratified and self.labels is not None:
            splits = list(self.kfold.split(indices, self.labels))
        else:
            splits = list(self.kfold.split(indices))
        
        train_indices, val_indices = splits[fold_idx]
        
        # Create subset datasets
        train_dataset = Subset(self.dataset, train_indices)
        val_dataset = Subset(self.dataset, val_indices)
        
        return train_dataset, val_dataset
    
    def get_all_folds(self):
        """
        Generator that yields all fold datasets
        
        Yields:
            fold_idx, train_dataset, val_dataset
        """
        for fold_idx in range(self.n_splits):
            train_dataset, val_dataset = self.get_fold_datasets(fold_idx)
            yield fold_idx, train_dataset, val_dataset

def create_model(model_type="autoSam", checkpoint_path=None, device="cuda"):
    """Create and initialize model"""
    model = sam_model_registry[model_type](checkpoint=checkpoint_path)
    model.to(device)
    return model

def create_optimizer_scheduler(model, args, total_epochs):
    """Create optimizer and scheduler"""
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, 
                                  momentum=args.momentum, weight_decay=args.weightdecay)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, 
                                   weight_decay=args.weightdecay)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, 
                                    weight_decay=args.weightdecay)
    
    # Scheduler setup
    if args.scheduler == 'Cosine':
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
        warmup_epochs = 5
        warmup_scheduler = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, 
                                  total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, 
                                           eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                               milestones=[warmup_epochs])
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    return optimizer, scheduler

def train_single_fold(fold_idx, train_dataset, val_dataset, args, device, 
                     saving_path, model_checkpoint):
    """
    Train a single fold
    
    Args:
        fold_idx: Current fold index
        train_dataset, val_dataset: Datasets for this fold
        args: Training arguments
        device: Device to use
        saving_path: Path to save results
        model_checkpoint: Path to model checkpoint
    
    Returns:
        Dictionary with fold results
    """
    print(f"\n{'='*50}")
    print(f"Training Fold {fold_idx + 1}/{args.n_splits}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"{'='*50}")
    
    # Create fold-specific saving directory
    fold_path = saving_path / f"fold_{fold_idx + 1}"
    fold_path.mkdir(exist_ok=True)
    
    # Data loaders
    g = torch.Generator()
    g.manual_seed(0)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, 
                            num_workers=args.workers, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False,
                          num_workers=args.workers, generator=g)
    
    # Model setup
    model = create_model("autoSam", model_checkpoint, device)
    optimizer, scheduler = create_optimizer_scheduler(model, args, args.epochs)
    
    # Loss function
    criterion = create_loss_criterion()  # Use your fixed loss function
    
    # Training variables
    best_val_loss = float('inf')
    epochs_no_improve = 0
    fold_results = {
        'fold': fold_idx + 1,
        'train_losses': [],
        'val_losses': [],
        'train_dice_losses': [],
        'val_dice_losses': [],
        'train_focal_losses': [],
        'val_focal_losses': [],
        'best_val_loss': None,
        'best_epoch': None
    }
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"Fold {fold_idx + 1}, Epoch {epoch + 1}: LR = {scheduler.get_last_lr()}")
        
        # Training
        model.train()
        train_stats, diceT, focalT = train_one_epoch_fine(
            model=model, dataloader=train_loader, optimizer=optimizer,
            device=device, epoch=epoch, criterion=criterion,
            path=fold_path, headless=args.headless, mixed_precision=True
        )
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss, diceV, focalV = validate_one_epoch_fine(
                model=model, dataloader=val_loader, device=device,
                epoch=epoch, criterion=criterion, path=fold_path,
                headless=args.headless, mixed_precision=True
            )
        
        # Store results
        fold_results['train_losses'].append(train_stats if isinstance(train_stats, float) else train_stats['loss'])
        fold_results['val_losses'].append(val_loss)
        fold_results['train_dice_losses'].append(diceT if isinstance(diceT, float) else diceT.item())
        fold_results['val_dice_losses'].append(diceV if isinstance(diceV, float) else diceV.item())
        fold_results['train_focal_losses'].append(focalT if isinstance(focalT, float) else focalT.item())
        fold_results['val_focal_losses'].append(focalV if isinstance(focalV, float) else focalV.item())
        
        print(f"Fold {fold_idx + 1}, Epoch {epoch + 1} - Train Loss: {fold_results['train_losses'][-1]:.4f}, "
              f"Val Loss: {val_loss:.4f}")
        
        # Save best model for this fold
        if val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}")
            best_val_loss = val_loss
            epochs_no_improve = 0
            fold_results['best_val_loss'] = val_loss
            fold_results['best_epoch'] = epoch + 1
            torch.save(model.state_dict(), fold_path / "best_model.pth")
        else:
            epochs_no_improve += 1
        
        # Early stopping
        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered for fold {fold_idx + 1}")
            break
        
        scheduler.step()
    
    # Plot fold results
    plot_fold_results(fold_results, fold_path)
    
    # Save fold results
    with open(fold_path / "fold_results.json", 'w') as f:
        json.dump(fold_results, f, indent=2)
    
    return fold_results

def plot_fold_results(fold_results, save_path):
    """Plot and save fold training curves"""
    plt.figure(figsize=(15, 10))
    
    # Main loss plot
    plt.subplot(2, 2, 1)
    plt.plot(fold_results['train_losses'], label='Train Loss', color='blue')
    plt.plot(fold_results['val_losses'], label='Val Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Fold {fold_results["fold"]} - Total Loss')
    plt.legend()
    plt.grid(True)
    
    # Dice loss plot
    plt.subplot(2, 2, 2)
    plt.plot(fold_results['train_dice_losses'], label='Train Dice', color='green')
    plt.plot(fold_results['val_dice_losses'], label='Val Dice', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Loss')
    plt.title(f'Fold {fold_results["fold"]} - Dice Loss')
    plt.legend()
    plt.grid(True)
    
    # Focal loss plot
    plt.subplot(2, 2, 3)
    plt.plot(fold_results['train_focal_losses'], label='Train Focal', color='purple')
    plt.plot(fold_results['val_focal_losses'], label='Val Focal', color='brown')
    plt.xlabel('Epoch')
    plt.ylabel('Focal Loss')
    plt.title(f'Fold {fold_results["fold"]} - Focal Loss')
    plt.legend()
    plt.grid(True)
    
    # Combined plot
    plt.subplot(2, 2, 4)
    plt.plot(fold_results['train_losses'], label='Train Total', alpha=0.7)
    plt.plot(fold_results['val_losses'], label='Val Total', alpha=0.7)
    plt.plot(fold_results['train_dice_losses'], label='Train Dice', alpha=0.7)
    plt.plot(fold_results['val_dice_losses'], label='Val Dice', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Fold {fold_results["fold"]} - All Losses')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path / "fold_training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_cv_summary(all_fold_results, save_path):
    """Plot summary of all folds"""
    n_folds = len(all_fold_results)
    
    # Collect final validation losses from each fold
    final_val_losses = [results['val_losses'][-1] for results in all_fold_results]
    best_val_losses = [results['best_val_loss'] for results in all_fold_results]
    
    plt.figure(figsize=(15, 10))
    
    # Plot validation loss for all folds
    plt.subplot(2, 2, 1)
    for i, results in enumerate(all_fold_results):
        plt.plot(results['val_losses'], label=f'Fold {i+1}', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss - All Folds')
    plt.legend()
    plt.grid(True)
    
    # Plot mean and std of validation losses
    plt.subplot(2, 2, 2)
    max_epochs = max(len(results['val_losses']) for results in all_fold_results)
    val_losses_padded = []
    for results in all_fold_results:
        losses = results['val_losses']
        # Pad with last value if needed
        while len(losses) < max_epochs:
            losses.append(losses[-1])
        val_losses_padded.append(losses)
    
    val_losses_array = np.array(val_losses_padded)
    mean_val_loss = np.mean(val_losses_array, axis=0)
    std_val_loss = np.std(val_losses_array, axis=0)
    
    epochs = range(1, len(mean_val_loss) + 1)
    plt.plot(epochs, mean_val_loss, label='Mean Val Loss', color='red', linewidth=2)
    plt.fill_between(epochs, mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, 
                     alpha=0.3, color='red', label='±1 Std')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Mean Validation Loss Across Folds')
    plt.legend()
    plt.grid(True)
    
    # Box plot of final validation losses
    plt.subplot(2, 2, 3)
    plt.boxplot([final_val_losses, best_val_losses], labels=['Final Val Loss', 'Best Val Loss'])
    plt.ylabel('Loss')
    plt.title('Distribution of Validation Losses')
    plt.grid(True, alpha=0.3)
    
    # Fold performance comparison
    plt.subplot(2, 2, 4)
    fold_numbers = [f'Fold {i+1}' for i in range(n_folds)]
    plt.bar(fold_numbers, best_val_losses, alpha=0.7, color='skyblue', edgecolor='navy')
    plt.ylabel('Best Validation Loss')
    plt.title('Best Validation Loss by Fold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add mean line
    mean_best_loss = np.mean(best_val_losses)
    plt.axhline(y=mean_best_loss, color='red', linestyle='--', 
                label=f'Mean: {mean_best_loss:.4f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path / "cv_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return mean_best_loss, std_val_loss

def main_kfold():
    parser = argparse.ArgumentParser(description="K-Fold Cross-Validation for SAM Fine-tuning")
    parser.add_argument("--input", type=str, default="dataset_mmi_3107")
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--batch", default=8, type=int)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds for CV")
    parser.add_argument("--stratified", action='store_true', help="Use stratified k-fold")
    parser.add_argument("--optimizer", type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument("--weightdecay", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--scheduler", type=str, default='Cosine')
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--headless", action='store_true')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Seed for reproducibility
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    
    # Create experiment directory
    root_dir = Path("runs/kfold_cv_experiment")
    root_dir.mkdir(parents=True, exist_ok=True)
    test_n = len(list(n for n in os.listdir(root_dir) if n.startswith('exp_')))
    saving_path = root_dir / f"exp_{test_n + 1}"
    saving_path.mkdir(exist_ok=True)
    
    print(f"Starting {args.n_splits}-Fold Cross-Validation")
    print(f"Results will be saved to: {saving_path}")
    
    # Load full dataset (without train/val split)
    prob = 0.5
    full_transform = A.Compose([
        A.Resize(1024, 1024),
        A.HorizontalFlip(p=prob),
        A.VerticalFlip(p=prob),
        A.Rotate(limit=30, p=prob),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=prob),
        A.GaussianBlur(blur_limit=(3, 7), p=prob),
        #A.CoarseDropout(max_holes=12, max_height=64, max_width=64, fill_value=0, p=0.3),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
    
    # Load the full dataset (you might need to modify MMIDataset to load all data)
    full_dataset = MMIDataset(root_dir=args.input, split='all', 
                             transform=full_transform, num_classes=args.num_classes)
    
    print(f"Total dataset size: {len(full_dataset)}")
    
    # Initialize K-Fold cross-validator
    cv = KFoldCrossValidator(full_dataset, n_splits=args.n_splits, 
                           stratified=args.stratified, random_state=42)
    
    # Model checkpoint path
    #model_checkpoint = "/home/shared-nearmrs/mdezenDatasets/autoSamVitH4BeJo.pth"
    model_checkpoint = "/home/shared-nearmrs/mdezenDatasets/autoSamFineVitHSuOkl.pth"
    
    # Store results from all folds
    all_fold_results = []
    
    # Run cross-validation
    for fold_idx, train_dataset, val_dataset in cv.get_all_folds():
        fold_results = train_single_fold(
            fold_idx, train_dataset, val_dataset, args, 
            device, saving_path, model_checkpoint
        )
        all_fold_results.append(fold_results)
        
        # Clear GPU memory between folds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Analyze and summarize results
    print(f"\n{'='*60}")
    print("K-FOLD CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    best_val_losses = [results['best_val_loss'] for results in all_fold_results]
    final_val_losses = [results['val_losses'][-1] for results in all_fold_results]
    
    print(f"Best validation losses by fold: {[f'{loss:.4f}' for loss in best_val_losses]}")
    print(f"Mean best validation loss: {np.mean(best_val_losses):.4f} ± {np.std(best_val_losses):.4f}")
    print(f"Final validation losses by fold: {[f'{loss:.4f}' for loss in final_val_losses]}")
    print(f"Mean final validation loss: {np.mean(final_val_losses):.4f} ± {np.std(final_val_losses):.4f}")
    
    # Plot summary
    mean_loss, std_loss = plot_cv_summary(all_fold_results, saving_path)
    
    # Save summary results
    summary = {
        'n_splits': args.n_splits,
        'stratified': args.stratified,
        'best_val_losses': best_val_losses,
        'final_val_losses': final_val_losses,
        'mean_best_val_loss': float(np.mean(best_val_losses)),
        'std_best_val_loss': float(np.std(best_val_losses)),
        'mean_final_val_loss': float(np.mean(final_val_losses)),
        'std_final_val_loss': float(np.std(final_val_losses)),
        'args': vars(args)
    }
    
    with open(saving_path / "cv_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Cross-validation completed! Results saved to {saving_path}")
    
    return all_fold_results, summary

if __name__ == '__main__':
    main_kfold()