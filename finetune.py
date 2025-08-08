#!/usr/bin/python3

import albumentations as A
from torch.utils.data import DataLoader
from modeling.build_sam import sam_model_registry

from utils import *
from albumentations.pytorch import ToTensorV2

import numpy as np
import torch.nn as nn
from engine import train_one_epoch_fine, validate_one_epoch_fine
from torch.utils.data import ConcatDataset

from timm.optim import create_optimizer_v2
from timm.utils import NativeScaler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import torch
import gc
from datasets import load_dataset
from utility import generate_random_name, contains_instrument

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchmetrics
import torchvision.utils
import torchvision.transforms as transforms
import torch
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import albumentations as albu
from tqdm import tqdm
import random
import argparse
from pathlib import Path
from Dataset import MMIDataset
from losses import *

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0.000005, path=None, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model,):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def seed_all(seed):
    if not seed:
        seed = 10

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def check_model_weights(model, epoch):
    """Check model weights for NaN or extreme values"""
    nan_params = []
    extreme_params = []
    
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            nan_params.append(name)
        if param.abs().max() > 100:
            extreme_params.append((name, param.abs().max().item()))
    
    if nan_params:
        print(f"Epoch {epoch}: NaN parameters found in: {nan_params}")
        return False
    
    if extreme_params:
        print(f"Epoch {epoch}: Extreme parameter values: {extreme_params}")
    
    return True

def reinitialize_problematic_layers(model):
    """Reinitialize layers that might be causing NaN"""
    print("Reinitializing potentially problematic layers...")
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            if hasattr(module, 'weight') and torch.isnan(module.weight).any():
                print(f"Reinitializing {name}")
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            if hasattr(module, 'weight') and torch.isnan(module.weight).any():
                print(f"Reinitializing {name}")
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

def main():
    parser = argparse.ArgumentParser(description="Train a model for image semantic classification (PyTorch)")
    parser.add_argument("--input", type = str, default = "dataset_mmi_3107")
    parser.add_argument("--workers", default = 8, type=int, help="Number of workers for data loading")
    parser.add_argument("--batch", default = 8, type=int)
    parser.add_argument("--epochs", type = int, default = 100)
    parser.add_argument("--num_classes", type = int, default = 1, help="Number of classes in the dataset")
    parser.add_argument("--optimizer", type = str, default = 'AdamW', choices=['SGD', 'Adam', 'AdamW'])
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument("--weightdecay", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--scheduler", type = str, default = 'Cosine')
    parser.add_argument("--patience", type = int, default = 10, help="Early stopping patience")
    parser.add_argument("--headless", default = False, type=bool, help="Run without displaying plots")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed
    g = torch.Generator()
    g.manual_seed(0)
    seed_all(seed=123)

    # Creation of runs file for the experiment
    root_dir = Path("runs/mmi_experiment")
    if not root_dir.exists():
        root_dir.mkdir(parents=True, exist_ok=True)
    test_n = len(list(n for n in os.listdir("runs/mmi_experiment") if n.startswith('exp_')))
    os.makedirs(root_dir / ("exp_" + str(test_n+1)), exist_ok=True)

    saving_path = root_dir / ("exp_" + str(test_n+1))
   
    print("Loading dataset...")
    dataset_mmi = args.input
    
    prob = 0.5
    train_transform = A.Compose([
        A.Resize(1024, 1024),
        A.HorizontalFlip(p=prob),
        A.VerticalFlip(p=prob),
        A.Rotate(limit=15, p=prob),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=prob),
        A.GaussianBlur(blur_limit=(3, 7), p=prob),
        #A.CoarseDropout(max_holes=8, max_height=128, max_width=128, fill_value=0, p=prob),
        #A.GridDistortion(p=prob),  # Add elastic deformation
        #A.OpticalDistortion(p=prob),
        #A.GaussNoise(var_limit=(10.0, 50.0), p=prob),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    validation_transform = A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    train_dataset = MMIDataset(root_dir=dataset_mmi, split='train', transform=train_transform, num_classes=args.num_classes)
    validation_dataset = MMIDataset(root_dir=dataset_mmi, split='valid', transform=validation_transform, num_classes=args.num_classes)

    train_loader = DataLoader(train_dataset, batch_size = args.batch, shuffle = True, num_workers = args.workers, worker_init_fn = seed_worker, generator=g)
    validation_loader = DataLoader(validation_dataset, batch_size = args.batch, shuffle = False, num_workers = args.workers, worker_init_fn = seed_worker, generator=g)
    
    print("Dataset loaded!")
    print("Number of training samples:", len(train_dataset))
    print("Number of validation samples:", len(validation_dataset))

    # Model setup
    print("Setting up model...")
    model_type = "autoSam"
  
    autosam_checkpoint = "/home/shared-nearmrs/mdezenDatasets/autoSamVitH4BeJo.pth"
    #autosam_checkpoint = "/home/shared-nearmrs/mdezenDatasets/autoSamFineVitHSuOkl.pth"
    model = sam_model_registry[model_type](checkpoint=autosam_checkpoint)
    model.to(device)

    print("Checking initial model weights...")
    check_model_weights(model, -1)
    # Reinitialize problematic layers if needed
    reinitialize_problematic_layers(model)
    #criterion = lambda pred, target: 0.3 * DiceLoss()(pred, target) + 0.7 * FocalBCELoss()(pred, target)
    alpha = 0.3
    beta = 0.7
    
    #criterion = create_loss_criterion(dice_weight=alpha, focal_weight=beta, use_tversky=True)
    criterion = create_robust_loss_criterion(dice_weight=alpha, focal_weight=beta)

    if args.optimizer == 'SGD' :
        optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum=args.momentum, weight_decay=args.weightdecay)
    elif args.optimizer == 'Adam' :
        optimizer = torch.optim.Adam(model.parameters(), lr= args.lr, weight_decay= args.weightdecay)
    elif args.optimizer == 'AdamW' :
        optimizer = torch.optim.AdamW(model.parameters(), lr= args.lr, weight_decay= args.weightdecay)
    else :
        print("Something wrong with the optimizer! Please check")

    if args.scheduler == 'stepLR' :
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer = optimizer, step_size=30, gamma=1e-5)
    elif args.scheduler == 'Cosine' :
        warmup_epochs = 5
        total_epochs = args.epochs
   
        warmup_scheduler = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6)

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )

    elif args.scheduler == 'Linear' :
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer = optimizer, start_factor=1.0, total_iters=args.epochs)
    else :
        print("Something wrong with the scheduler of lr! Please check")

    checkpoint_path = root_dir / ("exp_" + str(test_n+1)) / "best_model.pth"

    print("Model setup complete with the following parameters:")
    print(f"Optimizer: {args.optimizer} (lr={args.lr}, weight_decay={args.weightdecay}, momentum={args.momentum})")
    print(f"Scheduler: {args.scheduler}")

    #TRAINING
    print("Starting training...")
    best_val_loss = float('inf')
    epochs_no_improve = 0

    torch.cuda.empty_cache()
    gc.collect()

    summary = {
        'args': vars(args),
        'augmentation probabilities': prob,
        'autosam_checkpoint': autosam_checkpoint,
        'dice loss weights': alpha,
        'bce loss weights': beta
    }
    
    with open(saving_path / "cv_summary.json", 'w') as f:
        import json
        json.dump(summary, f, indent=2)
    
    for epoch in range(0, args.epochs):
        print(f"Epoch {epoch}: LR = {scheduler.get_last_lr()}")
        # Check model weights before training
        if not check_model_weights(model, epoch):
            print("NaN detected in model weights! Reinitializing...")
            reinitialize_problematic_layers(model)
        
        torch.cuda.empty_cache()
        gc.collect()
        
        model.train()
        train_stats, diceT, focalT = train_one_epoch_fine(model=model,
                                    dataloader=train_loader,
                                    optimizer=optimizer,
                                    device=device,
                                    epoch=epoch,
                                    criterion=criterion,
                                    path=saving_path,
                                    headless=args.headless,
                                    mixed_precision=True,
                                    grad_accum_steps=1)

        if np.isnan(train_stats):
            print(f"Training produced NaN at epoch {epoch}, skipping validation and reducing LR")
            # Emergency LR reduction
            for param_group in optimizer.optimizer.param_groups:
                param_group['lr'] *= 0.1
            continue
        
        # Validation
        model.eval()
        with torch.no_grad():
            valid_loss, diceV, focalV = validate_one_epoch_fine(model=model,
                                                  dataloader=validation_loader,
                                                  device=device,
                                                  epoch=epoch,
                                                  criterion=criterion,
                                                  path=saving_path,
                                                  headless=args.headless,
                                                  mixed_precision=True)

            if np.isnan(valid_loss):
                print(f"Validation produced NaN at epoch {epoch}, skipping early stopping and reducing LR")
                          
        print(f"Epoch {epoch} - Train Loss: {train_stats:.4f}, Valid Loss: {valid_loss:.4f}")
        
        if valid_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {valid_loss:.4f}. Saving model...")
            best_val_loss = valid_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)  # Save the best model
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")
        
        if epochs_no_improve >= args.patience:
            print("Early stopping triggered.")
            break

        scheduler.step() 

        # Save losses for plotting
        if epoch == 0:
            train_losses = []
            valid_losses = []
            train_dice_loss = []
            train_focal_loss = []
            valid_dice_loss = []
            valid_focal_loss = []

        train_losses.append(train_stats['loss'] if isinstance(train_stats, dict) and 'loss' in train_stats else train_stats)
        valid_losses.append(valid_loss)
        train_dice_loss.append(diceT if isinstance(diceT, float) else diceT.item())
        train_focal_loss.append(focalT if isinstance(focalT, float) else focalT.item())
        valid_dice_loss.append(diceV if isinstance(diceV, float) else diceV.item())
        valid_focal_loss.append(focalV if isinstance(focalV, float) else focalV.item())

        # Plot and save the loss curves
        plt.figure()
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
        plt.plot(range(1, len(valid_losses) + 1), valid_losses, label='Valid Loss')
        plt.plot(range(1, len(train_dice_loss) + 1), train_dice_loss, label='Train Dice Loss')
        plt.plot(range(1, len(valid_dice_loss) + 1), valid_dice_loss, label='Valid Dice Loss')
        plt.plot(range(1, len(train_focal_loss) + 1), train_focal_loss, label='Train Focal Loss')
        plt.plot(range(1, len(valid_focal_loss) + 1), valid_focal_loss, label='Valid Focal Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(root_dir / ("exp_" + str(test_n+1)) / "loss_curve.png")
        plt.close()

        # Clear memory
        #torch.cuda.empty_cache()
        #gc.collect()

if __name__ == '__main__':
    main()