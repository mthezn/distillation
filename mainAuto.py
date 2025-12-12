import copy
import albumentations as A
from torch.utils.data import DataLoader
from Dataset import CholecDataset
from losses import dice_loss

from modeling.build_sam import sam_model_registry
from Dataset import ImageMaskDataset
from utils import *
from albumentations.pytorch import ToTensorV2
import wandb
import numpy as np
import torch.nn as nn
from engine import train_one_epoch_auto, validate_one_epoch_auto
from torch.utils.data import ConcatDataset
import os
from timm.optim import create_optimizer_v2
from timm.utils import NativeScaler
import torch
import gc
from datasets import load_dataset
from utility import generate_random_name, contains_instrument

############################################################################################################

"""WANDB SETTINGS"""
wandb.login(key='14497a5de45116d579bde37168ccf06f78c2928e')  # Replace 'your_api_key' with your actual API key
name = "autoSamVitHUnet"+generate_random_name(5)
####################################################################################################################



"SETTINGS"
seed = 42
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.manual_seed(seed)
np.random.seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
##############################################################################################################









"""TEACHER LOADING"""

 # Path to the teacher model checkpoint
sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"  # Path to the SAM checkpoint
sam= sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to(device=device)
sam.load_state_dict(torch.load(sam_checkpoint, map_location=device), strict=False)  # Load the state dict into the model
sam.eval()


# CONGELO TUTTO
for param in sam.parameters():
    param.requires_grad = False
for param in sam.mask_decoder.parameters():
    param.requires_grad = False
#####################################################################################################################




"""STUDENT LOADING"""
encoder_checkpoint="checkpointsLight/decoupledVitH22ps1.pth"  #MODELLO STUDENT (encoder distillato e decoder non trainato)

model = sam_model_registry["autoSamUnet"](checkpoint=None)

model.load_state_dict(torch.load(encoder_checkpoint, map_location=device), strict=False)  # Load the state dict into the model

student = sam_model_registry["autoSamUnet"]()

cloned_image_encoder = copy.deepcopy(model.image_encoder)  # Clone the image encoder
cloned_image_encoder.load_state_dict(model.image_encoder.state_dict())  # Copy the weights  #qusta parte si potra togliere
student.image_encoder = cloned_image_encoder

student.to(device=device)
student.train()

for param in student.parameters():
    param.requires_grad = False
for param in student.mask_decoder.parameters():
    param.requires_grad = True
#################################################################################################################################




"""AUGMENTATIONS"""
train_transform = A.Compose([
    A.Resize(1024, 1024),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=45, p=0.5),
    #A.ColorJitter(p=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    #A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

validation_transform = A.Compose([
    A.Resize(1024, 1024),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])
#########################################################################################################################



"""DIRECTORIES FOR MICCAI DATASET"""
#DIRECTORIES
image_dirs_val = ["/home/mdezen/distillation/MICCAI/instrument_1_4_training/instrument_dataset_4/left_frames"]
mask_dirs_val = ["/home/mdezen/distillation/MICCAI/instrument_1_4_training/instrument_dataset_4/ground_truth/Large_Needle_Driver_Left_labels",
                 "/home/mdezen/distillation/MICCAI/instrument_1_4_training/instrument_dataset_4/ground_truth/Large_Needle_Driver_Right_labels",
                 "/home/mdezen/distillation/MICCAI/instrument_1_4_training/instrument_dataset_4/ground_truth/Prograsp_Forceps_labels",]

image_dirs_train = [
    #"/home/mdezen/distillation/MICCAI/instrument_1_4_training/test",
    "/home/mdezen/distillation/MICCAI/instrument_1_4_training/instrument_dataset_1/left_frames",

    "/home/mdezen/distillation/MICCAI/instrument_1_4_training/instrument_dataset_2/left_frames",
    "/home/mdezen/distillation/MICCAI/instrument_1_4_training/instrument_dataset_3/left_frames",
    "/home/mdezen/distillation/MICCAI/instrument_5_8_training/instrument_dataset_5/left_frames",
    "/home/mdezen/distillation/MICCAI/instrument_5_8_training/instrument_dataset_6/left_frames",
    "/home/mdezen/distillation/MICCAI/instrument_5_8_training/instrument_dataset_7/left_frames",
    "/home/mdezen/distillation/MICCAI/instrument_5_8_training/instrument_dataset_8/left_frames",

]
mask_dirs_train = [
    "/home/mdezen/distillation/MICCAI/instrument_1_4_training/instrument_dataset_1/ground_truth/Left_Prograsp_Forceps_labels",
    "/home/mdezen/distillation/MICCAI/instrument_1_4_training/instrument_dataset_1/ground_truth/Maryland_Bipolar_Forceps_labels",
    "/home/mdezen/distillation/MICCAI/instrument_1_4_training/instrument_dataset_1/ground_truth/Right_Prograsp_Forceps_labels",
    "/home/mdezen/distillation/MICCAI/instrument_1_4_training/instrument_dataset_2/ground_truth/Left_Prograsp_Forceps_labels",
    "/home/mdezen/distillation/MICCAI/instrument_1_4_training/instrument_dataset_2/ground_truth/Right_Prograsp_Forceps_labels",
    "/home/mdezen/distillation/MICCAI/instrument_1_4_training/instrument_dataset_3/ground_truth/Right_Large_Needle_Driver_labels",
    "/home/mdezen/distillation/MICCAI/instrument_1_4_training/instrument_dataset_3/ground_truth/Left_Large_Needle_Driver_labels",
    "/home/mdezen/distillation/MICCAI/instrument_5_8_training/instrument_dataset_5/ground_truth/Bipolar_Forceps_labels",
    "/home/mdezen/distillation/MICCAI/instrument_5_8_training/instrument_dataset_5/ground_truth/Grasping_Retractor_labels",
    "/home/mdezen/distillation/MICCAI/instrument_5_8_training/instrument_dataset_5/ground_truth/Vessel_Sealer_labels",
    "/home/mdezen/distillation/MICCAI/instrument_5_8_training/instrument_dataset_6/ground_truth/Monopolar_Curved_Scissors_labels",
    "/home/mdezen/distillation/MICCAI/instrument_5_8_training/instrument_dataset_6/ground_truth/Prograsp_Forceps",
    "/home/mdezen/distillation/MICCAI/instrument_5_8_training/instrument_dataset_6/ground_truth/Right_Large_Needle_Driver_labels",
    "/home/mdezen/distillation/MICCAI/instrument_5_8_training/instrument_dataset_7/ground_truth/Left_Bipolar_Forceps",
    "/home/mdezen/distillation/MICCAI/instrument_5_8_training/instrument_dataset_7/ground_truth/Right_Vessel_Sealer",
    "/home/mdezen/distillation/MICCAI/instrument_5_8_training/instrument_dataset_8/ground_truth/Bipolar_Forceps_labels",
    "/home/mdezen/distillation/MICCAI/instrument_5_8_training/instrument_dataset_8/ground_truth/Left_Grasping_Retractor_labels",
    "/home/mdezen/distillation/MICCAI/instrument_5_8_training/instrument_dataset_8/ground_truth/Monopolar_Curved_Scissors_labels",
    "/home/mdezen/distillation/MICCAI/instrument_5_8_training/instrument_dataset_8/ground_truth/Right_Grasping_Retractor_labels",





    #"/home/mdezen/distillation/MICCAI/instrument_1_4_training/testGT"


]
##########################################################################################################################
"""TRAINING SETTINGS"""

batch_size = 2
lr = 0.0001
optimizer_cfg = {
    'opt': 'adamw',
    'lr': lr,
    'weight_decay': 1e-4,
}
optimizer = create_optimizer_v2(student,**optimizer_cfg)
criterion = torch.nn.BCEWithLogitsLoss()
epochs = 30
loss_scaler = NativeScaler()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,
    eta_min=1e-6
)
patience = 5  # Number of epochs to wait for improvement
best_val_loss = float('inf')
epochs_no_improve = 0
checkpoint_path = "checkpoints/28_07/" + name+".pth"

torch.cuda.empty_cache()
gc.collect()

#####################################################################


"""WANDB INITIALIZATION"""
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).

    # Set the wandb project where this run will be logged.
    project="autoSamDistillationNew",
    name=name,
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": lr,
        "architecture": "CMT/Unet",
        "dataset": "Miccai",
        "epochs": epochs,
        "criterion": "BCELoss",
        "batch_size": batch_size,
        "optimizer": optimizer_cfg['opt'],
        "weight_decay": optimizer_cfg['weight_decay'],
        "augmentation": str(train_transform),


    }

)
###########################################################################################



"""DATASETS LOADING"""

datasetCholec = load_dataset("minwoosun/CholecSeg8k", trust_remote_code=True)
filtered_ds = datasetCholec["train"].filter(contains_instrument)

datasetVal = ImageMaskDataset(image_dirs=image_dirs_val,mask_dirs=mask_dirs_val,transform=validation_transform,increase=False)
dataloaderVal = DataLoader(datasetVal,batch_size=batch_size,shuffle=True)

#dataset_cholec = CholecDataset(filtered_ds, transform=train_transform)
datasetMiccai = ImageMaskDataset(image_dirs=image_dirs_train,mask_dirs=mask_dirs_train,transform=train_transform,increase=True)

#dataset_finale = ConcatDataset([dataset_cholec, datasetMiccai])

dataloader = DataLoader(datasetMiccai,batch_size=batch_size,shuffle=True,pin_memory=True)
for images, masks in dataloader:
    print(f"Batch di immagini: {images.shape}")  # (batch_size, 3, 224, 224)
    print(f"Batch di maschere: {masks.shape}")  # (batch_size, 1, 224, 224)
    break
####################################################################################################



"""TRAINING LOOP"""

for epoch in range(0, epochs):
    print(f"Epoch {epoch + 1}/{epochs}")


    train_stats = train_one_epoch_auto(sam,student,dataloader,optimizer,device,run,epoch,criterion)

    torch.cuda.empty_cache()
    gc.collect()
    #print(epoch)
    val_loss = validate_one_epoch_auto(sam,student,dataloaderVal,criterion ,device,epoch,run)
    scheduler.step(val_loss)  # Update the learning rate scheduler based on validation loss
    print(
        f"Epoch {epoch} loss: {val_loss}")
    if val_loss < best_val_loss:
        print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(student.state_dict(), checkpoint_path)  # Save the best model
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epoch(s).")

        # Early stopping condition
    if epochs_no_improve >= patience:
        print("Early stopping triggered.")
        break

############################################################################################################

