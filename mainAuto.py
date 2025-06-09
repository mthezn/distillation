import copy
import albumentations as A
import pandas as pd
from sympy.stats.rv import sampling_E
from torch.nn import BCEWithLogitsLoss
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from Dataser import CholecDataset
from losses import DistillationLoss, distillation_loss, dice_loss, iou_loss

from model import CMT_Ti
from build_CMT_sam import sam_model_registry
from repvit_sam import SamPredictor
from Dataser import ImageMaskDataset
from torch.cuda.amp import GradScaler, autocast
from repvit_sam.build_sam import build_sam_repvit
from utils import *
from albumentations.pytorch import ToTensorV2
import wandb
from matplotlib import pyplot as plt
import numpy as np
import time
import torch.nn as nn
from torchvision   import transforms
import utils
from engine import train_one_epoch_auto, validate_one_epoch_auto
import cv2
import timm.layers
from torch.utils.data import ConcatDataset

import timm.optim
from timm.loss import LabelSmoothingCrossEntropy
from timm.optim import create_optimizer, create_optimizer_v2
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler,get_state_dict,ModelEma
import torch
from timm.models import create_model
from PIL import Image
import gc
from datasets import load_dataset
import random
import string
def generate_random_name(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def contains_instrument(example):
    mask = np.array(example["color_mask"])  # o "segmentation" se diverso
    return np.any((mask == 169) | (mask == 170))

def get_train_augmentation(image_size=256):
    return A.Compose([
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.5, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8),
        A.GaussianBlur(p=0.3),
        #A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.3),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
def get_val_augmentation(image_size=256):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
############################################################################################################


wandb.login(key='14497a5de45116d579bde37168ccf06f78c2928e')  # Replace 'your_api_key' with your actual API key
name = "autoSam"+generate_random_name(5)

datasetCholec = load_dataset("minwoosun/CholecSeg8k", trust_remote_code=True)



filtered_ds = datasetCholec["train"].filter(contains_instrument)


seed = 42
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.manual_seed(seed)
np.random.seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"




#CREO UN MODELLO SAM CON ENCODER CMT CHE USERO  COME TEACHER FREEZANDO IL DECODER


teacher_checkpoint = "checkpoints/13_05/decoupledVitBDGfFE.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
sam_checkpoint = "checkpoints/sam_vit_b_01ec64.pth"
sam= sam_model_registry["vit_b"](checkpoint=sam_checkpoint).to(device=device)
print(device)

model = sam_model_registry["CMT"](checkpoint=None)
model.load_state_dict(torch.load(teacher_checkpoint, map_location=device), strict=False)  # Load the state dict into the model


model.to(device=device)

#model.mask_decoder = copy.deepcopy(sam.mask_decoder)
#model.mask_decoder.load_state_dict(sam.mask_decoder.state_dict(), strict=False)  # Load the state dict into the model
#model.prompt_encoder = copy.deepcopy(sam.prompt_encoder)
#model.prompt_encoder.load_state_dict(sam.prompt_encoder.state_dict(), strict=False)  # Load the state dict into the model
sam.image_encoder = (model.image_encoder)  # Clone the image encoder



sam.eval()


# CONGELO TUTTO E SBLOCCO SOLO L'ENCODER
for param in sam.parameters():
    param.requires_grad = False
for param in sam.mask_decoder.parameters():
    param.requires_grad = False


#MODELLO STUDENT

student = sam_model_registry["autoSam"]()
cloned_image_encoder = copy.deepcopy(model.image_encoder)  # Clone the image encoder
cloned_image_encoder.load_state_dict(model.image_encoder.state_dict())  # Copy the weights
student.image_encoder = cloned_image_encoder

student.to(device=device)
student.train()

for param in student.parameters():
    param.requires_grad = False
for param in student.mask_decoder.parameters():
    param.requires_grad = True

batch_size = 4
lr = 0.0001

print("caricato tutto")
optimizer_cfg = {
    'opt': 'adamw',
    'lr': lr,
    'weight_decay': 1e-4,
}
optimizer = create_optimizer_v2(student,**optimizer_cfg)
loss_scaler = NativeScaler()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode = 'min',factor = 0.1,patience = 3,threshold=0.000001)

criterion = iou_loss
epochs = 30




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
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).

    # Set the wandb project where this run will be logged.
    project="autoSamDistillationNew",
    name=name,
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": lr,
        "architecture": "CMT/autoSam",
        "dataset": "Miccai",
        "epochs": epochs,
        "criterion": "IouLoss",
        "batch_size": batch_size,
        "optimizer": optimizer_cfg['opt'],
        "weight_decay": optimizer_cfg['weight_decay'],
        "augmentation": str(train_transform),


    }

)
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

image_test = ["MICCAI/test/frame"]
mask_test = ["MICCAI/test/gt"]

datasetVal = ImageMaskDataset(image_dirs=image_dirs_val,mask_dirs=mask_dirs_val,transform=validation_transform,increase=False)
dataloaderVal = DataLoader(datasetVal,batch_size=batch_size,shuffle=True)

dataset_cholec = CholecDataset(filtered_ds, transform=train_transform)
datasetMiccai = ImageMaskDataset(image_dirs=image_dirs_train,mask_dirs=mask_dirs_train,transform=train_transform,increase=False)

dataset_finale = ConcatDataset([dataset_cholec, datasetMiccai])

dataloader = DataLoader(datasetMiccai,batch_size=batch_size,shuffle=True,pin_memory=True)
for images, masks in dataloader:
    print(f"Batch di immagini: {images.shape}")  # (batch_size, 3, 224, 224)
    print(f"Batch di maschere: {masks.shape}")  # (batch_size, 1, 224, 224)
    break

#TRAINING
patience = 5  # Number of epochs to wait for improvement
best_val_loss = float('inf')
epochs_no_improve = 0
checkpoint_path = "checkpoints/26_05/" + name+".pth"

torch.cuda.empty_cache()
gc.collect()
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



