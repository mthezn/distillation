
import pandas as pd
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from Dataser import CholecDataset
from losses import DistillationLoss
from model import CMT_Ti
from build_CMT_sam import sam_model_registry
from repvit_sam import SamPredictor
from Dataser import ImageMaskDataset
from torch.cuda.amp import GradScaler, autocast
from repvit_sam.build_sam import build_sam_repvit
from utils import *
import wandb
from matplotlib import pyplot as plt
import numpy as np
import time
import torch.nn as nn
from torchvision   import transforms
import utils
from engine import train_one_epoch, evaluate, validate_one_epoch
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

wandb.login(key='14497a5de45116d579bde37168ccf06f78c2928e')  # Replace 'your_api_key' with your actual API key
name = "autoSam"+generate_random_name(5)

datasetCholec = load_dataset("minwoosun/CholecSeg8k", trust_remote_code=True)
def contains_instrument(example):
    mask = np.array(example["color_mask"])  # o "segmentation" se diverso
    return np.any((mask == 169) | (mask == 170))

filtered_ds = datasetCholec["train"].filter(contains_instrument)


seed = 42
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.manual_seed(seed)
np.random.seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"




#CREO UN MODELLO SAM CON ENCODER CMT CHE USERO  COME TEACHER FREEZANDO IL DECODER

teacher_checkpoint = "checkpoints/13_05/decoupledVitBg4SXZ.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"


model = sam_model_registry["CMT"](teacher_checkpoint)
model.to(device=device)



model.train()
# CONGELO TUTTO E SBLOCCO SOLO L'ENCODER
for param in model.parameters():
    param.requires_grad = False
for param in model.mask_decoder.parameters():
    param.requires_grad = True


#MODELLO STUDENT

student = sam_model_registry["autoSam"]()
cloned_image_encoder = type(model.image_encoder)(model.image_encoder.embed_dim, model.image_encoder.depth, model.image_encoder.num_heads, model.image_encoder.mlp_ratio, model.image_encoder.qkv_bias, model.image_encoder.norm_layer)
cloned_image_encoder.load_state_dict(model.image_encoder.state_dict())  # Copy the weights
model.image_encoder = cloned_image_encoder

student.to(device=device)
student.train()

for param in student.parameters():
    param.requires_grad = False
for param in student.mask_decoder.parameters():
    param.requires_grad = True

batch_size = 2
lr = 0.001
linear_scaled_lr = lr * batch_size * utils.get_world_size() / 512.0

optimizer_cfg = {
    'opt': 'adamw',
    'lr': lr,
    'weight_decay': 0.1,
}
optimizer = create_optimizer_v2(model,**optimizer_cfg)
loss_scaler = NativeScaler()


criterion = nn.MSELoss()
epochs = 20


image_transform = transforms.Compose([
    transforms.Resize((1024,1024)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std = [0.5,0.5,0.5])



])
mask_transform  = transforms.Compose([

    transforms.Resize((1024,1024)),
    transforms.RandomHorizontalFlip(p=0.5),  # Ensure the same flip as the image
    transforms.RandomVerticalFlip(p=0.5),  # Ensure the same flip as the image
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor()
])
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).

    # Set the wandb project where this run will be logged.
    project="decoupledMSE",
    name=name,
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": lr,
        "architecture": "CMT/autoSam",
        "dataset": "MICCAI",
        "epochs": epochs,
        "criterion": "MSE",
        "batch_size": batch_size,
        "optimizer": optimizer_cfg['opt'],
        "weight_decay": optimizer_cfg['weight_decay'],
        "augmentation": str(image_transform),


    }

)
#DIRECTORIES
image_dirs_val = ["/home/mdezen/distillation/MICCAI/instrument_1_4_training/instrument_dataset_4/left_frames"]
mask_dirs_val = ["/home/mdezen/distillation/MICCAI/instrument_1_4_training/instrument_dataset_4/ground_truth/Large_Needle_Driver_Left_labels"]

image_dirs_train = [
    "/home/mdezen/distillation/MICCAI/instrument_1_4_training/test",
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



datasetVal = ImageMaskDataset(image_dirs=image_dirs_val,mask_dirs=mask_dirs_val,transform=image_transform,mask_transform=mask_transform)
dataloaderVal = DataLoader(datasetVal,batch_size=batch_size,shuffle=True)

#dataset_cholec = CholecDataset(filtered_ds, transform=image_transform, mask_transform=mask_transform)
datasetMiccai = ImageMaskDataset(image_dirs=image_dirs_train,mask_dirs=mask_dirs_train,transform=image_transform,mask_transform=mask_transform)

#dataset_finale = ConcatDataset([dataset_cholec, datasetMiccai])

dataloader = DataLoader(datasetMiccai,batch_size=batch_size,shuffle=True,pin_memory=True)
for images, masks in dataloader:
    print(f"Batch di immagini: {images.shape}")  # (batch_size, 3, 224, 224)
    print(f"Batch di maschere: {masks.shape}")  # (batch_size, 1, 224, 224)
    break

