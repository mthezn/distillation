
import pandas as pd
from torch.utils.data import DataLoader

from losses import DistillationLoss
from model import CMT_Ti
from build_CMT_sam import sam_model_registry
from repvit_sam import SamPredictor
from Dataser import ImageMaskDataset
from torch.cuda.amp import GradScaler, autocast
from repvit_sam.build_sam import build_sam_repvit
from utils import *
from matplotlib import pyplot as plt
import numpy as np
import time
import torch.nn as nn
from torchvision   import transforms
import utils
from engine import train_one_epoch, evaluate, train_one_epoch_coupled
import cv2
import timm.layers

import timm.optim
from timm.loss import LabelSmoothingCrossEntropy
from timm.optim import create_optimizer, create_optimizer_v2
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler,get_state_dict,ModelEma
import torch
from timm.models import create_model
from PIL import Image
import gc

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = np.array([coords[i] for i in range(len(coords)) if labels[i] == 1])
    # neg_points = np.array([coords[i] for i in range(len(coords)) if labels[i] == 0])
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    # ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_bbox(bbox, ax):
    for box in bbox:
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))



sam = sam_model_registry["vit_h"](
    checkpoint="C:/Users/User/OneDrive - Politecnico di Milano/Documenti/POLIMI/Tesi\SAM/checkpoints/sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)
device = "cuda" if torch.cuda.is_available() else "cpu"
sam.to(device=device)


model = sam_model_registry["CMT"]()
model.to(device=device)
predictorStud = SamPredictor(model)


model.train()
lr = 0.001
batch_size = 32

linear_scaled_lr = lr * batch_size * utils.get_world_size() / 512.0
lr = linear_scaled_lr
optimizer_cfg = {
    'opt': 'adamw',
    'lr': lr,
    'weight_decay': 0.01,
}
optimizer = create_optimizer_v2(model,**optimizer_cfg)
loss_scaler = NativeScaler()


image_dirs_val = ["C:/Users/User/OneDrive - Politecnico di Milano/Documenti/POLIMI/Tesi/distillation/MICCAI/instrument_1_4_training/instrument_dataset_4/left_frames"]
mask_dirs_val = ["C:/Users/User/OneDrive - Politecnico di Milano/Documenti/POLIMI/Tesi/distillation/MICCAI/instrument_1_4_training/instrument_dataset_4/ground_truth/Large_Needle_Driver_Left_labels" ,
                 "C:/Users/User/OneDrive - Politecnico di Milano/Documenti/POLIMI/Tesi/distillation/MICCAI/instrument_1_4_training/instrument_dataset_4/ground_truth/Large_Needle_Driver_Right_labels" ,
                 "C:/Users/User/OneDrive - Politecnico di Milano/Documenti/POLIMI/Tesi/distillation/MICCAI/instrument_1_4_training/instrument_dataset_4/ground_truth/Prograsp_Forceps_labels",]

image_dirs_train = [
    #"C:/Users/User/OneDrive - Politecnico di Milano/Documenti/POLIMI/Tesi/distillation/MICCAI/instrument_1_4_training/test",
    "C:/Users/User/OneDrive - Politecnico di Milano/Documenti/POLIMI/Tesi/distillation/MICCAI/instrument_1_4_training/instrument_dataset_1/left_frames",

    #"C:/Users/User/OneDrive - Politecnico di Milano/Documenti/POLIMI/Tesi/distillation/MICCAI/instrument_1_4_training/instrument_dataset_2/left_frames",
    #"C:/Users/User/OneDrive - Politecnico di Milano/Documenti/POLIMI/Tesi/distillation/MICCAI/instrument_1_4_training/instrument_dataset_3/left_frames",

]
mask_dirs_train = [
    "C:/Users/User/OneDrive - Politecnico di Milano/Documenti/POLIMI/Tesi/distillation/MICCAI/instrument_1_4_training/instrument_dataset_1/ground_truth/Left_Prograsp_Forceps_labels",
    "C:/Users/User/OneDrive - Politecnico di Milano/Documenti/POLIMI/Tesi/distillation/MICCAI/instrument_1_4_training/instrument_dataset_1/ground_truth/Maryland_Bipolar_Forceps_labels",
"C:/Users/User/OneDrive - Politecnico di Milano/Documenti/POLIMI/Tesi/distillation/MICCAI/instrument_1_4_training/instrument_dataset_1/ground_truth/Right_Prograsp_Forceps_labels"
    #"C:/Users/User/OneDrive - Politecnico di Milano/Documenti/POLIMI/Tesi/distillation/MICCAI/instrument_1_4_training/instrument_dataset_2/ground_truth/Left_Prograsp_Forceps_labels",
    # "C:/Users/User/OneDrive - Politecnico di Milano/Documenti/POLIMI/Tesi/distillation/MICCAI/instrument_1_4_training/instrument_dataset_2/ground_truth/Right_Prograsp_Forceps_label",
    # "C:/Users/User/OneDrive - Politecnico di Milano/Documenti/POLIMI/Tesi/distillation/MICCAI/instrument_1_4_training/instrument_dataset_3/ground_truth/Right_Large_Needle_Driver_labels",
    #"C:/Users/User/OneDrive - Politecnico di Milano/Documenti/POLIMI/Tesi/distillation/MICCAI/instrument_1_4_training/instrument_dataset_3/ground_truth/Left_Large_Needle_Driver_labels",
    #"C:/Users/User/OneDrive - Politecnico di Milano/Documenti/POLIMI/Tesi/distillation/MICCAI/instrument_1_4_training/testGT"


]
criterion = nn.MSELoss()

image_transform = transforms.Compose([
    transforms.Resize((1024,1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std = [0.5,0.5,0.5])



])
mask_transform  = transforms.Compose([

    transforms.Resize((1024,1024)),
    transforms.ToTensor()
])
#datasetVal = ImageMaskDataset(image_dirs=image_dirs_val,mask_dirs=mask_dirs_val,transform=image_transform,mask_transform=mask_transform)
#dataloaderVal = DataLoader(datasetVal,batch_size=2,shuffle=True)

dataset = ImageMaskDataset(image_dirs=image_dirs_train,mask_dirs=mask_dirs_train,transform=image_transform,mask_transform=mask_transform)
dataloader = DataLoader(dataset,batch_size=2,shuffle=True,pin_memory=True)
for images, masks in dataloader:
    print(f"Batch di immagini: {images.shape}")  # (batch_size, 3, 224, 224)
    print(f"Batch di maschere: {masks.shape}")  # (batch_size, 1, 224, 224)

    break

epoch = 1
predictorTeach = SamPredictor(sam)
for epoch in range(epoch):

    train_one_epoch_coupled(model,predictorStud, predictorTeach, epoch,criterion, dataloader,optimizer, device)


