
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

import timm.optim
from timm.loss import LabelSmoothingCrossEntropy
from timm.optim import create_optimizer, create_optimizer_v2
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler,get_state_dict,ModelEma
import torch
from timm.models import create_model
from PIL import Image
import gc
wandb.login(key='14497a5de45116d579bde37168ccf06f78c2928e')  # Replace 'your_api_key' with your actual API key



seed = 42
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.manual_seed(seed)
np.random.seed(seed)


#sam = sam_model_registry["vit_b"](
 #   checkpoint="/home/mdezen/distillation/checkpoints/sam_vit_b_01ec64.pth")
#predictor = SamPredictor(sam)
#device = "cuda" if torch.cuda.is_available() else "cpu"
#sam.to(device=device)
#sam.eval( )

sam_checkpoint = "/home/mdezen/distillation/checkpoints/repvit_sam.pt"
model_type = "repvit"

device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)


predictor = SamPredictor(sam)


# model = create_model('CMT_Ti',img_size=1024,output_dim=1280)
# model.to(device=device)
# model.train()
#carico i pesi del decoder e li assegno al modello, il decoder restera freezato e al limite fine tunnato
model = sam_model_registry["CMT"]()
model.to(device=device)

transformer_dim = model.mask_decoder.transformer_dim
transformer = model.mask_decoder.transformer


cloned_mask_decoder = type(sam.mask_decoder)(transformer_dim=transformer_dim, transformer=transformer)
cloned_mask_decoder.load_state_dict(sam.mask_decoder.state_dict())  # Copy the weights
model.mask_decoder = cloned_mask_decoder
# Congela tutto
for param in model.parameters():
    param.requires_grad = False
for param in model.image_encoder.parameters():
    param.requires_grad = True

for original_param, cloned_param in zip(sam.mask_decoder.parameters(), model.mask_decoder.parameters()):
    assert torch.equal(original_param.to(device=device), cloned_param.to(device=device)), "The weights do not match!"



model.train()
lr = 0.001
batch_size = 2

linear_scaled_lr = lr * batch_size * utils.get_world_size() / 512.0
lr = linear_scaled_lr
optimizer_cfg = {
    'opt': 'adamw',
    'lr': lr,
    'weight_decay': 0.01,
}
optimizer = create_optimizer_v2(model,**optimizer_cfg)
loss_scaler = NativeScaler()


criterion = nn.MSELoss()
epochs = 20

run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).

    # Set the wandb project where this run will be logged.
    project="decoupledDistillation",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": lr,
        "architecture": "CMT/EdgeSAm",
        "dataset": "MICCAI",
        "epochs": epochs,
        "batch_size": batch_size,
        "optimizer": optimizer_cfg['opt'],
        "weight_decay": optimizer_cfg['weight_decay'],


    }

)
#criterion = DistillationLoss(criterion,sam.image_encoder,'soft',0.5,0.5)

# out = sam.image_encoder(torch.randn(1,3,1024,1024).to(device)) #64,256,256
# print(out.shape)
# out = model(torch.randn(1,3,1024,1024).to(device))
# print(out.shape)

image_dirs_val = ["/home/mdezen/distillation/MICCAI/instrument_1_4_training/instrument_dataset_4/left_frames"]
mask_dirs_val = ["/home/mdezen/distillation/MICCAI/instrument_1_4_training/instrument_dataset_4/ground_truth/Large_Needle_Driver_Left_labels"]

image_dirs_train = [
    "/home/mdezen/distillation/MICCAI/instrument_1_4_training/test",
    "/home/mdezen/distillation/MICCAI/instrument_1_4_training/instrument_dataset_1/left_frames",

    "/home/mdezen/distillation/MICCAI/instrument_1_4_training/instrument_dataset_2/left_frames",
    "/home/mdezen/distillation/MICCAI/instrument_1_4_training/instrument_dataset_3/left_frames",

]
mask_dirs_train = [
    "/home/mdezen/distillation/MICCAI/instrument_1_4_training/instrument_dataset_1/ground_truth/Left_Prograsp_Forceps_labels",
    "/home/mdezen/distillation/MICCAI/instrument_1_4_training/instrument_dataset_1/ground_truth/Maryland_Bipolar_Forceps_labels",
    "/home/mdezen/distillation/MICCAI/instrument_1_4_training/instrument_dataset_1/ground_truth/Right_Prograsp_Forceps_labels",
    "/home/mdezen/distillation/MICCAI/instrument_1_4_training/instrument_dataset_2/ground_truth/Left_Prograsp_Forceps_labels",
     "/home/mdezen/distillation/MICCAI/instrument_1_4_training/instrument_dataset_2/ground_truth/Right_Prograsp_Forceps_labels",
   "/home/mdezen/distillation/MICCAI/instrument_1_4_training/instrument_dataset_3/ground_truth/Right_Large_Needle_Driver_labels",
    "/home/mdezen/distillation/MICCAI/instrument_1_4_training/instrument_dataset_3/ground_truth/Left_Large_Needle_Driver_labels",

    #"/home/mdezen/distillation/MICCAI/instrument_1_4_training/testGT"


]


image_transform = transforms.Compose([
    transforms.Resize((1024,1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std = [0.5,0.5,0.5])



])
mask_transform  = transforms.Compose([

    transforms.Resize((1024,1024)),
    transforms.ToTensor()
])
datasetVal = ImageMaskDataset(image_dirs=image_dirs_val,mask_dirs=mask_dirs_val,transform=image_transform,mask_transform=mask_transform)
dataloaderVal = DataLoader(datasetVal,batch_size=2,shuffle=True)

dataset = ImageMaskDataset(image_dirs=image_dirs_train,mask_dirs=mask_dirs_train,transform=image_transform,mask_transform=mask_transform)
dataloader = DataLoader(dataset,batch_size=2,shuffle=True,pin_memory=True)
for images, masks in dataloader:
    print(f"Batch di immagini: {images.shape}")  # (batch_size, 3, 224, 224)
    print(f"Batch di maschere: {masks.shape}")  # (batch_size, 1, 224, 224)
    break



# print(f"Start training for {epochs} epochs")
# start_time = time.time()
# max_accuracy = 0.0
# max_accuracy_ema = 0.

patience = 3  # Number of epochs to wait for improvement
best_val_loss = float('inf')
epochs_no_improve = 0
checkpoint_path = "checkpoints/student_checkpoint1704.pth"

torch.cuda.empty_cache()
gc.collect()
for epoch in range(0, epochs):
    print("sto trainando")

    train_stats = train_one_epoch(model.image_encoder,sam.image_encoder,epoch,criterion,dataloader,optimizer,device,run)

    torch.cuda.empty_cache()
    gc.collect()
    #print(epoch)
    val_loss = validate_one_epoch(model.image_encoder,sam.image_encoder,dataloaderVal,criterion ,device,epoch,run)
    print(
        f"Epoch {epoch} loss: {val_loss}")
    if val_loss < best_val_loss:
        print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), checkpoint_path)  # Save the best model
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epoch(s).")

        # Early stopping condition
    if epochs_no_improve >= patience:
        print("Early stopping triggered.")
        break



    #fare un early stopping e continuare solo se migliora
    # printare la loss ogni volta
    # fare una prova solo con 2 immagini

    #wandb e TEnsorboard
    #usare dataset 1-3 training e 4 in validation
    #verificare le GT del training
    #salvare alla fine il checkpoint del modello

    # checkpoint_paths = [ckpt_path]
    # print("Saving checkpoint to {}".format(ckpt_path))
    # for checkpoint_path in checkpoint_paths:
    #     utils.save_on_master({
    #         'model': model_without_ddp.state_dict(),
    #         'optimizer': optimizer.state_dict(),
    #         'lr_scheduler': lr_scheduler.state_dict(),
    #         'epoch': epoch,
    #         'model_ema': get_state_dict(model_ema),
    #         'scaler': loss_scaler.state_dict(),
    #         'args': args,
    #     }, checkpoint_path)
