
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
from engine import train_one_epoch, evaluate
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

seed = 42
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.manual_seed(seed)
np.random.seed(seed)


sam = sam_model_registry["vit_b"](
    checkpoint="/home/mdezen/distillation/checkpoints/sam_vit_b_01ec64.pth")
predictor = SamPredictor(sam)
device = "cuda" if torch.cuda.is_available() else "cpu"
sam.to(device=device)
sam.eval( )

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

for original_param, cloned_param in zip(sam.mask_decoder.parameters(), model.mask_decoder.parameters()):
    assert torch.equal(original_param.to(device=device), cloned_param.to(device=device)), "The weights do not match!"



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


criterion = nn.MSELoss()
#criterion = DistillationLoss(criterion,sam.image_encoder,'soft',0.5,0.5)

# out = sam.image_encoder(torch.randn(1,3,1024,1024).to(device)) #64,256,256
# print(out.shape)
# out = model(torch.randn(1,3,1024,1024).to(device))
# print(out.shape)

image_dirs_val = ["/home/mdezen/distillation/images/left_frames","/home/mdezen/distillation/images/left_frames2"]
mask_dirs_val = ["/home/mdezen/distillation/masks/instrument_dataset_1","/home/mdezen/distillation/masks/instrument_dataset_2"]

image_dirs_train = [
    "/home/mdezen/distillation/MICCAI/instrument_1_4_training/test",
    #"/home/mdezen/distillation/MICCAI/instrument_1_4_training/instrument_dataset_1/left_frames",

    #"/home/mdezen/distillation/MICCAI/instrument_1_4_training/instrument_dataset_2/left_frames",
    #"/home/mdezen/distillation/MICCAI/instrument_1_4_training/instrument_dataset_3/left_frames",
    #"/home/mdezen/distillation/MICCAI/instrument_1_4_training/instrument_dataset_4/left_frames"
]
mask_dirs_train = [
    #"/home/mdezen/distillation/MICCAI/instrument_1_4_training/instrument_dataset_1/ground_truth/Left_Prograsp_Forceps_labels",
    #"/home/mdezen/distillation/MICCAI/instrument_1_4_training/instrument_dataset_2/ground_truth/Left_Prograsp_Forceps_labels",
    #"/home/mdezen/distillation/MICCAI/instrument_1_4_training/instrument_dataset_3/ground_truth/Left_Large_Needle_Driver_labels",
    "/home/mdezen/distillation/MICCAI/instrument_1_4_training/testGT"
#"/home/mdezen/distillation/MICCAI/instrument_1_4_training/instrument_dataset_4/ground_truth/Large_Needle_Driver_Left_labels"

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
dataloader = DataLoader(dataset,batch_size=2,shuffle=True)
for images, masks in dataloader:
    print(f"Batch di immagini: {images.shape}")  # (batch_size, 3, 224, 224)
    print(f"Batch di maschere: {masks.shape}")  # (batch_size, 1, 224, 224)
    break



# print(f"Start training for {epochs} epochs")
# start_time = time.time()
# max_accuracy = 0.0
# max_accuracy_ema = 0.
epochs = 10

torch.cuda.empty_cache()

for epoch in range(0, epochs):
    print("sto trainando")

    train_stats = train_one_epoch(model.image_encoder,sam.image_encoder,epoch,criterion,dataloader,optimizer,device)

    #test_stats = evaluate(dataloaderVal, model, device)
    #print(
       # f"Accuracy of the network on the {len(datasetVal)} test images: {test_stats['acc1']:.1f}%")


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
