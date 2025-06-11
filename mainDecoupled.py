from torch.utils.data import DataLoader
from Dataset import CholecDataset
from modeling.build_sam import sam_model_registry
from repvit_sam import SamPredictor
from Dataset import ImageMaskDataset
from utils import *
import wandb
import numpy as np
import torch.nn as nn
import utils
import albumentations as A
from albumentations.pytorch import ToTensorV2

from engine import train_one_epoch, validate_one_epoch

from timm.optim import create_optimizer_v2
from timm.utils import NativeScaler
import torch
import gc
from datasets import load_dataset
from utility import generate_random_name, contains_instrument

wandb.login(key='14497a5de45116d579bde37168ccf06f78c2928e')  # Replace 'your_api_key' with your actual API key
name = "decoupledVitB"+generate_random_name(5)

datasetCholec = load_dataset("minwoosun/CholecSeg8k", trust_remote_code=True)
filtered_ds = datasetCholec["train"].filter(contains_instrument)

seed = 42
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.manual_seed(seed)
np.random.seed(seed)


#CARICO IL MODELLO SAM
sam_checkpoint = "/home/mdezen/distillation/checkpoints/sam_vit_b_01ec64.pth"
model_type = "vit_b"

device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)



#CREO UN MODELLO SAM CON ENCODER CMT
model = sam_model_registry["CMT"]()
model.to(device=device)

transformer_dim = model.mask_decoder.transformer_dim
transformer = model.mask_decoder.transformer


cloned_mask_decoder = type(sam.mask_decoder)(transformer_dim=transformer_dim, transformer=transformer)
cloned_mask_decoder.load_state_dict(sam.mask_decoder.state_dict())  # Copy the weights
model.mask_decoder = cloned_mask_decoder


# CONGELO TUTTO E SBLOCCO SOLO L'ENCODER
for param in model.parameters():
    param.requires_grad = False
for param in model.image_encoder.parameters():
    param.requires_grad = True

for original_param, cloned_param in zip(sam.mask_decoder.parameters(), model.mask_decoder.parameters()):
    assert torch.equal(original_param.to(device=device), cloned_param.to(device=device)), "The weights do not match!"



model.train()

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
    project="decoupledMSE",
    name=name,
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": lr,
        "architecture": "CMT/VitB",
        "dataset": "CholecSeg8k(169,170)",
        "epochs": epochs,
        "criterion": "MSE",
        "batch_size": batch_size,
        "optimizer": optimizer_cfg['opt'],
        "weight_decay": optimizer_cfg['weight_decay'],
        "augmentation": str(train_transform),


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

]



datasetVal = ImageMaskDataset(image_dirs=image_dirs_val,mask_dirs=mask_dirs_val,transform=validation_transform)
dataloaderVal = DataLoader(datasetVal,batch_size=batch_size,shuffle=True)

dataset_cholec = CholecDataset(filtered_ds, transform=train_transform)
datasetMiccai = ImageMaskDataset(image_dirs=image_dirs_train,mask_dirs=mask_dirs_train,transform=train_transform)

#dataset_finale = ConcatDataset([dataset_cholec, datasetMiccai])

dataloader = DataLoader(dataset_cholec,batch_size=batch_size,shuffle=True,pin_memory=True)
for images, masks in dataloader:
    print(f"Batch di immagini: {images.shape}")  # (batch_size, 3, 224, 224)
    print(f"Batch di maschere: {masks.shape}")  # (batch_size, 1, 224, 224)
    break




#TRAINING
patience = 3  # Number of epochs to wait for improvement
best_val_loss = float('inf')
epochs_no_improve = 0
checkpoint_path = "checkpoints/13_05/" + name+".pth"

torch.cuda.empty_cache()
gc.collect()
for epoch in range(0, epochs):


    train_stats = train_one_epoch(model.image_encoder,sam.image_encoder,epoch,criterion,dataloader,optimizer,device,run)

    torch.cuda.empty_cache()
    gc.collect()

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



