"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

from PIL.ImageChops import logical_or
from tqdm import tqdm
import torch
import cv2

import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
import os
from PIL import Image
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils
from repvit_sam import SamPredictor
import pandas as pd
from torch.utils.checkpoint import checkpoint

from Dataser import ImageMaskDataset
from repvit_sam import SamPredictor, sam_model_registry
import copy
from matplotlib import pyplot as plt
import numpy as np
import time
import cv2
from datasets import load_dataset

from Dataser import ImageMaskDataset,CholecDataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from build_CMT_sam import sam_model_registry
import torch.nn.functional as F
from datasets import load_dataset_builder
from PIL import Image




def display_image(dataset, image_index):
    '''Display the image and corresponding three masks.'''

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    for ax in axs.flat:
        ax.axis('off')

    # Display each image in its respective subplot
    axs[0, 0].imshow(dataset['train'][image_index]['image'])
    axs[0, 1].imshow(dataset['train'][image_index]['color_mask'])
    axs[1, 0].imshow(dataset['train'][image_index]['watershed_mask'])
    axs[1, 1].imshow(dataset['train'][image_index]['annotation_mask'])

    # Adjust spacing between images
    plt.subplots_adjust(wspace=0.01, hspace=-0.6)

    plt.show()


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


def calculate_iou(mask_pred, mask_gt):
    # Ensure the inputs are NumPy arrays
    if isinstance(mask_pred, torch.Tensor):
        mask_pred = mask_pred.cpu().numpy()
    if isinstance(mask_gt, torch.Tensor):
        mask_gt = mask_gt.cpu().numpy()

    # Calculate the intersection (common pixels in both masks)
    intersection = np.logical_and(mask_pred, mask_gt).sum()

    # Calculate the union (all pixels that are 1 in at least one of the masks)
    union = np.logical_or(mask_pred, mask_gt).sum()

    # Calculate IoU (Intersection over Union)
    iou = intersection / union if union != 0 else 0  # Avoid division by zero

    return iou

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def refining(mask):
    # 1. Rimuovi rumore (morphological opening)
    #mask = mask.detach().cpu().numpy()
    mask = (mask * 255).astype(np.uint8)
    while mask.ndim > 2:
        mask = mask[0]
    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)




    # 2. Chiudi buchi interni (closing)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    # 3. (opzionale) Gaussian blur per bordi morbidi
    mask_blurred = cv2.GaussianBlur(mask_clean, (5, 5), 0)
    mask_blurred = mask_blurred/255

    return mask_blurred


def save_binary_mask(mask_tensor, epoch, batch_idx, output_dir="binary_masks"):
    os.makedirs(output_dir, exist_ok=True)

    # Convert to numpy array
    if isinstance(mask_tensor, torch.Tensor):
        mask_np = mask_tensor.detach().cpu().numpy()
    else:
        mask_np = mask_tensor

    # Ensure shape
    if mask_np.ndim == 4:
        mask_np = mask_np[0, 0]
    elif mask_np.ndim == 3:
        mask_np = mask_np[0]

    # Ensure binary uint8 image
    mask_np = (mask_np > 0).astype(np.uint8) * 255

    # Save
    Image.fromarray(mask_np).save(
        os.path.join(output_dir, f"mask_epoch{epoch}_batch{batch_idx}.png")
    )

def calculate_iou(mask_pred, mask_gt):
    # Ensure the inputs are NumPy arrays
    if isinstance(mask_pred, torch.Tensor):
        mask_pred = mask_pred.cpu().numpy()
    if isinstance(mask_gt, torch.Tensor):
        mask_gt = mask_gt.cpu().numpy()

    # Calculate the intersection (common pixels in both masks)
    intersection = np.logical_and(mask_pred, mask_gt).sum()

    # Calculate the union (all pixels that are 1 in at least one of the masks)
    union = np.logical_or(mask_pred, mask_gt).sum()

    # Calculate IoU (Intersection over Union)
    iou = intersection / union if union != 0 else 0  # Avoid division by zero

    return iou
def set_bn_state(model):
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()

def predict_boxes(predictor, boxes):


    masks, _, low_res = predictor.predict_torch(
        # predict_torch serve quando ho le bboxes altrimenti predictor.predict
        point_coords=None,
        point_labels=None,
        boxes=boxes,
        multimask_output=False,
    )
    return masks, _, low_res


def predict_points_boxes_manual(model, image_embedding, boxes, centroids, input_label):
    all_masks = []
    all_scores = []
    all_low_res = []
    model_device = next(model.prompt_encoder.parameters()).device


    for i in range(boxes.shape[0]):
        # Estrai singola box, punto e label
        box = boxes[i].unsqueeze(0).to(device=model_device) # [1, 4]
        point = centroids[:, i, :].unsqueeze(0).to(device=model_device) # [1, 1, 2]
        label = input_label[:, i].unsqueeze(0).to(device = model_device) # [1, 1]

        # Encode prompt: box + point
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=(point, label),
            boxes=box,
            masks=None,
        )

        # Usa mask decoder
        low_res_logits, score = model.mask_decoder(
            image_embeddings=image_embedding,          # [1, C, H', W']
            image_pe=model.prompt_encoder.get_dense_pe(),  # positional encoding
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )

        # Upscale maschera a risoluzione originale
        mask = model.postprocess_masks(low_res_logits, input_size=(image_embedding.shape[-2], image_embedding.shape[-1]),original_size=(1024, 1024))

        all_masks.append(mask)
        all_scores.append(score)
        all_low_res.append(low_res_logits)

    # Concatenazione dei risultati
    if all_masks == []:
        return torch.zeros((1, 1, 1024, 1024)).to(device=model_device), torch.zeros((1, 1)).to(device=model_device), torch.zeros((1, 1, 1024, 1024)).to(device=model_device)
    final_masks = torch.cat(all_masks, dim=0)  # [N, 1, H, W]
    final_scores = torch.cat(all_scores, dim=0)  # [N, 1]
    final_low_res = torch.cat(all_low_res, dim=0)

    return final_masks, final_scores, final_low_res
def predict_points_boxes(predictor,image,boxes,centroids,input_label):
    all_masks = []
    all_scores = []
    all_low_res = []
    #image_array =(image[0])
    #image_array = np.transpose(image_array, (1, 2, 0))
    #predictor.set_image(image_array)
    model_device = next(predictor.model.parameters()).device  # Assicura coerenza col modello

    for i in range(boxes.shape[0]):
        box = boxes[i].unsqueeze(0).to(model_device)  # shape: [1, 4]
        centroid = centroids[:, i, :].unsqueeze(0).to(model_device)  # shape: [1, 1, 2]
        label = input_label[:, i].unsqueeze(0).to(model_device)  # shape: [1, 1]

        masks, scores, low_res = predictor.predict_torch(
            point_coords=centroid,
            point_labels=label,
            boxes=box,
            multimask_output=False
        )

        all_masks.append(masks)
        all_scores.append(scores)
        all_low_res.append(low_res)
    if all_masks == []:
            return torch.zeros((1, 1, 1024, 1024)).to(device=model_device), torch.zeros((1, 1)).to(
                device=model_device), torch.zeros((1, 1, 1024, 1024)).to(device=model_device)
    # Concatenazione dei risultati
    final_masks = torch.cat(all_masks, dim=0)
    final_scores = torch.cat(all_scores, dim=0)
    final_low_res = torch.cat(all_low_res, dim=0)

    return final_masks, final_scores, final_low_res


image_dirs_val = ["MICCAI/test/frame"]
mask_dirs_val = ["MICCAI/test/gt"]
image_transform = transforms.Compose([
    transforms.Resize((1024,1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std = [0.5,0.5,0.5])



])
mask_transform  = transforms.Compose([

    transforms.Resize((1024,1024)),
    transforms.ToTensor()
])
def contains_instrument(example):
    mask = np.array(example["color_mask"])  # o "segmentation" se diverso
    return np.any((mask == 169) | (mask == 170))

#datasetCholec = load_dataset("minwoosun/CholecSeg8k", trust_remote_code=True)

#filtered_ds = datasetCholec['train'].filter(contains_instrument)
datasetTest = ImageMaskDataset(image_dirs=image_dirs_val,mask_dirs=mask_dirs_val,transform=image_transform,mask_transform=mask_transform)
#datasetTest = CholecDataset(hf_dataset=filtered_ds, transform=image_transform, mask_transform=mask_transform)
dataloaderTest = DataLoader(datasetTest,batch_size=2,shuffle=True)
student_checkpoint = "checkpoints/13_05/decoupledVitBDGfFE.pth"
state_dict = torch.load(student_checkpoint, map_location=torch.device('cpu'))
model = sam_model_registry["CMT"](checkpoint=None)
model.load_state_dict(state_dict)

#print("Missing keys:", model.load_state_dict(state_dict, strict=False))
#CARICO UN MODELLO SAM
#sam_checkpoint = "C:/Users/User/OneDrive - Politecnico di Milano/Documenti/POLIMI/Tesi/distillation/checkpoints/sam_vit_b_01ec64.pth"

sam_checkpoint = "checkpoints/sam_vit_b_01ec64.pth"
model_type = "vit_b"

device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
#ASSEGNO L'IMAGE ENCODER DISTILLATO A SAM
sam.image_encoder = model.image_encoder
sam.eval()
model.eval()
predictor = SamPredictor(sam)




for images, labels in dataloaderTest:  # i->batch index, images->batch of images, labels->batch of labels

    images = images.to(device)
    labels = labels.to(device)

    results_teach = []
    logits_teach = []
    results_stud = []

    for image, label in zip(images, labels):
        # Convert the mask to a binary mask
        label = label.detach().cpu().numpy()

        label = (label > 0).astype(np.uint8)

        image = np.array(image)
        # print(image.shape)

        # Create contours from the gt
        contours, _ = cv2.findContours(label.squeeze(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

        centroids = []
        input_label = []
        bbox = []
        if contours:
            for countour in contours:
                M = cv2.moments(countour)
                if M["m00"] != 0:
                    centroid_x = int(M["m10"] / M["m00"])
                    centroid_y = int(M["m01"] / M["m00"])
                    centroids.append([centroid_x, centroid_y])
                    input_label.append(1)
                    x, y, w, h = cv2.boundingRect(countour)
                    bbox.append([x, y, x + w, y + h])
        centroids = np.array(centroids)
        print(centroids)
        bbox = torch.tensor(bbox).float()
        centroids = torch.tensor(centroids).float().unsqueeze(0)

        original_size = tuple(map(int, images[0].shape[-2:]))

        input_label = torch.tensor(input_label, dtype=torch.int64).unsqueeze(0)
        # print(image.shape)

        # image_embedding_model = model.image_encoder(image)  # in teoria posso passare n batch di immagini

        image_array = np.transpose(image, (1, 2, 0))
        image = np.transpose(image, (1, 2, 0))
        image = (image * 0.5 + 0.5) * 255
        image = image.astype(np.uint8)
        predictor.set_image(image)
        masks, _, low_res = predict_points_boxes(predictor,image, bbox,centroids,input_label)  # masks_model -> binary masks, low_res -> logits

        low_res = model.postprocess_masks(low_res, (1024, 1024), (1024, 1024))

        # vec = torch.sigmoid(low_res)
        # vec = F.interpolate(vec, (1024,1024), mode="bilinear", align_corners=False)
        end_time = time.time()
        # unique,values = np.unique(low_res, return_counts=True)


        plt.figure(figsize=(10, 10))
        #image = np.transpose(image, (1, 2, 0))  # Convert to HWC format for display
        plt.imshow(image)
        #maskunion = np.zeros_like(masks[0].cpu().numpy())
        logits_list = []
        maskunion = np.zeros_like(masks[0].cpu().numpy())
        for mask in masks:
            mask = mask.cpu().numpy()  # ricordarsi .foat con Bcelogits
            unique, values = np.unique(mask, return_counts=True)
            print("unique", unique)
            print("values", values)
            mask = refining(mask)
            save_binary_mask(mask, 1, 3, output_dir="binary_masks_teacher")

            maskunion = np.logical_or(maskunion, mask)

            show_mask(mask, plt.gca(), random_color=True)


        for box in bbox:
            show_box(box.cpu().numpy(), plt.gca())
        plt.axis('off')
        plt.show()
        plt.figure(figsize=(10, 10))

        plt.imshow(np.transpose(maskunion,(1,2,0)), cmap='gray')
        plt.axis('off')




