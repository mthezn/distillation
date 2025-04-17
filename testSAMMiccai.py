import math
import sys
from typing import Iterable, Optional
from tqdm import tqdm
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from Dataser import ImageMaskDataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from build_CMT_sam import sam_model_registry
from PIL import Image
from repvit_sam import SamPredictor
import pandas as pd
from torch.cuda.amp import GradScaler, autocast


from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils
image_dirs_val = ["MICCAI/instrument_1_4_testing/instrument_dataset_1/left_frames"]
mask_dirs_val = ["MICCAI/instrument_2017_test/instrument_2017_test/instrument_dataset_1/BinarySegmentation"]
image_transform = transforms.Compose([
    transforms.Resize((1024,1024)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.5,0.5,0.5],std = [0.5,0.5,0.5])



])
mask_transform  = transforms.Compose([

    transforms.Resize((1024,1024)),
    transforms.ToTensor()
])
datasetTest = ImageMaskDataset(image_dirs=image_dirs_val,mask_dirs=mask_dirs_val,transform=image_transform,mask_transform=mask_transform)
dataloaderTest = DataLoader(datasetTest,batch_size=2,shuffle=True)
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



sam_checkpoint = "C:/Users/User/OneDrive - Politecnico di Milano/Documenti/POLIMI/Tesi/EdgeSAM/RepViT/sam/weights/repvit_sam.pt"
model_type = "repvit"

device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
sam.eval()

predictor = SamPredictor(sam)

timeDf = pd.DataFrame(columns=['time', 'index', 'iou'])




for images, labels in dataloaderTest:  # i->batch index, images->batch of images, labels->batch of labels



        images = images.to(device)
        print(images.shape)
        labels = labels.to(device)
        print(labels.shape)

        for image, label in zip(images, labels):
            # Convert the mask to a binary mask
            label = label.squeeze(0).cpu().numpy()
            # print("label",label)
            print(label.shape)
            image = np.array(image)
            # Convert to binary mask
            label = (label > 0).astype(np.uint8)
            # Convert to binary mask
            contours, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
            # print("contours",contours)

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
            transformed_boxes = predictor.transform.apply_boxes_torch(bbox, images[0].shape[:2])
            plt.figure(figsize=(10, 10))

            image = np.transpose(image, (1, 2, 0))
            print(image.shape)
            predictor.set_image(image)

            masks, _, low_res_teach = predictor.predict_torch(
                # predict_torch serve quando ho le bboxes altrimenti predictor.predict
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            maskunion = np.zeros_like(masks[0].cpu().numpy())
            for mask in masks:
                show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
                print(mask.shape)
                maskunion = np.logical_or(maskunion, mask.cpu().numpy())
            for box in bbox:
                show_box(box.cpu().numpy(), plt.gca())
            plt.axis('off')
            plt.show()

            #latency = (end_time - start_time) * 1000
            iou = calculate_iou(maskunion, label)

            #timeDf.loc[len(timeDf)] = [latency, len(timeDf), iou]
timeDf.to_csv('TimeDfBBoxStudent.csv', index=False)
print(timeDf)


