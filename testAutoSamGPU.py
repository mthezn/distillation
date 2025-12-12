import pandas as pd

import albumentations as A
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
import numpy as np
import time
import cv2
import os
from Dataset import ImageMaskDataset,CholecDataset,LeedsDataset
import torch
from torch.utils.data import DataLoader
from modeling.build_sam import sam_model_registry
from utility import dice_coefficient,sensitivity,specificity

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
    if mask_pred.ndim == 3:
        mask_pred = np.any(mask_pred != 0, axis=-1)
    if mask_gt.ndim == 3:
        mask_gt = np.any(mask_gt != 0, axis=-1)

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
    # mask = mask.detach().cpu().numpy()
    mask = (mask * 255).astype(np.uint8)
    while mask.ndim > 2:
        mask = mask[0]
    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 2. Chiudi buchi interni (closing)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    # 3. (opzionale) Gaussian blur per bordi morbidi
    mask_blurred = cv2.GaussianBlur(mask_clean, (3, 3), 0)
    mask_blurred = mask_blurred / 255

    return mask_blurred




########################################################################################################

image_dirs_test = ["MICCAI/instrument_1_4_testing/instrument_dataset_1/left_frames",
                   "MICCAI/instrument_1_4_testing/instrument_dataset_2/left_frames",
                   "MICCAI/instrument_1_4_testing/instrument_dataset_3/left_frames",
                   "MICCAI/instrument_1_4_testing/instrument_dataset_4/left_frames",
                   ]
mask_dirs_test = ["MICCAI/instrument_2017_test/instrument_2017_test/instrument_dataset_1/gt/TypeSegmentationRescaled",
                  "MICCAI/instrument_2017_test/instrument_2017_test/instrument_dataset_2/gt/TypeSegmentationRescaled",
                  "MICCAI/instrument_2017_test/instrument_2017_test/instrument_dataset_3/gt/TypeSegmentationRescaled",
                  "MICCAI/instrument_2017_test/instrument_2017_test/instrument_dataset_4/gt/TypeSegmentationRescaled",
]
#image_dirs_val = ["MICCAI/instrument_1_4_training/instrument_dataset_4/left_frames"]
#mask_dirs_val = ["MICCAI/instrument_1_4_training/instrument_dataset_4/ground_truth/Large_Needle_Driver_Left_labels"]
image_dirs_train = [

    "MICCAI/instrument_1_4_training/instrument_dataset_1/left_frames",
]
mask_dirs_train = [
    "MICCAI/instrument_1_4_training/instrument_dataset_1/ground_truth/Left_Prograsp_Forceps_labels",
    "MICCAI/instrument_1_4_training/instrument_dataset_1/ground_truth/Maryland_Bipolar_Forceps_labels",
    "MICCAI/instrument_1_4_training/instrument_dataset_1/ground_truth/Right_Prograsp_Forceps_labels"]

validation_transform = A.Compose([
    A.Resize(1024,1024),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])





def contains_instrument(example):
    mask = np.array(example["color_mask"])  # o "segmentation" se diverso
    return np.any((mask == 169) | (mask == 170))


img_dir=["/home/shared-nearmrs/MS_dataset/real"]
#datasetTest = LeedsDataset(image_dirs=img_dir, transform=validation_transform,)
datasetTest = ImageMaskDataset(image_dirs=image_dirs_test,mask_dirs=mask_dirs_test, transform=validation_transform)
dataloaderTest = DataLoader(datasetTest, batch_size=2, shuffle=True)


# CARICO UN MODELLO SAM
# sam_checkpoint = "C:/Users/User/OneDrive - Politecnico di Milano/Documenti/POLIMI/Tesi/distillation/checkpoints/sam_vit_b_01ec64.pth"
autosam_checkpoint = "checkpointsLight/autoSamFineUnetk57VL.pth"
model_type = "autoSamUnet"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = sam_model_registry[model_type](checkpoint=None)
model.load_state_dict(torch.load(autosam_checkpoint, map_location=device))


model.to(device=device)


model.eval()


# predictor = SamPredictor(model)
timeDf = pd.DataFrame(columns=['time', 'index', 'iou','dice','sensitivity','specificity'])
save_dir = "results_seg2"
os.makedirs(save_dir, exist_ok=True)
n = 0
for images, labels in dataloaderTest:  # i->batch index, images->batch of images, labels->batch of labels

    images = images.to(device)
    print(images.shape)
    labels = labels.to(device)
    print(labels.shape)
    results_teach = []
    results_stud = []
    for image, label in zip(images, labels):
        # Convert the mask to a binary mask
        label = label.detach().cpu().numpy()
        label = (label > 0).astype(np.uint8)
        # print("label",label)
        print(label.shape)
        unique,values = np.unique(label, return_counts=True)
        print("unique", unique)
        print("values", values)

        image = image.to(device = device).float()

        # Assicurati che sia float e abbia batch dimensione
        # converti in float32 se necessario
     # manca batch dimensione
        image = image.unsqueeze(0)



        start_time = time.time()
        image_embedding = model.image_encoder(image)

        low_res = model.mask_decoder(image_embedding)
        low_res = model.postprocess_masks(low_res,(1024,1024),(1024,1024))
        end_time = time.time()
        mask = low_res >0
        mask = mask.detach().cpu().numpy()
        mask = refining(mask)

        #mask = mask.cpu().numpy()


        values, counts = np.unique(low_res.detach().cpu().numpy(), return_counts=True)
        print("unique", values)
        print("values", counts)
        # Visualizza la maschera raw


        # Applica soglia
        binary_mask = (low_res > 0)








        values, counts = np.unique(mask, return_counts=True)  # mask.cpu().numpy()
        #print("unique", values)
        #print("counts", counts)
        img_vis = image.squeeze(0).cpu().permute(1, 2, 0).numpy()
        img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min())  # normali

        # plot side by side
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].imshow(img_vis)
        axs[0].set_title("Image")
        axs[0].axis("off")



        axs[1].imshow(mask, cmap='gray')
        axs[1].set_title("Prediction")
        axs[1].axis("off")

        # salva
        save_path = os.path.join(save_dir, f"result_{n}.png")
        n = n +1
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)



        latency = (end_time - start_time) * 1000
        iou = calculate_iou(mask, label)
        dice = dice_coefficient(mask, label)
        sens = sensitivity(mask, label)
        spec = specificity(mask, label)

        timeDf.loc[len(timeDf)] = [latency, len(timeDf), iou,dice,sens,spec]


timeDf.to_csv('RISULTATI/TimeDfBBoxAutoSam.csv', index=False)
pd.set_option('display.max_rows', None)
print(timeDf)

