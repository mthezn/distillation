import cv2
import torch
import numpy as np
from torchvision import transforms
import pandas as pd

import albumentations as A
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
import numpy as np
import time
import cv2

from Dataset import ImageMaskDataset,CholecDataset
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
    mask_blurred = cv2.GaussianBlur(mask_clean, (5, 5), 0)
    mask_blurred = mask_blurred / 255

    return mask_blurred


# === CONFIG ===
video_input = "cat1_test_set_public/1_fps1.mp4"
video_output = "cat1_test_set_public/output_segmented_videoRandom1.mp4"
device = "cuda" if torch.cuda.is_available() else "cpu"

cap = cv2.VideoCapture(video_input)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
transform = transforms.Compose([
    transforms.ToTensor(),             # da HxWxC a CxHxW
    transforms.Resize((1024, 1024)),     # adatta alla dimensione del modello
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# === CREA VIDEO WRITER ===
out = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))

autosam_checkpoint = "checkpoints/23_06/autoSamRandom68v4F.pth"
model_type = "autoSam"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = sam_model_registry[model_type](checkpoint=autosam_checkpoint)

model.to(device=device)


model.eval()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    original = frame.copy()

    # === Preprocessing ===
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        image_embedding = model.image_encoder(img)
        pred, _ = model.mask_decoder(
            image_embeddings=image_embedding,  # dict
            image_pe=model.prompt_encoder.get_dense_pe(),

            multimask_output=False
        )             # output tipo [B, 1, H, W] o logits

    # === Post-processing maschera ===
    mask =  model.postprocess_masks(pred,(1024,1024),(1024,1024)).cpu().numpy() # [B, 1, H, W] -> [H, W]
    mask = (mask > 0).astype(np.uint8)
    mask = mask.squeeze()  # da (1,1024,1024) a (1024,1024)
    mask = refining(mask)  # applica refining

    # === Resize della maschera alla dimensione del frame originale ===
    mask_resized = cv2.resize(mask, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)

    # === Crea maschera colorata ===
    colored_mask = np.zeros_like(original)
    colored_mask[:, :, 1] = mask_resized * 255  # canale verde

    # === Fonde maschera + immagine originale ===
    blended = cv2.addWeighted(original, 0.8, colored_mask, 0.4, 0)

    # === Scrivi nel nuovo video ===
    out.write(blended)

# === CHIUDI ===
cap.release()
out.release()
print(f"Video salvato in {video_output}")