import pandas as pd

from repvit_sam import SamPredictor, sam_model_registry
from datasets import load_dataset
from matplotlib import pyplot as plt
import numpy as np
import time
import cv2
import torch
from PIL import Image

dataset = load_dataset("minwoosun/CholecSeg8k", trust_remote_code=True)


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
    # Calcola l'intersezione (pixel comuni nelle due maschere)
    intersection = np.logical_and(mask_pred, mask_gt).sum()

    # Calcola l'unione (tutti i pixel che sono 1 in almeno una delle due maschere)
    union = np.logical_or(mask_pred, mask_gt).sum()

    # Calcola IoU (Intersection over Union)
    iou = intersection / union if union != 0 else 0  # Evita la divisione per zero

    return iou


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


# display_image(dataset, 800)   #video index from 0 to 8079
# image = dataset['train'][800]['image']
#
#
#
# # Converti l'immagine in un array NumPy
# image_array = np.array(image)
# plt.figure(figsize=(10,10))
# plt.imshow(image_array)
# plt.axis('on')
# plt.show()
#
# input_point = np.array([[300, 100],[600,400],[700,200],[200,300]]) #x,y
# input_label = np.array([1,1,0,0])
# plt.figure(figsize=(10,10))
# plt.imshow(image)
# show_points(input_point, input_label, plt.gca())
# plt.axis('on')
# plt.show()
# start_time = time.time()
sam = sam_model_registry["vit_h"](
    checkpoint="checkpoints/sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)
device = "cuda" if torch.cuda.is_available() else "cpu"
sam.to(device=device)
idx = [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 80, 81, 82, 83, 84, 85, 86, 87, 88]
timeDf = pd.DataFrame(columns=['time', 'index', 'iou'])
for i in idx:
    image = dataset['train'][i]['image']
    image_array = np.array(image)
    mask = np.array(dataset['train'][i]['color_mask'])
    instrument_mask = ((mask == 170) | (mask == 169)).astype(np.uint8) * 255
    instrument_mask = instrument_mask.astype(np.uint8)
    if len(instrument_mask.shape) == 3:
        instrument_mask = cv2.cvtColor(instrument_mask, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(instrument_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

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

    bbox = torch.tensor(bbox).float()
    transformed_boxes = predictor.transform.apply_boxes_torch(bbox, image_array.shape[:2])
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    show_points(centroids, input_label, plt.gca())
    show_bbox(bbox, plt.gca())
    plt.axis('off')
    plt.show()

    start_time = time.time()
    print("shape immagine ")
    print(image_array.shape )
    predictor.set_image(image_array)

    masks, _, _ = predictor.predict_torch(  # predict_torch serve quando ho le bboxes altrimenti predictor.predict
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    end_time = time.time()
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    maskunion = np.zeros_like(masks[0].cpu().numpy())
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        maskunion = np.logical_or(maskunion, mask.cpu().numpy())
    for box in bbox:
        show_box(box.cpu().numpy(), plt.gca())
    plt.axis('off')
    plt.show()

    latency = (end_time - start_time) * 1000
    iou = calculate_iou(maskunion, instrument_mask)

    timeDf.loc[i] = [latency, i, iou]
#timeDf.to_csv('TimeDfBBoxSAM.csv', index=False)
print(timeDf)

