from ultralytics import YOLO
from yolo import YOLOSegmentationFineTuner
from Dataset import ImageMaskDataset,InstrumentDataset,InstrumentDatasetTest

import pandas as pd

import albumentations as A
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
import numpy as np
import time
import torch
import cv2
import os

from torch.utils.data import DataLoader

from utility import dice_coefficient,sensitivity,specificity,calculate_iou
# Carica modello pretrained
print("\n5️⃣ Predizione")
print("-" * 70)
class_names = { #class_names per il print su immagine
    0: 'Large_Needle_Driver',
    1:'Forceps', #qui viene fatto in modo che tutte le calssi 2 e 3 vengano scrittte come froceps nell'immagine


    2: 'Grasping_Retractor',
    3: 'Maryland_Bipolar_Forceps' ,
     4: 'Monopolar_Curved_Scissors',
     5: 'Other',
     6 : 'Vessel_Sealer'
}

GLOBAL_CLASS_MAPPING = {#class names per il dataset che legge le cartelle e da un numero
    'Large_Needle_Driver': 1,
    'Prograsp_Forceps': 2,

    'Bipolar_Forceps': 3,#cambia a 2 se vuoi unificare le forceps
    'Grasping_Retractor': 4,
    'Maryland_Bipolar_Forceps': 5,
    'Monopolar_Curved_Scissors': 6,
    #'Other': 7,
    'Vessel_Sealer': 7
}
validation_transform = A.Compose([
    A.Resize(1024,1024),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])
image_dirs_test = ["/home/mdezen/distillation/MICCAI/instrument_1_4_testing/instrument_dataset_2/left_frames"]
mask_dirs_test = ["/home/mdezen/distillation/MICCAI/instrument_2017_test/instrument_2017_test/instrument_dataset_2/gt/TypeSegmentationRescaled"]

    # Carica best model
best_model = YOLOSegmentationFineTuner(
        model_path='runs/segment/instrument_seg/weights/best.pt',
        num_classes=7
    )
datasetTest = InstrumentDatasetTest(image_dirs=image_dirs_test, gt_dirs=mask_dirs_test, transform=validation_transform, class_to_id=GLOBAL_CLASS_MAPPING)
#datasetTest = CholecDataset(hf_dataset=filtered_ds, transform=validation_transform)
print(len(datasetTest))
#datasetTest = InstrumentDataset(image_dirs=image_dirs_train, gt_dirs=mask_dirs_train, transform=validation_transform, class_to_id=GLOBAL_CLASS_MAPPING)
dataloaderTest = DataLoader(datasetTest, batch_size=2, shuffle=True)
    # Predici su nuova immagine
# predictor = SamPredictor(model)
timeDf = pd.DataFrame(columns=['time', 'index', 'iou','dice','sensitivity','specificity'])
save_dir = "results_yolo"
os.makedirs(save_dir, exist_ok=True)
positive_classes = [1, 2,3,4 ,5,6,7]
n= 0
colors = [np.array([255, 0, 0]), np.array([0, 255, 0]),np.array([0, 0, 255]),np.array([0, 255, 255]),np.array([255, 255, 0])]  # es. rosso e verde
for bi,(images,labels) in enumerate(dataloaderTest):
    for image,label in zip(images,labels):

        binary_mask = np.zeros(label.shape[:2], dtype=np.uint8)
        label = label.cpu().numpy()

        for c in positive_classes:
            binary_mask |= (label == c)

        start= time.time()
        results = best_model.predict(
        image_path=image,
        conf=0.25
    )
        end= time.time()
        if results  and results[0].masks is not None:
            res = results
            masks = res.masks.data
            binary_masks = (masks> 0.5).int()

        # Combine multiple predicted masks into a single mask
            combined_mask = torch.any(binary_masks.bool(),dim=0).int()
            combined_mask = combined_mask.cpu().numpy().astype(np.uint8)



    # Estrai maschere
        if results.masks is not None:
            masks = results.masks.data.cpu().numpy()  # [N, H, W]
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()

            print(f"Trovati {len(masks)} strumenti")
        img_vis = image.squeeze(0).cpu().permute(1, 2, 0).numpy()
        img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min())  # normali

        # plot side by side
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].imshow(img_vis)
        axs[0].set_title("Image")
        axs[0].axis("off")

        axs[1].imshow(combined_mask,cmap='jet', alpha=0.5)
        axs[1].set_title("Prediction")
        axs[1].axis("off")

        # salva
        save_path = os.path.join(save_dir, f"result_{n}.png")
        n = n + 1
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        latency = (end - start) * 1000
        iou = calculate_iou(combined_mask, label)
        dice = dice_coefficient(combined_mask, label)
        sens = sensitivity(combined_mask, label)
        spec = specificity(combined_mask, label)

        timeDf.loc[len(timeDf)] = [latency, len(timeDf), iou, dice, sens, spec]



timeDf.to_csv('TimeDfYOLO.csv', index=False)
pd.set_option('display.max_rows', None)
print(timeDf)