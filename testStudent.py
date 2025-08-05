import pandas as pd

from repvit_sam import SamPredictor
from matplotlib import pyplot as plt
import numpy as np
import time
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from Dataset import ImageMaskDataset,CholecDataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from modeling.build_sam import sam_model_registry
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from utility import dice_coefficient,sensitivity,specificity
from display import show_mask, show_points, show_box
from utility import generate_random_name,refining, predict_points_boxes,contains_instrument, calculate_iou,get_bbox_centroids






########################################################################################################

image_dirs_val = ["MICCAI/instrument_1_4_testing/instrument_dataset_4/left_frames"]
mask_dirs_val = ["MICCAI/instrument_2017_test/instrument_2017_test/instrument_dataset_4/gt/BinarySegmentation"]


image_transform = A.Compose([
    A.Resize(1024, 1024),
    #A.HorizontalFlip(p=0.5),
    #A.VerticalFlip(p=0.5),
    #A.Rotate(limit=45, p=0.5),
    #A.ColorJitter(p=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    #A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])




datasetCholec = load_dataset("minwoosun/CholecSeg8k", trust_remote_code=True)

filtered_ds = datasetCholec['train'].filter(contains_instrument)
datasetTest = ImageMaskDataset(image_dirs=image_dirs_val,mask_dirs=mask_dirs_val,transform=image_transform)
#datasetTest = CholecDataset(hf_dataset=filtered_ds, transform=image_transform)
dataloaderTest = DataLoader(datasetTest,batch_size=2,shuffle=True)

student_checkpoint = "checkpoints/checkpoints_mmagro/decoupledVitHhkuYf.pth"
state_dict = torch.load(student_checkpoint, map_location=torch.device('cpu'))
model = sam_model_registry["CMT"](checkpoint=None)
model.load_state_dict(state_dict)


#CARICO UN MODELLO SAM


sam_checkpoint = "checkpoints/repvit_sam.pt"
model_type = "repvit"


device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
#ASSEGNO L'IMAGE ENCODER DISTILLATO A SAM
#sam.image_encoder = model.image_encoder
sam.eval()
model.eval()
predictor = SamPredictor(sam) 




model.eval()




timeDf = pd.DataFrame(columns=['time', 'index', 'iou', 'dice', 'sensitivity', 'specificity'])

for images, labels in dataloaderTest:  # i->batch index, images->batch of images, labels->batch of labels



        images = images.to(device)
        print(images.shape)
        labels = labels.to(device)
        print(labels.shape)
        results_teach = []
        results_stud = []
        for image, label in zip(images, labels):
            # Convert the mask to a binary mask
            label = np.array(label.cpu())
            label = (label>0).astype(np.uint8)
            # print("label",label)
            print(label.shape)
            image_array = np.array(image.cpu())
            image = image.unsqueeze(0)
            # Convert to binary mask

            plt.figure(figsize=(10, 10))
            plt.imshow(label.squeeze(), cmap='gray')


            centroids,bbox,input_label = get_bbox_centroids(label,5)
            centroids = torch.tensor(centroids).float()
            print(centroids)


            bbox = torch.tensor(bbox).float()
            original_size = tuple(map(int, images[0].shape[-2:]))





            #input_label = ([1] * len(centroids))
            input_label = torch.tensor(input_label,dtype = torch.int64).unsqueeze(0)
            print(image.shape)
            print("Image shape:", image.shape)
            print("Image min/max values:", image.min(), image.max())

            start_time = time.time()

            masks, _, low_res = predict_points_boxes(predictor,image, boxes=bbox,centroids = centroids.unsqueeze(0), input_label=input_label)


            #vec = torch.sigmoid(low_res)
            #vec = F.interpolate(vec, (1024,1024), mode="bilinear", align_corners=False)
            end_time = time.time()


            plt.figure(figsize=(10, 10))

            plt.imshow(np.transpose(image_array, (1, 2, 0)) * 0.5 + 0.5)  # Normalizza l'immagine per visualizzazione
            maskunion = np.zeros_like(masks[0].cpu().numpy())
            for mask in masks:
                mask = mask.cpu().numpy()
                mask = refining(mask)
                show_mask(mask, plt.gca(), random_color=True)




                maskunion = np.logical_or(maskunion, mask)
            for box in bbox:
                show_box(box.cpu().numpy(), plt.gca())
            plt.axis('off')
            plt.show()


            latency = (end_time - start_time) * 1000
            iou = calculate_iou(maskunion, label)
            dice = dice_coefficient(maskunion, label)
            sens = sensitivity(maskunion, label)
            spec = specificity(maskunion, label)

            timeDf.loc[len(timeDf)] = [latency, len(timeDf), iou, dice, sens, spec]
timeDf.to_csv('RISULTATI DECOUPLED/TimeDfBBoxStudent.csv', index=False)
print(timeDf)

