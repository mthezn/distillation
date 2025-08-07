import pandas as pd

import albumentations as A
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
import numpy as np
import time
import cv2
from datasets import load_dataset

from Dataset import ImageMaskDataset, CholecDataset
import torch
from torch.utils.data import DataLoader
from modeling.build_sam import sam_model_registry
from utility import dice_coefficient,sensitivity,specificity
import os

from Dataset import MMIDataset
from pathlib import Path
import random
import tqdm
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def seed_all(seed):
    if not seed:
        seed = 10

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
    # mask = mask.detach().cpu().numpy()
    mask = (mask * 255).astype(np.uint8)
    while mask.ndim > 2:
        mask = mask[0]
    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    mask_blurred = cv2.GaussianBlur(mask_clean, (5, 5), 0)
    mask_blurred = mask_blurred / 255

    return mask_blurred

def main():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed
    g = torch.Generator()
    g.manual_seed(0)
    seed_all(seed=123)

    # Creation of runs file for the experiment
    root_dir = Path("test/runs/mmi_experiment")
    if not root_dir.exists():
        root_dir.mkdir(parents=True, exist_ok=True)
    test_n = len(list(n for n in os.listdir("test/runs/mmi_experiment") if n.startswith('exp_')))
    os.makedirs(root_dir / ("exp_" + str(test_n+1)), exist_ok=True)

    saving_path = root_dir / ("exp_" + str(test_n+1))
   
    print("Loading dataset...")
    dataset_mmi = "./dataset_mmi_1807"
    
    test_transform = A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    #image_dirs_val = ["/home/shared-nearmrs/mdezenDatasets/MICCAI/instrument_1_4_testing/instrument_dataset_4/left_frames"]
    #mask_dirs_val = ["/home/shared-nearmrs/mdezenDatasets/MICCAI/instrument_2017_test/instrument_2017_test/instrument_dataset_4/gt/BinarySegmentation"]

    #test_dataset = ImageMaskDataset(image_dirs=image_dirs_val, mask_dirs=mask_dirs_val, transform=test_transform,)
    #test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    test_dataset = MMIDataset(root_dir=dataset_mmi, split='test', transform=test_transform, num_classes=1)
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False, num_workers = 8, worker_init_fn = seed_worker, generator=g)

    print("Dataset loaded!")
    print("Number of test samples:", len(test_dataset))

    autosam_checkpoint = "runs/mmi_experiment/exp_2/best_model.pth"
    #autosam_checkpoint = "/home/shared-nearmrs/mdezenDatasets/sam_vit_b_01ec64.pth"
    #autosam_checkpoint = "/home/shared-nearmrs/mdezenDatasets/sam_vit_h_4b8939.pth"
    model_type = "autoSam"
    #model_type = "vit_b"
    print("Loading model...")
    model = sam_model_registry[model_type](checkpoint=autosam_checkpoint)
    #state_dict = torch.load(autosam_checkpoint, map_location=device)
    #model.load_state_dict(state_dict, strict=False)  # Load the state dict into the model
    model.to(device=device)

    model.eval()
    print("Model loaded!")
    print("Size of model.image_encoder:", sum(p.numel() for p in model.image_encoder.parameters() if p.requires_grad))
    input("Press Enter to continue...")  
    timeDf = pd.DataFrame(columns=['time', 'index', 'iou','dice','sensitivity','specificity'])
    pbar = tqdm.tqdm(test_loader, desc="Processing test samples", unit="sample")
    for images, labels in pbar: 
    
        original_shape = (1080, 1920)  # Original shape of the images in the dataset
        images = images.to(device)
        labels = labels.to(device)
        
        for image, label in zip(images, labels):
            image = image.to(device=device).float()

            image = image.unsqueeze(0)
            print(f"Image shape: {image.shape}, dtype: {image.dtype}")
            image_memory = image.element_size() * image.nelement() / (1024 ** 2)
            print(f"Memory occupied by the image: {image_memory:.2f} MB")
            input("Press Enter to continue...")
            start_time = time.time()
            image_embedding = model.image_encoder(image)
            t1 = time.time()
            low_res, _ = model.mask_decoder(
                image_embeddings=image_embedding,  # dict
                image_pe=model.prompt_encoder.get_dense_pe(),
                multimask_output=False
            )
            t2 = time.time()
            postprocessed = model.postprocess_masks(low_res, (1024, 1024), (1024, 1024))
            end_time = time.time()
            latency = (end_time - start_time) * 1000
            print("Encoder:", (t1 - start_time)*1000, "Decoder:", (t2 - t1)*1000, "Postprocess:", (end_time - t2)*1000, "Total:", latency)
           
            print("Memory Allocated (MB):", torch.cuda.memory_allocated(device_id) / 1024**2)
            print("Memory Cached (MB):", torch.cuda.memory_reserved(device_id) / 1024**2)
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            print("Memory Allocated (MB):", torch.cuda.memory_allocated(device_id) / 1024**2)
            print("Memory Cached (MB):", torch.cuda.memory_reserved(device_id) / 1024**2)
            input("Press Enter to continue...")

    timeDf.to_csv(saving_path/ 'Results.csv', index=False)
    pd.set_option('display.max_rows', None)

if __name__ == "__main__":
    main()