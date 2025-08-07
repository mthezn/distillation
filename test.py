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
    """
    Refine a binary mask to extract the two largest sharp blobs with clean contours.
    """
    # Ensure 2D and uint8
    if mask.ndim > 2:
        mask = mask[0]
    mask = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.copy()

    # Morphological cleaning
    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    # Connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)
    if num_labels <= 1:
        return mask_clean / 255.0

    # Keep 2 largest components (exclude background)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_indices = np.argsort(areas)[-2:] + 1  # Add 1 to skip background

    # Create binary mask for top blobs
    top_mask = np.isin(labels, largest_indices).astype(np.uint8)

    # Optional: find contours for sharp edges
    contours, _ = cv2.findContours(top_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_mask = np.zeros_like(top_mask)
    cv2.drawContours(contour_mask, contours, -1, color=1, thickness=-1)  # Fill in contours

    return contour_mask.astype(np.float32)



def main():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    device_id = torch.cuda.current_device()
    print("GPU memory cleaned.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # Seed
    g = torch.Generator()
    g.manual_seed(0)
    seed_all(seed=123)
   
    print("Loading dataset...")
    dataset_mmi = "./dataset_mmi_3107"
    
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

    autosam_checkpoint = "/home/mmagro/distillation/runs/mmi_experiment/exp_10/best_model.pth"
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
    saving_path = Path(os.path.dirname(autosam_checkpoint), "test_results")
    saving_path.mkdir(parents=True, exist_ok=True)

    print("Starting inference...")
    # Initialize a DataFrame to store results

    timeDf = pd.DataFrame(columns=['time', 'index', 'iou','dice','sensitivity'])
    pbar = tqdm.tqdm(test_loader, desc="Processing test samples", unit="sample")
    for images, labels in pbar: 
        #print("Memory Allocated (MB):", torch.cuda.memory_allocated(device_id) / 1024**2)
        #print("Memory Cached (MB):", torch.cuda.memory_reserved(device_id) / 1024**2)
        original_shape = (1080, 1920)  # Original shape of the images in the dataset
        images = images.to(device)
        labels = labels.to(device)
        for image, label in zip(images, labels):
            # Convert the mask to a binary mask
            label = torch.nn.functional.interpolate(label.unsqueeze(0).unsqueeze(0).float(), size=original_shape, mode='bilinear', align_corners=False)
            label = label.detach().cpu().numpy()
            label = (label > 0).astype(np.uint8)

            #image = torch.Tensor(image.cpu())
            #image = image.float()  # converti in float32 se necessario
            #image = image.unsqueeze(0)
            image = image.to(device=device).float()

            image = image.unsqueeze(0)
            #image_to_show = torch.nn.functional.interpolate(image, size=original_shape, mode='bilinear', align_corners=False)
            
            #start_time = time.time()

            #image_embedding = model.image_encoder(image)
            
            #low_res, _ = model.mask_decoder(
            #    image_embeddings=image_embedding,  # dict
            #    image_pe=model.prompt_encoder.get_dense_pe(),
            #    #sparse_prompt_embeddings=torch.zeros(1, 64, 256).to(device),  # Empty tensor for sparse prompts
            #    #dense_prompt_embeddings=torch.zeros(1, 1, 64).to(device),  # Tensor of size (1, 256, 64) for dense prompts
            #    multimask_output=False
            #)

            #low_res = model.postprocess_masks(low_res,(1024,1024),(1024,1024))
            #end_time = time.time()
            start_time = time.time()
            image_embedding = model.image_encoder(image)
            t1 = time.time()
            #print("Time for image encoding:", (t1 - start_time) * 1000, "ms")
            low_res, _ = model.mask_decoder(
                image_embeddings=image_embedding,  # dict
                image_pe=model.prompt_encoder.get_dense_pe(),
                multimask_output=False
            )
            #t2 = time.time()
            postprocessed = model.postprocess_masks(low_res, (1024, 1024), (1024, 1024))
            end_time = time.time()

            low_res = torch.nn.functional.interpolate(low_res, size=original_shape, mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

            mask = low_res > 0
            mask = mask.detach().cpu().numpy()
  
            #plt.figure(figsize=(12, 6))
            #plt.subplot(1, 3, 1)
            #plt.imshow(low_res.detach().cpu().numpy().squeeze(), cmap='gray')  # Display in grayscale
            #plt.title('Low Resolution Logits')
            #plt.axis('off')  # Remove axes for better visualization

            # Applica soglia
            #binary_mask = (low_res >= 0.0)
            # Visualizza maschera sogliata
            #plt.subplot(1, 3, 2)
            #plt.imshow(binary_mask.squeeze().detach().cpu().numpy(), cmap='gray')
            #plt.title('Thresholded mask (>0.5)')
            #plt.axis('off')
            
            #plt.subplot(1, 3, 3)
            #image_to_show = image_to_show[0].permute(1, 2, 0).cpu().numpy()
            #image = (image_to_show * 0.5 + 0.5) * 255
            #image = image.astype(np.uint8)
            #plt.imshow(image)

            #show_mask(mask, plt.gca(), random_color=True)
            #plt.title('Image with Predicted Mask')
            #plt.axis('off')
            #plt.tight_layout()
            #plt.savefig(saving_path / f"output_{len(timeDf)}.png")
            #plt.close()

            latency = (end_time - start_time) * 1000
            iou = calculate_iou(mask, label)
            dice = dice_coefficient(mask, label)
            sens = sensitivity(mask, label)
            #print("Encoder:", (t1 - start_time)*1000, "Decoder:", (t2 - t1)*1000, "Postprocess:", (end_time - t2)*1000, "Total:", latency)
            #print(f"Latency: {latency:.2f} ms, IoU: {iou:.4f}, Dice: {dice:.4f}, Sensitivity: {sens:.4f}, Specificity: {spec:.4f}")
            #timeDf.loc[len(timeDf)] = [latency, len(timeDf), iou,dice,sens,spec]

            # Resize the predicted mask and the original image to the original shape
            resized_mask = cv2.resize(mask.squeeze().astype(np.uint8), (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
            #resized_image = cv2.resize(image.astype(np.uint8), (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
            resized_gt = cv2.resize(label.squeeze().astype(np.uint8), (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)

            resized_mask_ref = refining(resized_mask)

            # Recompute metrics for resized and refined mask
            iou = calculate_iou(resized_mask, resized_gt)
            dice = dice_coefficient(resized_mask, resized_gt)
            sens = sensitivity(resized_mask, resized_gt)
            #print(f"Original Mask - IoU: {iou:.4f}, Dice: {dice:.4f}, Sensitivity: {sens:.4f}")    
            iou_ref = calculate_iou(resized_mask_ref, resized_gt)
            dice_ref = dice_coefficient(resized_mask_ref, resized_gt)
            sens_ref = sensitivity(resized_mask_ref, resized_gt)
            #print(f"Refined Mask - IoU: {iou_ref:.4f}, Dice: {dice_ref:.4f}, Sensitivity: {sens_ref:.4f}")
            timeDf.loc[len(timeDf)] = [latency, len(timeDf), iou_ref,dice_ref,sens_ref]
            #timeDf.loc[len(timeDf)] = [latency, len(timeDf), iou,dice,sens,spec]
            # Display resized image and mask
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 3, 1)
            plt.title("Resized Mask refined")
            plt.imshow(resized_mask_ref)
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.title("Resized Mask")
            plt.imshow(resized_mask, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.title("Resized Refined Mask")
            plt.imshow(resized_gt, cmap='gray')
            plt.axis('off')

            plt.tight_layout()
            #plt.savefig(saving_path / f"output_{len(timeDf)}.png")
            plt.close()
            #print("Memory Allocated (MB):", torch.cuda.memory_allocated(device_id) / 1024**2)
            #print("Memory Cached (MB):", torch.cuda.memory_reserved(device_id) / 1024**2)
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            #print("Memory Allocated (MB):", torch.cuda.memory_allocated(device_id) / 1024**2)
            #print("Memory Cached (MB):", torch.cuda.memory_reserved(device_id) / 1024**2)
            #input("Press Enter to continue...")

    timeDf.to_csv(saving_path/ 'results.csv', index=False)
    pd.set_option('display.max_rows', None)

    # Calculate mean and std for each column (excluding 'index')
    mean_row = timeDf.drop(columns=['index']).mean()
    std_row = timeDf.drop(columns=['index']).std()

    # Prepare rows for appending
    mean_row['index'] = 'mean'
    std_row['index'] = 'std'

    # Append to DataFrame
    timeDf = pd.concat([timeDf, pd.DataFrame([mean_row, std_row], columns=timeDf.columns)], ignore_index=True)

    # Save again with mean and std
    timeDf.to_csv(saving_path / 'results_with_stats.csv', index=False)
    print(timeDf.tail(2))

if __name__ == "__main__":
    main()