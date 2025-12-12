

import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from repvit_sam import SamPredictor
from matplotlib import pyplot as plt
import numpy as np
import time
import cv2
import os

from Dataset import ImageMaskDataset,DatasetTest
from utility import dice_coefficient, sensitivity, specificity
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from modeling.build_sam import sam_model_registry


def collate_fn(batch):
    images = [item[0] for item in batch]
    bboxes = [item[1] for item in batch]
    return images, bboxes


def calculate_iou_single(boxA, boxB):
    """Calcola IoU tra due singole bounding box"""
    xA1, yA1, xA2, yA2 = boxA[:4]
    xB1, yB1, xB2, yB2 = boxB[:4]

    x_left = max(xA1, xB1)
    y_top = max(yA1, yB1)
    x_right = min(xA2, xB2)
    y_bottom = min(yA2, yB2)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    inter_area = (x_right - x_left) * (y_bottom - y_top)
    areaA = (xA2 - xA1) * (yA2 - yA1)
    areaB = (xB2 - xB1) * (yB2 - yB1)
    union_area = areaA + areaB - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def calculate_metrics_per_image(preds, gts, iou_threshold=0.5):
    """
    Calcola TP, FP, FN per una singola immagine usando Hungarian matching

    Args:
        preds: lista di [x1, y1, x2, y2, score]
        gts: lista di [x1, y1, x2, y2]
        iou_threshold: soglia IoU per considerare un match

    Returns:
        dict con tp, fp, fn, matched_pairs
    """
    if len(preds) == 0 and len(gts) == 0:
        return {'tp': 0, 'fp': 0, 'fn': 0, 'matched_pairs': []}

    if len(preds) == 0:
        return {'tp': 0, 'fp': 0, 'fn': len(gts), 'matched_pairs': []}

    if len(gts) == 0:
        return {'tp': 0, 'fp': len(preds), 'fn': 0, 'matched_pairs': []}

    # Matrice IoU tra tutte le predizioni e tutti i GT
    iou_matrix = np.zeros((len(preds), len(gts)))
    for i, pred in enumerate(preds):
        for j, gt in enumerate(gts):
            iou_matrix[i, j] = calculate_iou_single(pred, gt)

    # Greedy matching: assegna ogni GT alla predizione con IoU massimo
    matched_preds = set()
    matched_gts = set()
    matched_pairs = []

    # Ordina tutte le coppie per IoU decrescente
    matches = []
    for i in range(len(preds)):
        for j in range(len(gts)):
            if iou_matrix[i, j] >= iou_threshold:
                matches.append((i, j, iou_matrix[i, j]))

    matches.sort(key=lambda x: x[2], reverse=True)

    # Assegna match uno alla volta (greedy)
    for pred_idx, gt_idx, iou_val in matches:
        if pred_idx not in matched_preds and gt_idx not in matched_gts:
            matched_preds.add(pred_idx)
            matched_gts.add(gt_idx)
            matched_pairs.append((pred_idx, gt_idx, iou_val))

    tp = len(matched_pairs)
    fp = len(preds) - tp
    fn = len(gts) - tp

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'matched_pairs': matched_pairs
    }


def compute_pr_curve_and_ap(all_preds, all_gts, iou_threshold=0.5):
    """
    Calcola curva Precision-Recall e Average Precision (AP)

    Args:
        all_preds: lista di [x1, y1, x2, y2, score, img_id]
        all_gts: lista di [x1, y1, x2, y2, img_id]
        iou_threshold: soglia IoU

    Returns:
        recalls, precisions, ap
    """
    if len(all_preds) == 0:
        return np.array([0]), np.array([0]), 0.0

    # Ordina predizioni per confidence decrescente
    all_preds_sorted = sorted(all_preds, key=lambda x: x[4], reverse=True)

    # Raggruppa GT per immagine
    gt_by_image = {}
    for gt in all_gts:
        img_id = gt[4]
        if img_id not in gt_by_image:
            gt_by_image[img_id] = []
        gt_by_image[img_id].append(gt[:4])

    # Per ogni immagine, traccia quali GT sono già stati matchati
    matched_gt = {img_id: set() for img_id in gt_by_image.keys()}

    tp_list = []
    fp_list = []
    confidence_scores = []

    for pred in all_preds_sorted:
        px1, py1, px2, py2, score, img_id = pred
        confidence_scores.append(score)

        # Se l'immagine non ha GT, è un FP
        if img_id not in gt_by_image:
            tp_list.append(0)
            fp_list.append(1)
            continue

        # Trova il GT con IoU massimo
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(gt_by_image[img_id]):
            iou = calculate_iou_single([px1, py1, px2, py2], gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        # Se IoU >= threshold e GT non già matchato -> TP
        if best_iou >= iou_threshold and best_gt_idx not in matched_gt[img_id]:
            tp_list.append(1)
            fp_list.append(0)
            matched_gt[img_id].add(best_gt_idx)
        else:
            tp_list.append(0)
            fp_list.append(1)

    # Calcola cumulative TP e FP
    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum(fp_list)

    total_gt = len(all_gts)

    # Calcola precision e recall
    recalls = tp_cumsum / total_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

    # Calcola AP usando interpolazione a 11 punti o integrale
    # Metodo: interpolazione Pascal VOC
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11

    return recalls, precisions, ap




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

def predict_boxes(predictor,boxes):
    all_masks = []
    all_scores = []
    all_low_res = []
    input = torch.ones(1, boxes.shape[0]).to(boxes.device)  # shape: [1, N]

    # centroids: torch.Size([1, N, 2])
    # input_label: torch.Size([1, N])
    # boxes: torch.Size([num_boxes, 4]) — num_boxes > 1

    #devo creare le maschere dando in nput una box e un punto per volta perche non è possible dare piu boxe
    #e allo stesso tempo piu punti

    for i in range(boxes.shape[0]):
        box = boxes[i].unsqueeze(0)  # shape: [1, 4], batch size 1
        print(box.shape)

        #print(input.shape)
        masks, scores, low_res = predictor.predict_torch(
            point_coords=None,
            point_labels=input,
            boxes=box,
            multimask_output=False
        )

        all_masks.append(masks)  # masks: [1, C, H, W]

        all_scores.append(scores)  # scores: [1, C]
        all_low_res.append(low_res)  # low_res: [1, C, H, W]

    # Concatenazione lungo la dimensione delle maschere (C)
    final_masks = torch.cat(all_masks, dim=0)  # [1, total_C, H, W]

    final_scores = torch.cat(all_scores, dim=0)  # [1, total_C]
    final_low_res = torch.cat(all_low_res, dim=0)

    return final_masks, final_scores, final_low_res

def predict_points_boxes(predictor,boxes,centroids,input_label):
    all_masks = []
    all_scores = []
    all_low_res = []
    print(transformed_boxes.shape)
    print(input_label.shape)
    print(centroids.shape)

    # centroids: torch.Size([1, N, 2])
    # input_label: torch.Size([1, N])
    # boxes: torch.Size([num_boxes, 4]) — num_boxes > 1

    #devo creare le maschere dando in nput una box e un punto per volta perche non è possible dare piu boxe
    #e allo stesso tempo piu punti

    for i in range(boxes.shape[0]):
        box = boxes[i].unsqueeze(0)  # shape: [1, 4], batch size 1
        #print(box.shape)
        centroid = centroids[:,i,:].unsqueeze(0) # shape: [1,1,2]
        #print(centroid.shape)
        input = input_label[:,i].unsqueeze(0) # shape: [1, N]
        #print(input.shape)
        masks, scores, low_res = predictor.predict_torch(
            point_coords=centroid,
            point_labels=input,
            boxes=box,
            multimask_output=False
        )

        all_masks.append(masks)  # masks: [1, C, H, W]
        all_scores.append(scores)  # scores: [1, C]
        all_low_res.append(low_res)  # low_res: [1, C, H, W]

    # Concatenazione lungo la dimensione delle maschere (C)
    final_masks = torch.cat(all_masks, dim=0)  # [1, total_C, H, W]
    print(final_masks.shape)
    final_scores = torch.cat(all_scores, dim=0)  # [1, total_C]
    final_low_res = torch.cat(all_low_res, dim=0)

    return final_masks, final_scores, final_low_res
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
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 2. Chiudi buchi interni (closing)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    # 3. (opzionale) Gaussian blur per bordi morbidi
    mask_blurred = cv2.GaussianBlur(mask_clean, (5, 5), 0)
    mask_blurred = mask_blurred/255

    return mask_blurred




device = "cuda" if torch.cuda.is_available() else "cpu"

#student_checkpoint = "checkpoints/13_05/decoupledVitBDGfFE.pth"

#model = sam_model_registry["CMT"](checkpoint=student_checkpoint)
#model.to(device=device)
#model.load_state_dict(state_dict)
#print("Missing keys:", model.load_state_dict(state_dict, strict=False))
#CARICO UN MODELLO SAM
#sam_checkpoint = "C:/Users/User/OneDrive - Politecnico di Milano/Documenti/POLIMI/Tesi/distillation/checkpoints/sam_vit_b_01ec64.pth"
sam_checkpoint = "checkpoints/mobile_sam.pt"
model_type = "vit_t"



sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
img_dir = ["/home/mdezen/distillation/testSetBbox/test"]
coco = "/home/mdezen/distillation/testSetBbox/test/_annotations.coco.json"

validation_transform = A.Compose([
    A.Resize(1024, 1024),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])
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
datasetTest = ImageMaskDataset(image_dirs=image_dirs_test,mask_dirs=mask_dirs_test, transform=validation_transform)
dataloaderTest = DataLoader(datasetTest, batch_size=2, shuffle=True)

datasetTest = DatasetTest(image_dirs=img_dir, coco=coco, transform=validation_transform)
dataloaderTest = DataLoader(datasetTest, batch_size=2, shuffle=False, collate_fn=collate_fn)


#ASSEGNO L'IMAGE ENCODER DISTILLATO A SAM
#sam.image_encoder = model.image_encoder
sam.eval()
#model.eval()
predictor = SamPredictor(sam)
#print("State dict keys:", state_dict.keys())
"""
checkpoint = torch.load("C:/Users/User/OneDrive - Politecnico di Milano/Documenti/POLIMI/Tesi/distillation/checkpoints/student_checkpoint.pth", map_location="cpu")
 
image_encoder_state_dict = {
    k.replace("image_encoder.", ""): v
    for k, v in checkpoint.items()
    if k.startswith("image_encoder.")
}

model.image_encoder.load_state_dict(image_encoder_state_dict)
sam = sam_model_registry["vit_b"](checkpoint="checkpoints/sam_vit_b_01ec64.pth")
transformer_dim = model.mask_decoder.transformer_dim
transformer = model.mask_decoder.transformer


cloned_mask_decoder = type(sam.mask_decoder)(transformer_dim=transformer_dim, transformer=transformer)
cloned_mask_decoder.load_state_dict(sam.mask_decoder.state_dict())  # Copy the weights
model.mask_decoder = cloned_mask_decoder
cloned_prompt_encoder = copy.deepcopy(sam.prompt_encoder)
 # Copy the weights

# Assign the cloned prompt encoder to the model
model.prompt_encoder = cloned_prompt_encoder 
"""
#model.to(device=device)
#model.eval()



#predictor = SamPredictor(model)
timeDf = pd.DataFrame(columns=['time', 'index', 'iou','dice','sensitivity','specificity'])
iou_threshold = 0.5
save_dir = "results_bBoxMobile"
os.makedirs(save_dir, exist_ok=True)

# Accumulatori globali
all_preds_global = []  # [x1, y1, x2, y2, score, img_id]
all_gts_global = []  # [x1, y1, x2, y2, img_id]

# Metriche per immagine
per_image_metrics = []

img_id = 0

n = 0
for images, labels in dataloaderTest:  # i->batch index, images->batch of images, labels->batch of labels

        images = torch.stack(images).to(device)
        results_teach = []
        results_stud = []

        for image, label in zip(images, labels):
            labelraw = label



            label = torch.tensor(label)
            label = [[int(x), int(y), int(x + w), int(y + h)]
                         for (x, y, w, h) in label]
            label = torch.tensor(label)
            #print(label)
            image = image.detach().cpu()
            #print(image.shape)
            image = np.transpose(np.squeeze(image), (1, 2, 0))  # (C,H,W) → (H,W,C)
            image = image.detach().cpu().numpy()
            image = (image * 0.5 + 0.5) * 255
            image = image.astype(np.uint8)


            start_time = time.time()

            predictor.set_image(image)
            masks, _, low_res = predict_boxes(predictor, label)
            end_time = time.time()
            latency = ( end_time - start_time)*1000  # in ms
            maskunion = np.zeros_like(masks[0].cpu().numpy())
            for mask in masks:
                mask = mask.detach().cpu().numpy()
                #print(mask.shape)
                mask = refining(mask)


                #show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
                #values, counts = np.unique(mask.cpu().numpy(), return_counts=True)
                #print("unique", values)
                #print("counts", counts)


                maskunion = np.logical_or(maskunion, mask)
            # -------------------- ESTRAI BBOX --------------------
            print(maskunion.shape)
            maskunion = (maskunion * 255).astype(np.uint8)
            maskunion = maskunion.squeeze()
            contours, _ = cv2.findContours(maskunion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_contours = [c for c in contours if cv2.contourArea(c) >= 1500]

            # Predizioni con confidence
            bboxes_pred = []
            for contour in filtered_contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                bbox_area = w * h
                # Confidence basata sul rapporto area_contorno/area_bbox
                conf = float(area / bbox_area) if bbox_area > 0 else 0.5
                bboxes_pred.append([x, y, x + w, y + h, conf])

            # Ground truth
            bboxes_gt = [[int(x), int(y), int(x + w), int(y + h)]
                         for (x, y, w, h) in labelraw]

            # -------------------- CALCOLA METRICHE PER IMMAGINE --------------------
            metrics = calculate_metrics_per_image(bboxes_pred, bboxes_gt, iou_threshold)
            tp = metrics['tp']
            fp = metrics['fp']
            fn = metrics['fn']

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            # Calcola IoU medio sui match
            mean_iou = 0.0
            if metrics['matched_pairs']:
                ious = [calculate_iou_single(bboxes_pred[i], bboxes_gt[j])
                        for i, j, _ in metrics['matched_pairs']]
                mean_iou = np.mean(ious)

            # Salva metriche per immagine
            per_image_metrics.append({
                'img_id': img_id,
                'latency_ms': latency,
                'num_preds': len(bboxes_pred),
                'num_gts': len(bboxes_gt),
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'mean_iou': mean_iou
            })

            # -------------------- AGGIUNGI A LISTE GLOBALI --------------------
            for bbox in bboxes_pred:
                all_preds_global.append(bbox + [img_id])  # [x1,y1,x2,y2,score,img_id]

            for bbox in bboxes_gt:
                all_gts_global.append(bbox + [img_id])  # [x1,y1,x2,y2,img_id]

            # -------------------- VISUALIZZAZIONE --------------------
            img_vis = np.transpose(image, (1, 2, 0)) if image.shape[0] == 3 else image
            img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min() + 1e-8)
            img_vis = (img_vis * 255).astype(np.uint8)
            img_vis = np.ascontiguousarray(img_vis)

            if img_vis.ndim == 2:
                img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2RGB)

            # Draw GT (verde)
            for box in bboxes_gt:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw predictions (blu)
            for bbox in bboxes_pred:
                x1, y1, x2, y2 = map(int, bbox[:4])
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Plot
            fig, axs = plt.subplots(1, 2, figsize=(12, 4))
            axs[0].imshow(img_vis)
            axs[0].set_title(f"Image {img_id} | P={precision:.2f} R={recall:.2f}")
            axs[0].axis("off")

            axs[1].imshow(maskunion, cmap='gray')
            axs[1].set_title("Segmentation Mask")
            axs[1].axis("off")

            save_path = os.path.join(save_dir, f"result_{img_id}.png")
            plt.savefig(save_path, bbox_inches="tight", dpi=100)
            plt.close(fig)

            if img_id % 10 == 0:
                print(f"Processed {img_id} images...")

            img_id += 1

########################################################################################################
# CALCOLA METRICHE GLOBALI
########################################################################################################

print("\n" + "=" * 70)
print("📊 CALCOLO METRICHE GLOBALI")
print("=" * 70)

# Converti a DataFrame
df_metrics = pd.DataFrame(per_image_metrics)

# Calcola precision/recall globale
total_tp = df_metrics['tp'].sum()
total_fp = df_metrics['fp'].sum()
total_fn = df_metrics['fn'].sum()

global_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
global_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
global_f1 = 2 * global_precision * global_recall / (global_precision + global_recall) \
    if (global_precision + global_recall) > 0 else 0.0

# Calcola curva PR e AP
recalls, precisions, ap = compute_pr_curve_and_ap(
    all_preds_global, all_gts_global, iou_threshold
)

# Plot curva PR
plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, 'b-', linewidth=2, label=f'AP@{iou_threshold} = {ap:.4f}')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title(f'Precision-Recall Curve (IoU threshold = {iou_threshold})', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.savefig(os.path.join(save_dir, 'pr_curve.png'), dpi=150, bbox_inches='tight')
plt.close()

########################################################################################################
# STAMPA RISULTATI
########################################################################################################

print(f"\n{'=' * 70}")
print("📈 RISULTATI FINALI")
print(f"{'=' * 70}\n")

print(f"📊 Dataset:")
print(f"   Totale immagini: {len(df_metrics)}")
print(f"   Totale GT boxes: {len(all_gts_global)}")
print(f"   Totale predizioni: {len(all_preds_global)}")

print(f"\n🎯 Metriche Globali (somma TP/FP/FN su tutto il dataset):")
print(f"   TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
print(f"   Precision: {global_precision:.4f}")
print(f"   Recall: {global_recall:.4f}")
print(f"   F1-Score: {global_f1:.4f}")
print(f"   AP@{iou_threshold}: {ap:.4f}")

print(f"\n📊 Metriche Medie (media su singole immagini):")
print(f"   Precision: {df_metrics['precision'].mean():.4f} ± {df_metrics['precision'].std():.4f}")
print(f"   Recall: {df_metrics['recall'].mean():.4f} ± {df_metrics['recall'].std():.4f}")
print(f"   F1-Score: {df_metrics['f1'].mean():.4f} ± {df_metrics['f1'].std():.4f}")
print(f"   IoU medio: {df_metrics['mean_iou'].mean():.4f} ± {df_metrics['mean_iou'].std():.4f}")

print(f"\n⏱️  Performance:")
print(f"   Latenza media: {df_metrics['latency_ms'].mean():.2f} ± {df_metrics['latency_ms'].std():.2f} ms")

print(f"\n{'=' * 70}\n")

########################################################################################################
# SALVA RISULTATI
########################################################################################################

# Salva metriche dettagliate
df_metrics.to_csv(os.path.join(save_dir, 'metrics_per_image.csv'), index=False)

# Salva summary
summary = {
    'total_images': len(df_metrics),
    'total_gt_boxes': len(all_gts_global),
    'total_predictions': len(all_preds_global),
    'global_precision': global_precision,
    'global_recall': global_recall,
    'global_f1': global_f1,
    'ap_at_iou_threshold': ap,
    'iou_threshold': iou_threshold,
    'mean_precision': df_metrics['precision'].mean(),
    'mean_recall': df_metrics['recall'].mean(),
    'mean_f1': df_metrics['f1'].mean(),
    'mean_iou': df_metrics['mean_iou'].mean(),
    'mean_latency_ms': df_metrics['latency_ms'].mean()
}

with open(os.path.join(save_dir, 'summary.txt'), 'w') as f:
    for key, value in summary.items():
        f.write(f"{key}: {value}\n")

print(f"✅ Risultati salvati in '{save_dir}/'")
print(f"   - metrics_per_image.csv: metriche dettagliate per ogni immagine")
print(f"   - pr_curve.png: curva Precision-Recall")
print(f"   - summary.txt: riepilogo metriche globali")
