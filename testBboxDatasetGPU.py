import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
import numpy as np
import time
import cv2
import os
from sklearn.metrics import auc
from Dataset import ImageMaskDataset, CholecDataset, LeedsDataset, DatasetTest
import torch
from torch.utils.data import DataLoader
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


def refining(mask):
    """Applica post-processing morfologico alla maschera"""
    mask = (mask * 255).astype(np.uint8)
    while mask.ndim > 2:
        mask = mask[0]
    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
    mask_blurred = cv2.GaussianBlur(mask_clean, (5, 5), 0)
    mask_blurred = mask_blurred / 255
    return mask_blurred


########################################################################################################
# CONFIGURAZIONE
########################################################################################################

img_dir = ["/home/mdezen/distillation/testSetBbox/test"]
coco = "/home/mdezen/distillation/testSetBbox/test/_annotations.coco.json"

validation_transform = A.Compose([
    A.Resize(1024, 1024),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

datasetTest = DatasetTest(image_dirs=img_dir, coco=coco, transform=validation_transform)
dataloaderTest = DataLoader(datasetTest, batch_size=2, shuffle=False, collate_fn=collate_fn)

autosam_checkpoint = "checkpointsLight/autoSamFineUnetk57VL.pth"

model_type = "autoSamUnet"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = sam_model_registry[model_type](checkpoint=None)
model.load_state_dict(torch.load(autosam_checkpoint, map_location=device))
model.to(device=device)
model.eval()

iou_threshold = 0.5
save_dir = "results_bBoxLarge"
os.makedirs(save_dir, exist_ok=True)

# Accumulatori globali
all_preds_global = []  # [x1, y1, x2, y2, score, img_id]
all_gts_global = []  # [x1, y1, x2, y2, img_id]

# Metriche per immagine
per_image_metrics = []

img_id = 0

########################################################################################################
# LOOP DI INFERENZA
########################################################################################################

print("\n" + "=" * 70)
print("🚀 INIZIO INFERENZA")
print("=" * 70)

for images, labels in dataloaderTest:
    images = torch.stack(images).to(device)

    for image, label in zip(images, labels):
        image = image.unsqueeze(0).float().to(device)
        print(label)

        # -------------------- INFERENZA --------------------
        start_time = time.time()
        image_embedding = model.image_encoder(image)
        low_res = model.mask_decoder(image_embedding)
        low_res = model.postprocess_masks(low_res, (1024, 1024), (1024, 1024))
        end_time = time.time()
        latency = (end_time - start_time) * 1000

        mask = (low_res > 0).detach().cpu().numpy()
        mask = refining(mask).astype(np.uint8) * 255

        # -------------------- ESTRAI BBOX --------------------
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
                     for (x, y, w, h) in label]

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
        img_vis = image.squeeze(0).cpu().permute(1, 2, 0).numpy()
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

        axs[1].imshow(mask, cmap='gray')
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