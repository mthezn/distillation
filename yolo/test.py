import torch
import numpy as np
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import pandas as pd
import time
import cv2
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
    ConfusionMatrixDisplay
)
import os
from collections import defaultdict
from PIL import Image
from reportMulticlass import YOLOStyleReporter

class YOLOInferenceWrapper:
    """
    Wrapper per inferenza YOLO con tensori PyTorch.
    """

    def __init__(self, model_path, num_classes, device='cuda'):
        self.model = YOLO(model_path)
        self.num_classes = num_classes
        self.device = device

    def predict_from_tensor(self, image_tensor, conf=0.25, iou=0.7, imgsz=1024):
        """
        Predizione da tensor PyTorch normalizzato.

        Args:
            image_tensor: Tensor [C, H, W] normalizzato con mean=0.5, std=0.5
            conf: Confidence threshold
            iou: IoU threshold for NMS
            imgsz: Image size

        Returns:
            combined_mask: Maschera binaria combinata [H, W]
            masks: Lista di maschere individuali
            boxes: Bounding boxes
            classes: Class IDs
        """
        # 1. Converti tensor normalizzato in numpy array [0, 255]
        img_np = self._tensor_to_numpy(image_tensor)

        # 2. Predizione YOLO
        results = self.model.predict(
            source=img_np,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            save=False,
            verbose=False
        )

        # 3. Estrai maschere
        if results and len(results) > 0 and results[0].masks is not None:
            masks = results[0].masks.data  # [N, H, W] tensor
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            # Combina tutte le maschere in una binaria
            binary_masks = (masks > 0.5).int()
            combined_mask = torch.any(binary_masks.bool(), dim=0).int()
            combined_mask = combined_mask.cpu().numpy().astype(np.uint8)

            # Ridimensiona a dimensione originale se necessario
            h, w = image_tensor.shape[1:]
            if combined_mask.shape != (h, w):
                combined_mask = cv2.resize(
                    combined_mask,
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                )

            return combined_mask, masks.cpu().numpy(), boxes, classes
        else:
            # Nessuna detection
            h, w = image_tensor.shape[1:]
            return np.zeros((h, w), dtype=np.uint8), None, None, None

    def _tensor_to_numpy(self, tensor):
        """
        Converte tensor normalizzato [C, H, W] in numpy array [H, W, C] in range [0, 255].
        """
        # Denormalizza (mean=0.5, std=0.5)
        img = tensor.cpu().numpy()
        img = img.transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
        img = (img * 0.5 + 0.5)  # Denormalizza da [-1, 1] a [0, 1]
        img = (img * 255).clip(0, 255).astype(np.uint8)

        return img


import torch
import numpy as np
import cv2
import os
import pandas as pd
from collections import defaultdict
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import time
from matplotlib import pyplot as plt


def extract_gt_classes_from_mask(mask_label, pred_boxes):
    """
    Estrae le classi GT dalle maschere per ogni bounding box predetta.

    Args:
        mask_label: Maschera numpy [H, W] con class_id
        pred_boxes: Bounding boxes predette [[x1, y1, x2, y2], ...]

    Returns:
        gt_classes: Lista di class_id GT per ogni box
        gt_classes_all: Set di tutte le classi presenti nell'immagine
    """
    gt_classes = []
    gt_classes_all = set()

    # Trova tutte le classi presenti nell'immagine (escluso background=-1 o 0)
    unique_classes = np.unique(mask_label)
    unique_classes = unique_classes[(unique_classes > 0) & (unique_classes != -1)]
    gt_classes_all = set(unique_classes.tolist())

    # Per ogni box predetta, trova la classe GT dominante
    if pred_boxes is not None and len(pred_boxes) > 0:
        for box in pred_boxes:
            x1, y1, x2, y2 = map(int, box[:4])

            # Clip alle dimensioni immagine
            h, w = mask_label.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Estrai regione
            roi = mask_label[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            # Trova classe dominante (escludi background)
            unique, counts = np.unique(roi, return_counts=True)

            # Filtra background
            valid_mask = (unique > 0) & (unique != -1)
            unique = unique[valid_mask]
            counts = counts[valid_mask]

            if len(unique) > 0:
                # Prendi classe con più pixel
                dominant_class = unique[np.argmax(counts)]
                gt_classes.append(int(dominant_class))
            else:
                # Nessuna classe valida nella box (probabilmente false positive)
                gt_classes.append(-1)

    return gt_classes, gt_classes_all


def match_predictions_to_gt(pred_boxes, pred_classes, mask_label, iou_threshold=0.3):
    """
    Matching più sofisticato tra predizioni e GT usando IoU.

    Args:
        pred_boxes: Box predette [[x1, y1, x2, y2], ...]
        pred_classes: Classi predette [class_id, ...]
        mask_label: Maschera GT [H, W]
        iou_threshold: Soglia IoU minima per match

    Returns:
        matched_pairs: Lista di (pred_class, gt_class) per box matchate
        all_gt_classes: Tutte le classi GT presenti
    """
    if pred_boxes is None or len(pred_boxes) == 0:
        # Nessuna predizione, ma ci potrebbero essere GT
        unique_gt = np.unique(mask_label)
        unique_gt = unique_gt[(unique_gt > 0) & (unique_gt != -1)]
        return [], set(unique_gt.tolist())

    # Estrai tutte le istanze GT dalla maschera
    gt_instances = []
    unique_classes = np.unique(mask_label)
    unique_classes = unique_classes[(unique_classes > 0) & (unique_classes != -1)]

    for class_id in unique_classes:
        # Trova componenti connesse per questa classe
        class_mask = (mask_label == class_id).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(class_mask)

        for inst_id in range(1, num_labels):  # Skip background (0)
            inst_mask = (labels == inst_id)

            # Trova bounding box di questa istanza GT
            ys, xs = np.where(inst_mask)
            if len(xs) == 0:
                continue

            gt_box = [xs.min(), ys.min(), xs.max(), ys.max()]
            gt_instances.append({
                'box': gt_box,
                'class': class_id,
                'mask': inst_mask
            })

    # Match predizioni con GT usando IoU
    matched_pairs = []
    matched_gt = set()

    for pred_box, pred_class in zip(pred_boxes, pred_classes):
        best_iou = 0
        best_gt_class = None
        best_gt_idx = None

        # Trova GT con maggior IoU
        for idx, gt_inst in enumerate(gt_instances):
            if idx in matched_gt:
                continue  # Già matchata

            iou = calculate_box_iou(pred_box, gt_inst['box'])

            if iou > best_iou:
                best_iou = iou
                best_gt_class = gt_inst['class']
                best_gt_idx = idx

        # Se trovato match sopra soglia
        if best_iou >= iou_threshold and best_gt_class is not None:
            matched_pairs.append((int(pred_class), int(best_gt_class)))
            matched_gt.add(best_gt_idx)
        else:
            # False positive (nessun match GT)
            matched_pairs.append((int(pred_class), -1))

    # Aggiungi GT non matchate come False Negatives
    all_gt_classes = set(unique_classes.tolist())

    return matched_pairs, all_gt_classes


def calculate_box_iou(box1, box2):
    """
    Calcola IoU tra due box [x1, y1, x2, y2].
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def calculate_metrics(pred_mask, gt_mask, positive_classes=None):
    """
    Calcola metriche di segmentazione binaria.
    """
    # Converti GT in binario se necessario
    if positive_classes:
        gt_binary = np.zeros_like(gt_mask, dtype=np.uint8)
        for c in positive_classes:
            gt_binary |= (gt_mask == c)
    else:
        gt_binary = (gt_mask > 0).astype(np.uint8)

    pred_binary = (pred_mask > 0).astype(np.uint8)

    # IoU
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    iou = intersection / (union + 1e-8)

    # Dice
    dice = 2 * intersection / (pred_binary.sum() + gt_binary.sum() + 1e-8)

    # Sensitivity
    tp = intersection
    fn = (gt_binary & ~pred_binary).sum()
    sensitivity = tp / (tp + fn + 1e-8)

    # Specificity
    tn = (~gt_binary & ~pred_binary).sum()
    fp = (pred_binary & ~gt_binary).sum()
    specificity = tn / (tn + fp + 1e-8)

    return {
        'iou': iou,
        'dice': dice,
        'sensitivity': sensitivity,
        'specificity': specificity
    }


def visualize_results(image_tensor, pred_mask, gt_mask, save_path, class_names=None):
    """
    Visualizza risultati side-by-side.
    """
    # Denormalizza immagine
    img_vis = image_tensor.cpu().permute(1, 2, 0).numpy()
    img_vis = (img_vis * 0.5 + 0.5).clip(0, 1)

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Immagine
    axs[0].imshow(img_vis)
    axs[0].set_title("Original Image", fontsize=14)
    axs[0].axis("off")

    # Ground Truth
    axs[1].imshow(img_vis)
    gt_vis = np.ma.masked_where(gt_mask <= 0, gt_mask)
    axs[1].imshow(gt_vis, cmap='jet', alpha=0.6, vmin=1, vmax=7)
    axs[1].set_title("Ground Truth", fontsize=14)
    axs[1].axis("off")

    # Predizione
    #axs[2].imshow(img_vis)
    pred_vis = np.ma.masked_where(pred_mask == 1, pred_mask)
    axs[2].imshow(pred_vis, cmap='gray')
    axs[2].set_title("Prediction", fontsize=14)
    axs[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_yolo_inference_with_classification():
    """
    Script completo di inferenza YOLO con estrazione corretta GT classes.
    """

    from reportMulticlass import YOLOStyleReporter
    from Dataset import InstrumentDatasetTest

    print("=" * 70)
    print("🚀 YOLO SEGMENTATION + CLASSIFICATION INFERENCE")
    print("=" * 70)

    # Configurazione
    class_names = {
        1: 'Large_Needle_Driver',
        2: 'Forceps',
        3: 'Grasping_Retractor',
        4: 'Maryland_Bipolar_Forceps',
        5: 'Monopolar_Curved_Scissors',
        6: 'Vessel_Sealer'
    }
    """  
    configurazione in datasettest
    self.class_colors = {
            0: np.array((0, 0, 0)),  # background = nero
            1: np.array((0, 0, 255)),  # Large_Needle_Driver= blu
            2: np.array((0, 255, 255)),  # Prograsp_Forceps= ciano

            4: np.array((255, 255, 0)),  # Maryland_Bipolar_Forceps = giallo (nuovi)
            5: np.array((0, 255, 0)),  # monopolar curved scissors = verde
            6: np.array((255, 0, 0))  # vessel sealer = rosso

    }"""

    positive_classes = [1, 2, 3, 4, 5, 6]

    # Dataset
    validation_transform = A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    image_dirs_test = ["/home/mdezen/multiclass/MICCAImod/instrument_1_4_testing/instrument_dataset_1/left_frames",
                       "/home/mdezen/multiclass/MICCAImod/instrument_1_4_testing/instrument_dataset_2/left_frames2",
                       "/home/mdezen/multiclass/MICCAImod/instrument_1_4_testing/instrument_dataset_3/left_frames",
                       "/home/mdezen/multiclass/MICCAImod/instrument_1_4_testing/instrument_dataset_4/left_frames",
                       "/home/mdezen/multiclass/MICCAImod/instrument_5_8_testing/instrument_dataset_6/left_frames",
                       "/home/mdezen/multiclass/MICCAImod/instrument_5_8_testing/instrument_dataset_7/left_frames",
                       "/home/mdezen/multiclass/MICCAImod/instrument_5_8_testing/instrument_dataset_8/left_frames"
                       ]
    mask_dirs_test = [
            "/home/mdezen/multiclass/MICCAImod/instrument_2017_test/instrument_2017_test/instrument_dataset_1/gt/TypeSegmentationRescaled",
        "/home/mdezen/multiclass/MICCAImod/instrument_2017_test/instrument_2017_test/instrument_dataset_2/gt/TypeSegmentationRescaled",
        "/home/mdezen/multiclass/MICCAImod/instrument_2017_test/instrument_2017_test/instrument_dataset_3/gt/TypeSegmentationRescaled",
        "/home/mdezen/multiclass/MICCAImod/instrument_2017_test/instrument_2017_test/instrument_dataset_4/gt/TypeSegmentationRescaled",
        "/home/mdezen/multiclass/MICCAImod/instrument_2017_test/instrument_2017_test/instrument_dataset_6/gt/TypeSegmentationRescaled",
        "/home/mdezen/multiclass/MICCAImod/instrument_2017_test/instrument_2017_test/instrument_dataset_7/gt/TypeSegmentationRescaled",
        "/home/mdezen/multiclass/MICCAImod/instrument_2017_test/instrument_2017_test/instrument_dataset_8/gt/TypeSegmentationRescaled"
                    ]

    datasetTest = InstrumentDatasetTest(
        image_dirs=image_dirs_test,
        gt_dirs=mask_dirs_test,
        transform=validation_transform
    )

    print(f"✓ Dataset size: {len(datasetTest)}")

    dataloaderTest = DataLoader(datasetTest, batch_size=1, shuffle=True)

    # Carica modello
    print("\n📦 Caricamento modello...")
    model = YOLOInferenceWrapper(
        model_path='runs/segment/instrument_seg/weights/best.pt',
        num_classes=6
    )
    print("✓ Modello caricato")

    # Setup risultati
    save_dir = "results_yolo_classification"
    os.makedirs(save_dir, exist_ok=True)

    # Reporter per classificazione
    reporter = YOLOStyleReporter(
        class_names=class_names,
        save_dir=os.path.join(save_dir, "classification_report")
    )

    results_df = pd.DataFrame(columns=[
        'index', 'time_ms', 'iou', 'dice', 'sensitivity', 'specificity',
        'num_pred', 'num_gt', 'pred_classes', 'gt_classes', 'matched_pairs'
    ])

    # Inferenza
    print(f"\n🔍 Inizio inferenza su {len(datasetTest)} immagini...")
    print("-" * 70)
    n=0
    for idx, (images, labels) in enumerate(dataloaderTest):
        image = images[0]  # [C, H, W]
        mask_label = labels[0].cpu().numpy()  # [H, W]

        # Predizione
        start_time = time.time()
        pred_mask, masks, pred_boxes, pred_classes = model.predict_from_tensor(
            image,
            conf=0.25,
            iou=0.7,
            imgsz=1024
        )
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # ms

        # Estrai GT classes usando matching IoU
        if pred_boxes is not None and len(pred_boxes) > 0:
            # Converti pred_classes da YOLO (0-6) a GLOBAL_CLASS_MAPPING (1-7)
            pred_classes_adjusted = (pred_classes + 1).astype(int)

            # Match predizioni con GT
            matched_pairs, all_gt_classes = match_predictions_to_gt(
                pred_boxes,
                pred_classes_adjusted,
                mask_label,
                iou_threshold=0.3
            )

            # Separa in pred e gt per il reporter
            pred_list = [p for p, g in matched_pairs]
            gt_list = [g for p, g in matched_pairs if g != -1]  # Escludi FP

        else:
            # Nessuna predizione
            pred_classes_adjusted = np.array([])
            pred_list = []

            # Ma potrebbero esserci GT (False Negatives)
            unique_gt = np.unique(mask_label)
            unique_gt = unique_gt[(unique_gt > 0) & (unique_gt != -1)]
            all_gt_classes = set(unique_gt.tolist())
            gt_list = list(all_gt_classes)
            matched_pairs = []

        # Calcola metriche segmentazione
        mask_binary = mask_label > 0
        metrics = calculate_metrics(pred_mask, mask_binary, positive_classes)

        # Aggiorna reporter (solo per box matchate)
        if len(pred_list) > 0 and len(gt_list) > 0:
            reporter.update(
                predictions=pred_list,
                ground_truths=gt_list,
                processing_time=latency
            )

        # Log
        print(f"[{idx + 1}/{len(datasetTest)}] "
              f"Time: {latency:.1f}ms | "
              f"IoU: {metrics['iou']:.3f} | "
              f"Pred: {pred_list} | "
              f"GT: {gt_list}")

        # Salva risultati
        results_df.loc[len(results_df)] = [
            idx,
            latency,
            metrics['iou'],
            metrics['dice'],
            metrics['sensitivity'],
            metrics['specificity'],
            len(pred_list),
            len(gt_list),
            str(pred_list),
            str(gt_list),
            str(matched_pairs)
        ]

        # Visualizza
        #if idx % 10 == 0 or metrics['iou'] < 0.5:
        save_path = os.path.join(save_dir, f"result_{n:04d}.png")
        n = n + 1
        visualize_results(image, pred_mask, mask_label, save_path, class_names)

    # Salva CSV
    csv_path = os.path.join(save_dir, 'results_yolo.csv')
    results_df.to_csv(csv_path, index=False)

    # Genera report classificazione
    print("\n📊 Generazione report classificazione...")
    reporter.generate_all_reports()

    # Statistiche finali
    print("\n" + "=" * 70)
    print("📊 STATISTICHE FINALI")
    print("=" * 70)
    print(f"{'Metrica':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 70)

    for col in ['time_ms', 'iou', 'dice', 'sensitivity', 'specificity']:
        mean = results_df[col].mean()
        std = results_df[col].std()
        min_val = results_df[col].min()
        max_val = results_df[col].max()
        print(f"{col:<20} {mean:>10.3f} {std:>10.3f} {min_val:>10.3f} {max_val:>10.3f}")

    print("=" * 70)
    print(f"✅ Risultati salvati in: {save_dir}")
    print(f"📄 CSV: {csv_path}")
    print(f"📊 Report classificazione: {save_dir}/classification_report/")
    print("=" * 70)


if __name__ == "__main__":
    run_yolo_inference_with_classification()