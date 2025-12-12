
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import LOGGER
import numpy as np
import cv2
from tqdm import tqdm
import os
from PIL import Image


class YOLOSegmentationFineTuner:
    """
    Fine-tuning di YOLOv8-seg usando dataset con maschere di segmentazione.
    Converte automaticamente dal tuo InstrumentDataset al formato YOLO.
    """

    def __init__(
            self,
            model_path='yolov8n-seg.pt',  # Modello per segmentazione
            num_classes=8,
            class_names=None
    ):
        """
        Args:
            model_path: Path al modello YOLOv8-seg pretrained
            num_classes: Numero di classi (escluso background)
            class_names: Dict {class_id: 'class_name'} o lista ['class1', 'class2', ...]
        """
        print(f" Caricamento YOLOv8-seg: {model_path}")
        self.model = YOLO(model_path)
        self.num_classes = num_classes

        # Nomi delle classi
        if class_names is None:
            self.class_names = {i: f'class_{i}' for i in range(num_classes)}
        elif isinstance(class_names, dict):
            self.class_names = class_names
        else:
            self.class_names = {i: name for i, name in enumerate(class_names)}

        print(f"✓ Numero classi: {num_classes}")
        print(f"✓ Classi: {list(self.class_names.values())}")

    def prepare_yolo_dataset(
            self,
            train_loader,
            val_loader=None,
            output_dir='./yolo_dataset',
            min_mask_area=100
    ):
        """
        Converte il dataset con maschere in formato YOLO.

        Args:
            train_loader: DataLoader di training (InstrumentDataset)
            val_loader: DataLoader di validation (opzionale)
            output_dir: Directory per salvare il dataset YOLO
            min_mask_area: Area minima per una maschera (pixel)

        Returns:
            Path al file di configurazione YAML
        """
        print("\n" + "=" * 70)
        print(" CONVERSIONE DATASET IN FORMATO YOLO")
        print("=" * 70)

        os.makedirs(output_dir, exist_ok=True)

        # Converte train
        train_dir = self._convert_split(
            train_loader,
            os.path.join(output_dir, 'train'),
            'train',
            min_mask_area
        )

        # Converte val se presente
        val_dir = None
        if val_loader is not None:
            val_dir = self._convert_split(
                val_loader,
                os.path.join(output_dir, 'val'),
                'val',
                min_mask_area
            )

        # Crea file YAML di configurazione
        yaml_path = self._create_yaml_config(output_dir, train_dir, val_dir)

        print(f"\nDataset convertito!")
        print(f"Salvato in: {output_dir}")
        print(f" Config: {yaml_path}")

        return yaml_path

    def _convert_split(self, dataloader, output_dir, split_name, min_mask_area):
        """
        Converte un singolo split (train/val) in formato YOLO.
        """
        img_dir = os.path.join(output_dir, 'images')
        label_dir = os.path.join(output_dir, 'labels')

        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        print(f"\n Conversione {split_name}...")

        total_instances = 0
        skipped_instances = 0

        for batch_idx, (images, masks) in enumerate(tqdm(dataloader, desc=f"Converting {split_name}")):
            # images: [B, C, H, W] - tensor normalizzato
            # masks: [B, H, W] - tensor con class_id (0=background)

            batch_size = images.shape[0]

            for i in range(batch_size):
                img_tensor = images[i]  # [C, H, W]
                mask_tensor = masks[i]  # [H, W]

                # Genera nome file univoco
                img_name = f"{split_name}_{batch_idx:06d}_{i:02d}"

                # Salva immagine
                img_path = os.path.join(img_dir, f"{img_name}.jpg")
                self._save_image_tensor(img_tensor, img_path)

                # Converte maschera in formato YOLO (poligoni)
                label_path = os.path.join(label_dir, f"{img_name}.txt")
                #print(label_dir)
                n_inst, n_skip = self._mask_to_yolo_labels(
                    mask_tensor,
                    label_path,
                    min_mask_area
                )

                total_instances += n_inst
                skipped_instances += n_skip

        print(f"  ✓ {split_name}: {total_instances} istanze, {skipped_instances} skipped (troppo piccole)")

        return output_dir

    def _save_image_tensor(self, img_tensor, save_path):
        """
        Salva tensor immagine come file JPG.
        """
        # Denormalizza (assumo mean=0.5, std=0.5)
        img = img_tensor.cpu().numpy()
        img = img.transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
        img = (img * 0.5 + 0.5) * 255  # Denormalizza
        img = np.clip(img, 0, 255).astype(np.uint8)

        # Converti RGB a BGR per OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img_bgr)

    def _mask_to_yolo_labels(self, mask_tensor, label_path, min_area):
        """
        Converte maschera di segmentazione in formato YOLO.

        YOLO Segmentation format:
        <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
        Coordinate normalizzate [0, 1]
        """
        mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
        h, w = mask_np.shape

        labels = []
        total_instances = 0
        skipped_instances = 0

        # Trova tutte le classi presenti (escludi background=0)
        unique_classes = np.unique(mask_np)
        unique_classes = unique_classes[unique_classes > 0]
        print(unique_classes)

        for class_id in unique_classes:
            # Crea maschera binaria per questa classe
            binary_mask = (mask_np == class_id).astype(np.uint8)

            # Trova contorni
            contours, _ = cv2.findContours(
                binary_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                # Calcola area
                area = cv2.contourArea(contour)

                if area < min_area:
                    skipped_instances += 1
                    continue

                # Semplifica contorno (riduce numero di punti)
                epsilon = 0.001 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Converti contorno in formato YOLO
                # Flatten e normalizza
                points = approx.reshape(-1, 2)
                points_norm = points.astype(float)
                points_norm[:, 0] /= w  # Normalizza x
                points_norm[:, 1] /= h  # Normalizza y

                # Crea stringa: class_id x1 y1 x2 y2 ...
                # YOLO usa class_id - 1 (perché 0=background nel nostro caso)

                yolo_class_id = class_id -1 #sposta le calssi in 0-6 seguendo e indicazioni di yolo, il background non vine considerato come classe
                points_str = ' '.join([f"{x:.6f} {y:.6f}" for x, y in points_norm])
                label_line = f"{yolo_class_id} {points_str}"
                #print(label_line)

                labels.append(label_line)
                total_instances += 1

        # Salva labels
        with open(label_path, 'w') as f:
            f.write('\n'.join(labels))

        return total_instances, skipped_instances

    def _create_yaml_config(self, base_dir, train_dir, val_dir):
        """
        Crea file YAML di configurazione per YOLO.
        """
        config = {
            'path': os.path.abspath(base_dir),
            'train': 'train/images',
            'val': 'val/images' if val_dir else None,
            'nc': self.num_classes,
            'names': self.class_names
        }

        yaml_path = os.path.join(base_dir, 'config.yaml')

        import yaml
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        return yaml_path

    def train(
            self,
            yaml_config,
            epochs=100,
            imgsz=640,
            batch=16,
            lr0=0.01,
            patience=50,
            project='runs/segment',
            name='custom_segment',
            **kwargs
    ):
        """
        Esegue il fine-tuning.

        Args:
            yaml_config: Path al file YAML di configurazione
            epochs: Numero di epoche
            imgsz: Dimensione immagine
            batch: Batch size
            lr0: Learning rate iniziale
            patience: Early stopping patience
            project: Directory progetto
            name: Nome esperimento
            **kwargs: Altri parametri per model.train()
        """
        print("\n" + "=" * 70)
        print("🚀 INIZIO TRAINING YOLO SEGMENTATION")
        print("=" * 70)

        results = self.model.train(
            data=yaml_config,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=lr0,
            patience=patience,
            project=project,
            name=name,
            save=True,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            **kwargs
        )

        print("\n✅ Training completato!")
        print(f"📁 Risultati salvati in: {project}/{name}")

        return results

    def predict(self, image_path, conf=0.25, iou=0.7):
        """
        Esegue predizione su una singola immagine.

        Returns:
            masks: Array numpy con le maschere [H, W, N]
            boxes: Bounding boxes
            classes: Class IDs
        """
        results = self.model.predict(
            source=image_path,
            conf=conf,
            iou=iou,
            save=True
        )

        return results[0]
