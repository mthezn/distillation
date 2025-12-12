import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
import json
from albumentations.pytorch import ToTensorV2
import torch
from torchvision.transforms import ToTensor
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import re



class ImageMaskDataset(Dataset):
    """
    Class: ImageMaskDataset

    Purpose:
        Custom PyTorch Dataset for loading images and their corresponding segmentation masks.
        Designed specifically to handle datasets with multiple annotations per image and
        compatible with the structure used in the MICCAI dataset (e.g., MICCAI 2023 challenges).
        Supports flexible image/mask pairing and data augmentation.

    Constructor Arguments:
        image_dirs (List[str]):
            List of paths to directories containing input RGB images (.png files).

        mask_dirs (List[str]):
            List of paths to directories containing corresponding mask images (.png files).
            Multiple masks can correspond to the same image and will be summed together.

        transform (albumentations.Compose, optional):
            A composed albumentations transformation to apply jointly on the image and mask.
            If None, a default normalization and tensor conversion is applied.

        increase (bool):
            If True, artificially increases the dataset size by repeating the entries 3 times.

    Dataset Organization:
        - Image and mask keys are matched using a pattern that includes the dataset number
          (parsed from folder names) and filename.
        - Only samples that exist in both the image and mask directories are retained.
        - Masks corresponding to the same image are stacked and summed pixel-wise.

    Returns (per sample):
        image (torch.Tensor):
            A normalized RGB image tensor of shape [3, H, W], dtype=torch.float32,
            with values typically in [-1, 1] if normalized using mean=0.5, std=0.5.

        combined_mask (torch.Tensor):
            A 2D segmentation mask of shape [H, W], dtype=torch.float32.
            Mask values are clipped to [0, 255] and optionally scaled depending on the transform.
            Containing all summed masks for the corresponding image.

    Use Case:
        Directory organization should follow the structure:
        image_dirs_val = ["MICCAI/instrument_1_4_testing/instrument_dataset_4/left_frames"]
        mask_dirs_val = ["MICCAI/instrument_2017_test/instrument_2017_test/instrument_dataset_4/gt/BinarySegmentation"]
    """


    def __init__(self, image_dirs, mask_dirs=None, transform=None, increase=False):
            self.image_dirs = image_dirs
            self.mask_dirs = mask_dirs if mask_dirs is not None else []
            self.transform = transform
            self.increase = increase

            self.image_paths = {}
            self.mask_paths = {}

            for img_dir in image_dirs:
                dataset_number = img_dir.split('/')[-2]
                for filename in os.listdir(img_dir):
                    if filename.endswith('.png'):
                        key = f"{dataset_number}_{filename}"
                        self.image_paths[key] = os.path.join(img_dir, filename)

            if self.mask_dirs:
                for mask_dir in self.mask_dirs:
                    dataset_number = mask_dir.split('/')[-3]
                    for filename in os.listdir(mask_dir):
                        if filename.endswith('.png'):
                            key = f"{dataset_number}_{filename}"
                            self.mask_paths.setdefault(key, []).append(os.path.join(mask_dir, filename))

                # Solo immagini con maschere
                self.image_filenames = sorted(set(self.image_paths.keys()) & set(self.mask_paths.keys()))

            else:
                # Nessuna maschera disponibile, restituisci tutte le immagini
                self.image_filenames = sorted(self.image_paths.keys())

            if self.increase:
                self.image_filenames = self.image_filenames * 3

    def __len__(self):
            return len(self.image_filenames)

    def __getitem__(self, idx):
            img_name = self.image_filenames[idx]
            img_path = self.image_paths[img_name]
            image = np.array(Image.open(img_path).convert("RGB"))

            # Se non ci sono maschere → maschera vuota
            if img_name in self.mask_paths:
                masks = [np.array(Image.open(p).convert("L")) for p in self.mask_paths[img_name]]
                masks = np.array(masks, dtype=np.uint8)
                combined_mask = np.sum(masks, axis=0).clip(0, 255).astype(np.uint8)
            else:
                combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

            if self.transform:
                augmented = self.transform(image=image, mask=combined_mask)
                image = augmented["image"]
                combined_mask = augmented["mask"]
            else:
                transform_basic = A.Compose([
                    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                    ToTensorV2()
                ])
                augmented = transform_basic(image=image, mask=combined_mask)
                image = augmented["image"]
                combined_mask = augmented["mask"]

            return image, combined_mask

class LeedsDataset(Dataset):
    def __init__(self, image_dirs, transform=None, increase=False):
            self.image_dirs = image_dirs
            self.transform = transform
            self.increase = increase

            self.image_paths = []

            for img_dir in image_dirs:
                for filename in os.listdir(img_dir):
                    if filename.endswith('.png'):
                        self.image_paths.append(os.path.join(img_dir, filename))

            if self.increase:
                self.image_paths = self.image_paths * 3

    def __len__(self):
            return len(self.image_paths)

    def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            image = np.array(Image.open(img_path).convert("RGB"))

            if self.transform:
                augmented = self.transform(image=image)
                image = augmented["image"]
            else:
                transform_basic = A.Compose([
                    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                    ToTensorV2()
                ])
                augmented = transform_basic(image=image)
                image = augmented["image"]
            mask = torch.zeros((image.shape[1], image.shape[2]), dtype=torch.float32)

            return image,mask


class Kvasir(Dataset):

    def __init__(self, image_dirs, mask_dirs=None, transform=None, increase=False):
            self.image_dirs = image_dirs
            self.mask_dirs = mask_dirs if mask_dirs is not None else []
            self.transform = transform
            self.increase = increase

            self.image_paths = {}
            self.mask_paths = {}

            for img_dir in image_dirs:
                for filename in os.listdir(img_dir):
                    if filename.endswith('.jpg'):
                        name_base = os.path.splitext(filename)[0]  # es: img1
                        self.image_paths[name_base] = os.path.join(img_dir, filename)

            # Popola self.mask_paths con chiave = nome base senza estensione
            if self.mask_dirs:
                for mask_dir in self.mask_dirs:
                    for filename in os.listdir(mask_dir):
                        if filename.endswith('.png'):
                            name_base = os.path.splitext(filename)[0]  # es: img1
                            self.mask_paths.setdefault(name_base, []).append(os.path.join(mask_dir, filename))

            # Trova solo le immagini che hanno una maschera corrispondente (match sul nome base)
            common_keys = sorted(set(self.image_paths.keys()) & set(self.mask_paths.keys()))

            # Ora puoi salvarle come lista di tuple (image_path, mask_path)
            self.image_filenames = [
                (self.image_paths[key], self.mask_paths[key][0])  # se più maschere, prende la prima
                for key in common_keys
            ]



            if self.increase:
                self.image_filenames = self.image_filenames * 3

    def __len__(self):
            return len(self.image_filenames)

    def __getitem__(self, idx):


            img_path, mask_path = self.image_filenames[idx]

            image = np.array(Image.open(img_path).convert("RGB"))
            mask = np.array(Image.open(mask_path).convert("L"))




            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                combined_mask = augmented["mask"]
            else:
                transform_basic = A.Compose([
                    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                    ToTensorV2()
                ])
                augmented = transform_basic(image=image, mask=mask)
                image = augmented["image"]
                combined_mask = augmented["mask"]

            return image, combined_mask



class CholecDataset(Dataset):
        """
        Class: CholecDataset

        Purpose:
            A PyTorch-compatible dataset class for loading and preprocessing surgical video frames
            and their corresponding instrument segmentation masks from the CholecSeg dataset
            (or Hugging Face-compatible derivatives).

            The dataset expects each sample to contain an RGB image and a color-encoded mask
            (under the keys "image" and "color_mask", respectively). Instrument masks are extracted
            by filtering specific color codes in the mask (169 and 170), which correspond to surgical tools.

        Constructor Arguments:
            hf_dataset (Dataset or DatasetDict):
                A Hugging Face dataset object containing samples with fields:
                    - "image": the RGB image (PIL.Image or numpy.ndarray)
                    - "color_mask": a color-encoded segmentation mask (PIL.Image)

            transform (albumentations.Compose, optional):
                A joint image-mask transformation pipeline (e.g., resizing, flipping, normalization).
                Applied to both the image and the binary mask.

        Sample Processing:
            - Converts the image to RGB format if necessary.
            - Converts grayscale images to 3-channel RGB by stacking.
            - Converts the color mask into a binary mask, selecting instrument labels (169 and 170).
            - Applies the provided transformation to both image and mask.
            - Ensures the mask is a float32 tensor of shape [H, W].

        Returns (per sample):
            image (torch.Tensor):
                A 3-channel RGB image of shape [3, H, W], normalized if using a transform.

            instrument_mask (torch.Tensor):
                A binary segmentation mask of shape [H, W], dtype=torch.float32.
                Values are 1 for instrument pixels, 0 elsewhere.


        """

        def __init__(self, hf_dataset, transform=None):
            self.dataset = hf_dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            sample = self.dataset[idx]

            # === IMMAGINE ===
            image = sample["image"]
            if isinstance(image, Image.Image):
                image = np.array(image.convert("RGB"))
            elif isinstance(image, np.ndarray):
                if image.ndim == 2:
                    image = np.stack([image] * 3, axis=-1)

            # === MASCHERA ===
            mask = sample["color_mask"]
            if isinstance(mask, Image.Image):
                mask = np.array(mask)

            instrument_mask = np.isin(mask, [169, 170]).astype(np.uint8)

            # === TRASFORMAZIONI ===
            if self.transform:
                transformed = self.transform(image=image, mask=instrument_mask)
                image = transformed["image"]  # [3, H, W] tensor
                instrument_mask = transformed["mask"]  # [H, W] numpy o tensor

                if isinstance(instrument_mask, np.ndarray):
                    instrument_mask = torch.tensor(instrument_mask, dtype=torch.float32)
                # print("mask unique after transform:", np.unique(instrument_mask)) #maschera tutta  0
                instrument_mask = torch.tensor(instrument_mask[:, :, 0], dtype=torch.float32)
            else:
                image = ToTensor()(image)  # [3, H, W]

                mask_pil = Image.fromarray(instrument_mask)
                instrument_mask = mask_pil.convert("L")

            return image, instrument_mask  # immmagine [3,h,w] mask [h,w] torch.float32


class DatasetTest(Dataset):
    def __init__(self, image_dirs, coco, transform=None, top_n=None, min_area=None):
        """
        Dataset per immagini con annotazioni COCO.

        Args:
            image_dirs: Lista di directory contenenti le immagini
            coco_file: Path al file annotations.json in formato COCO
            transform: Trasformazioni Albumentations (opzionale)
            top_n: Numero massimo di bbox da restituire (le più grandi)
            min_area: Area minima per filtrare le bbox
        """
        self.image_dirs = image_dirs if isinstance(image_dirs, list) else [image_dirs]
        self.transform = transform
        self.top_n = top_n
        self.min_area = min_area

        # Carica COCO
        with open(coco, "r") as f:
            self.coco_data = json.load(f)
        self.coco = COCO(coco)

        # Crea mappatura filename -> image_id per ricerca veloce
        self.filename_to_id = {
            img["file_name"]: img["id"]
            for img in self.coco_data["images"]
        }

        # Trova tutti i percorsi delle immagini
        self.image_paths = []
        for img_dir in self.image_dirs:
            if not os.path.exists(img_dir):
                print(f"⚠️  Directory non trovata: {img_dir}")
                continue

            for filename in os.listdir(img_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(img_dir, filename)
                    # Verifica che l'immagine sia nel file COCO
                    if filename in self.filename_to_id:
                        self.image_paths.append(full_path)
                    else:
                        print(f"⚠️  Immagine non nel COCO: {filename}")

        print(f"✓ Trovate {len(self.image_paths)} immagini con annotazioni")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Carica immagine
        img_path = self.image_paths[idx]
        image = np.array(Image.open(img_path).convert("RGB"))

        # Estrai filename
        filename = os.path.basename(img_path)

        # Trova image_id
        img_id = self.filename_to_id.get(filename)

        if img_id is None:
            print(f"⚠️  Nessuna annotazione per: {filename}")
            # Ritorna immagine senza bbox
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented["image"]
            else:
                image = self._default_transform(image)
            return image, []

        # Carica annotazioni
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)

        # Estrai bboxes: [x_min, y_min, width, height] (formato COCO)
        bboxes_with_area = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            area = ann.get("area", w * h)
            bboxes_with_area.append({
                "bbox": [x, y, w, h],
                "bbox_xyxy": [x, y, x + w, y + h],  # Formato [x1, y1, x2, y2]
                "area": area,
                "category_id": ann["category_id"]
            })

        # Filtra per area minima
        if self.min_area is not None:
            bboxes_with_area = [b for b in bboxes_with_area if b["area"] >= self.min_area]

        # Ordina per area (più grande prima) e prendi top N
        bboxes_with_area.sort(key=lambda x: x["area"], reverse=True)
        if self.top_n is not None:
            bboxes_with_area = bboxes_with_area[:self.top_n]

        # Estrai bbox in formato COCO [x, y, w, h]
        bboxes = [b["bbox"] for b in bboxes_with_area]

        # Applica trasformazioni
        if self.transform:
            # Albumentations richiede bbox in formato [x_min, y_min, x_max, y_max] normalizzato
            h, w = image.shape[:2]
            bbox_params = A.BboxParams(
                format='coco',  # COCO format: [x, y, width, height]
                label_fields=['category_ids'],
                min_area=0,
                min_visibility=0
            )

            transform_with_bbox = A.Compose(
                self.transform.transforms if hasattr(self.transform, 'transforms') else [self.transform],
                bbox_params=bbox_params
            )

            category_ids = [b["category_id"] for b in bboxes_with_area]


            augmented = transform_with_bbox(
                image=image,
                bboxes=bboxes,
                category_ids=category_ids
            )
            image = augmented["image"]
            bboxes = augmented["bboxes"]
        else:
            image = self._default_transform(image)

        return image, bboxes
class InstrumentDataset(Dataset):
    def __init__(self, image_dirs, gt_dirs, transform=None, class_to_id=None, increase=False):
        """
        image_dirs: lista di cartelle con i frame RGB
        gt_dirs: lista di cartelle 'ground_truth' (una per dataset)
        """
        self.image_dirs = image_dirs
        self.gt_dirs = gt_dirs
        self.transform = transform
        self.increase = increase

        self.image_paths = {}
        self.mask_paths = {}
        for img_dir in image_dirs:
            dataset_number = img_dir.split('/')[-2]
            for filename in os.listdir(img_dir):
                if filename.endswith('.png'):
                    key = f"{dataset_number}_{filename}"
                    self.image_paths[key] = os.path.join(img_dir, filename)
        # print(self.image_paths)
        """
        for d in image_dirs:
            self.image_paths += [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".png")]"""

        # estrai classi dagli strumenti nelle cartelle ground_truth
        self.class_folders = {}
        class_names = set()

        if gt_dirs:
            # trovo tutti i folder con i nomi degli strumenti
            for gt in gt_dirs:
                for folder in os.listdir(gt):
                    #print(folder)

                    if folder != "Other_labels": #evito la classe other labels per consistnza con gl altri modelli distillati che non la cosniderano
                        print(folder)

                        folder_path = os.path.join(gt, folder)

                        if os.path.isdir(folder_path):
                            norm_name = self.normalize_name(folder)
                            class_names.add(norm_name)
                            self.class_folders.setdefault(norm_name, []).append(folder_path)

            # creo le chiavi delle maschere,assegnadno ad ogni maschera il suo dataset
            for mask_dir in self.gt_dirs:
                dataset_number = mask_dir.split('/')[-2]
                #print(dataset_number)
                for folder in os.listdir(mask_dir):

                    if folder != "Other_labels":
                        folder_path = os.path.join(mask_dir, folder)
                        print(folder)

                        for filename in os.listdir(folder_path):
                            if filename.endswith('.png'):
                                key = f"{dataset_number}_{filename}"
                                self.mask_paths.setdefault(key, []).append(os.path.join(folder_path,
                                                                                    filename))  # le chiavi sono del tipo instrument_dataset_2_frame163.png, e il contenuto sono i percorsi per arrivare all immagine o maschera

        # assegna ID univoco alle classi
        self.class_to_id = class_to_id if class_to_id is not None else {
            cls: i + 1 for i, cls in enumerate(sorted(class_names))
        }
        print("Class mapping:", self.class_to_id)
        #print(self.mask_paths.keys())
        #print(self.image_paths.keys())
        self.image_filenames = sorted(set(self.image_paths.keys()) & set(self.mask_paths.keys()))
        # print("Number of images with masks:", len(self.image_filenames))

        if self.increase:
            self.image_filenames = self.image_filenames * 3

    def normalize_name(self, folder_name):
        """Rimuove Right_, Left_, e _labels dai nomi delle cartelle"""
        name = re.sub(r'^(Right_|Left_|_Left|_Right)', '', folder_name)
        name = re.sub(r'_labels$', '', name)
        return name

    def __len__(self):
        return len(self.image_filenames)

    def getNumClasses(self):
        return len(self.class_to_id) + 1  # +1 background

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        # print(img_name)
        img_path = self.image_paths[img_name]
        image = np.array(Image.open(img_path).convert("RGB"))

        # maschera iniziale = background
        if img_name in self.mask_paths:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

            # nome file (solo basename!)
            fname = os.path.basename(img_path)
            # print(fname)
            # print(img_name)
            dataset = img_name.split('_')[2]  # instrument_dataset_N_frameN
            # print(dataset)
            self.instruments = [0] * self.getNumClasses()
            # cerca la maschera per ogni classe
            for cls, folders in self.class_folders.items():

                class_id = self.class_to_id[cls]
                for folder in folders:
                    mask_path = os.path.join(folder, fname)
                    # filtro dataset corretto(perche i frameN sono in piu cartelle di dataset diversi)
                    match = re.search(r"instrument_dataset_(\d+)",
                                      mask_path)  # MICCAI/instrument_1_4_training/instrument_dataset_2/ground_truth\Left_Prograsp_Forceps_labels\frame163.png
                    if (match is None):
                        match = re.search(r"validation_dataset_(\d+)", mask_path)
                    if (os.path.exists(mask_path) & (match and match.group(1) == dataset)):
                        inst_mask = np.array(Image.open(mask_path).convert("L"))
                        # print(mask_path)
                        mask[inst_mask > 0] = class_id

        """
        label = (mask * 255).astype(np.uint8)
        # Convert to a PIL Image
        label_image = Image.fromarray(label)
        # Save the image
        label_image.save("label.png")
"""

        # trasformazioni
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
        else:
            basic_transform = A.Compose([
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2()
            ])
            augmented = basic_transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        mask = torch.as_tensor(mask, dtype=torch.long)

        return image, mask

    def getInstruments(self):
        return self.instruments



class InstrumentDatasetTest(Dataset):
    def __init__(self, image_dirs, gt_dirs, transform=None):
        """
        image_dirs: lista di cartelle con i frame RGB
        gt_dirs: lista di cartelle 'ground_truth' (una per dataset)
        """
        self.image_dirs = image_dirs
        self.gt_dirs = gt_dirs
        self.transform = transform
        self.class_colors = {
            0: np.array((0, 0, 0)),  # background = nero
            1: np.array((0, 0, 255)),  # Large_Needle_Driver= blu
            2: np.array((0, 255, 255)),  # Prograsp_Forceps= ciano

            4: np.array((255, 255, 0)),  # Maryland_Bipolar_Forceps = giallo (nuovi)
            5: np.array((0, 255, 0)),  # monopolar curved scissors = verde
            6: np.array((255, 0, 0))  # vessel sealer = rosso

        }
        # ciano prograsp froceps
        # blu large needle driver
        # giallo maryland
        self.colors = [(0, 255, 255), (0, 0, 255), (255, 255, 0)]  # ciano, blu, giallo

        self.image_paths = {}
        self.mask_paths = {}
        for img_dir in image_dirs:
            dataset_number = img_dir.split('/')[-2]
            print(dataset_number)

            for filename in os.listdir(img_dir):
                if filename.endswith('.png'):
                    key = f"{dataset_number}_{filename}"
                    self.image_paths[key] = os.path.join(img_dir, filename)

        for gt_dir in gt_dirs:
            dataset_number = gt_dir.split('/')[-3]
            print(dataset_number)

            for filename in os.listdir(gt_dir):
                if filename.endswith('.png'):

                    key = f"{dataset_number}_{filename}"
                    self.mask_paths.setdefault(key, []).append(os.path.join(gt_dir, filename))

        self.image_filenames = sorted(set(self.image_paths.keys()) & set(self.mask_paths.keys()))
        print(self.image_filenames)

    def __len__(self):

        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = self.image_paths[img_name]
        image = np.array(Image.open(img_path).convert("RGB"))
        is_instrument_dataset_1 = "instrument_dataset_1" in img_path
        if img_name in self.mask_paths:
            mask = np.array(Image.open(self.mask_paths[img_name][0]))
            mask_label = np.full((image.shape[0], image.shape[1]), -1, dtype=np.int32)  # Inizializza con background

            for id_color, color in self.class_colors.items():
                if mask.ndim == 2:
                    raise ValueError(f"Expected RGB mask but got shape {mask.shape}")
                match = np.all(mask == color, axis=-1)
                match = np.all((mask == color), axis=-1)
                if is_instrument_dataset_1:
                    # Assumendo che il giallo sia [255, 255, 0] o simile
                    is_yellow = np.all(color == [255, 255, 0])  # Adatta ai tuoi valori RGB del giallo

                    if is_yellow:
                        mask_label[match] = 2  # Se è giallo, assegna 2 (Prograsp_Forceps)(ildataset 1 è sbagliato,maryland in realta sono prograsp)


                    else:

                        mask_label[match] = id_color #se non è giallo regola normale
                else:
                    mask_label[match] = id_color

        if self.transform:
            augmented = self.transform(image=image, mask=mask_label)
            image = augmented["image"]
            mask_label = augmented["mask"]
        else:
            transform_basic = A.Compose([
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])
            augmented = transform_basic(image=image, mask=mask_label)
            image = augmented["image"]
            combined_mask = augmented["mask"]

        return image, mask_label
