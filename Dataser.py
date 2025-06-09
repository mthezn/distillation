import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torchvision.transforms import ToTensor

class ImageMaskDataset(Dataset):
    def __init__(self, image_dirs, mask_dirs, transform=None, increase=False):
        self.image_dirs = image_dirs
        self.mask_dirs = mask_dirs
        self.transform = transform
        self.image_paths = {}
        self.increase = increase
        self.mask_paths = {}

        for img_dir in image_dirs:
            dataset_number = img_dir.split('/')[-2]  # Extract dataset number from the directory path
            for filename in os.listdir(img_dir):
                if filename.endswith('.png'):
                    key = f"{dataset_number}_{filename}"  # Combine dataset number and filename
                    self.image_paths[key] = os.path.join(img_dir, filename)

        # Map masks
        for mask_dir in mask_dirs:
            dataset_number = mask_dir.split('/')[-3]  # Extract dataset number from the directory path
            for filename in os.listdir(mask_dir):
                if filename.endswith('.png'):
                    key = f"{dataset_number}_{filename}"  # Combine dataset number and filename
                    if key not in self.mask_paths:
                        self.mask_paths[key] = []
                    self.mask_paths[key].append(os.path.join(mask_dir, filename))



        # Filtra solo i file comuni tra immagini e maschere
        self.image_filenames = sorted(list(set(self.image_paths.keys()) & set(self.mask_paths.keys())))


        if self.increase:
            self.image_filenames = self.image_filenames * 3

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]

        # Percorsi immagine e maschere
        img_path = self.image_paths[img_name]
        mask_paths = self.mask_paths[img_name]

        # Carica immagine
        image = np.array(Image.open(img_path).convert("RGB"))

        # Carica e somma maschere
        masks = [np.array(Image.open(mask_path).convert("L")) for mask_path in mask_paths]
        # Convert masks to a smaller data type before summing
        masks = np.array(masks, dtype=np.uint8)  # Use uint8 to reduce memory usage
        combined_mask = np.sum(masks, axis=0).clip(0, 255).astype(np.uint8)




        # Applica trasformazioni
        if self.transform:
            augmented = self.transform(image=image, mask=combined_mask)
            image = augmented["image"]
            combined_mask = augmented["mask"]

        else:
            # Fallback: solo tensor conversion
            transform_basic = A.Compose([
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2()
            ])
            augmented = transform_basic(image=image, mask=combined_mask)
            image = augmented["image"]
            combined_mask = augmented["mask"]


        return image, combined_mask #immmagine [3,h,w] mask [h,w] torch.float32




class CholecDataset(Dataset):
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
            unique,values = np.unique(mask, return_inverse=True)

        instrument_mask = np.isin(mask, [169, 170]).astype(np.uint8)  # (H, W) --> maschera corretta
        unique,values = np.unique(instrument_mask, return_counts=True)
        print("unique post", unique)
        print("values post", values)

        #mask_pil = Image.fromarray(instrument_mask)
        #mask_pil = mask_pil.convert("L")
        #print(mask_pil.size)
        #senza quest aparte dopo la trsformazione ho dei valori ad 1fors eprobaema di shape bisogna aggiungere un canale

        # === TRASFORMAZIONI ===
        if self.transform:
            transformed = self.transform(image=image, mask=instrument_mask)
            image = transformed["image"]  # [3, H, W] tensor
            instrument_mask = transformed["mask"]  # [H, W] numpy o tensor

            if isinstance(instrument_mask, np.ndarray):
                instrument_mask = torch.tensor(instrument_mask, dtype=torch.float32)
            print("mask unique after transform:", np.unique(instrument_mask)) #maschera tutta  0
            instrument_mask = torch.tensor(instrument_mask[:,:,0], dtype=torch.float32)
        else:
            image = ToTensor()(image)  # [3, H, W]
            #instrument_mask = torch.tensor(instrument_mask, dtype=torch.float32)  # [H, W]
            mask_pil = Image.fromarray(instrument_mask)
            instrument_mask = mask_pil.convert("L")

        return image, instrument_mask #immmagine [3,h,w] mask [h,w] torch.float32