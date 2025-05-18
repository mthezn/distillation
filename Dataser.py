import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import random


class ImageMaskDataset(Dataset):
    def __init__(self, image_dirs, mask_dirs, transform=None, mask_transform=None,increase = False):
        self.image_dirs = image_dirs
        self.mask_dirs = mask_dirs
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_paths = {}
        self.increase = increase

        # Mappa immagini
        for img_dir in image_dirs:
            for filename in os.listdir(img_dir):
                if filename.endswith('.png'):
                    self.image_paths[filename] = os.path.join(img_dir, filename)

        # Mappa maschere
        self.mask_paths = {}
        for mask_dir in mask_dirs:
            for filename in os.listdir(mask_dir):
                if filename.endswith('.png'):
                    if filename not in self.mask_paths:
                        self.mask_paths[filename] = []
                    self.mask_paths[filename].append(os.path.join(mask_dir, filename))

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
        image = Image.open(img_path).convert("RGB")

        # Load and combine masks
        masks = [np.array(Image.open(mask_path).convert("L")) for mask_path in mask_paths]
        combined_mask = np.sum(masks, axis=0)  # Sum the masks element-wise

        # Convert back to PIL.Image if needed
        combined_mask = Image.fromarray(combined_mask.astype(np.uint8)) # Somma le maschere (puoi cambiare logica se necessario)

        # Applica trasformazioni
        if self.transform:
            seed = random.randint(0, 9999)
            random.seed(seed)
            image = self.transform(image)
            random.seed(seed)
            combined_mask = self.mask_transform(combined_mask) if self.mask_transform else combined_mask
        else:
            image = transforms.ToTensor()(image)
            combined_mask = transforms.ToTensor()(combined_mask)

        return image, combined_mask

class CholecDataset(Dataset):
    def __init__(self, hf_dataset, transform=None, mask_transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        image = sample["image"]
        if isinstance(image, Image.Image):
            image = image.convert("RGB")

        mask = sample["color_mask"]
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
        elif not isinstance(mask, np.ndarray):
            raise TypeError(f"Unexpected mask type: {type(mask)}")

        # Binarizza strumenti
        instrument_mask = ((mask == 169) | (mask == 170)).astype(np.uint8) * 255
        mask_pil = Image.fromarray(instrument_mask)
        mask_pil = mask_pil.convert("L")

        # Trasformazioni
        if self.transform:
            seed = random.randint(0, 99999)
            random.seed(seed)
            image = self.transform(image)
            random.seed(seed)
            mask_pil = self.mask_transform(mask_pil) if self.mask_transform else mask_pil
        else:
            image = transforms.ToTensor()(image)
            mask_pil = transforms.ToTensor()(mask_pil)

        return image, mask_pil

