import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np


class ImageMaskDataset(Dataset):
    def __init__(self, image_dirs, mask_dirs, transform=None, mask_transform=None):
        self.image_dirs = image_dirs
        self.mask_dirs = mask_dirs
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_paths = {}

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
            image = self.transform(image)
        if self.mask_transform:
            combined_mask = self.mask_transform(combined_mask)

        return image, combined_mask





