import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import os
from PIL import Image

class ImageMaskDataset(Dataset):
    def __init__(self,image_dirs,mask_dirs,transform=None,mask_transform=None):
        self.image_dirs = image_dirs
        self.mask_dirs = mask_dirs
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_paths = {}
        for img_dir in image_dirs:
            for filename in os.listdir(img_dir):
                if filename.endswith('.png'):
                    self.image_paths[filename] = os.path.join(img_dir,filename)

        self.mask_paths = {}
        for mask_dir in mask_dirs:
            for filename in os.listdir(mask_dir):
                if filename.endswith('.png'):
                    self.mask_paths[filename] = os.path.join(mask_dir,filename)

        self.image_filenames = sorted(list(set(self.image_paths.keys()) & set(self.mask_paths.keys())))


    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self,idx):
        img_name = self.image_filenames[idx]

        # Percorsi immagine e maschera
        img_path = self.image_paths[img_name]
        mask_path = self.mask_paths[img_name]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") #scala di grigi

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image,mask





