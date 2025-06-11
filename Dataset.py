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
            print("mask unique after transform:", np.unique(instrument_mask)) #maschera tutta  0
            instrument_mask = torch.tensor(instrument_mask[:,:,0], dtype=torch.float32)
        else:
            image = ToTensor()(image)  # [3, H, W]

            mask_pil = Image.fromarray(instrument_mask)
            instrument_mask = mask_pil.convert("L")

        return image, instrument_mask #immmagine [3,h,w] mask [h,w] torch.float32