import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torchvision.transforms import ToTensor
import cv2
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


class MMIDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, num_classes=3):
        """
        Args:
            root_dir (str): Root folder with 'images' and 'labels' subfolders.
            split (str): 'train', 'valid', or 'test'.
            transform (callable, optional): Image transforms.
            target_transform (callable, optional): Label transforms.
        """
        self.image_dir = os.path.join(root_dir, 'images', split)
        self.label_dir = os.path.join(root_dir, 'labels', split)
        self.label_dir_png = os.path.join(root_dir, 'labels_png', split)
        self.transform = transform
        self.png_labels = False  # Default to text labels
        self.split = split
        self.num_classes = num_classes

        if not os.path.exists(self.label_dir_png):
            print("Warning: No PNG labels directory provided, using text labels instead.")
            self.png_labels = True
        self.samples = []
        
        for fname in sorted(os.listdir(self.image_dir)):#[:100]:
            if fname.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                img_path = os.path.join(self.image_dir, fname)
                if self.png_labels:
                    label_path = os.path.join(self.label_dir_png, os.path.splitext(fname)[0] + '_mask.png')
                    if os.path.exists(label_path):
                        #print(f"Found label for {fname} in PNG format.")
                        self.samples.append((img_path, label_path))
                    else:
                        print(f"Warning: no label for {fname} in PNG format, skipping.")
                else:
                    label_path = os.path.join(self.label_dir, os.path.splitext(fname)[0] + '.txt')
                    if os.path.exists(label_path):
                        self.samples.append((img_path, label_path))
                    else:
                        print(f"Warning: no label for {fname}, skipping.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Load first image and mask
        img_path, label_path = self.samples[idx]
        image = np.array(Image.open(img_path).convert('RGB'))

        if self.png_labels:
            mask = np.array(Image.open(label_path).convert('L'), dtype=np.uint8)
            mask = (mask > 0).astype(np.uint8)
        else:
            polygons = []
            with open(label_path, 'r') as f:
                for line in f:
                    values = line.strip().split()
                    if len(values) >= 7 and (len(values) - 1) % 2 == 0:
                        class_id = int(values[0])
                        coords = list(map(float, values[1:]))
                        polygon = [class_id] + coords
                        polygons.append(polygon)
            image_size = (image.shape[0], image.shape[1])
            if self.num_classes > 1:
                mask = self.polygon_to_mask(image.shape[:2], polygons, self.num_classes)
            else:
                mask = self.polygon_to_mask(image.shape[:2], polygons)
                #mask = self.polygon_to_mask(image_size, polygons)

        # ----- CutMix augmentation -----
        if self.split == 'train' and random.random() < 0.0:
            # Sample a second random image and mask
            idx2 = random.randint(0, len(self.samples) - 1)
            img_path2, label_path2 = self.samples[idx2]
            image2 = np.array(Image.open(img_path2).convert('RGB'))

            if self.png_labels:
                mask2 = np.array(Image.open(label_path2).convert('L'), dtype=np.uint8)
                mask2 = (mask2 > 0).astype(np.uint8)
            else:
                polygons2 = []
                with open(label_path2, 'r') as f:
                    for line in f:
                        values = line.strip().split()
                        if len(values) >= 7 and (len(values) - 1) % 2 == 0:
                            class_id = int(values[0])
                            coords = list(map(float, values[1:]))
                            polygon = [class_id] + coords
                            polygons2.append(polygon)
                image_size2 = (image2.shape[0], image2.shape[1])
                if self.num_classes > 1:
                    mask2 = self.polygon_to_mask(image_size2, polygons2, self.num_classes)
                else:   
                    mask2 = self.polygon_to_mask(image_size2, polygons2)

            # Apply CutMix
            image, mask = self.apply_cutmix(image, mask, image2, mask2)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask


    @staticmethod
    def polygon_to_mask(image_size, polygons):
        """
        Creates a binary mask from polygons, ignoring class ID.

        Args:
            image_size (tuple): (height, width)
            polygons (list of list): Each sublist = [class_id, x1, y1, ..., xn, yn]

        Returns:
            np.ndarray: Binary mask (height, width), dtype=np.uint8
        """
        mask = np.zeros(image_size, dtype=np.uint8)
        for poly in polygons:
            coords = poly[1:]  # skip class_id
            # Rescale points to fit image_size
            coords = np.array(coords, dtype=np.float32).reshape((-1, 2))
            h, w = image_size
            # Assume original coordinates are in [0, 1] range, scale to image size
            if coords.max() <= 1.0:
                coords[:, 0] *= w
                coords[:, 1] *= h
            pts = coords.astype(np.int32)
            cv2.fillPoly(mask, [pts], color=1)

        return mask

    def polygon_to_multimask(image_size, polygons, num_classes=3):
        """
        Creates a multiclass mask from polygon annotations.

        Args:
            image_size (tuple): (height, width)
            polygons (list of list): Each sublist = [class_id, x1, y1, ..., xn, yn]
            num_classes (int): Number of classes (excluding background)

        Returns:
            np.ndarray: Multiclass mask (H, W), dtype=np.uint8
        """
        mask = np.zeros(image_size, dtype=np.uint8)  # background is 0

        for poly in polygons:
            class_id = poly[0] + 1  # +1 to reserve 0 for background
            coords = poly[1:]
            coords = np.array(coords, dtype=np.float32).reshape((-1, 2))
            h, w = image_size
            if coords.max() <= 1.0:
                coords[:, 0] *= w
                coords[:, 1] *= h
            pts = coords.astype(np.int32)
            cv2.fillPoly(mask, [pts], color=class_id)

        return mask

    def apply_cutmix(self, image1, mask1, image2, mask2):
        """Apply CutMix augmentation between two image-mask pairs."""
        h, w, _ = image1.shape

        cut_w = random.randint(w // 4, w // 2)
        cut_h = random.randint(h // 4, h // 2)
        x1 = random.randint(0, w - cut_w)
        y1 = random.randint(0, h - cut_h)
        x2, y2 = x1 + cut_w, y1 + cut_h

        image1[y1:y2, x1:x2, :] = image2[y1:y2, x1:x2, :]
        mask1[y1:y2, x1:x2] = mask2[y1:y2, x1:x2]

        return image1, mask1

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