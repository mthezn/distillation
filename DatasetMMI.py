# Add this modification to your Dataset.py file
from torch.utils.data import Dataset
from pathlib import Path
import cv2

class MMIDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, num_classes=1):
        """
        Modified to support 'all' split for k-fold cross-validation
        
        Args:
            root_dir: Root directory of dataset
            split: 'train', 'valid', 'test', or 'all'
            transform: Albumentations transform
            num_classes: Number of classes
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.num_classes = num_classes
        
        # Define split directories
        if split == 'all':
            # Load all data for k-fold CV
            self.image_dirs = []
            self.mask_dirs = []
            
            # Combine train and valid data
            for split_name in ['train', 'valid']:
                if (self.root_dir / 'images' / split_name).exists():
                    split_image_dir = self.root_dir / 'images' / split_name
                    split_mask_dir = self.root_dir / 'labels_png' / split_name

                    if split_image_dir.exists() and split_mask_dir.exists():
                        self.image_dirs.append(split_image_dir)
                        self.mask_dirs.append(split_mask_dir)
            
            # Get all image files from all splits
            self.image_files = []
            self.mask_files = []
            
            for img_dir, mask_dir in zip(self.image_dirs, self.mask_dirs):
                img_files = sorted([f for f in img_dir.iterdir() 
                                  if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']])
                
                for img_file in img_files:
                    # Find corresponding mask
                    mask_file = mask_dir / (img_file.stem + '_mask.png')  # Adjust extension as needed
                    if not mask_file.exists():
                        mask_file = mask_dir / (img_file.stem + '.jpg')
                    if not mask_file.exists():
                        mask_file = mask_dir / (img_file.stem + img_file.suffix)
                    
                    if mask_file.exists():
                        self.image_files.append(img_file)
                        self.mask_files.append(mask_file)
        
        else:
            # Original logic for train/valid/test splits
            self.image_dir = self.root_dir / split / 'images'
            self.mask_dir = self.root_dir / split / 'labels'
            
            if not self.image_dir.exists() or not self.mask_dir.exists():
                raise ValueError(f"Split '{split}' directories not found in {root_dir}")
            
            # Get all image files
            self.image_files = sorted([f for f in self.image_dir.iterdir() 
                                     if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']])
            
            # Find corresponding mask files
            self.mask_files = []
            for img_file in self.image_files:
                mask_file = self.mask_dir / (img_file.stem + '_mask.png')  # Adjust extension as needed
                if not mask_file.exists():
                    mask_file = self.mask_dir / (img_file.stem + '.jpg')
                if not mask_file.exists():
                    mask_file = self.mask_dir / (img_file.stem + img_file.suffix)
                
                if mask_file.exists():
                    self.mask_files.append(mask_file)
                else:
                    print(f"Warning: No mask found for {img_file}")
            
            # Ensure we have matching pairs
            assert len(self.image_files) == len(self.mask_files), \
                f"Mismatch: {len(self.image_files)} images vs {len(self.mask_files)} masks"
        
        print(f"Loaded {len(self.image_files)} samples for split '{split}'")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image and mask
        image_path = self.image_files[idx]
        mask_path = self.mask_files[idx]
        
        # Read image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return image, mask