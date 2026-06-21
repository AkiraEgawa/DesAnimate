# src/dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import scipy.io as sio
import torchvision.transforms.v2 as transforms

class EdgeDetectionDataset(Dataset):
    def __init__(self, image_dir, mask_dir, is_train=True, img_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        
        raw_images = os.listdir(image_dir)
        raw_masks = os.listdir(mask_dir)
        
        # Filter for jpg/jpeg for images, and mat/png/jpg for ground truth
        self.images = sorted([f for f in raw_images if f.lower().endswith(('.jpg', '.jpeg'))])
        self.masks = sorted([f for f in raw_masks if f.lower().endswith(('.mat', '.png', '.jpg', '.jpeg'))])
        
        assert len(self.images) == len(self.masks), (
            f"Mismatch! Images: {len(self.images)}, Masks: {len(self.masks)}. "
            f"Check if filenames match up."
        )

        # Image geometric augmentations
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True)
            ])

    def __len__(self):
        return len(self.images)

    def _load_mat_mask(self, path):
        """Extracts and averages human boundary maps from a BSDS500 .mat file."""
        mat = sio.loadmat(path)
        # 'groundTruth' is the standard key inside BSDS500 .mat structures
        ground_truth = mat['groundTruth'][0]
        num_annotations = len(ground_truth)
        
        # Pull the boundary matrix from the first human annotator to establish shape
        # [0][0][0][0][1] navigates the nested MATLAB struct layer to get the boundary logical array
        boundaries = ground_truth[0][0][0][0][1].astype(np.float32)
        
        # Accumulate trimmings from all other human annotators
        for i in range(1, num_annotations):
            boundaries += ground_truth[i][0][0][0][0][1].astype(np.float32)
            
        # Average the edges across all human scores
        boundaries /= num_annotations
        return Image.fromarray((boundaries * 255).astype(np.uint8))

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        # Load RGB image
        image = Image.open(img_path).convert("RGB")
        
        # Load mask (handle .mat extraction vs standard image files)
        if mask_path.lower().endswith('.mat'):
            mask = self._load_mat_mask(mask_path)
        else:
            mask = Image.open(mask_path).convert("L") 

        # Apply aligned transforms
        image, mask = self.transform(image, mask)
        
        # Threshold: if more than 10% of humans agreed it's an edge, count it as a 1.0
        mask = (mask > 0.1).float()

        return image, mask