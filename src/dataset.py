import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.v2 as transforms

class EdgeDetectionDataset(Dataset):
    def __init__(self, image_dir, mask_dir, is_train=True, img_size=(256,256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size

        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

        assert len(self.images) = len(self.masks), "Mismatch between images and mask length"

        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
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
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image, mask = self.transform(image, mask)
        mask = (mask > 0.5).float()

        return image, mask