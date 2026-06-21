# test_data.py
import os
import torch
from src.dataset import EdgeDetectionDataset
from src.model import GraphCVNet

DATA_BASE_DIR = "datasets"

train_dataset = EdgeDetectionDataset(
    image_dir=os.path.join(DATA_BASE_DIR, "images/train"),
    mask_dir=os.path.join(DATA_BASE_DIR, "ground_truth/train"),
    is_train=True
)

# 1. Create a mock batch of 4 images using a PyTorch DataLoader
from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Grab one batch
images, masks = next(iter(train_loader))
print(f"Batch Image Shape: {images.shape}") # [4, 3, 256, 256]
print(f"Batch Mask Shape:  {masks.shape}")  # [4, 1, 256, 256]

# 2. Instantiate your neural network
model = GraphCVNet()

# 3. Feed the batch forward through the architecture
output = model(images)
print(f"Model Output Shape: {output.shape}") # Should match the mask shape exactly!

if output.shape == masks.shape:
    print("\n✓ SUCCESS: Network structural geometry matches target data completely!")
else:
    print("\n✕ ERROR: Tensor shape mismatch between output and target mask.")