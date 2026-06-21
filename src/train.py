import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import EdgeDetectionDataset
from src.model import GraphCVNet

class HybridEdgeLoss(nn.Module):
    def __init__(self, pos_weight_val=10.0, smooth=1e-5):
        super().__init__()
        # We force the model to weigh blank mistake more than 
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]))
        self.smooth=smooth
    
    def forward(self, pred_logits, targets):
        bce_loss = self.bce(pred_logits, targets)

        pred_probs = torch.sigmoid(pred_logits)

        pred_flat = pred_probs.view(-1)
        target_flat = targets.view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice_coef = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        dice_loss = 1.0 - dice_coef

        # basic loss calc
        return bce_loss + dice_loss
    

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)

        # forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)

        # backwards pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)

@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)

def main():
    # Config
    DATA_BASE = "datasets"
    BATCH_SIZE = 8
    EPOCHS = 50
    LR = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using execution device: {device}")

    # Datasets and Loaders
    train_dataset = EdgeDetectionDataset(f"{DATA_BASE}/images/train", f"{DATA_BASE}/ground_truth/train", is_train=True)
    val_dataset = EdgeDetectionDataset(f"{DATA_BASE}/images/val", f"{DATA_BASE}/ground_truth/val", is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model Setup
    model = GraphCVNet().to(device)
    criterion = HybridEdgeLoss(pos_weight_val=15.0).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    os.makedirs("weights", exist_ok=True)
    print("Strating training loops\n" + "-"*30)

    best_val_loss = float("inf")

    for epoch in range(1,EPOCHS+1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch:02d}/{EPOCHS}] -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "weights/graphcv_best.pth")
            print("Saved new optimal checkpoint") # We are save scumming our AI

if __name__ == "__main__":
    main()