import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np
import random

# ==========================================
# 1. DATASET & AUGMENTATION
# ==========================================
class NyandoFloodDataset(Dataset):
    def __init__(self, dataset_dir, split="train", transform=True):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.patches = []
        
        csv_path = os.path.join(dataset_dir, f"nyando_split_{split}.csv")
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    self.patches.append(row[0])
                    
    def __len__(self):
        return len(self.patches)
        
    def __getitem__(self, idx):
        patch_id = self.patches[idx]
        
        sar_path = os.path.join(self.dataset_dir, "S1Hand", f"{patch_id}_S1Hand.tif")
        mask_path = os.path.join(self.dataset_dir, "LabelHand", f"{patch_id}_LabelHand.tif")
        
        with rasterio.open(sar_path) as src:
            # Sentinel-1 is VV, VH (2 bands). Read and convert to Float32
            sar = src.read().astype(np.float32)
            sar = np.nan_to_num(sar)
            # Per-channel Z-score normalization — must match 01_base_training.py
            for c in range(sar.shape[0]):
                mean, std = np.mean(sar[c]), np.std(sar[c])
                if std > 0:
                    sar[c] = (sar[c] - mean) / std
            
        with rasterio.open(mask_path) as src:
            # Mask is 1 band (0=dry, 1=flood)
            mask = src.read(1).astype(np.float32)
            
        # DATA AUGMENTATION
        if self.transform:
            if random.random() > 0.5:
                sar = np.flip(sar, axis=1) # Horizontal flip
                mask = np.flip(mask, axis=0)
            if random.random() > 0.5:
                sar = np.flip(sar, axis=2) # Vertical flip
                mask = np.flip(mask, axis=1)
                
            # Random 90-degree rotations
            k = random.randint(0, 3)
            sar = np.rot90(sar, k, axes=(1, 2))
            mask = np.rot90(mask, k, axes=(0, 1))

        # Ensure correct memory layout for PyTorch
        sar = np.ascontiguousarray(sar)
        mask = np.ascontiguousarray(mask)
            
        return torch.from_numpy(sar), torch.from_numpy(mask).unsqueeze(0)

# ==========================================
# 2. U-NET ARCHITECTURE (From Phase 1)
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(256, 512)

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.down3(self.pool2(x2))
        x_btn = self.bottleneck(self.pool3(x3))

        x = self.conv1(torch.cat([self.up1(x_btn), x3], dim=1))
        x = self.conv2(torch.cat([self.up2(x), x2], dim=1))
        x = self.conv3(torch.cat([self.up3(x), x1], dim=1))
        return self.sigmoid(self.out_conv(x))

# ==========================================
# 3. LOSS FUNCTION & METRICS
# ==========================================
class MaskedBCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, pred, target):
        valid_mask = (target != -1).float()
        safe_target = target.clamp(min=0)
        
        # Masked BCE
        bce_loss = self.bce(pred, safe_target)
        bce_loss = (bce_loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)

        # Masked Dice
        smooth = 1e-5
        pred_flat = pred.view(-1)
        target_flat = safe_target.view(-1)
        mask_flat = valid_mask.view(-1)

        intersection = (pred_flat * target_flat * mask_flat).sum()
        denominator = (pred_flat * mask_flat).sum() + (target_flat * mask_flat).sum()
        dice_loss = 1 - ((2. * intersection + smooth) / (denominator + smooth))

        return bce_loss + dice_loss

def calculate_metrics(pred, target):
    """Calculate IoU, F1-score, Precision, and Recall on valid pixels only."""
    valid_mask = (target != -1)
    preds_bin = (pred > 0.5).bool() & valid_mask
    targets_bin = (target > 0).bool() & valid_mask

    tp = (preds_bin & targets_bin).float().sum()
    fp = (preds_bin & ~targets_bin & valid_mask).float().sum()
    fn = (~preds_bin & targets_bin & valid_mask).float().sum()

    precision = (tp / (tp + fp + 1e-8)).item()
    recall = (tp / (tp + fn + 1e-8)).item()
    f1 = (2 * tp / (2 * tp + fp + fn + 1e-8)).item()
    iou = (tp / (tp + fp + fn + 1e-8)).item()

    return iou, f1, precision, recall

# ==========================================
# 4. FINE-TUNING LOOP WITH EARLY STOPPING
# ==========================================
def train_model(dataset_dir, base_weights_path="unet_sen1floods_best.pth", epochs=50, batch_size=8, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataloaders
    train_dataset = NyandoFloodDataset(dataset_dir, split="train", transform=True)
    val_dataset = NyandoFloodDataset(dataset_dir, split="val", transform=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    model = UNet(in_channels=2, out_channels=1).to(device)
    
    # Load Pre-trained weights
    if os.path.exists(base_weights_path):
        model.load_state_dict(torch.load(base_weights_path, map_location=device))
        print(f"Successfully loaded base weights from {base_weights_path}")
    else:
        print(f"WARNING: Base weights '{base_weights_path}' not found.")
    
    criterion = MaskedBCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Early Stopping tracking
    patience = 10
    best_iou = 0.0
    epochs_no_improve = 0
    save_path = os.path.join(dataset_dir, "nyando_finetuned_model.pth")
    
    print(f"Starting Fine-Tuning for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for sar, mask in train_loader:
            sar, mask = sar.to(device), mask.to(device)
            
            optimizer.zero_grad()
            outputs = model(sar)
            loss = criterion(outputs, mask)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss, val_iou, val_f1 = 0, 0, 0
        with torch.no_grad():
            for sar, mask in val_loader:
                sar, mask = sar.to(device), mask.to(device)
                outputs = model(sar)
                
                loss = criterion(outputs, mask)
                val_loss += loss.item()
                
                iou, f1, _, _ = calculate_metrics(outputs, mask)
                val_iou += iou
                val_f1 += f1
                
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        avg_val_f1 = val_f1 / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f} | Val F1: {avg_val_f1:.4f}")
        
        # Early Stopping Logic
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f" -> Best model saved with IoU: {best_iou:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # ==========================================
    # FINAL TEST SET EVALUATION
    # ==========================================
    print("\n" + "="*50)
    print("FINAL TEST SET EVALUATION")
    print("="*50)

    # Reload the best saved model (not the last epoch)
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    test_dataset = NyandoFloodDataset(dataset_dir, split="test", transform=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_iou_sum, test_f1_sum, test_prec_sum, test_rec_sum = 0, 0, 0, 0
    num_batches = 0

    with torch.no_grad():
        for sar, mask in test_loader:
            sar, mask = sar.to(device), mask.to(device)
            outputs = model(sar)
            iou, f1, prec, rec = calculate_metrics(outputs, mask)
            test_iou_sum += iou
            test_f1_sum += f1
            test_prec_sum += prec
            test_rec_sum += rec
            num_batches += 1

    print(f"Test IoU:       {test_iou_sum / num_batches:.4f}")
    print(f"Test F1-Score:  {test_f1_sum / num_batches:.4f}")
    print(f"Test Precision: {test_prec_sum / num_batches:.4f}")
    print(f"Test Recall:    {test_rec_sum / num_batches:.4f}")
    print("="*50)


if __name__ == "__main__":
    DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "Nyando_Dataset")
    BASE_WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "models", "unet_sen1floods_best.pth")
    train_model(DATASET_DIR, base_weights_path=BASE_WEIGHTS)

