import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import rasterio
import random

# ── Colab Setup ──────────────────────────────────────────────────────────────
# Uncomment when running in Colab:
# from google.colab import drive
# drive.mount('/content/drive')

S1_DIR    = '/content/drive/MyDrive/Main dataset/HandLabeled/S1Hand'
LABEL_DIR = '/content/drive/MyDrive/Main dataset/HandLabeled/LabelHand'


# ── Dataset ──────────────────────────────────────────────────────────────────
class Sen1Floods11Dataset(Dataset):
    def __init__(self, s1_dir, label_dir, african_countries_only=False, patch_size=256, overlap=50):
        self.s1_dir = s1_dir
        self.label_dir = label_dir
        self.patch_size = patch_size
        self.stride = patch_size - overlap  # 206

        all_files = [f for f in os.listdir(s1_dir) if f.endswith('.tif')]
        if african_countries_only:
            african_prefixes = ('Ghana', 'Nigeria', 'Somalia')
            self.s1_files = [f for f in all_files if f.startswith(african_prefixes)]
        else:
            self.s1_files = all_files

        self.patches = []
        for file_idx, s1_name in enumerate(self.s1_files):
            s1_path = os.path.join(self.s1_dir, s1_name)
            with rasterio.open(s1_path) as src:
                h, w = src.height, src.width
            if h >= patch_size and w >= patch_size:
                for top in range(0, h - patch_size + 1, self.stride):
                    for left in range(0, w - patch_size + 1, self.stride):
                        self.patches.append((file_idx, top, left))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        file_idx, top, left = self.patches[idx]
        s1_name = self.s1_files[file_idx]
        label_name = s1_name.replace('_S1Hand.tif', '_LabelHand.tif')

        s1_path = os.path.join(self.s1_dir, s1_name)
        label_path = os.path.join(self.label_dir, label_name)

        with rasterio.open(s1_path) as src:
            s1_img = src.read()
            s1_img = np.nan_to_num(s1_img)
            for c in range(s1_img.shape[0]):
                mean, std = np.mean(s1_img[c]), np.std(s1_img[c])
                if std > 0:
                    s1_img[c] = (s1_img[c] - mean) / std

        with rasterio.open(label_path) as src:
            label_img = src.read(1)

        s1_img    = s1_img[:, top:top+self.patch_size, left:left+self.patch_size]
        label_img = label_img[top:top+self.patch_size, left:left+self.patch_size]

        s1_tensor    = torch.tensor(s1_img, dtype=torch.float32)
        label_tensor = torch.tensor(label_img, dtype=torch.float32).unsqueeze(0)

        return s1_tensor, label_tensor


# ── Model ─────────────────────────────────────────────────────────────────────
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
        self.down1   = DoubleConv(in_channels, 64)
        self.pool1   = nn.MaxPool2d(2)
        self.down2   = DoubleConv(64, 128)
        self.pool2   = nn.MaxPool2d(2)
        self.down3   = DoubleConv(128, 256)
        self.pool3   = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(256, 512)

        self.up1   = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256)
        self.up2   = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up3   = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        x1    = self.down1(x)
        x2    = self.down2(self.pool1(x1))
        x3    = self.down3(self.pool2(x2))
        x_btn = self.bottleneck(self.pool3(x3))

        x = self.conv1(torch.cat([self.up1(x_btn), x3], dim=1))
        x = self.conv2(torch.cat([self.up2(x),    x2], dim=1))
        x = self.conv3(torch.cat([self.up3(x),    x1], dim=1))
        return self.sigmoid(self.out_conv(x))


# ── Loss & Metrics ────────────────────────────────────────────────────────────
class MaskedBCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, pred, target):
        valid_mask  = (target != -1).float()
        safe_target = target.clamp(min=0)

        bce_loss = self.bce(pred, safe_target)
        bce_loss = (bce_loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)

        smooth      = 1e-5
        pred_flat   = pred.view(-1)
        target_flat = safe_target.view(-1)
        mask_flat   = valid_mask.view(-1)

        intersection = (pred_flat * target_flat * mask_flat).sum()
        denominator  = (pred_flat * mask_flat).sum() + (target_flat * mask_flat).sum()
        dice_loss    = 1 - ((2. * intersection + smooth) / (denominator + smooth))

        return bce_loss + dice_loss


def calculate_metrics(pred, target):
    valid_mask  = (target != -1)
    preds_bin   = (pred > 0.5).bool() & valid_mask
    targets_bin = (target > 0).bool() & valid_mask

    tp = (preds_bin  & targets_bin).float().sum()
    fp = (preds_bin  & ~targets_bin & valid_mask).float().sum()
    fn = (~preds_bin & targets_bin  & valid_mask).float().sum()

    precision = (tp / (tp + fp + 1e-8)).item()
    recall    = (tp / (tp + fn + 1e-8)).item()
    f1        = (2 * tp / (2 * tp + fp + fn + 1e-8)).item()
    iou       = (tp / (tp + fp + fn + 1e-8)).item()

    return iou, f1, precision, recall


# ── Training ──────────────────────────────────────────────────────────────────
def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    full_dataset = Sen1Floods11Dataset(S1_DIR, LABEL_DIR, patch_size=256, overlap=50)
    print(f"Total patches: {len(full_dataset)} (from {len(full_dataset.s1_files)} images)")

    total_len = len(full_dataset)
    train_len = int(0.6 * total_len)
    val_len   = int(0.2 * total_len)
    test_len  = total_len - train_len - val_len

    train_set, val_set, test_set = random_split(full_dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=8, shuffle=False)

    model     = UNet(in_channels=2, out_channels=1).to(device)
    criterion = MaskedBCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs      = 100
    patience        = 15
    best_iou        = 0.0
    epochs_no_improve = 0
    save_path       = '/content/drive/MyDrive/unet_sen1floods_best.pth'

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = val_iou = val_f1 = val_prec = val_rec = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                iou, f1, prec, rec = calculate_metrics(outputs, labels)
                val_iou += iou; val_f1 += f1; val_prec += prec; val_rec += rec

        n = len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss/n:.4f} | Val IoU: {val_iou/n:.4f} | "
              f"Val F1: {val_f1/n:.4f} | Prec: {val_prec/n:.4f} | Rec: {val_rec/n:.4f}")

        if val_iou / n > best_iou:
            best_iou = val_iou / n
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print("  [*] Best model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    # ── Test Evaluation ───────────────────────────────────────────────────────
    print("\n--- Test Set Evaluation ---")
    model.load_state_dict(torch.load(save_path))
    model.eval()

    test_loader = DataLoader(test_set, batch_size=8, shuffle=False)
    test_iou = test_f1 = test_prec = test_rec = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            iou, f1, prec, rec = calculate_metrics(model(images), labels)
            test_iou += iou; test_f1 += f1; test_prec += prec; test_rec += rec

    n = len(test_loader)
    print(f"Test IoU:       {test_iou/n:.4f}")
    print(f"Test F1:        {test_f1/n:.4f}")
    print(f"Test Precision: {test_prec/n:.4f}")
    print(f"Test Recall:    {test_rec/n:.4f}")


if __name__ == '__main__':
    train_model()
