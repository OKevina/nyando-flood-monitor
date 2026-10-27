"""
05_inference.py — U-Net inference on a SAR GeoTIFF scene + PostGIS ingestion.

Usage:
    python pipeline/05_inference.py \
        --scene data/scenes/my_scene.tif \
        --scene_id 1 \
        --model models/nyando_finetuned_model.pth \
        --out data/predictions/flood_mask.tif
"""
import argparse
import subprocess
import os
import numpy as np
import torch
import torch.nn as nn
import rasterio
from rasterio.transform import from_bounds

# Exact architecture from 04_finetune_nyando.py
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 64);  self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128);           self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256);          self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(256, 512)
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2); self.conv1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2); self.conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128,  64, 2, stride=2); self.conv3 = DoubleConv(128,  64)
        self.out_conv = nn.Conv2d(64, out_channels, 1)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.down3(self.pool2(x2))
        xb = self.bottleneck(self.pool3(x3))
        x  = self.conv1(torch.cat([self.up1(xb), x3], 1))
        x  = self.conv2(torch.cat([self.up2(x),  x2], 1))
        x  = self.conv3(torch.cat([self.up3(x),  x1], 1))
        return self.sigmoid(self.out_conv(x))

# Helpers
PATCH = 256

def normalize_patch(patch: np.ndarray) -> np.ndarray:
    """Per-channel Z-score — must match training normalization."""
    out = patch.astype(np.float32)
    for c in range(out.shape[0]):
        mean, std = out[c].mean(), out[c].std()
        if std > 0:
            out[c] = (out[c] - mean) / std
    return out

def run_inference(scene_path: str, model_path: str, out_path: str, threshold: float = 0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = UNet(in_channels=2, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")

    # Load SAR scene
    with rasterio.open(scene_path) as src:
        sar   = src.read().astype(np.float32)   # shape (2, H, W)
        meta  = src.meta.copy()
        H, W  = sar.shape[1], sar.shape[2]

    # Pad so H and W are multiples of PATCH
    pad_h = (PATCH - H % PATCH) % PATCH
    pad_w = (PATCH - W % PATCH) % PATCH
    sar_padded = np.pad(sar, ((0,0),(0,pad_h),(0,pad_w)), mode='reflect')
    PH, PW = sar_padded.shape[1], sar_padded.shape[2]

    prediction = np.zeros((PH, PW), dtype=np.float32)

    # Tile inference
    total = (PH // PATCH) * (PW // PATCH)
    done  = 0
    with torch.no_grad():
        for row in range(0, PH, PATCH):
            for col in range(0, PW, PATCH):
                patch = sar_padded[:, row:row+PATCH, col:col+PATCH]
                patch = normalize_patch(patch)
                tensor = torch.from_numpy(patch).unsqueeze(0).to(device)  # (1,2,256,256)
                pred   = model(tensor).squeeze().cpu().numpy()             # (256,256)
                prediction[row:row+PATCH, col:col+PATCH] = pred
                done += 1
                if done % 10 == 0:
                    print(f"  Processed {done}/{total} patches")

    # Crop padding back, apply threshold
    mask = (prediction[:H, :W] >= threshold).astype(np.uint8)

    # Write output GeoTIFF
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    meta.update(count=1, dtype='uint8', nodata=255)
    with rasterio.open(out_path, 'w', **meta) as dst:
        dst.write(mask, 1)
    print(f"Flood mask saved to {out_path}")
    return out_path

def ingest_to_postgis(tif_path: str, scene_id: int, db_url: str):
    """Load flood mask raster into PostGIS flood_masks table via raster2pgsql."""
    print("Ingesting raster into PostGIS...")
    # raster2pgsql: -a = append, -s 4326 = SRID, -t 256x256 = tile size
    cmd_raster = [
        "raster2pgsql", "-a", "-s", "4326", "-t", "256x256",
        tif_path, "flood_masks"
    ]
    cmd_psql = ["psql", db_url]
    r2p = subprocess.Popen(cmd_raster, stdout=subprocess.PIPE)
    psql = subprocess.Popen(cmd_psql, stdin=r2p.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    r2p.stdout.close()
    out, err = psql.communicate()
    if psql.returncode != 0:
        print(f"ERROR ingesting raster:\n{err.decode()}")
    else:
        print(f"Raster ingested successfully for scene_id={scene_id}")

# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U-Net flood inference")
    parser.add_argument("--scene",    required=True,  help="Path to input SAR GeoTIFF (2-band VV+VH)")
    parser.add_argument("--scene_id", required=True,  type=int, help="Scene ID in DB")
    parser.add_argument("--model",    default="models/nyando_finetuned_model.pth")
    parser.add_argument("--out",      default="data/predictions/flood_mask.tif")
    parser.add_argument("--db",       default=os.getenv("DATABASE_URL", ""), help="PostgreSQL connection string")
    parser.add_argument("--threshold",default=0.5, type=float)
    parser.add_argument("--no-ingest",action="store_true", help="Skip PostGIS ingestion")
    args = parser.parse_args()

    tif = run_inference(args.scene, args.model, args.out, args.threshold)

    if not args.no_ingest and args.db:
        ingest_to_postgis(tif, args.scene_id, args.db)
    elif args.no_ingest:
        print("Skipping PostGIS ingestion (--no-ingest flag set).")
    else:
        print("No DATABASE_URL provided — skipping PostGIS ingestion.")