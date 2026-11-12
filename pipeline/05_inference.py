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
import os

# Point PROJ/GDAL at rasterio's bundled data files before any rasterio use.
try:
    import importlib.util as _ilu
    _spec = _ilu.find_spec("rasterio")
    if _spec and _spec.submodule_search_locations:
        _rio = _spec.submodule_search_locations[0]
        os.environ["PROJ_LIB"] = os.path.join(_rio, "proj_data")
        os.environ["GDAL_DATA"] = os.path.join(_rio, "gdal_data")
except Exception:
    pass

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
    """Fixed dB scaling — must match NyandoFloodDataset in the Colab notebook.

    Converts linear sigma0 to dB, clips to [-30, 0], scales to [0, 1].
    Preserves absolute backscatter (open water = low dB), unlike the old
    per-patch z-score which erased it.
    """
    out = 10.0 * np.log10(np.where(patch > 0, patch, np.nan))
    out = np.nan_to_num(out, nan=-15.0)
    out = np.clip(out, -30.0, 0.0)
    return ((out + 30.0) / 30.0).astype(np.float32)

def run_inference(scene_path: str, model_path: str, out_path: str,
                  threshold: float = 0.5, min_area: int = 200):
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

    # Remove speckle: drop connected components smaller than min_area pixels
    # (~22 m/px -> 200 px ≈ 0.1 km²)
    if min_area > 0:
        from scipy import ndimage
        lab, _ = ndimage.label(mask)
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        mask[np.isin(lab, np.nonzero(sizes < min_area)[0])] = 0
        print(f"Removed components < {min_area} px; "
              f"{int(mask.sum())} flood px remain")

    # Write output GeoTIFF
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    meta.update(count=1, dtype='uint8', nodata=255)
    with rasterio.open(out_path, 'w', **meta) as dst:
        dst.write(mask, 1)
    print(f"Flood mask saved to {out_path}")
    return out_path

async def ingest_to_postgis(tif_path: str, scene_id: str, db_url: str, model_version: str = "U-Net v2.0-dB", threshold: float = 0.5):
    """Load flood mask raster into PostGIS flood_masks table using asyncpg (no raster2pgsql dependency)."""
    import asyncpg
    print("Ingesting raster into PostGIS via asyncpg...")
    try:
        with open(tif_path, 'rb') as f:
            raster_bytes = f.read()
        conn = await asyncpg.connect(db_url)
        try:
            await conn.execute(
                """
                INSERT INTO flood_masks (scene_id, model_version, flood_raster, classification_thr)
                VALUES ($1, $2, ST_FromGDALRaster($3, 4326), $4)
                """,
                scene_id,
                model_version,
                raster_bytes,
                threshold,
            )
            print(f"Raster ingested successfully for scene_id={scene_id}")
        finally:
            await conn.close()
    except Exception as e:
        print(f"ERROR ingesting raster: {e}")
        raise

# Entry point
if __name__ == "__main__":
    import asyncio
    parser = argparse.ArgumentParser(description="U-Net flood inference")
    parser.add_argument("--scene",    required=True,  help="Path to input SAR GeoTIFF (2-band VV+VH)")
    parser.add_argument("--scene_id", required=True,  help="Scene ID in DB (UUID string)")
    parser.add_argument("--model",    default="models/nyando_finetuned_model.pth")
    parser.add_argument("--out",      default="data/predictions/flood_mask.tif")
    parser.add_argument("--db",       default=os.getenv("DATABASE_URL", ""), help="PostgreSQL connection string")
    parser.add_argument("--threshold",default=0.5, type=float)
    parser.add_argument("--min-area", default=200, type=int,
                        help="Min flood component size in pixels (0 disables)")
    parser.add_argument("--model-version", default="U-Net v2.0-dB")
    parser.add_argument("--no-ingest",action="store_true", help="Skip PostGIS ingestion")
    args = parser.parse_args()

    tif = run_inference(args.scene, args.model, args.out, args.threshold,
                        min_area=args.min_area)

    if not args.no_ingest and args.db:
        asyncio.run(ingest_to_postgis(tif, args.scene_id, args.db,
                                      model_version=args.model_version,
                                      threshold=args.threshold))
    elif args.no_ingest:
        print("Skipping PostGIS ingestion (--no-ingest flag set).")
    else:
        print("No DATABASE_URL provided — skipping PostGIS ingestion.")