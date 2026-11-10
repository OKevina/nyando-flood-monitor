"""
Build a permanent-water baseline mask for the Nyando pipeline.

Composite rule: a pixel is "permanent water" if it was classified as water
(Otsu mask == 1) in more than half of the valid scene dates. Flood water is
transient, so event pixels never reach majority; Lake Victoria, rivers and
wetlands do.

Only dates with a cached raw scene are used, and degenerate masks (>90% flood,
e.g. leftover artifacts from the old linear-scale Otsu bug) are skipped.

Output: <project_dir>/permanent_water_mask.tif (uint8, same grid as scenes).
"""

import os
import sys
import glob

# PROJ/GDAL must point at rasterio's bundled data, not the PostGIS copies.
import rasterio
os.environ.setdefault("PROJ_LIB", os.path.join(os.path.dirname(rasterio.__file__), "proj_data"))
os.environ.setdefault("GDAL_DATA", os.path.join(os.path.dirname(rasterio.__file__), "gdal_data"))

import numpy as np


def main(project_dir="data/scenes", flood_frac_skip=0.90, majority=0.5):
    mask_files = sorted(glob.glob(os.path.join(project_dir, "nyando_flood_mask_*.tif")))
    raw_dates = {
        os.path.basename(f)[len("nyando_sar_"):-len("_raw.tif")]
        for f in glob.glob(os.path.join(project_dir, "nyando_sar_*_raw.tif"))
    }

    water_sum = None
    meta = None
    used = []
    skipped = []

    for path in mask_files:
        date_str = os.path.basename(path)[len("nyando_flood_mask_"):-len(".tif")]
        if date_str not in raw_dates:
            skipped.append((date_str, "no raw scene"))
            continue
        with rasterio.open(path) as src:
            mask = src.read(1)
            if meta is None:
                meta = src.meta.copy()
            flood_frac = float((mask == 1).mean())
            if flood_frac > flood_frac_skip:
                skipped.append((date_str, f"degenerate mask ({flood_frac:.0%} flood)"))
                continue
            if water_sum is None:
                water_sum = np.zeros(mask.shape, dtype=np.uint32)
            water_sum += (mask == 1).astype(np.uint32)
            used.append((date_str, flood_frac))

    if water_sum is None or not used:
        raise SystemExit("No valid masks found — run the pipeline first.")

    n = len(used)
    permanent = (water_sum > majority * n).astype(np.uint8)

    out_path = os.path.join(project_dir, "permanent_water_mask.tif")
    meta.update(count=1, dtype=rasterio.uint8, nodata=0)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(permanent, 1)

    print(f"Used {n} scene dates:")
    for d, f in used:
        print(f"  {d}  flood={f:.1%}")
    print(f"Skipped {len(skipped)}:")
    for d, why in skipped:
        print(f"  {d}  ({why})")
    print(f"\nPermanent water covers {100 * permanent.mean():.2f}% of the scene.")
    print(f"Saved baseline to {out_path}")


if __name__ == "__main__":
    project_dir = sys.argv[1] if len(sys.argv) > 1 else "data/scenes"
    main(project_dir)
