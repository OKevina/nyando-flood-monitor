"""
Build per-orbit temporal median VV (dB) baselines across filtered scenes.

Sentinel-1 passes over Nyando come from (at least) two relative orbits with
different incidence angles, which shifts VV backscatter by ~1.5-2 dB and
breaks any single-scene threshold. A single mixed-orbit median fails the
same way, so we build one median per relative orbit (from orbit_map.json,
produced by 08_build_orbit_map.py) and label each scene against its own
orbit's history:

    flood = (vv_db < median_db - DELTA_DB) & (vv_db < CEIL_DB)

Median is computed on a ::STRIDE subsample and nearest-upsampled.

Output: <project_dir>/temporal_median_db_<orbit>.tif (float32, full res, dB).
"""

import os
import sys
import glob
import json

import rasterio
os.environ.setdefault("PROJ_LIB", os.path.join(os.path.dirname(rasterio.__file__), "proj_data"))
os.environ.setdefault("GDAL_DATA", os.path.join(os.path.dirname(rasterio.__file__), "gdal_data"))

import numpy as np

STRIDE = 4


def scene_date(path):
    return os.path.basename(path)[len("nyando_sar_"):-len("_filtered.tif")]


def main(project_dir="data/scenes"):
    orbit_path = os.path.join(project_dir, "orbit_map.json")
    orbit_map = json.load(open(orbit_path)) if os.path.exists(orbit_path) else {}

    files = sorted(glob.glob(os.path.join(project_dir, "nyando_sar_*_filtered.tif")))
    groups = {}
    for path in files:
        orbit = orbit_map.get(scene_date(path), "unknown")
        groups.setdefault(orbit, []).append(path)

    for orbit, paths in sorted(groups.items()):
        stack = []
        meta = None
        for path in paths:
            with rasterio.open(path) as src:
                vv = src.read(1).astype(np.float32)
                if meta is None:
                    meta = src.meta.copy()
                if vv.shape != (meta["height"], meta["width"]):
                    continue
                vv_db = 10.0 * np.log10(np.where(vv > 0, vv, np.nan))
                stack.append(vv_db[::STRIDE, ::STRIDE])

        if len(stack) < 3:
            print(f"orbit {orbit}: only {len(stack)} scenes — skipping")
            continue

        arr = np.stack(stack)
        with np.errstate(invalid="ignore"):
            med = np.nanmedian(arr, axis=0).astype(np.float32)
        del arr, stack

        med_full = np.repeat(np.repeat(med, STRIDE, axis=0), STRIDE, axis=1)
        med_full = med_full[: meta["height"], : meta["width"]]
        med_full = np.where(np.isnan(med_full), -9999.0, med_full).astype(np.float32)

        out_path = os.path.join(project_dir, f"temporal_median_db_{orbit}.tif")
        meta.update(count=1, dtype="float32", nodata=-9999.0, compress="deflate")
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(med_full, 1)

        valid = med_full[med_full > -9999]
        print(f"orbit {orbit}: {len(paths)} scenes -> median dB p50 "
              f"{np.percentile(valid, 50):.1f} -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/scenes")
