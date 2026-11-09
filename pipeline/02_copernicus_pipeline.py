import os
import requests
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
import geopandas as gpd
import numpy as np
from scipy.ndimage import uniform_filter
from skimage.filters import threshold_otsu
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# CONFIGURATION
# ==========================================
from dotenv import load_dotenv
load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_API_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
CATALOG_API_URL = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"

# Nyando Basin Bounding Box [lon_min, lat_min, lon_max, lat_max]
BBOX = [34.7, -0.3, 35.2, 0.2]


def _check_credentials():
    if not CLIENT_ID or not CLIENT_SECRET:
        raise ValueError("Missing CDSE API credentials in .env file!")

# ==========================================
# 1. AUTHENTICATION & CATALOG
# ==========================================
def get_cdse_token():
    _check_credentials()
    payload = {"grant_type": "client_credentials", "client_id": CLIENT_ID, "client_secret": CLIENT_SECRET}
    response = requests.post(TOKEN_URL, data=payload)
    response.raise_for_status()
    return response.json()["access_token"]

def get_available_dates(token, bbox, start_date, end_date, min_coverage=0.5):
    """Return pass dates whose scene footprint covers at least `min_coverage`
    of the AOI. The catalog bbox query returns scenes that only clip the
    corner of the bbox — their swaths miss the basin and download as
    all-nodata rasters."""
    print(f"\nQuerying CDSE Catalog for Sentinel-1 scenes over Nyando between {start_date} and {end_date}...")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "collections": ["sentinel-1-grd"],
        "bbox": bbox,
        "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
        "limit": 100
    }
    response = requests.post(CATALOG_API_URL, headers=headers, json=payload)
    response.raise_for_status()

    from shapely.geometry import box, shape
    aoi = box(bbox[0], bbox[1], bbox[2], bbox[3])

    features = response.json().get("features", [])
    best_cover = {}
    for f in features:
        date_str = f["properties"]["datetime"].split("T")[0]
        try:
            cov = aoi.intersection(shape(f["geometry"])).area / aoi.area
        except Exception:
            cov = 0.0
        best_cover[date_str] = max(best_cover.get(date_str, 0.0), cov)

    sorted_dates = sorted(d for d, c in best_cover.items() if c >= min_coverage)
    skipped = sorted(d for d, c in best_cover.items() if c < min_coverage)
    print(f"Found {len(sorted_dates)} usable pass dates: {', '.join(sorted_dates)}")
    if skipped:
        print(f"Skipped {len(skipped)} dates with <{min_coverage:.0%} AOI coverage: {', '.join(skipped)}\n")
    return sorted_dates


def scene_has_valid_data(tif_path, min_valid_frac=0.01):
    """True if the scene's VV band has at least `min_valid_frac` non-zero pixels.
    Catches both swath-miss dates and transient empty RTC products."""
    import rasterio
    with rasterio.open(tif_path) as src:
        vv = src.read(1)
    return float((vv > 0).mean()) >= min_valid_frac

# ==========================================
# 2. CDSE PROCESS API (RTC DATA)
# ==========================================
def download_rtc_sar_image(token, output_path, date_str):
    print(f"[{date_str}] Downloading RTC Sentinel-1 image from Copernicus API...")
    
    evalscript = """
    //VERSION=3
    function setup() {
      return {
        input: [{"bands": ["VV", "VH", "dataMask"]}],
        output: { bands: 2, sampleType: "FLOAT32" }
      };
    }
    function evaluatePixel(sample) {
      if (sample.dataMask == 0) return [0, 0];
      return [sample.VV, sample.VH];
    }
    """
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "image/tiff"
    }
    
    payload = {
        "input": {
            "bounds": {
                "bbox": BBOX,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
            },
            "data": [{
                "type": "sentinel-1-grd",
                "dataFilter": {
                    "timeRange": {
                        "from": f"{date_str}T00:00:00Z",
                        "to": f"{date_str}T23:59:59Z"
                    }
                },
                "processing": {
                    "backCoeff": "SIGMA0_ELLIPSOID",
                    "orthorectify": True,
                    "demInstance": "COPERNICUS_30"
                }
            }]
        },
        "evalscript": evalscript,
        "output": {
            "width": 2500,
            "height": 2500,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]
        }
    }

    import time
    import os
    max_retries = 15
    for attempt in range(max_retries):
        response = None
        try:
            # (connect_timeout, read_timeout)
            response = requests.post(PROCESS_API_URL, headers=headers, json=payload, timeout=(60, 600), stream=True)
            if response.status_code == 401:
                print(f"[{date_str}] Token expired on attempt {attempt+1}. Fetching a new one...")
                token = get_cdse_token()
                headers["Authorization"] = f"Bearer {token}"
                continue
                
            response.raise_for_status()
            
            temp_path = output_path + ".tmp"
            with open(temp_path, 'wb') as f:
                # 1MB chunks for stability
                for chunk in response.iter_content(chunk_size=1048576):
                    if chunk:
                        f.write(chunk)
                        
            # Validate by actually opening the file — a valid but empty
            # (all-nodata) scene compresses to a tiny TIFF and must not be
            # rejected by a size heuristic.
            try:
                with rasterio.open(temp_path) as _src:
                    _src.read(1)
            except Exception:
                raise Exception("Downloaded file is not a readable GeoTIFF.")
                
            break
        except Exception as e:
            print(f"[{date_str}] Download attempt {attempt + 1} failed: {e}")
            try:
                if response is not None and hasattr(response, 'text') and response.text:
                    print(f"API Error details: {response.text[:200]}")
            except Exception:
                pass
            if attempt == max_retries - 1:
                raise e
            time.sleep(10 + attempt * 5)
        
    with rasterio.open(temp_path) as src:
        data = src.read()
        transform = from_bounds(*BBOX, src.width, src.height)
        meta = src.meta.copy()
        meta.update(crs="EPSG:4326", transform=transform)
        
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(data)
        
    os.remove(temp_path)
    print(f"[{date_str}] Saved georeferenced SAR image to {output_path}")

# ==========================================
# 3. LOCAL PREPROCESSING: LEE SPECKLE FILTER
# ==========================================
def lee_filter(img, size=5):
    img_mean = uniform_filter(img, size)
    img_sqr_mean = uniform_filter(img**2, size)
    img_variance = img_sqr_mean - img_mean**2
    overall_variance = np.var(img)
    img_weights = img_variance / (img_variance + overall_variance + 1e-8)
    return img_mean + img_weights * (img - img_mean)

def apply_speckle_filter(input_tiff, output_tiff, date_str):
    print(f"[{date_str}] Applying Lee speckle filter locally...")
    with rasterio.open(input_tiff) as src:
        meta = src.meta
        vv = src.read(1)
        vh = src.read(2)

    vv_filtered = lee_filter(vv, size=5)
    vh_filtered = lee_filter(vh, size=5)

    with rasterio.open(output_tiff, 'w', **meta) as dst:
        dst.write(vv_filtered, 1)
        dst.write(vh_filtered, 2)

# ==========================================
# 4. DYNAMIC MASK GENERATION
# ==========================================
def generate_mask(filtered_sar_path, shapefile_path, output_mask, date_str):
    print(f"[{date_str}] Generating binary flood mask...")
    with rasterio.open(filtered_sar_path) as src:
        meta = src.meta.copy()
        transform = src.transform
        shape = (src.height, src.width)
        sar_crs = src.crs
        vv_band = src.read(1)

    valid_vv = vv_band[vv_band > 0]
    if len(valid_vv) == 0:
        print(f"[{date_str}] Warning: No valid SAR data. Generating blank mask.")
        refined_mask = np.zeros(shape, dtype=np.uint8)
    else:
        thresh = threshold_otsu(valid_vv)
        otsu_mask = (vv_band < thresh).astype(np.uint8)
        
        # Check if date falls in the waterlogged interlude (April 24-28)
        if "2024-04-24" <= date_str <= "2024-04-28":
            print(f"[{date_str}]  -> Using Hybrid Mask (Otsu + VIIRS spatial prior)")
            gdf = gpd.read_file(shapefile_path)
            if gdf.crs != sar_crs:
                gdf = gdf.to_crs(sar_crs)
                
            geometries = []
            for geom in gdf.geometry:
                if geom is not None:
                    if not geom.is_valid:
                        geom = geom.buffer(0)
                    geometries.append(geom)
                    
            if geometries:
                viirs_mask = rasterize(geometries, out_shape=shape, transform=transform, fill=0, default_value=1, dtype=np.uint8)
            else:
                viirs_mask = np.zeros(shape, dtype=np.uint8)
                
            refined_mask = otsu_mask & viirs_mask
        else:
            print(f"[{date_str}]  -> Using Pure Otsu Mask")
            refined_mask = otsu_mask

    meta.update(count=1, dtype=rasterio.uint8, nodata=0)
    with rasterio.open(output_mask, 'w', **meta) as dst:
        dst.write(refined_mask, 1)

# ==========================================
# 5. TILING & FILTERING
# ==========================================
def tile_and_filter(sar_tiff, mask_tiff, output_dir, date_str, patch_size=256, overlap=50):
    print(f"[{date_str}] Tiling and filtering patches (< 0.5% flood discarded)...")
    os.makedirs(os.path.join(output_dir, 'S1Hand'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'LabelHand'), exist_ok=True)

    stride = patch_size - overlap
    
    total_scanned = 0
    retained_patches = 0
    total_flood_pixels = 0

    with rasterio.open(sar_tiff) as src_sar, rasterio.open(mask_tiff) as src_mask:
        sar_data = src_sar.read()
        mask_data = src_mask.read(1)
        
        h, w = sar_data.shape[1], sar_data.shape[2]
        
        for top in range(0, h - patch_size + 1, stride):
            for left in range(0, w - patch_size + 1, stride):
                total_scanned += 1
                mask_patch = mask_data[top:top+patch_size, left:left+patch_size]
                
                # Filter logic
                flood_pixels = np.count_nonzero(mask_patch == 1)
                flood_pct = (flood_pixels / (patch_size*patch_size)) * 100
                
                if flood_pct < 0.5:
                    continue
                    
                retained_patches += 1
                total_flood_pixels += flood_pixels
                
                sar_patch = sar_data[:, top:top+patch_size, left:left+patch_size]
                
                # File paths with date embedded
                sar_patch_path = os.path.join(output_dir, 'S1Hand', f'Nyando_{date_str}_{retained_patches}_S1Hand.tif')
                mask_patch_path = os.path.join(output_dir, 'LabelHand', f'Nyando_{date_str}_{retained_patches}_LabelHand.tif')
                
                # Write SAR patch
                sar_meta = src_sar.meta.copy()
                sar_meta.update(width=patch_size, height=patch_size, transform=src_sar.window_transform(((top, top+patch_size), (left, left+patch_size))))
                with rasterio.open(sar_patch_path, 'w', **sar_meta) as dst:
                    dst.write(sar_patch)
                    
                # Write Mask patch
                mask_meta = src_mask.meta.copy()
                mask_meta.update(width=patch_size, height=patch_size, transform=src_mask.window_transform(((top, top+patch_size), (left, left+patch_size))))
                with rasterio.open(mask_patch_path, 'w', **mask_meta) as dst:
                    dst.write(mask_patch, 1)
                    
    avg_flood_pct = (total_flood_pixels / (retained_patches * patch_size * patch_size) * 100) if retained_patches > 0 else 0
    return total_scanned, retained_patches, avg_flood_pct

# ==========================================
# PIPELINE EXECUTION
# ==========================================
def run_pipeline_for_date(token, date_str, project_dir, shapefile_path, output_tiles_dir):
    """Process a single Sentinel-1 scene date end-to-end: download, filter, mask, tile."""
    print("\n--- Processing Scene: " + date_str + " ---")

    raw_sar_path = os.path.join(project_dir, "nyando_sar_" + date_str + "_raw.tif")
    filtered_sar_path = os.path.join(project_dir, "nyando_sar_" + date_str + "_filtered.tif")
    mask_path = os.path.join(project_dir, "nyando_flood_mask_" + date_str + ".tif")

    # Download
    if not os.path.exists(raw_sar_path):
        download_rtc_sar_image(token, raw_sar_path, date_str)
    else:
        print("[" + date_str + "] Raw SAR already exists, skipping download.")

    # Skip scenes that are (nearly) all nodata — swath misses or empty RTC products
    if not scene_has_valid_data(raw_sar_path):
        print("[" + date_str + "] Skipping — scene has no valid SAR data.")
        return {"date": date_str, "total": 0, "retained": 0, "pct": 0.0}

    # Filter
    apply_speckle_filter(raw_sar_path, filtered_sar_path, date_str)

    # Mask
    generate_mask(filtered_sar_path, shapefile_path, mask_path, date_str)

    # Tile & Filter Empty Patches
    total, retained, avg_pct = tile_and_filter(filtered_sar_path, mask_path, output_tiles_dir, date_str)

    return {
        "date": date_str,
        "total": total,
        "retained": retained,
        "pct": avg_pct,
    }


def run_pipeline(start_date, end_date, project_dir, shapefile_path, output_tiles_dir, bbox=None, skip_dates=None):
    """Run the full CDSE pipeline for a date range."""
    if bbox is None:
        bbox = BBOX
    if skip_dates is None:
        skip_dates = []

    print("Authenticating with Copernicus CDSE...")
    token = get_cdse_token()
    print("Authentication successful!")

    # 1. Query available dates
    available_dates = get_available_dates(token, bbox, start_date, end_date)

    # Filter out skipped dates
    available_dates = [d for d in available_dates if d not in skip_dates]

    # Stats tracking
    stats = []

    # 2. Process each date
    for date_str in available_dates:
        result = run_pipeline_for_date(token, date_str, project_dir, shapefile_path, output_tiles_dir)
        stats.append(result)

    # 3. Print Summary Table
    print("\n" + "=" * 60)
    print("NYANDO FLOOD EVENT SUMMARY TABLE")
    print("=" * 60)
    print("{:<15} | {:<15} | {:<18} | {:<15}".format("Date", "Total Patches", "Retained (>0.5%)", "Avg Flood %"))
    print("-" * 60)
    total_dataset_patches = 0
    for s in stats:
        print("{:<15} | {:<15} | {:<18} | {:.2f}%".format(s["date"], s["total"], s["retained"], s["pct"]))
        total_dataset_patches += s["retained"]
    print("=" * 60)
    print("Total Dataset Size: " + str(total_dataset_patches) + " training patches.")
    print("=" * 60)
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CDSE Sentinel-1 pipeline for Nyando Basin")
    parser.add_argument("--start-date", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--project-dir", required=True, help="Output directory for SAR/mask tiles")
    parser.add_argument("--shapefile-path", required=True, help="Path to VIIRS flood extent shapefile")
    parser.add_argument("--output-tiles-dir", required=True, help="Directory for tiled training patches")
    parser.add_argument("--skip-dates", nargs="*", default=[], help="Dates to skip (e.g. 2024-04-24)")
    args = parser.parse_args()

    run_pipeline(
        start_date=args.start_date,
        end_date=args.end_date,
        project_dir=args.project_dir,
        shapefile_path=args.shapefile_path,
        output_tiles_dir=args.output_tiles_dir,
        skip_dates=args.skip_dates,
    )
