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

if not CLIENT_ID or not CLIENT_SECRET:
    raise ValueError("Missing CDSE API credentials in .env file!")
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_API_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
CATALOG_API_URL = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"

# Nyando Basin Bounding Box [lon_min, lat_min, lon_max, lat_max]
BBOX = [34.7, -0.3, 35.2, 0.2]

# ==========================================
# 1. AUTHENTICATION & CATALOG
# ==========================================
def get_cdse_token():
    payload = {"grant_type": "client_credentials", "client_id": CLIENT_ID, "client_secret": CLIENT_SECRET}
    response = requests.post(TOKEN_URL, data=payload)
    response.raise_for_status()
    return response.json()["access_token"]

def get_available_dates(token, bbox, start_date, end_date):
    print(f"\nQuerying CDSE Catalog for Sentinel-1 scenes over Nyando between {start_date} and {end_date}...")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "collections": ["sentinel-1-grd"],
        "bbox": bbox,
        "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z"
    }
    response = requests.post(CATALOG_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    
    features = response.json().get("features", [])
    dates = set()
    for f in features:
        dt = f["properties"]["datetime"]
        date_str = dt.split("T")[0]
        dates.add(date_str)
        
    sorted_dates = sorted(list(dates))
    print(f"Found {len(sorted_dates)} available pass dates: {', '.join(sorted_dates)}\n")
    return sorted_dates

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
                        
            if os.path.getsize(temp_path) < 100000:
                raise Exception(f"File too small ({os.path.getsize(temp_path)} bytes), likely corrupted.")
                
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
if __name__ == "__main__":
    PROJECT_DIR = r"C:\Users\KO\Desktop\Sem project"
    SHAPEFILE_PATH = r"D:\Documents\School\ICS Project 2\FL20240426KEN_SHP\FL20240426KEN_SHP\VIIRS_20240424_20240428_MaximumFloodWaterExtent_Kenya.shp"
    OUTPUT_TILES_DIR = os.path.join(PROJECT_DIR, "Nyando_Dataset")
    
    START_DATE = "2024-04-14"
    END_DATE = "2024-05-08"

    print("Authenticating with Copernicus CDSE...")
    token = get_cdse_token()
    print("Authentication successful!")
    
    # 1. Query available dates
    available_dates = get_available_dates(token, BBOX, START_DATE, END_DATE)
    
    # HARD SKIP the buggy 2024-04-24 scene that causes CDSE Process API to deadlock/timeout
    available_dates = [d for d in available_dates if d != '2024-04-24']
    
    # Stats tracking
    stats = []

    # 2. Process each date
    for date_str in available_dates:
        print(f"\n--- Processing Scene: {date_str} ---")
        
        raw_sar_path = os.path.join(PROJECT_DIR, f"nyando_sar_{date_str}_raw.tif")
        filtered_sar_path = os.path.join(PROJECT_DIR, f"nyando_sar_{date_str}_filtered.tif")
        mask_path = os.path.join(PROJECT_DIR, f"nyando_flood_mask_{date_str}.tif")
        
        # Download
        if not os.path.exists(raw_sar_path):
            download_rtc_sar_image(token, raw_sar_path, date_str)
        else:
            print(f"[{date_str}] Raw SAR already exists, skipping download.")
            
        # Filter
        apply_speckle_filter(raw_sar_path, filtered_sar_path, date_str)
        
        # Mask
        generate_mask(filtered_sar_path, SHAPEFILE_PATH, mask_path, date_str)
        
        # Tile & Filter Empty Patches
        total, retained, avg_pct = tile_and_filter(filtered_sar_path, mask_path, OUTPUT_TILES_DIR, date_str)
        
        stats.append({
            "date": date_str,
            "total": total,
            "retained": retained,
            "pct": avg_pct
        })
        
    # 3. Print Summary Table
    print("\n" + "="*60)
    print("NYANDO FLOOD EVENT SUMMARY TABLE")
    print("="*60)
    print(f"{'Date':<15} | {'Total Patches':<15} | {'Retained (>0.5%)':<18} | {'Avg Flood %':<15}")
    print("-" * 60)
    total_dataset_patches = 0
    for s in stats:
        print(f"{s['date']:<15} | {s['total']:<15} | {s['retained']:<18} | {s['pct']:.2f}%")
        total_dataset_patches += s['retained']
    print("="*60)
    print(f"Total Dataset Size: {total_dataset_patches} training patches.")
    print("="*60)
