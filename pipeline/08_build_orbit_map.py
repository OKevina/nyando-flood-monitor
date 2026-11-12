"""One-off: map each usable pass date to its Sentinel-1 orbit state
(ascending/descending) using the CDSE catalog. Saves
<data_dir>/orbit_map.json as {"YYYY-MM-DD": "ascending"|"descending"}.

For dates with multiple products (S1A/S1C, multiple bursts) the feature
with the largest AOI footprint coverage wins — same criterion the
pipeline effectively uses.
"""

import os
import sys
import json
import importlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from dotenv import load_dotenv
from shapely.geometry import box, shape

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

pipe = importlib.import_module("pipeline.02_copernicus_pipeline")


def main():
    token = pipe.get_cdse_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    aoi = box(*pipe.BBOX)
    orbit_map = {}

    # 6-month chunks keep each catalog response under the 100-feature limit.
    chunks = [
        ("2023-01-01", "2023-06-30"), ("2023-07-01", "2023-12-31"),
        ("2024-01-01", "2024-06-30"), ("2024-07-01", "2024-12-31"),
        ("2025-01-01", "2025-06-30"), ("2025-07-01", "2025-12-31"),
    ]
    for start, end in chunks:
        payload = {
            "collections": ["sentinel-1-grd"],
            "bbox": pipe.BBOX,
            "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
            "limit": 100,
        }
        r = requests.post(pipe.CATALOG_API_URL, headers=headers, json=payload, timeout=90)
        r.raise_for_status()
        best = {}
        for f in r.json().get("features", []):
            date_str = f["properties"]["datetime"].split("T")[0]
            orbit = f["properties"].get("sat:relative_orbit") or f["properties"].get("sat:orbit_state", "unknown")
            try:
                cov = aoi.intersection(shape(f["geometry"])).area / aoi.area
            except Exception:
                cov = 0.0
            if cov >= 0.5 and cov > best.get(date_str, (0, None))[0]:
                best[date_str] = (cov, orbit)
        for d, (cov, orbit) in best.items():
            orbit_map[d] = orbit
        print(f"{start}..{end}: {len(best)} dates mapped")

    out = os.path.join("data", "scenes", "orbit_map.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(orbit_map, fh, indent=1, sort_keys=True)
    asc = sum(1 for v in orbit_map.values() if v == "ascending")
    print(f"Saved {len(orbit_map)} dates ({asc} ascending, {len(orbit_map)-asc} descending) -> {out}")


if __name__ == "__main__":
    main()
