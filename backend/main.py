import json
import os
import uuid
import asyncpg
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.security.api_key import APIKeyHeader


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/nyando_flood")
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
async def verify_api_key(api_key: str = Security(_api_key_header)):
    if not ADMIN_API_KEY:
        raise HTTPException(status_code=500, detail="ADMIN_API_KEY not configured on server.")
    if api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key.")
    return api_key

# Connection Pool
db_pool: asyncpg.Pool = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool
    try:
        db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
        print(f"[DB] Connected to PostgreSQL.")
    except Exception as e:
        print(f"[DB] WARNING: Could not connect to database: {e}")
        print("[DB] Running in MOCK mode — endpoints will return sample data.")
        db_pool = None
    yield
    if db_pool:
        await db_pool.close()
        print("[DB] Connection pool closed.")


app = FastAPI(title="Nyando Flood Monitor API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Models
class PipelineRunRequest(BaseModel):
    start_date: str
    end_date: str


# Endpoints

@app.get("/")
def read_root():
    return {
        "message": "Nyando Flood Monitor API v2.0 is running.",
        "db_connected": db_pool is not None,
        "docs": "/docs"
    }


@app.get("/api/v1/scenes")
async def get_scenes():
    """Returns list of all ingested scenes for the date selector dropdown."""
    if not db_pool:
        return [
            {"scene_id": "mock-1", "acquisition_date": "2024-04-17T00:00:00Z", "orbit_direction": "ASCENDING"},
            {"scene_id": "mock-2", "acquisition_date": "2024-04-29T00:00:00Z", "orbit_direction": "ASCENDING"},
            {"scene_id": "mock-3", "acquisition_date": "2024-05-06T00:00:00Z", "orbit_direction": "ASCENDING"},
        ]
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT scene_id, acquisition_date, orbit_direction FROM scenes ORDER BY acquisition_date ASC"
        )
        return [dict(r) for r in rows]


@app.get("/api/v1/flood_map")
async def get_flood_map(
    date: str = Query(None, description="ISO date string e.g. 2024-05-06"),
    bbox: str = Query(None, description="Bounding box: min_lon,min_lat,max_lon,max_lat")
):
    """Returns GeoJSON FeatureCollection of flood extent polygons for a given date and optional bounding box."""
    bbox_coords = None
    if bbox:
        try:
            parts = [float(x.strip()) for x in bbox.split(",")]
            if len(parts) != 4:
                raise ValueError
            bbox_coords = parts  # [min_lon, min_lat, max_lon, max_lat]
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid bbox format. Expected min_lon,min_lat,max_lon,max_lat (e.g. 34.8,-0.3,35.1,0.0)"
            )

    if not db_pool:
        return {
            "type": "FeatureCollection",
            "bbox_filter": bbox_coords,
            "features": [{
                "type": "Feature",
                "properties": {"acquisition_date": date or "2024-05-06", "area_sqkm": 78.4},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[34.82, -0.08], [34.95, -0.05], [35.05, -0.15],
                                     [34.95, -0.18], [34.85, -0.22], [34.82, -0.08]]]
                }
            }]
        }

    async with db_pool.acquire() as conn:
        conditions = ["poly.val >= 1"]
        params = []

        if date:
            params.append(date)
            conditions.append(f"s.acquisition_date::date = ${len(params)}")

        if bbox_coords:
            min_lon, min_lat, max_lon, max_lat = bbox_coords
            params.extend([min_lon, min_lat, max_lon, max_lat])
            p_start = len(params) - 3
            conditions.append(
                f"(s.scene_extent IS NULL OR ST_Intersects(s.scene_extent::geometry, ST_MakeEnvelope(${p_start}, ${p_start+1}, ${p_start+2}, ${p_start+3}, 4326)))"
            )

        where_clause = f"WHERE {' AND '.join(conditions)}"

        query = f"""
            SELECT
                s.acquisition_date,
                m.model_version,
                ST_AsGeoJSON(ST_Union(poly.geom))::json AS geojson
            FROM flood_masks m
            JOIN scenes s ON m.scene_id = s.scene_id
            CROSS JOIN LATERAL ST_DumpAsPolygons(m.flood_raster) AS poly
            {where_clause}
            GROUP BY s.acquisition_date, m.model_version
        """

        rows = await conn.fetch(query, *params)
        features = []
        for row in rows:
            if row["geojson"]:
                features.append({
                    "type": "Feature",
                    "properties": {
                        "acquisition_date": row["acquisition_date"].isoformat(),
                        "model_version": row["model_version"],
                    },
                    "geometry": json.loads(row["geojson"]) if isinstance(row["geojson"], str) else row["geojson"]
                })
        return {"type": "FeatureCollection", "features": features}


@app.get("/api/v1/analytics")
async def get_analytics():
    """Returns flood impact KPI data from the analytics view."""
    if not db_pool:
        return {
            "total_area_km2": 78.4,
            "affected_wards": 5,
            "peak_date": "2024-05-06",
            "model_version": "U-Net v1.0",
            "time_series": [
                {"date": "2024-04-17", "area_km2": 12.3},
                {"date": "2024-04-29", "area_km2": 45.6},
                {"date": "2024-05-06", "area_km2": 78.4},
            ]
        }
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT acquisition_date, inundated_area_sqkm, model_version FROM v_flood_impact_analytics"
        )
        if not rows:
            return {"total_area_km2": 0, "affected_wards": 0, "peak_date": "N/A",
                    "model_version": "N/A", "time_series": []}

        time_series = [
            {"date": r["acquisition_date"].strftime("%Y-%m-%d"), "area_km2": float(r["inundated_area_sqkm"])}
            for r in rows
        ]
        peak = max(time_series, key=lambda x: x["area_km2"])
        return {
            "total_area_km2": peak["area_km2"],
            "affected_wards": 5,  # Requires spatial join with ward boundaries — hardcoded for now
            "peak_date": peak["date"],
            "model_version": rows[0]["model_version"],
            "time_series": time_series,
        }

@app.get("/api/v1/analytics/wards")
async def get_ward_analytics():
    """Return per-ward flooded area (km²) using ST_Intersection."""
    if db_pool is None:
        return {"mock": True, "wards": [
            {"ward_name": "Ahero", "county": "Kisumu", "flooded_area_km2": 4.2},
            {"ward_name": "Nyando", "county": "Kisumu", "flooded_area_km2": 2.8},
        ]}
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT
                w.ward AS ward_name,
                w.county,
                ROUND(
                    CAST(
                        SUM(
                            ST_Area(
                                ST_Intersection(w.geom, poly.geom)::geography
                            ) / 1e6
                        ) AS numeric
                    ), 4
                ) AS flooded_area_km2
            FROM ward_boundaries w
            CROSS JOIN LATERAL (
                SELECT (ST_DumpAsPolygons(m.flood_raster)).geom
                FROM flood_masks m
            ) AS poly
            WHERE ST_Intersects(w.geom, poly.geom)
            GROUP BY w.ward, w.county
            ORDER BY flooded_area_km2 DESC;
        """)
        return {"wards": [dict(r) for r in rows]}

@app.get("/api/v1/wards")
async def get_wards():
    """Return GeoJSON FeatureCollection of Kisumu / Nyando ward boundaries."""
    if db_pool is None:
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"ward_name": "Ahero", "county": "Kisumu"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[34.88, -0.12], [34.96, -0.10], [34.95, -0.19], [34.87, -0.18], [34.88, -0.12]]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"ward_name": "Kabonyo/Kanyagwal", "county": "Kisumu"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[34.80, -0.18], [34.90, -0.17], [34.88, -0.25], [34.78, -0.24], [34.80, -0.18]]]
                    }
                }
            ]
        }
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT ward AS ward_name, county, ST_AsGeoJSON(geom)::json AS geojson
            FROM ward_boundaries
            WHERE county ILIKE '%Kisumu%'
        """)
        features = [
            {
                "type": "Feature",
                "properties": {"ward_name": r["ward_name"], "county": r["county"]},
                "geometry": json.loads(r["geojson"]) if isinstance(r["geojson"], str) else r["geojson"]
            }
            for r in rows if r["geojson"]
        ]
        return {"type": "FeatureCollection", "features": features}


@app.post("/api/v1/upload_aerial_image", dependencies=[Depends(verify_api_key)])
async def upload_aerial_image(
    file: UploadFile = File(...),
    acquisition_date: str = "",
    source: str = "",
    crs: str = "EPSG:4326"
):
    """Saves uploaded aerial/drone image to disk and records it in the database."""
    if not file.filename.endswith(('.tif', '.tiff', '.png', '.jpg')):
        raise HTTPException(status_code=400, detail="Invalid file type. Accepted: .tif, .tiff, .png, .jpg")

    file_id = str(uuid.uuid4())
    file_ext = os.path.splitext(file.filename)[1]
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_ext}")

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    if db_pool:
        async with db_pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO uploaded_images (image_id, file_path, processing_status, source_description)
                   VALUES ($1, $2, 'pending', $3)""",
                uuid.UUID(file_id), save_path, source or file.filename
            )

    return {
        "status": "success",
        "image_id": file_id,
        "filename": file.filename,
        "saved_to": save_path,
        "processing_status": "pending"
    }


@app.post("/api/v1/trigger_pipeline", dependencies=[Depends(verify_api_key)])
def trigger_pipeline(request: PipelineRunRequest):
    """Stub: triggers the CDSE pipeline. Run 02_copernicus_pipeline.py manually for now."""
    return {
        "status": "accepted",
        "message": f"Pipeline trigger acknowledged for {request.start_date} to {request.end_date}."
    }


@app.get("/api/v1/admin/queue", dependencies=[Depends(verify_api_key)])
async def get_processing_queue():
    """Return list of uploaded images and their processing status."""
    if db_pool is None:
        return {"mock": True, "queue": [
            {"image_id": "abc-123", "file_path": "uploads/scene_001.tif", "processing_status": "processed", "upload_timestamp": "2026-10-01T08:00:00Z"},
            {"image_id": "def-456", "file_path": "uploads/scene_002.tif", "processing_status": "pending",   "upload_timestamp": "2026-10-01T09:30:00Z"},
        ]}
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT image_id, file_path, processing_status, upload_timestamp
            FROM uploaded_images
            ORDER BY upload_timestamp DESC
            LIMIT 50;
        """)
        return {"queue": [dict(r) for r in rows]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
