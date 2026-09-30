import os
import uuid
import asyncpg
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/nyando_flood")
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Connection Pool ──────────────────────────────────────────────────────────
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


# ── Pydantic Models ──────────────────────────────────────────────────────────
class PipelineRunRequest(BaseModel):
    start_date: str
    end_date: str


# ── Endpoints ────────────────────────────────────────────────────────────────

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
async def get_flood_map(date: str = Query(None, description="ISO date string e.g. 2024-05-06")):
    """Returns GeoJSON FeatureCollection of flood extent polygons for a given date."""
    if not db_pool:
        return {
            "type": "FeatureCollection",
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
        query = """
            SELECT
                s.acquisition_date,
                m.model_version,
                ST_AsGeoJSON(ST_Union(ST_DumpAsPolygons(m.flood_raster)::geometry))::json AS geojson
            FROM flood_masks m
            JOIN scenes s ON m.scene_id = s.scene_id
        """
        params = []
        if date:
            query += " WHERE s.acquisition_date::date = $1"
            params.append(date)
        query += " GROUP BY s.acquisition_date, m.model_version"

        rows = await conn.fetch(query, *params)
        features = []
        for row in rows:
            features.append({
                "type": "Feature",
                "properties": {
                    "acquisition_date": row["acquisition_date"].isoformat(),
                    "model_version": row["model_version"],
                },
                "geometry": row["geojson"]
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


@app.post("/api/v1/upload_aerial_image")
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


@app.post("/api/v1/trigger_pipeline")
def trigger_pipeline(request: PipelineRunRequest):
    """Stub: triggers the CDSE pipeline. Run 02_copernicus_pipeline.py manually for now."""
    return {
        "status": "accepted",
        "message": f"Pipeline trigger acknowledged for {request.start_date} to {request.end_date}."
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
