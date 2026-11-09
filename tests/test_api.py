"""
Smoke tests for the Nyando Flood Monitor API (Issue #21).

These tests run against the FastAPI TestClient in mock/no-DB mode
(db_pool = None), so they execute without any PostgreSQL connection.
Every public GET endpoint and the upload POST are covered.
"""
import io
import os
import sys

import pytest
from fastapi.testclient import TestClient

# Ensure the backend package is importable when running from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

# Force no-DB mode so tests never need a live Postgres instance
os.environ.setdefault("DATABASE_URL", "postgresql://none:none@localhost:5432/none")
os.environ.setdefault("ADMIN_API_KEY", "test-secret-key")

from main import app  # noqa: E402  (import after env setup)

client = TestClient(app, raise_server_exceptions=True)
ADMIN_HEADERS = {"X-API-Key": "test-secret-key"}


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------
class TestRoot:
    def test_status_200(self):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_response_has_message(self):
        data = client.get("/").json()
        assert "message" in data
        assert "Nyando Flood Monitor" in data["message"]

    def test_docs_key_present(self):
        data = client.get("/").json()
        assert data.get("docs") == "/docs"


# ---------------------------------------------------------------------------
# GET /api/v1/scenes
# ---------------------------------------------------------------------------
class TestScenes:
    def test_status_200(self):
        resp = client.get("/api/v1/scenes")
        assert resp.status_code == 200

    def test_returns_list(self):
        data = client.get("/api/v1/scenes").json()
        assert isinstance(data, list)

    def test_mock_items_have_scene_id(self):
        data = client.get("/api/v1/scenes").json()
        assert len(data) > 0
        for item in data:
            assert "scene_id" in item
            assert "acquisition_date" in item


# ---------------------------------------------------------------------------
# GET /api/v1/flood_map
# ---------------------------------------------------------------------------
class TestFloodMap:
    def test_status_200_no_params(self):
        resp = client.get("/api/v1/flood_map")
        assert resp.status_code == 200

    def test_geojson_structure(self):
        data = client.get("/api/v1/flood_map").json()
        assert data["type"] == "FeatureCollection"
        assert isinstance(data["features"], list)

    def test_with_date_param(self):
        resp = client.get("/api/v1/flood_map", params={"date": "2024-05-06"})
        assert resp.status_code == 200

    def test_with_valid_bbox(self):
        resp = client.get("/api/v1/flood_map",
                          params={"bbox": "34.8,-0.3,35.1,0.0"})
        assert resp.status_code == 200

    def test_invalid_bbox_returns_400(self):
        resp = client.get("/api/v1/flood_map", params={"bbox": "bad-bbox"})
        assert resp.status_code == 400

    def test_bbox_wrong_part_count_returns_400(self):
        resp = client.get("/api/v1/flood_map", params={"bbox": "34.8,-0.3"})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# GET /api/v1/analytics
# ---------------------------------------------------------------------------
class TestAnalytics:
    def test_status_200(self):
        resp = client.get("/api/v1/analytics")
        assert resp.status_code == 200

    def test_has_total_area(self):
        data = client.get("/api/v1/analytics").json()
        assert "total_area_km2" in data
        assert isinstance(data["total_area_km2"], (int, float))

    def test_has_time_series(self):
        data = client.get("/api/v1/analytics").json()
        assert "time_series" in data
        assert isinstance(data["time_series"], list)
        assert len(data["time_series"]) > 0


# ---------------------------------------------------------------------------
# GET /api/v1/analytics/wards
# ---------------------------------------------------------------------------
class TestWardAnalytics:
    def test_status_200(self):
        resp = client.get("/api/v1/analytics/wards")
        assert resp.status_code == 200

    def test_has_wards_key(self):
        data = client.get("/api/v1/analytics/wards").json()
        assert "wards" in data
        assert isinstance(data["wards"], list)

    def test_ward_items_have_name_and_area(self):
        wards = client.get("/api/v1/analytics/wards").json()["wards"]
        assert len(wards) > 0
        for w in wards:
            assert "ward_name" in w
            assert "flooded_area_km2" in w


# ---------------------------------------------------------------------------
# GET /api/v1/wards
# ---------------------------------------------------------------------------
class TestWards:
    def test_status_200(self):
        resp = client.get("/api/v1/wards")
        assert resp.status_code == 200

    def test_geojson_feature_collection(self):
        data = client.get("/api/v1/wards").json()
        assert data["type"] == "FeatureCollection"
        assert isinstance(data["features"], list)

    def test_features_have_ward_name(self):
        features = client.get("/api/v1/wards").json()["features"]
        assert len(features) > 0
        for f in features:
            assert "ward_name" in f["properties"]


# ---------------------------------------------------------------------------
# POST /api/v1/upload_aerial_image (authenticated)
# ---------------------------------------------------------------------------
class TestUploadAerialImage:
    def _dummy_tif(self, name="test_scene.tif"):
        return {"file": (name, io.BytesIO(b"FAKE_TIFF_BYTES"), "image/tiff")}

    def test_missing_api_key_returns_403(self):
        resp = client.post(
            "/api/v1/upload_aerial_image",
            files=self._dummy_tif(),
        )
        assert resp.status_code == 403

    def test_wrong_api_key_returns_403(self):
        resp = client.post(
            "/api/v1/upload_aerial_image",
            files=self._dummy_tif(),
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 403

    def test_valid_tif_upload_succeeds(self):
        resp = client.post(
            "/api/v1/upload_aerial_image",
            files=self._dummy_tif("scene_001.tif"),
            headers=ADMIN_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert "image_id" in data
        assert data["processing_status"] == "pending"

    def test_invalid_file_type_returns_400(self):
        resp = client.post(
            "/api/v1/upload_aerial_image",
            files={"file": ("malware.exe", io.BytesIO(b"bad"), "application/octet-stream")},
            headers=ADMIN_HEADERS,
        )
        assert resp.status_code == 400

    def test_png_upload_accepted(self):
        resp = client.post(
            "/api/v1/upload_aerial_image",
            files={"file": ("aerial.png", io.BytesIO(b"PNG_BYTES"), "image/png")},
            headers=ADMIN_HEADERS,
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /api/v1/trigger_pipeline (authenticated stub)
# ---------------------------------------------------------------------------
class TestTriggerPipeline:
    def test_requires_auth(self):
        resp = client.post("/api/v1/trigger_pipeline",
                           json={"start_date": "2024-04-01", "end_date": "2024-05-01"})
        assert resp.status_code == 403

    def test_stub_returns_accepted(self):
        resp = client.post(
            "/api/v1/trigger_pipeline",
            json={"start_date": "2024-04-01", "end_date": "2024-05-01"},
            headers=ADMIN_HEADERS,
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "accepted"


# ---------------------------------------------------------------------------
# GET /api/v1/admin/queue (authenticated)
# ---------------------------------------------------------------------------
class TestAdminQueue:
    def test_requires_auth(self):
        resp = client.get("/api/v1/admin/queue")
        assert resp.status_code == 403

    def test_returns_queue_list(self):
        resp = client.get("/api/v1/admin/queue", headers=ADMIN_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert "queue" in data
        assert isinstance(data["queue"], list)
