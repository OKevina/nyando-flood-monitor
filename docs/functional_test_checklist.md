# Functional Test Checklist — Nyando Flood Monitor
**Issue #10 | Milestone M4: Testing & Documentation**

> This checklist covers manual end-to-end verification of all API endpoints, frontend dashboard views, and the image upload flow. Each test should be executed against a running local instance (`uvicorn backend.main:app --reload`) and recorded with a PASS / FAIL verdict.

---

## 1. API Endpoints

### 1.1 `GET /` — Health Check
| # | Test Scenario | Expected Result | Verdict |
|---|---|---|---|
| 1.1.1 | Request root endpoint with no parameters | HTTP 200; JSON body contains `"message"` and `"docs": "/docs"` | |
| 1.1.2 | Verify `db_connected` field is present | `true` when Postgres is running; `false` in mock mode | |

---

### 1.2 `GET /api/v1/scenes` — Scene List
| # | Test Scenario | Expected Result | Verdict |
|---|---|---|---|
| 1.2.1 | Request scenes with DB connected | HTTP 200; returns JSON array of scene objects | |
| 1.2.2 | Request scenes with no DB (mock mode) | HTTP 200; returns 3 mock scene objects with `scene_id` and `acquisition_date` | |
| 1.2.3 | Each item contains required fields | `scene_id`, `acquisition_date`, `orbit_direction` present | |

---

### 1.3 `GET /api/v1/flood_map` — Flood Map GeoJSON
| # | Test Scenario | Expected Result | Verdict |
|---|---|---|---|
| 1.3.1 | Request with no parameters | HTTP 200; GeoJSON `FeatureCollection` with at least 1 feature | |
| 1.3.2 | Request with valid `date=2024-05-06` | HTTP 200; feature `acquisition_date` matches requested date | |
| 1.3.3 | Request with valid `bbox=34.8,-0.3,35.1,0.0` | HTTP 200; `bbox_filter` echoed in mock response | |
| 1.3.4 | Request with malformed `bbox=bad-data` | HTTP 400; error detail describes correct format | |
| 1.3.5 | Request with incomplete bbox (only 2 values) | HTTP 400 | |
| 1.3.6 | GeoJSON geometry type is `Polygon` or `MultiPolygon` | Geometry field is valid GeoJSON | |

---

### 1.4 `GET /api/v1/analytics` — Flood Impact KPIs
| # | Test Scenario | Expected Result | Verdict |
|---|---|---|---|
| 1.4.1 | Request analytics endpoint | HTTP 200 | |
| 1.4.2 | Response contains `total_area_km2` as a number | Numeric value > 0 in mock mode | |
| 1.4.3 | Response contains `affected_wards` as an integer | Integer > 0 | |
| 1.4.4 | Response contains `time_series` as a non-empty array | Each item has `date` and `area_km2` | |
| 1.4.5 | Response contains `peak_date` as ISO date string | Valid date string | |

---

### 1.5 `GET /api/v1/analytics/wards` — Per-Ward Analytics
| # | Test Scenario | Expected Result | Verdict |
|---|---|---|---|
| 1.5.1 | Request ward analytics | HTTP 200 | |
| 1.5.2 | Response contains `wards` array | At least 1 ward object | |
| 1.5.3 | Each ward object has required fields | `ward_name`, `county`, `flooded_area_km2` present | |

---

### 1.6 `GET /api/v1/wards` — Ward Boundary GeoJSON
| # | Test Scenario | Expected Result | Verdict |
|---|---|---|---|
| 1.6.1 | Request ward boundaries | HTTP 200; GeoJSON `FeatureCollection` | |
| 1.6.2 | Each feature has geometry | `geometry.type` is `Polygon` | |
| 1.6.3 | Each feature has `ward_name` in properties | String field present | |

---

### 1.7 `POST /api/v1/upload_aerial_image` — Authenticated Upload
| # | Test Scenario | Expected Result | Verdict |
|---|---|---|---|
| 1.7.1 | Upload `.tif` file without API key | HTTP 403 Forbidden | |
| 1.7.2 | Upload `.tif` file with wrong API key | HTTP 403 Forbidden | |
| 1.7.3 | Upload valid `.tif` file with correct `X-API-Key` header | HTTP 200; `status: success`; `image_id` UUID present; `processing_status: pending` | |
| 1.7.4 | Upload valid `.png` file with correct API key | HTTP 200 | |
| 1.7.5 | Upload unsupported file type (e.g. `.exe`) | HTTP 400; error message describes accepted formats | |
| 1.7.6 | Verify uploaded file saved to `backend/uploads/` directory | File exists on disk with UUID filename | |

---

### 1.8 `POST /api/v1/trigger_pipeline` — Pipeline Trigger Stub
| # | Test Scenario | Expected Result | Verdict |
|---|---|---|---|
| 1.8.1 | POST without API key | HTTP 403 | |
| 1.8.2 | POST with valid API key and date range JSON body | HTTP 200; `status: accepted`; message contains the date range | |

---

### 1.9 `GET /api/v1/admin/queue` — Processing Queue (Authenticated)
| # | Test Scenario | Expected Result | Verdict |
|---|---|---|---|
| 1.9.1 | Request without API key | HTTP 403 | |
| 1.9.2 | Request with valid API key | HTTP 200; response contains `queue` array | |
| 1.9.3 | Each queue item has required fields | `image_id`, `file_path`, `processing_status`, `upload_timestamp` | |

---

## 2. Frontend Dashboard — Manual Browser Tests

> Open `frontend/dashboard.html` or the live dashboard URL in a browser.

### 2.1 Map View
| # | Test Scenario | Expected Result | Verdict |
|---|---|---|---|
| 2.1.1 | Dashboard loads without JS console errors | No `404` or `TypeError` errors in console | |
| 2.1.2 | Leaflet map renders centered on Nyando Basin | Map tiles visible; initial center approx. `(-0.15, 34.95)` | |
| 2.1.3 | Flood polygon overlay renders on map | Colored polygon(s) visible over the Nyando floodplain | |
| 2.1.4 | Date selector dropdown populates with scenes | Dropdown shows available acquisition dates | |
| 2.1.5 | Selecting a different date reloads flood layer | Map updates without full page refresh | |
| 2.1.6 | Ward boundary layer toggles on/off | Ward boundary polygons appear/disappear correctly | |

---

### 2.2 Analytics View
| # | Test Scenario | Expected Result | Verdict |
|---|---|---|---|
| 2.2.1 | KPI cards render: Total Inundated Area, Affected Wards, Peak Date | All 3 cards show non-zero values | |
| 2.2.2 | Time series chart renders | Line/bar chart displays flood area over time | |
| 2.2.3 | Per-ward breakdown table renders | Rows showing ward name and flooded area | |

---

### 2.3 Upload View (Admin)
| # | Test Scenario | Expected Result | Verdict |
|---|---|---|---|
| 2.3.1 | Upload form renders with file input field | Form visible with file input and API key input | |
| 2.3.2 | Submitting without an API key shows error | Error message displayed; upload blocked | |
| 2.3.3 | Submitting a `.tif` file with valid API key shows success | Success message with returned `image_id` | |
| 2.3.4 | Submitting an invalid file type shows error | Error message describes accepted formats | |

---

## 3. Automated Test Coverage Summary (pytest)

> Run: `pytest tests/ -v` from the repository root.

| Test Class | Tests | Expected |
|---|---|---|
| `TestRoot` | 3 | All PASS |
| `TestScenes` | 3 | All PASS |
| `TestFloodMap` | 6 | All PASS |
| `TestAnalytics` | 3 | All PASS |
| `TestWardAnalytics` | 3 | All PASS |
| `TestWards` | 3 | All PASS |
| `TestUploadAerialImage` | 5 | All PASS |
| `TestTriggerPipeline` | 2 | All PASS |
| `TestAdminQueue` | 2 | All PASS |
| **Total** | **30** | **30 PASS** |

---

## Sign-Off

| Item | Detail |
|---|---|
| **Tester** | Kevin Otieno |
| **Date** | |
| **API version tested** | v2.0.0 |
| **Test mode** | Mock (no-DB) + live browser |
| **Overall verdict** | |
