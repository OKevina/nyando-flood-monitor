# Nyando River Basin Flood Monitoring System

A web-based computer vision application for near-real-time flood extent mapping in the Nyando River Basin, Kisumu County, Kenya. The system uses Sentinel-1 SAR (Synthetic Aperture Radar) imagery and a U-Net deep learning model to detect and map flooded areas.

## Project Overview

The system automates flood monitoring by:
1. Fetching Sentinel-1 SAR imagery from the Copernicus Data Space Ecosystem (CDSE) API
2. Applying Lee speckle filtering and Otsu/VIIRS hybrid weak label generation
3. Training a U-Net model via two-phase transfer learning (Sen1Floods11 base → Nyando fine-tune)
4. Serving flood maps through a FastAPI backend with PostGIS spatial queries
5. Visualising results on a React.js dashboard with an interactive Leaflet map

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Pipeline | Python, PyTorch, Rasterio, GeoPandas, scikit-image |
| Backend | FastAPI, PostgreSQL + PostGIS, asyncpg |
| Frontend | React 18, Tailwind CSS, Leaflet.js |
| Data Source | Copernicus CDSE (Sentinel-1 RTC, April–May 2024) |

## Project Structure

```
nyando-flood-monitor/
├── pipeline/           # ML training and data acquisition scripts
│   ├── 01_base_training.py         # Phase 1: Sen1Floods11 U-Net training (Colab)
│   ├── 02_copernicus_pipeline.py   # CDSE data fetch, Lee filter, mask generation
│   ├── 03_prepare_dataset.py       # 60/20/20 train/val/test CSV split generator
│   └── 04_finetune_nyando.py       # Phase 2: Transfer learning fine-tuning
├── backend/
│   ├── main.py         # FastAPI REST API
│   ├── init_db.sql     # PostgreSQL + PostGIS schema DDL
│   └── .env.example    # Database credentials template
├── frontend/
│   └── dashboard.html  # React 18 SPA (Map, Analytics, Upload tabs)
├── docs/
│   └── diagrams/       # Architecture and methodology diagrams
├── notebooks/
│   └── 04_finetune_nyando_colab.ipynb  # Colab version of fine-tuning
├── data/               # gitignored — GeoTIFFs and tiled patches
├── models/             # gitignored — .pth model weights
├── .env.example        # CDSE credentials template
└── requirements.txt    # Python dependencies
```

## Setup

### Prerequisites
- Python 3.9+
- PostgreSQL 14+ with PostGIS extension
- A Copernicus Data Space Ecosystem account ([register here](https://dataspace.copernicus.eu/))

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/nyando-flood-monitor.git
cd nyando-flood-monitor

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
copy .env.example .env
# Edit .env with your CDSE credentials

copy backend\.env.example backend\.env
# Edit backend\.env with your PostgreSQL connection string

# 5. Initialise the database
psql -U postgres -d nyando_flood -f backend/init_db.sql

# 6. Run the backend
cd backend
python main.py
# API available at http://localhost:8000
# Swagger docs at http://localhost:8000/docs

# 7. Open the frontend
# Serve via a local HTTP server (required — fetch() is blocked on file://)
cd frontend
python -m http.server 3000
# Open http://localhost:3000/dashboard.html
```

## Model Performance (Phase 1 — Sen1Floods11 Base)

| Metric | Score |
|--------|-------|
| IoU | 0.6382 |
| F1-Score | 0.7639 |
| Precision | 0.8303 |
| Recall | 0.7289 |

## Author

**Otieno Kevin Ashly** — Admission No. 150487  
Supervisor: Prof. Vincent Omwenga  
ICS Project II Capstone — Strathmore University, 2026
