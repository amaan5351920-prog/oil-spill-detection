# 🛢️ Oil Spill Detection & Vessel Attribution System

An automated end-to-end pipeline for detecting marine oil spills from satellite imagery, modelling slick drift, and attributing spills to responsible vessels using AIS data.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Pipeline Orchestrator                      │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│  │ 1.DETECT │→│ 2.ANALYSE│→│ 3.DRIFT  │→│ 4.ATTRIBUTION│  │
│  │          │  │          │  │          │  │             │  │
│  │ U-Net    │  │ Geometric│  │Lagrangian│  │ AIS Scorer  │  │
│  │ Segment  │  │ Properties│ │ Particle  │  │ Spatio-     │  │
│  │ SAR/EO   │  │ Age Est. │  │ Tracking │  │ Temporal    │  │
│  └──────────┘  └──────────┘  └──────────┘  └────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Web Dashboard (Flask + Leaflet)          │   │
│  │  Map · Vessel Leaderboard · Geometry · Score Charts  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
oil-spill-detection/
├── main.py                          # CLI entry point
├── requirements.txt
├── configs/default.yaml
├── src/
│   ├── detection/
│   │   ├── oil_detector.py          # U-Net SAR image segmentation
│   │   └── geometric_analyzer.py    # Shape, age, fractal dimension
│   ├── drift/
│   │   ├── drift_model.py           # Lagrangian particle tracking
│   │   └── oceanographic_data.py    # Wind, current, wave fields
│   ├── attribution/
│   │   ├── ais_processor.py         # AIS data loading & filtering
│   │   └── vessel_scorer.py         # Multi-factor vessel ranking
│   ├── pipeline/
│   │   └── orchestrator.py          # End-to-end pipeline
│   └── web/
│       └── app.py                   # Flask web application
├── templates/
│   └── dashboard.html               # Web dashboard UI
└── tests/
    └── test_pipeline.py             # Unit tests
```

## 🚀 Quick Start

### Installation

```bash
cd oil-spill-detection
pip install -r requirements.txt
```

### Run Demo Pipeline (no real data needed)

```bash
python main.py demo
```

This generates synthetic SAR imagery, oceanographic data, and AIS records, then runs the full pipeline:

1. **Detect** → Segments oil slick from synthetic SAR image
2. **Analyse** → Computes geometric properties (area, compactness, elongation, age)
3. **Drift** → Hindcasts origin and forecasts future trajectory using 200 Lagrangian particles
4. **Attribute** → Scores 25 vessels based on proximity, trajectory, anomalies, vessel type, and flag risk

### Run Web Dashboard

```bash
python main.py web
```

Opens at `http://127.0.0.1:5000` with an interactive map showing trajectories and vessel tracks.

### Detect on a Real Image

```bash
python main.py detect path/to/sar_image.tif --pixel-size 10
```

## 🔬 Module Details

### 1. Detection (U-Net Segmentation)

- **Architecture**: 4-level U-Net encoder-decoder with skip connections
- **Input**: SAR (Synthetic Aperture Radar) or EO (Electro-Optical) imagery
- **Output**: Binary segmentation mask + per-pixel confidence map
- **Post-processing**: Morphological cleanup, connected component analysis

### 2. Geometric Analysis

Computes per-slick properties:
- Area (km²), perimeter (m)
- Compactness (circularity index)
- Elongation, aspect ratio (via PCA)
- Convexity, solidity
- Fractal dimension (box-counting)
- Principal axis angle
- Feret diameters (min/max)
- **Age estimation** via wind-decay model

### 3. Drift Model (Lagrangian Particle Tracking)

- **Wind drift**: 3.5% of 10m wind speed with 15° Coriolis deflection
- **Current advection**: Bilinear interpolation from HYCOM/ERA5 grids
- **Turbulent diffusion**: Random walk (K = 10 m²/s)
- **Hindcast**: Reverses vectors to trace slick back to origin
- **Forecast**: Forward simulation to predict spread

### 4. Vessel Attribution

Multi-factor scoring (weighted sum, configurable):
| Factor | Weight | Description |
|--------|--------|-------------|
| Proximity | 25% | Min distance to estimated origin |
| Trajectory | 20% | Path alignment with origin area |
| Anomaly | 20% | Speed drops, course changes, stop events |
| Temporal | 15% | Time proximity to spill event |
| Vessel Type | 10% | Tankers score highest |
| Flag Risk | 5% | Flag state regulatory risk |
| Draught | 5% | Cargo/heavy draught indicator |

### 5. Web Dashboard

- **Leaflet map** with dark basemap showing trajectories (hindcast/forecast) and vessel tracks
- **Vessel leaderboard** with score breakdown bars
- **Geometry panel** with slick measurements
- **Origin card** with confidence rating
- **Interactive controls** for coordinates and time

## ⚙️ Configuration

Edit `configs/default.yaml` or pass CLI flags:

```bash
python main.py demo --lat 19.0 --lon 72.8 --hindcast 48 --particles 500
```

## 🧪 Testing

```bash
cd oil-spill-detection
python -m pytest tests/ -v
```

## 📊 Data Sources (Production)

| Data | Source | Format |
|------|--------|--------|
| SAR imagery | Sentinel-1, RADARSAT | GeoTIFF |
| EO imagery | Sentinel-2, Landsat | GeoTIFF |
| Wind | ERA5, GFS | NetCDF |
| Currents | HYCOM, Copernicus Marine | NetCDF |
| Waves | ERA5 Wave | NetCDF |
| AIS | MarineTraffic, exactEarth | CSV/JSON |
| Bathymetry | GEBCO | NetCDF |

## 🎯 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Dashboard |
| POST | `/api/demo` | Run demo pipeline |
| POST | `/api/detect` | Upload image for detection |
| GET | `/api/result/<run_id>` | Get pipeline result |
| GET | `/api/trajectory/<run_id>` | Get GeoJSON trajectories |
| GET | `/api/vessels/<run_id>` | Get vessel scores |
