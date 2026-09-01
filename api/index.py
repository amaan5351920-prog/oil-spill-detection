"""
Vercel Serverless Entry Point — Flask app for Vercel deployment.
All routes (/api/*, /*) are handled by this single function.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add project root to Python path (Vercel runs from api/)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flask import Flask, jsonify, render_template, request
from src.pipeline.orchestrator import PipelineOrchestrator, PipelineConfig, PipelineResult
from datetime import datetime
import numpy as np
import cv2
import json
import logging

logger = logging.getLogger(__name__)

# In-memory result store
_results_store: dict[str, PipelineResult] = {}

# Create Flask app with correct paths for Vercel
app = Flask(
    __name__,
    template_folder=str(project_root / "templates"),
    static_folder=str(project_root / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "oil-spill-detection-dev-key")
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/demo", methods=["POST"])
def run_demo():
    data = request.get_json(silent=True) or {}
    center_lat = data.get("lat", 20.5)
    center_lon = data.get("lon", 60.3)
    detection_time = data.get("time", "2025-01-01T12:00:00")

    pipeline = PipelineOrchestrator(PipelineConfig(
        output_dir=str(project_root / "output"),
        detection_threshold=0.35,
        min_slick_area_px=200,
    ))

    result = pipeline.run_demo(
        center_lat=center_lat,
        center_lon=center_lon,
        detection_time=detection_time,
    )
    _results_store[result.run_id] = result

    return jsonify({
        "run_id": result.run_id,
        "summary": result.summary,
        "status": "completed" if not result.errors else "error",
        "errors": result.errors,
    })


@app.route("/api/detect", methods=["POST"])
def detect_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return jsonify({"error": "Cannot read image. Please upload a PNG or JPG file."}), 400

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    cv2.imwrite(tmp.name, img)
    tmp.close()

    pipeline = PipelineOrchestrator(PipelineConfig(
        output_dir=str(project_root / "output"),
        detection_threshold=0.35,
        min_slick_area_px=200,
    ))

    result = pipeline.run(
        image_path=tmp.name,
        detection_time=request.form.get("time", datetime.utcnow().isoformat()),
        origin_lat=float(request.form.get("lat", 20.5)),
        origin_lon=float(request.form.get("lon", 60.3)),
    )
    _results_store[result.run_id] = result

    return jsonify({
        "run_id": result.run_id,
        "summary": result.summary,
        "status": "completed" if not result.errors else "error",
        "errors": result.errors,
    })


@app.route("/api/result/<run_id>")
def get_result(run_id):
    result = _results_store.get(run_id)
    if not result:
        return jsonify({"error": "Result not found"}), 404
    return jsonify({
        "run_id": result.run_id,
        "summary": result.summary,
        "errors": result.errors,
    })


@app.route("/api/trajectory/<run_id>")
def get_trajectory(run_id):
    result = _results_store.get(run_id)
    if not result:
        return jsonify({"error": "Result not found"}), 404

    features = []

    if result.hindcast:
        for i, traj in enumerate(result.hindcast.trajectories[:50]):
            coords = [[pt.lon, pt.lat] for pt in traj]
            features.append({
                "type": "Feature",
                "properties": {"type": "hindcast", "particle": i, "color": "#e74c3c"},
                "geometry": {"type": "LineString", "coordinates": coords},
            })
        if result.hindcast.origin_estimate:
            features.append({
                "type": "Feature",
                "properties": {
                    "type": "origin",
                    "label": "Estimated Origin",
                    "color": "#e74c3c",
                    "radius": 8,
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": list(result.hindcast.origin_estimate)[::-1],
                },
            })

    if result.forecast:
        for i, traj in enumerate(result.forecast.trajectories[:50]):
            coords = [[pt.lon, pt.lat] for pt in traj]
            features.append({
                "type": "Feature",
                "properties": {"type": "forecast", "particle": i, "color": "#3498db"},
                "geometry": {"type": "LineString", "coordinates": coords},
            })

    if result.traffic_window and result.attribution:
        top_mmsis = [vs.mmsi for vs in result.attribution.scores[:5]]
        for mmsi in result.traffic_window.vessels:
            track = result.traffic_window.vessels[mmsi]
            if not track.path:
                continue
            coords = [[lon, lat] for lat, lon in track.path]
            is_suspect = top_mmsis and mmsi == top_mmsis[0]
            features.append({
                "type": "Feature",
                "properties": {
                    "type": "vessel",
                    "mmsi": mmsi,
                    "name": track.vessel_name,
                    "color": "#e67e22" if is_suspect else "#95a5a6",
                    "width": 3 if is_suspect else 1,
                },
                "geometry": {"type": "LineString", "coordinates": coords},
            })

    return jsonify({"type": "FeatureCollection", "features": features})


@app.route("/api/vessels/<run_id>")
def get_vessels(run_id):
    result = _results_store.get(run_id)
    if not result:
        return jsonify({"error": "Result not found"}), 404
    if not result.attribution:
        return jsonify({"error": "No attribution data"}), 404

    vessels = []
    for vs in result.attribution.scores:
        vessels.append({
            "rank": vs.rank,
            "mmsi": vs.mmsi,
            "name": vs.vessel_name,
            "type": vs.vessel_type,
            "imo": vs.imo,
            "flag": vs.flag,
            "total_score": round(vs.total_score, 4),
            "proximity_score": round(vs.proximity_score, 4),
            "temporal_score": round(vs.temporal_score, 4),
            "trajectory_score": round(vs.trajectory_score, 4),
            "anomaly_score": round(vs.anomaly_score, 4),
            "vessel_type_score": round(vs.vessel_type_score, 4),
            "flag_risk_score": round(vs.flag_risk_score, 4),
            "draught_score": round(vs.draught_score, 4),
            "min_distance_km": round(vs.min_distance_km, 2),
            "anomalies": vs.anomalies_detected[:5],
        })

    return jsonify({
        "run_id": run_id,
        "total_vessels": len(vessels),
        "vessels": vessels,
    })
