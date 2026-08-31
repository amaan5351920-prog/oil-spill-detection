"""
Pipeline Orchestrator — End-to-end orchestration of oil spill detection,
drift modelling, and vessel attribution.

This module connects the three core subsystems into a single automated
pipeline:

    1. DETECT   → Identify oil slicks in SAR/EO imagery
    2. ANALYSE  → Compute geometric properties & age
    3. DRIFT    → Hindcast to origin, forecast trajectory
    4. ATTRIBUTION → Score & rank suspect vessels via AIS data
    5. REPORT   → Generate summary with results and visualisation data
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

from ..detection.oil_detector import OilSpillDetector, DetectionResult
from ..detection.geometric_analyzer import GeometricAnalyzer, GeometricProperties
from ..drift.drift_model import DriftModel, TrajectoryResult
from ..drift.oceanographic_data import OceanographicDataManager
from ..attribution.ais_processor import AISDataProcessor, TrafficWindow
from ..attribution.vessel_scorer import VesselScorer, AttributionResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the full pipeline run."""
    # Detection
    model_weights_path: Optional[str] = None
    detection_threshold: float = 0.5
    min_slick_area_px: int = 50
    pixel_size_m: float = 10.0

    # Drift
    hindcast_hours: float = 36.0
    forecast_hours: float = 48.0
    n_particles: int = 200
    particle_spread_m: float = 500.0

    # Attribution
    search_radius_km: float = 100.0
    origin_radius_km: float = 5.0
    vessel_types_filter: Optional[list[str]] = None

    # Output
    output_dir: str = "./output"


@dataclass
class PipelineResult:
    """Complete result from a full pipeline run."""
    run_id: str = ""
    run_time: str = ""
    config: Optional[PipelineConfig] = None

    # Detection results
    detection: Optional[DetectionResult] = None
    geometric_properties: list[GeometricProperties] = field(default_factory=list)

    # Drift results
    hindcast: Optional[TrajectoryResult] = None
    forecast: Optional[TrajectoryResult] = None

    # Attribution results
    attribution: Optional[AttributionResult] = None
    traffic_window: Optional[TrafficWindow] = None

    # Summary
    summary: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


class PipelineOrchestrator:
    """End-to-end oil spill detection, drift, and attribution pipeline.

    Usage:
        config = PipelineConfig(pixel_size_m=10.0, hindcast_hours=36)
        pipeline = PipelineOrchestrator(config)

        # Option 1: Run on an image
        result = pipeline.run(image_path="sar_image.tif")

        # Option 2: Run on a numpy array
        result = pipeline.run_from_array(image_array)

        # Option 3: Run with demo data
        result = pipeline.run_demo(
            center_lat=20.5, center_lon=60.3,
            detection_time="2025-01-01T12:00:00",
        )

        print(result.summary)
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.detector = OilSpillDetector(
            model_weights_path=self.config.model_weights_path,
            threshold=self.config.detection_threshold,
            min_area=self.config.min_slick_area_px,
            pixel_size_m=self.config.pixel_size_m,
        )
        self.analyzer = GeometricAnalyzer(pixel_size_m=self.config.pixel_size_m)
        self.ocean_data = OceanographicDataManager()
        self.drift_model = DriftModel(ocean_data=self.ocean_data)
        self.ais_processor = AISDataProcessor()
        self.vessel_scorer = VesselScorer()

        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        logger.info("PipelineOrchestrator initialised (output=%s)", self.config.output_dir)

    # -----------------------------------------------------------------
    # Full pipeline run
    # -----------------------------------------------------------------

    def run(
        self,
        image_path: str,
        detection_time: Optional[str] = None,
        origin_lat: Optional[float] = None,
        origin_lon: Optional[float] = None,
        ais_data_path: Optional[str] = None,
        ocean_data_path: Optional[str] = None,
    ) -> PipelineResult:
        """Run the full pipeline on a SAR/EO image."""
        import uuid
        result = PipelineResult(
            run_id=str(uuid.uuid4())[:8],
            run_time=datetime.utcnow().isoformat(),
            config=self.config,
        )

        try:
            # Step 1: Detection
            logger.info("[Step 1/5] Detecting oil slicks...")
            result.detection = self.detector.detect(image_path, detection_time)

            # Step 2: Geometric analysis
            logger.info("[Step 2/5] Analysing geometry...")
            for slick in result.detection.slicks:
                props = self.analyzer.analyze(
                    mask=slick.mask,
                    image_time=detection_time,
                )
                result.geometric_properties.append(props)

            # Step 3: Load ocean data and run drift
            if ocean_data_path:
                self.ocean_data.load_netcdf(ocean_data_path, "wind")

            # Use detected slick centroid as the initial position
            if result.detection.slicks and origin_lat is None:
                slick = result.detection.slicks[0]
                origin_lat = slick.centroid[1] * self.config.pixel_size_m / 111_320 + 20  # approximate
                origin_lon = slick.centroid[0] * self.config.pixel_size_m / 111_320 + 60

            if origin_lat and origin_lon:
                # Generate fresh ocean and AIS data for this location
                dt = datetime.fromisoformat((detection_time or '2025-01-01T12:00:00').replace('Z', '+00:00'))
                self.ocean_data = OceanographicDataManager()
                self.ocean_data.generate_synthetic(
                    center_lat=origin_lat, center_lon=origin_lon,
                    start_time=dt, duration_hours=72,
                )
                self.drift_model = DriftModel(ocean_data=self.ocean_data)
                self.ais_processor = AISDataProcessor()
                import hashlib
                seed = int(hashlib.md5(f'{origin_lat}{origin_lon}'.encode()).hexdigest()[:8], 16) % 10000
                self.ais_processor.generate_synthetic(
                    center_lat=origin_lat, center_lon=origin_lon,
                    n_vessels=25, n_hours=48,
                    suspect_mmsi=str(1000000000 + seed),
                )
                self._run_drift(result, origin_lat, origin_lon, detection_time)

            # Step 4: Attribution
            self._run_attribution(result, origin_lat, origin_lon, detection_time, ais_data_path)

            # Step 5: Build summary
            self._build_summary(result)

        except Exception as exc:
            result.errors.append(str(exc))
            logger.error("Pipeline error: %s", exc, exc_info=True)

        self._save_result(result)
        return result

    def run_from_array(
        self,
        image: np.ndarray,
        detection_time: Optional[str] = None,
        origin_lat: float = 20.5,
        origin_lon: float = 60.3,
    ) -> PipelineResult:
        """Run pipeline on a numpy array."""
        import uuid
        result = PipelineResult(
            run_id=str(uuid.uuid4())[:8],
            run_time=datetime.utcnow().isoformat(),
            config=self.config,
        )

        try:
            result.detection = self.detector.detect_from_array(image, detection_time)

            for slick in result.detection.slicks:
                props = self.analyzer.analyze(
                    mask=slick.mask,
                    image_time=detection_time,
                )
                result.geometric_properties.append(props)

            self._run_drift(result, origin_lat, origin_lon, detection_time)
            self._run_attribution(result, origin_lat, origin_lon, detection_time)
            self._build_summary(result)

        except Exception as exc:
            result.errors.append(str(exc))
            logger.error("Pipeline error: %s", exc, exc_info=True)

        self._save_result(result)
        return result

    def run_demo(
        self,
        center_lat: float = 20.5,
        center_lon: float = 60.3,
        detection_time: str = "2025-01-01T12:00:00",
    ) -> PipelineResult:
        """Run a full demo pipeline using synthetic data.

        Generates synthetic SAR imagery, oceanographic data, and AIS
        records to demonstrate the complete pipeline without real data.
        """
        import uuid
        result = PipelineResult(
            run_id=str(uuid.uuid4())[:8],
            run_time=datetime.utcnow().isoformat(),
            config=self.config,
        )

        logger.info("Running demo pipeline at (%.2f, %.2f)", center_lat, center_lon)

        try:
            # Step 1: Generate synthetic SAR image and detect
            logger.info("[Step 1/5] Generating synthetic SAR image and detecting...")
            synthetic_image = self._generate_synthetic_sar(center_lat, center_lon)
            result.detection = self.detector.detect_from_array(synthetic_image, detection_time)

            # Step 2: Geometric analysis
            logger.info("[Step 2/5] Analysing geometry...")
            for slick in result.detection.slicks:
                props = self.analyzer.analyze(
                    mask=slick.mask,
                    wind_speed_knots=8.0,
                    image_time=detection_time,
                    spill_report_time=detection_time,
                    lat=center_lat,
                    lon=center_lon,
                )
                result.geometric_properties.append(props)

            # Step 3: Generate ocean data and run drift
            logger.info("[Step 3/5] Running drift model...")
            self.ocean_data.generate_synthetic(
                center_lat=center_lat,
                center_lon=center_lon,
                start_time=datetime.fromisoformat(detection_time),
                duration_hours=72,
            )

            self._run_drift(result, center_lat, center_lon, detection_time)

            # Step 4: Generate AIS data and attribute
            logger.info("[Step 4/5] Analysing vessel traffic...")
            suspect_mmsi = "2000000000"
            self.ais_processor.generate_synthetic(
                center_lat=center_lat,
                center_lon=center_lon,
                n_vessels=25,
                n_hours=48,
                suspect_mmsi=suspect_mmsi,
            )

            self._run_attribution(result, center_lat, center_lon, detection_time)
            self._build_summary(result)

        except Exception as exc:
            result.errors.append(str(exc))
            logger.error("Demo pipeline error: %s", exc, exc_info=True)

        self._save_result(result)
        return result

    # -----------------------------------------------------------------
    # Internal steps
    # -----------------------------------------------------------------

    def _run_drift(self, result, lat, lon, detection_time):
        """Run hindcast and forecast drift models."""
        if detection_time:
            dt = datetime.fromisoformat(detection_time.replace("Z", "+00:00"))
        else:
            dt = datetime(2025, 1, 1, 12, 0)

        result.hindcast = self.drift_model.hindcast(
            detected_lat=lat,
            detected_lon=lon,
            detected_time=dt,
            hours=self.config.hindcast_hours,
            n_particles=self.config.n_particles,
            spread_m=self.config.particle_spread_m,
        )

        result.forecast = self.drift_model.forecast(
            start_lat=lat,
            start_lon=lon,
            start_time=dt,
            hours=self.config.forecast_hours,
            n_particles=self.config.n_particles,
            spread_m=self.config.particle_spread_m,
        )

    def _run_attribution(self, result, lat, lon, detection_time, ais_path=None):
        """Run vessel attribution analysis."""
        if ais_path:
            self.ais_processor.load_csv(ais_path)

        dt = datetime.fromisoformat(detection_time.replace("Z", "+00:00"))
        time_window_start = dt - timedelta(hours=self.config.hindcast_hours)
        time_window_end = dt + timedelta(hours=self.config.forecast_hours)

        # Use hindcast origin if available
        origin_lat = lat
        origin_lon = lon
        origin_time = dt - timedelta(hours=self.config.hindcast_hours)
        if result.hindcast and result.hindcast.origin_estimate:
            origin_lat, origin_lon = result.hindcast.origin_estimate
            origin_time = result.hindcast.origin_time or origin_time

        result.traffic_window = self.ais_processor.build_traffic_window(
            center_lat=origin_lat,
            center_lon=origin_lon,
            radius_km=self.config.search_radius_km,
            start_time=time_window_start,
            end_time=time_window_end,
            vessel_types_filter=self.config.vessel_types_filter,
        )

        result.attribution = self.vessel_scorer.score_vessels(
            traffic_window=result.traffic_window,
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            origin_time=origin_time,
            origin_radius_km=self.config.origin_radius_km,
        )

    def _build_summary(self, result):
        """Build a human-readable and machine-readable summary."""
        s = result.summary
        s["run_id"] = result.run_id
        s["run_time"] = result.run_time
        s["errors"] = result.errors

        # Detection summary
        if result.detection:
            s["detection"] = {
                "num_slicks": result.detection.num_slicks,
                "total_area_km2": sum(
                    g.area_km2 for g in result.geometric_properties
                ),
            }
            for i, gp in enumerate(result.geometric_properties):
                s[f"slick_{i}"] = {
                    "area_km2": gp.area_km2,
                    "perimeter_m": gp.perimeter_m,
                    "compactness": gp.compactness,
                    "elongation": gp.elongation,
                    "aspect_ratio": gp.aspect_ratio,
                    "fractal_dimension": gp.fractal_dimension,
                    "spill_age_hours": gp.spill_age_hours,
                }

        # Drift summary
        if result.hindcast and result.hindcast.origin_estimate:
            s["hindcast"] = {
                "origin_lat": result.hindcast.origin_estimate[0],
                "origin_lon": result.hindcast.origin_estimate[1],
                "origin_time": str(result.hindcast.origin_time),
                "confidence": result.hindcast.confidence,
                "n_particles": len(result.hindcast.trajectories),
            }
        if result.forecast:
            end_lats = [t[-1].lat for t in result.forecast.trajectories]
            end_lons = [t[-1].lon for t in result.forecast.trajectories]
            s["forecast"] = {
                "predicted_lat": float(np.mean(end_lats)),
                "predicted_lon": float(np.mean(end_lons)),
                "spread_lat": float(np.std(end_lats)),
                "spread_lon": float(np.std(end_lons)),
            }

        # Attribution summary
        if result.attribution:
            s["attribution"] = {
                "vessels_analysed": result.attribution.total_vessels_analysed,
            }
            if result.attribution.top_suspect:
                ts = result.attribution.top_suspect
                s["attribution"]["top_suspect"] = {
                    "mmsi": ts.mmsi,
                    "name": ts.vessel_name,
                    "type": ts.vessel_type,
                    "flag": ts.flag,
                    "score": ts.total_score,
                    "min_distance_km": ts.min_distance_km,
                    "anomalies": ts.anomalies_detected[:5],
                }
            s["attribution"]["top_5"] = [
                {
                    "rank": vs.rank,
                    "name": vs.vessel_name,
                    "mmsi": vs.mmsi,
                    "score": round(vs.total_score, 3),
                }
                for vs in result.attribution.scores[:5]
            ]

    @staticmethod
    def _generate_synthetic_sar(lat: float, lon: float, size: int = 256) -> np.ndarray:
        """Generate a synthetic SAR image with an oil spill pattern."""
        rng = np.random.default_rng(42)
        # Background SAR speckle
        background = rng.gamma(2, 2, (size, size)).astype(np.float32)
        # Oil spill: dark elongated region (oil dampens capillary waves → lower backscatter)
        spill = np.zeros((size, size), dtype=np.float32)
        cy, cx = size // 2 + rng.integers(-20, 20), size // 2 + rng.integers(-20, 20)
        angle = rng.uniform(0, np.pi)
        yy, xx = np.mgrid[:size, :size]
        rx, ry = 40 + rng.integers(0, 20), 15 + rng.integers(0, 10)
        rot_xx = (xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)
        rot_yy = -(xx - cx) * np.sin(angle) + (yy - cy) * np.cos(angle)
        spill[((rot_xx / rx) ** 2 + (rot_yy / ry) ** 2) < 1] = 1.0
        # Darken the spill area
        image = background.copy()
        image[spill > 0] *= 0.15
        # Add noise
        image += rng.normal(0, 0.1, image.shape)
        image = np.clip(image, 0, None)
        # Normalize to [0, 1]
        image = image / (image.max() + 1e-8)
        return image[np.newaxis, ...]  # (1, H, W)

    def _save_result(self, result: PipelineResult):
        """Save pipeline result to JSON."""
        output_path = Path(self.config.output_dir) / f"result_{result.run_id}.json"
        try:
            # Convert to serialisable dict
            data = {
                "run_id": result.run_id,
                "run_time": result.run_time,
                "summary": result.summary,
                "errors": result.errors,
            }
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            logger.info("Result saved to %s", output_path)
        except Exception as exc:
            logger.warning("Could not save result: %s", exc)
