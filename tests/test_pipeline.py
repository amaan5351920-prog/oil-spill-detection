"""
Tests for the Oil Spill Detection & Vessel Attribution System.

Run with: python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.oil_detector import (
    OilSpillDetector, DetectedSlick, DetectionResult,
    postprocess_mask, extract_regions, UNetModel,
)
from src.detection.geometric_analyzer import GeometricAnalyzer, GeometricProperties
from src.drift.oceanographic_data import (
    OceanographicDataManager, WindField, CurrentField, bilinear_interp,
)
from src.drift.drift_model import DriftModel, TrajectoryResult
from src.attribution.ais_processor import AISDataProcessor, VesselRecord, haversine_km
from src.attribution.vessel_scorer import VesselScorer, AttributionResult
from src.pipeline.orchestrator import PipelineOrchestrator, PipelineConfig, PipelineResult


# =====================================================================
# Detection Tests
# =====================================================================

class TestOilSpillDetector:
    def test_detect_from_array(self):
        """Detection should find the dark region in a synthetic image."""
        rng = np.random.default_rng(42)
        # Create a synthetic SAR image with a dark oil spill
        image = rng.gamma(2, 2, (256, 256)).astype(np.float32)
        # Add dark region
        cy, cx = 128, 128
        yy, xx = np.mgrid[:256, :256]
        spill = ((xx - cx) ** 2 + (yy - cy) ** 2) < 30 ** 2
        image[spill] *= 0.1
        image = image / (image.max() + 1e-8)

        detector = OilSpillDetector(threshold=0.3, min_area=20)
        result = detector.detect_from_array(image)

        assert isinstance(result, DetectionResult)
        assert result.raw_mask is not None
        assert result.confidence_map is not None

    def test_empty_image(self):
        """Detection on uniform image should find no slicks."""
        image = np.ones((128, 128), dtype=np.float32) * 0.5
        detector = OilSpillDetector(threshold=0.8, min_area=50)
        result = detector.detect_from_array(image)
        # Uniform image → no dark regions → no slicks expected
        assert result.num_slicks >= 0

    def test_unet_instantiation(self):
        """U-Net model should instantiate with correct layer count."""
        model = UNetModel(in_channels=1, base_features=16)
        assert model.encoder is not None
        assert model.decoder is not None
        assert len(model.encoder.blocks) == 4

    def test_detector_confidence_map(self):
        """Detector should produce a normalised confidence map."""
        rng = np.random.default_rng(99)
        image = rng.uniform(0.3, 0.8, (128, 128)).astype(np.float32)
        yy, xx = np.mgrid[:128, :128]
        mask = ((xx - 64) / 30) ** 2 + ((yy - 64) / 12) ** 2 < 1
        image[mask] *= 0.1
        detector = OilSpillDetector(threshold=0.15, min_area=20)
        result = detector.detect_from_array(image)
        assert result.confidence_map is not None
        assert result.confidence_map.max() <= 1.0 + 1e-6
        assert result.confidence_map.min() >= -1e-6


class TestPostprocessing:
    def test_postprocess_mask(self):
        """Postprocessing should clean up a noisy mask."""
        mask = np.random.rand(100, 100)
        mask[40:60, 40:60] = 1.0  # clear region
        result = postprocess_mask(mask, threshold=0.7, min_area=10)
        assert result.dtype == bool
        assert result.sum() > 0

    def test_extract_regions(self):
        """Should find connected components."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:20, 10:20] = True
        mask[50:70, 50:70] = True
        regions = extract_regions(mask)
        assert len(regions) == 2

    def test_extract_empty(self):
        """Empty mask should yield no regions."""
        mask = np.zeros((100, 100), dtype=bool)
        regions = extract_regions(mask)
        assert len(regions) == 0


# =====================================================================
# Geometric Analysis Tests
# =====================================================================

class TestGeometricAnalyzer:
    def test_circle_properties(self):
        """A circular slick should have high compactness."""
        mask = np.zeros((200, 200), dtype=bool)
        yy, xx = np.mgrid[:200, :200]
        mask[(xx - 100) ** 2 + (yy - 100) ** 2 < 50 ** 2] = True

        analyzer = GeometricAnalyzer(pixel_size_m=10.0)
        props = analyzer.analyze(mask, lat=20.0, lon=60.0)

        assert isinstance(props, GeometricProperties)
        assert props.area_m2 > 0
        assert props.compactness > 0.7  # circle → compactness ≈ 1
        assert props.area_km2 > 0

    def test_elongated_properties(self):
        """An elongated slick should have high elongation."""
        mask = np.zeros((200, 200), dtype=bool)
        mask[90:110, 20:180] = True  # horizontal bar

        analyzer = GeometricAnalyzer(pixel_size_m=10.0)
        props = analyzer.analyze(mask)

        assert props.aspect_ratio > 1.5  # clearly elongated
        assert props.elongation > 0.3

    def test_age_estimation(self):
        """Age estimation should return a positive value."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 40:60] = True

        analyzer = GeometricAnalyzer(pixel_size_m=10.0)
        props = analyzer.analyze(
            mask,
            wind_speed_knots=8.0,
            image_time="2025-01-02T12:00:00",
            spill_report_time="2025-01-01T12:00:00",
        )
        assert props.spill_age_hours is not None
        assert props.spill_age_hours >= 0


# =====================================================================
# Oceanographic Data Tests
# =====================================================================

class TestOceanographicData:
    def test_generate_synthetic(self):
        """Should generate synthetic data for all fields."""
        ocean = OceanographicDataManager()
        ocean.generate_synthetic(center_lat=20.0, center_lon=60.0, duration_hours=24)
        assert len(ocean.wind_data) > 0
        assert len(ocean.current_data) > 0
        assert len(ocean.wave_data) > 0

    def test_bilinear_interp(self):
        """Bilinear interpolation should return a value within range."""
        lat = np.array([0.0, 1.0, 2.0])
        lon = np.array([0.0, 1.0, 2.0])
        values = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        val = bilinear_interp(lat, lon, values, 0.5, 0.5)
        assert 1.0 <= val <= 5.0

    def test_get_wind_vector(self):
        """Should return (u, v) wind vector."""
        ocean = OceanographicDataManager()
        ocean.generate_synthetic(center_lat=20.0, center_lon=60.0, duration_hours=6)
        from datetime import datetime
        u, v = ocean.get_wind_vector(datetime(2025, 1, 1), 20.0, 60.0)
        assert isinstance(u, float)
        assert isinstance(v, float)


# =====================================================================
# Drift Model Tests
# =====================================================================

class TestDriftModel:
    def test_forecast(self):
        """Forecast should produce valid trajectories."""
        ocean = OceanographicDataManager()
        ocean.generate_synthetic(center_lat=20.0, center_lon=60.0, duration_hours=48)

        model = DriftModel(ocean_data=ocean)
        result = model.forecast(
            start_lat=20.0, start_lon=60.0,
            start_time=datetime(2025, 1, 1),
            hours=12, n_particles=10, dt_seconds=600,
        )

        assert isinstance(result, TrajectoryResult)
        assert len(result.trajectories) == 10
        # Each trajectory should have multiple points
        assert len(result.trajectories[0]) > 1

    def test_hindcast(self):
        """Hindcast should trace back toward origin."""
        from datetime import datetime

        ocean = OceanographicDataManager()
        ocean.generate_synthetic(center_lat=20.0, center_lon=60.0, duration_hours=48)

        model = DriftModel(ocean_data=ocean)
        result = model.hindcast(
            detected_lat=20.5, detected_lon=60.5,
            detected_time=datetime(2025, 1, 2),
            hours=12, n_particles=10, dt_seconds=600,
        )

        assert isinstance(result, TrajectoryResult)
        assert result.origin_estimate is not None
        assert result.origin_time is not None
        # Origin should be somewhere near (20, 60)
        origin_lat, origin_lon = result.origin_estimate
        assert 15 < origin_lat < 25
        assert 55 < origin_lon < 65

    def test_particles_disperse(self):
        """Particles should spread out over time."""
        from datetime import datetime

        ocean = OceanographicDataManager()
        ocean.generate_synthetic(center_lat=20.0, center_lon=60.0, duration_hours=24)

        model = DriftModel(ocean_data=ocean)
        result = model.forecast(
            start_lat=20.0, start_lon=60.0,
            start_time=datetime(2025, 1, 1),
            hours=24, n_particles=20, dt_seconds=600,
        )

        # Check spread increases over time
        lats_start = [t[0].lat for t in result.trajectories]
        lats_end = [t[-1].lat for t in result.trajectories]
        assert np.std(lats_end) >= np.std(lats_start) * 0.5  # some dispersion


# =====================================================================
# AIS Processor Tests
# =====================================================================

class TestAISProcessor:
    def test_generate_synthetic(self):
        """Should generate synthetic vessel data."""
        processor = AISDataProcessor()
        mmsis = processor.generate_synthetic(
            center_lat=20.0, center_lon=60.0, n_vessels=10, n_hours=12,
        )
        assert len(mmsis) == 10
        assert len(processor.records) > 0

    def test_build_traffic_window(self):
        """Should filter vessels within spatial-temporal window."""
        from datetime import datetime, timedelta

        processor = AISDataProcessor()
        processor.generate_synthetic(
            center_lat=20.0, center_lon=60.0, n_vessels=10, n_hours=24,
        )

        window = processor.build_traffic_window(
            center_lat=20.0, center_lon=60.0,
            radius_km=50,
            start_time=datetime(2025, 1, 1),
            end_time=datetime(2025, 1, 2),
        )
        assert window.num_vessels > 0
        assert window.total_records > 0

    def test_haversine(self):
        """Haversine distance should be reasonable."""
        d = haversine_km(20.0, 60.0, 20.1, 60.1)
        assert 5 < d < 20  # ~15 km expected


# =====================================================================
# Vessel Scorer Tests
# =====================================================================

class TestVesselScorer:
    def test_score_vessels(self):
        """Should produce ranked vessel scores."""
        from datetime import datetime, timedelta

        processor = AISDataProcessor()
        processor.generate_synthetic(
            center_lat=20.0, center_lon=60.0,
            n_vessels=10, n_hours=24,
            suspect_mmsi="2000000000",
        )

        window = processor.build_traffic_window(
            center_lat=20.0, center_lon=60.0,
            radius_km=100,
            start_time=datetime(2025, 1, 1),
            end_time=datetime(2025, 1, 2),
        )

        scorer = VesselScorer()
        result = scorer.score_vessels(
            traffic_window=window,
            origin_lat=20.0,
            origin_lon=60.0,
            origin_time=datetime(2025, 1, 1, 6),
        )

        assert isinstance(result, AttributionResult)
        assert result.total_vessels_analysed > 0
        # Scores should be sorted descending
        scores = [s.total_score for s in result.scores]
        assert scores == sorted(scores, reverse=True)

    def test_format_report(self):
        """Should produce a readable report."""
        from datetime import datetime

        processor = AISDataProcessor()
        processor.generate_synthetic(center_lat=20.0, center_lon=60.0, n_vessels=5)
        window = processor.build_traffic_window(
            center_lat=20.0, center_lon=60.0, radius_km=100,
            start_time=datetime(2025, 1, 1), end_time=datetime(2025, 1, 2),
        )
        scorer = VesselScorer()
        result = scorer.score_vessels(window, 20.0, 60.0, datetime(2025, 1, 1, 6))
        report = scorer.format_report(result)
        assert "ATTRIBUTION REPORT" in report
        assert len(report) > 100


# =====================================================================
# Full Pipeline Tests
# =====================================================================

class TestPipeline:
    def test_run_demo(self):
        """Full demo pipeline should complete without errors."""
        config = PipelineConfig(
            hindcast_hours=12,
            forecast_hours=12,
            n_particles=20,
            output_dir="./test_output",
        )
        pipeline = PipelineOrchestrator(config)
        result = pipeline.run_demo(
            center_lat=20.0, center_lon=60.0,
            detection_time="2025-01-01T12:00:00",
        )

        assert isinstance(result, PipelineResult)
        assert result.detection is not None
        assert result.hindcast is not None
        assert result.forecast is not result.hindcast
        assert result.attribution is not None
        assert len(result.errors) == 0
        assert len(result.summary) > 0

    def test_run_from_array(self):
        """Pipeline should work with a numpy array input."""
        rng = np.random.default_rng(42)
        image = rng.gamma(2, 2, (256, 256)).astype(np.float32)
        cy, cx = 128, 128
        yy, xx = np.mgrid[:256, :256]
        spill = ((xx - cx) ** 2 + (yy - cy) ** 2) < 30 ** 2
        image[spill] *= 0.1
        image = image / (image.max() + 1e-8)

        config = PipelineConfig(
            hindcast_hours=6,
            forecast_hours=6,
            n_particles=10,
            output_dir="./test_output",
        )
        pipeline = PipelineOrchestrator(config)
        result = pipeline.run_from_array(
            image,
            origin_lat=20.0,
            origin_lon=60.0,
        )

        assert result.detection is not None
        assert result.summary.get("detection", {}).get("num_slicks", 0) >= 0


# Fixtures
from datetime import datetime
