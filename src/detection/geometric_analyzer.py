"""
Geometric Analyzer — Computes geometric properties and age estimates for
detected oil slicks.

From a binary mask and detection metadata the analyzer computes:
  • Area, perimeter, centroid
  • Bounding box, aspect ratio, elongation
  • Compactness (circularity) and convexity
  • Spill age estimation using wind-speed decay model
  • Wind-fan opening angle (downwind spread)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GeometricProperties:
    """Geometric descriptors of a single oil slick."""
    area_m2: float = 0.0
    perimeter_m: float = 0.0
    centroid_lat: Optional[float] = None
    centroid_lon: Optional[float] = None
    centroid_pixel: tuple[float, float] = (0.0, 0.0)
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    aspect_ratio: float = 0.0
    elongation: float = 0.0
    compactness: float = 0.0       # 4π·area / perimeter²  (1 for circle)
    convexity: float = 0.0         # convex_hull_perim / perim
    solidity: float = 0.0          # area / convex_hull_area
    fractal_dimension: float = 0.0
    principal_axis_angle: float = 0.0  # degrees, major axis from PCA
    min_feret_diameter_m: float = 0.0
    max_feret_diameter_m: float = 0.0
    area_km2: float = 0.0
    spill_age_hours: Optional[float] = None
    spill_age_confidence: str = "low"   # low / medium / high
    wind_fan_angle_deg: Optional[float] = None


@dataclass
class AgeEstimate:
    """Estimated age of the oil slick."""
    hours: float = 0.0
    confidence: str = "low"
    method: str = "wind_decay_model"
    details: str = ""


class GeometricAnalyzer:
    """Compute geometric properties and age for detected oil slicks.

    Usage:
        analyzer = GeometricAnalyzer(pixel_size_m=10.0)
        props = analyzer.analyze(slick_mask, wind_speed_knots=8.0)
    """

    def __init__(self, pixel_size_m: float = 10.0):
        self.pixel_size_m = pixel_size_m

    def analyze(
        self,
        mask: np.ndarray,
        wind_speed_knots: Optional[float] = None,
        wind_direction_deg: Optional[float] = None,
        image_time: Optional[str] = None,
        spill_report_time: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> GeometricProperties:
        """Analyze a binary slick mask and return geometric properties.

        Parameters
        ----------
        mask : np.ndarray
            Binary mask (H, W), dtype bool.
        wind_speed_knots : float, optional
            Current wind speed in knots.
        wind_direction_deg : float, optional
            Meteorological wind direction (direction *from* which wind blows).
        image_time, spill_report_time : str, optional
            ISO-8601 timestamps for age calculation.
        lat, lon : float, optional
            Geographic coordinates of the slick centroid.
        """
        props = GeometricProperties()

        area_px = int(mask.sum())
        props.area_m2 = area_px * (self.pixel_size_m ** 2)
        props.area_km2 = props.area_m2 / 1e6

        # Perimeter
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if contours:
            props.perimeter_m = cv2.arcLength(contours[0], True) * self.pixel_size_m
            props.bbox = cv2.boundingRect(contours[0])

            # Centroid
            moments = cv2.moments(contours[0])
            if moments["m00"] > 0:
                cx = moments["m10"] / moments["m00"]
                cy = moments["m01"] / moments["m00"]
                props.centroid_pixel = (cx, cy)

            # Convex hull
            hull = cv2.convexHull(contours[0])
            hull_area = cv2.contourArea(hull)
            hull_perim = cv2.arcLength(hull, True)

            # Solidity & convexity
            if hull_area > 0:
                props.solidity = area_px / hull_area
            if props.perimeter_m > 0:
                props.convexity = hull_perim * self.pixel_size_m / props.perimeter_m

        # Compactness (circularity)
        if props.perimeter_m > 0:
            props.compactness = (4 * np.pi * props.area_m2) / (props.perimeter_m ** 2)

        # Aspect ratio, elongation, principal axis via PCA
        props.aspect_ratio, props.elongation, props.principal_axis_angle = (
            self._pca_analysis(mask)
        )

        # Feret diameters
        props.min_feret_diameter_m, props.max_feret_diameter_m = (
            self._feret_diameters(mask)
        )

        # Fractal dimension (box-counting)
        props.fractal_dimension = self._box_counting_dimension(mask)

        # Geographic centroid
        if lat is not None and lon is not None:
            props.centroid_lat = lat
            props.centroid_lon = lon

        # Age estimation
        if image_time and spill_report_time:
            age = self._estimate_age(
                wind_speed_knots=wind_speed_knots or 5.0,
                image_time=image_time,
                report_time=spill_report_time,
            )
            props.spill_age_hours = age.hours
            props.spill_age_confidence = age.confidence

        # Wind fan angle
        if wind_direction_deg is not None:
            props.wind_fan_angle_deg = self._estimate_wind_fan_angle(mask, wind_direction_deg)

        logger.info(
            "Geometric analysis: area=%.2f km², perimeter=%.0f m, compactness=%.3f",
            props.area_km2, props.perimeter_m, props.compactness,
        )
        return props

    # -----------------------------------------------------------------
    # PCA-based shape analysis
    # -----------------------------------------------------------------

    def _pca_analysis(self, mask: np.ndarray) -> tuple[float, float, float]:
        """Return (aspect_ratio, elongation, principal_axis_angle)."""
        ys, xs = np.where(mask)
        if len(xs) < 10:
            return 1.0, 0.0, 0.0

        coords = np.column_stack([xs, ys]).astype(np.float64)
        coords -= coords.mean(axis=0)
        cov = np.cov(coords.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues = np.maximum(eigenvalues, 1e-10)

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        aspect_ratio = np.sqrt(eigenvalues[0] / eigenvalues[1])
        elongation = 1.0 - np.sqrt(eigenvalues[1] / eigenvalues[0]) if eigenvalues[0] > 0 else 0.0
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

        return float(aspect_ratio), float(max(elongation, 0.0)), float(angle)

    # -----------------------------------------------------------------
    # Feret diameters
    # -----------------------------------------------------------------

    def _feret_diameters(self, mask: np.ndarray) -> tuple[float, float]:
        """Min and max Feret diameters (bounding calipers)."""
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return 0.0, 0.0

        pts = contours[0].squeeze()
        if pts.ndim != 2:
            return 0.0, 0.0

        # Rotating calipers approximation via convex hull
        hull = cv2.convexHull(pts)
        hull_pts = hull.squeeze()
        if hull_pts.ndim != 2:
            return 0.0, 0.0

        n = len(hull_pts)
        min_d = float("inf")
        max_d = 0.0
        for i in range(n):
            p1 = hull_pts[i]
            p2 = hull_pts[(i + 1) % n]
            d = np.linalg.norm(p2 - p1) * self.pixel_size_m
            min_d = min(min_d, d)
            max_d = max(max_d, d)

        return float(min_d), float(max_d)

    # -----------------------------------------------------------------
    # Fractal dimension (box counting)
    # -----------------------------------------------------------------

    @staticmethod
    def _box_counting_dimension(mask: np.ndarray, n_sizes: int = 10) -> float:
        """Estimate fractal dimension via box-counting."""
        binary = mask.astype(np.uint8)
        h, w = binary.shape
        size = min(h, w)
        sizes = []
        counts = []
        for _ in range(n_sizes):
            if size < 2:
                break
            kernel = size
            resized = cv2.resize(binary, (max(w // kernel, 1), max(h // kernel, 1)),
                                 interpolation=cv2.INTER_NEAREST)
            counts.append(np.sum(resized > 0))
            sizes.append(size)
            size //= 2

        if len(sizes) < 2 or min(counts) == 0:
            return 1.0

        log_sizes = np.log(sizes)
        log_counts = np.log(counts)
        coeffs = np.polyfit(log_sizes, log_counts, 1)
        return float(-coeffs[0])

    # -----------------------------------------------------------------
    # Age estimation (simplified wind-decay model)
    # -----------------------------------------------------------------

    def _estimate_age(
        self,
        wind_speed_knots: float,
        image_time: str,
        report_time: str,
    ) -> AgeEstimate:
        """Estimate slick age from reported sighting time and wind speed.

        Uses a simplified empirical model:
            visible_duration ≈ f(area, wind_speed)

        Oil slicks are typically visible in SAR for 24–72 hours depending
        on wind speed.  Higher wind speeds cause faster emulsification
        and dispersion, reducing visibility.
        """
        from datetime import datetime

        try:
            t_image = datetime.fromisoformat(image_time.replace("Z", "+00:00"))
            t_report = datetime.fromisoformat(report_time.replace("Z", "+00:00"))
            hours_since_report = abs((t_image - t_report).total_seconds()) / 3600
        except Exception as exc:
            return AgeEstimate(hours=0.0, confidence="low", details=str(exc))

        # Empirical decay: wind speed increases dispersion rate
        # At 5 kts → visible ~72 h; at 20 kts → visible ~24 h
        base_duration = 72.0
        wind_factor = max(1.0, wind_speed_knots / 5.0)
        estimated_age = min(hours_since_report, base_duration / wind_factor)

        confidence = "low"
        if wind_speed_knots < 3:
            confidence = "medium"  # Low wind → more predictable spread

        return AgeEstimate(
            hours=float(estimated_age),
            confidence=confidence,
            method="wind_decay_model",
            details=f"wind={wind_speed_knots:.1f} kts, base_duration={base_duration} h",
        )

    # -----------------------------------------------------------------
    # Wind-fan angle estimation
    # -----------------------------------------------------------------

    def _estimate_wind_fan_angle(self, mask: np.ndarray, wind_dir_deg: float) -> float:
        """Estimate the opening angle of the downwind spread fan."""
        ys, xs = np.where(mask)
        if len(xs) < 10:
            return 0.0

        cx, cy = xs.mean(), ys.mean()
        # Angles of all slick pixels relative to centroid
        angles = np.degrees(np.arctan2(ys - cy, xs - cx))
        # Wind direction in image coords (convert meteorological to image angle)
        wind_img = (wind_dir_deg + 180) % 360
        # Angular spread
        diff = (angles - wind_img + 180) % 360 - 180
        spread = np.percentile(np.abs(diff), 90) - np.percentile(np.abs(diff), 10)
        return float(max(spread, 0.0))
