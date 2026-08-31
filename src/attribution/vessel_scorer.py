"""
Vessel Scorer — Ranks suspect vessels based on spatio-temporal correlation
with a detected oil spill origin.

Scoring factors:
  1. Proximity to origin point
  2. Proximity in time to estimated spill start
  3. Trajectory alignment (passing through origin area)
  4. Behavioural anomalies (speed drops, course changes)
  5. Vessel type suitability (tankers are more likely sources)
  6. Vessel registration / flag state risk
  7. Draught / cargo state indicators
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from .ais_processor import (
    AISDataProcessor,
    TrafficWindow,
    VesselTrack,
    haversine_km,
)

logger = logging.getLogger(__name__)


@dataclass
class VesselScore:
    """Score breakdown for a single vessel."""
    mmsi: str
    vessel_name: str
    vessel_type: str
    imo: str
    flag: str

    # Component scores [0–1]
    proximity_score: float = 0.0
    temporal_score: float = 0.0
    trajectory_score: float = 0.0
    anomaly_score: float = 0.0
    vessel_type_score: float = 0.0
    flag_risk_score: float = 0.0
    draught_score: float = 0.0

    # Weighted total
    total_score: float = 0.0
    rank: int = 0

    # Supporting details
    min_distance_km: float = 0.0
    nearest_passage_time: Optional[datetime] = None
    anomalies_detected: list[str] = field(default_factory=list)


@dataclass
class AttributionResult:
    """Full result of vessel attribution analysis."""
    scores: list[VesselScore] = field(default_factory=list)
    origin_lat: float = 0.0
    origin_lon: float = 0.0
    origin_time: Optional[datetime] = None
    top_suspect: Optional[VesselScore] = None
    total_vessels_analysed: int = 0


class VesselScorer:
    """Score and rank vessels for oil spill attribution.

    Usage:
        scorer = VesselScorer()
        result = scorer.score_vessels(
            traffic_window=traffic,
            origin_lat=20.5,
            origin_lon=60.3,
            origin_time=datetime(2025, 1, 1, 6, 0),
        )
        for v in result.scores[:5]:
            print(f"{v.vessel_name}: {v.total_score:.3f}")
    """

    # Default weights for scoring components
    DEFAULT_WEIGHTS = {
        "proximity": 0.25,
        "temporal": 0.15,
        "trajectory": 0.20,
        "anomaly": 0.20,
        "vessel_type": 0.10,
        "flag_risk": 0.05,
        "draught": 0.05,
    }

    # Vessel types ranked by likelihood of causing oil spills
    VESSEL_TYPE_RISK = {
        "tanker": 1.0,
        "oil_tanker": 1.0,
        "chemical_tanker": 0.9,
        "cargo": 0.4,
        "bulk_carrier": 0.5,
        "container": 0.3,
        "fishing": 0.2,
        "pleasure_craft": 0.1,
        "sailing": 0.05,
        "tug": 0.2,
        "passenger": 0.15,
        "other": 0.1,
    }

    # Flag states with higher regulatory risk
    FLAG_RISK = {
        "PA": 0.7,   # Panama
        "LR": 0.6,   # Liberia
        "MH": 0.65,  # Marshall Islands
        "HK": 0.4,   # Hong Kong
        "SG": 0.3,   # Singapore
        "MT": 0.55,  # Malta
        "GB": 0.2,   # United Kingdom
        "NO": 0.1,   # Norway
        "US": 0.1,   # United States
        "JP": 0.1,   # Japan
        "DE": 0.1,   # Germany
    }

    def __init__(self, weights: Optional[dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        # Normalise weights
        total_w = sum(self.weights.values())
        self.weights = {k: v / total_w for k, v in self.weights.items()}
        logger.info("VesselScorer initialised with weights: %s", self.weights)

    def score_vessels(
        self,
        traffic_window: TrafficWindow,
        origin_lat: float,
        origin_lon: float,
        origin_time: Optional[datetime] = None,
        origin_radius_km: float = 5.0,
    ) -> AttributionResult:
        """Score all vessels in the traffic window for attribution.

        Parameters
        ----------
        traffic_window : TrafficWindow
            Filtered vessel traffic around the spill.
        origin_lat, origin_lon : float
            Estimated origin point of the spill.
        origin_time : datetime, optional
            Estimated time of the spill start.
        origin_radius_km : float
            Radius around origin considered "close" (affects proximity score).
        """
        scores = []

        for mmsi, track in traffic_window.vessels.items():
            if not track.records:
                continue
            vs = self._score_single_vessel(
                track, origin_lat, origin_lon, origin_time, origin_radius_km
            )
            scores.append(vs)

        # Sort by total score descending (higher = more suspect)
        scores.sort(key=lambda s: s.total_score, reverse=True)
        for i, s in enumerate(scores):
            s.rank = i + 1

        result = AttributionResult(
            scores=scores,
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            origin_time=origin_time,
            top_suspect=scores[0] if scores else None,
            total_vessels_analysed=len(scores),
        )

        logger.info(
            "Attribution: analysed %d vessels, top suspect: %s (%.3f)",
            len(scores),
            result.top_suspect.vessel_name if result.top_suspect else "none",
            result.top_suspect.total_score if result.top_suspect else 0,
        )
        return result

    # -----------------------------------------------------------------
    # Single vessel scoring
    # -----------------------------------------------------------------

    def _score_single_vessel(
        self,
        track: VesselTrack,
        origin_lat: float,
        origin_lon: float,
        origin_time: Optional[datetime],
        origin_radius_km: float,
    ) -> VesselScore:
        vs = VesselScore(
            mmsi=track.mmsi,
            vessel_name=track.vessel_name,
            vessel_type=track.vessel_type,
            imo=track.imo,
            flag=track.flag,
        )

        # 1. Proximity
        distances = []
        for rec in track.records:
            d = haversine_km(origin_lat, origin_lon, rec.lat, rec.lon)
            distances.append(d)
            if not vs.min_distance_km or d < vs.min_distance_km:
                vs.min_distance_km = d
                vs.nearest_passage_time = rec.timestamp

        if distances:
            min_d = min(distances)
            # Score: 1.0 if at origin, decays exponentially
            vs.proximity_score = float(np.exp(-min_d / origin_radius_km))

        # 2. Temporal
        if origin_time and track.records:
            time_diffs = [abs((r.timestamp - origin_time).total_seconds()) / 3600
                          for r in track.records]
            min_hours = min(time_diffs)
            # Score: 1.0 at origin time, decays over 24h
            vs.temporal_score = float(np.exp(-min_hours / 24.0))

        # 3. Trajectory alignment
        vs.trajectory_score = self._compute_trajectory_score(
            track, origin_lat, origin_lon, origin_radius_km
        )

        # 4. Behavioural anomalies
        anomalies = self._detect_anomalies(track, origin_lat, origin_lon, origin_time)
        vs.anomaly_score = min(len(anomalies) / 5.0, 1.0)
        vs.anomalies_detected = anomalies

        # 5. Vessel type
        vtype = track.vessel_type.lower().replace(" ", "_")
        vs.vessel_type_score = self.VESSEL_TYPE_RISK.get(vtype, 0.1)

        # 6. Flag risk
        vs.flag_risk_score = self.FLAG_RISK.get(track.flag, 0.3)

        # 7. Draught
        draughts = [r.draught for r in track.records if r.draught > 0]
        if draughts:
            mean_draught = np.mean(draughts)
            # Higher draught (heavier cargo) → higher score
            vs.draught_score = float(min(mean_draught / 15.0, 1.0))

        # Weighted total
        vs.total_score = (
            self.weights["proximity"] * vs.proximity_score +
            self.weights["temporal"] * vs.temporal_score +
            self.weights["trajectory"] * vs.trajectory_score +
            self.weights["anomaly"] * vs.anomaly_score +
            self.weights["vessel_type"] * vs.vessel_type_score +
            self.weights["flag_risk"] * vs.flag_risk_score +
            self.weights["draught"] * vs.draught_score
        )

        return vs

    # -----------------------------------------------------------------
    # Trajectory scoring
    # -----------------------------------------------------------------

    def _compute_trajectory_score(
        self,
        track: VesselTrack,
        origin_lat: float,
        origin_lon: float,
        radius_km: float,
    ) -> float:
        """Score how well the vessel trajectory aligns with the origin.

        A vessel that passes directly through the origin area scores high.
        Also considers whether the trajectory has a directionality
        consistent with the slick spread.
        """
        sorted_recs = sorted(track.records, key=lambda r: r.timestamp)
        if len(sorted_recs) < 2:
            return 0.0

        # Count how many positions are within the origin radius
        near_origin = 0
        for rec in sorted_recs:
            d = haversine_km(origin_lat, origin_lon, rec.lat, rec.lon)
            if d < radius_km:
                near_origin += 1

        # Trajectory through-origin ratio
        passage_ratio = near_origin / len(sorted_recs)
        # Also penalise if vessel just skirts the area
        min_d = min(haversine_km(origin_lat, origin_lon, r.lat, r.lon) for r in sorted_recs)
        distance_factor = np.exp(-min_d / (radius_km * 2))

        return float(min(passage_ratio * 3 + distance_factor, 1.0))

    # -----------------------------------------------------------------
    # Anomaly detection
    # -----------------------------------------------------------------

    def _detect_anomalies(
        self,
        track: VesselTrack,
        origin_lat: float,
        origin_lon: float,
        origin_time: Optional[datetime],
    ) -> list[str]:
        """Detect behavioural anomalies near the origin time and location."""
        anomalies = []
        sorted_recs = sorted(track.records, key=lambda r: r.timestamp)

        # Speed anomalies near origin
        for rec in sorted_recs:
            d = haversine_km(origin_lat, origin_lon, rec.lat, rec.lon)
            if d < 10:  # within 10 km of origin
                if rec.sog < 2.0:
                    anomalies.append(f"Low speed ({rec.sog:.1f} kts) near origin at {rec.timestamp}")
                if rec.sog < 0.5:
                    anomalies.append(f"Near-stationary vessel near origin at {rec.timestamp}")

        # Course changes near origin
        for i in range(1, len(sorted_recs)):
            d = haversine_km(origin_lat, origin_lon, sorted_recs[i].lat, sorted_recs[i].lon)
            if d < 20:
                delta_cog = abs(sorted_recs[i].cog - sorted_recs[i-1].cog)
                delta_cog = min(delta_cog, 360 - delta_cog)
                if delta_cog > 45:
                    anomalies.append(
                        f"Sharp course change ({delta_cog:.0f}°) near origin at {sorted_recs[i].timestamp}"
                    )

        # Stop event (speed drops to near zero then resumes)
        for i in range(1, len(sorted_recs) - 1):
            if sorted_recs[i-1].sog > 3 and sorted_recs[i].sog < 1 and sorted_recs[i+1].sog > 3:
                anomalies.append(f"Stop event at {sorted_recs[i].timestamp}")

        return anomalies

    # -----------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------

    def format_report(self, result: AttributionResult) -> str:
        """Format attribution result as a human-readable report."""
        lines = [
            "=" * 60,
            "OIL SPILL ATTRIBUTION REPORT",
            "=" * 60,
            f"Origin: ({result.origin_lat:.4f}, {result.origin_lon:.4f})",
            f"Estimated time: {result.origin_time}",
            f"Vessels analysed: {result.total_vessels_analysed}",
            "-" * 60,
        ]

        for vs in result.scores[:10]:
            lines.extend([
                f"\nRank #{vs.rank}: {vs.vessel_name} (MMSI: {vs.mmsi})",
                f"  Type: {vs.vessel_type}  |  Flag: {vs.flag}  |  IMO: {vs.imo}",
                f"  Total Score: {vs.total_score:.3f}",
                f"  Components:",
                f"    Proximity:  {vs.proximity_score:.3f}  (min dist: {vs.min_distance_km:.1f} km)",
                f"    Temporal:   {vs.temporal_score:.3f}",
                f"    Trajectory: {vs.trajectory_score:.3f}",
                f"    Anomaly:    {vs.anomaly_score:.3f}",
                f"    VesselType: {vs.vessel_type_score:.3f}",
                f"    Flag Risk:  {vs.flag_risk_score:.3f}",
                f"    Draught:    {vs.draught_score:.3f}",
            ])
            if vs.anomalies_detected:
                lines.append(f"  Anomalies:")
                for a in vs.anomalies_detected[:3]:
                    lines.append(f"    - {a}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
