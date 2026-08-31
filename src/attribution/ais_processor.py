"""
AIS Data Processor — Ingest, clean, and reconstruct vessel traffic from
Automatic Identification System data.

Supports loading from CSV/JSON AIS records, spatial and temporal filtering,
and building a traffic picture around a suspected spill origin.

AIS data fields handled:
  • MMSI, IMO, vessel name, type, flag
  • Latitude, longitude, SOG (speed), COG (course)
  • Draught, destination, ETA
  • Timestamp
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vessel record
# ---------------------------------------------------------------------------

@dataclass
class VesselRecord:
    """A single AIS position report."""
    mmsi: str
    timestamp: datetime
    lat: float
    lon: float
    sog: float = 0.0       # speed over ground (knots)
    cog: float = 0.0       # course over ground (degrees)
    heading: Optional[float] = None
    draught: float = 0.0   # metres
    vessel_name: str = ""
    vessel_type: str = ""
    imo: str = ""
    flag: str = ""
    destination: str = ""
    eta: Optional[datetime] = None

    @property
    def position(self) -> tuple[float, float]:
        return (self.lat, self.lon)


@dataclass
class VesselTrack:
    """Reconstructed track for a single vessel."""
    mmsi: str
    vessel_name: str
    vessel_type: str
    imo: str
    flag: str
    records: list[VesselRecord] = field(default_factory=list)

    @property
    def time_range(self) -> tuple[datetime, datetime]:
        if not self.records:
            raise ValueError("Empty track")
        times = [r.timestamp for r in self.records]
        return min(times), max(times)

    @property
    def path(self) -> list[tuple[float, float]]:
        """Ordered (lat, lon) positions."""
        sorted_recs = sorted(self.records, key=lambda r: r.timestamp)
        return [(r.lat, r.lon) for r in sorted_recs]

    @property
    def mean_speed(self) -> float:
        if not self.records:
            return 0.0
        return float(np.mean([r.sog for r in self.records]))

    @property
    def max_speed(self) -> float:
        if not self.records:
            return 0.0
        return float(max(r.sog for r in self.records))


@dataclass
class TrafficWindow:
    """Filtered vessel traffic around a suspected spill."""
    center_lat: float
    center_lon: float
    radius_km: float
    start_time: datetime
    end_time: datetime
    vessels: dict[str, VesselTrack] = field(default_factory=dict)
    total_records: int = 0

    @property
    def num_vessels(self) -> int:
        return len(self.vessels)

    def get_track(self, mmsi: str) -> Optional[VesselTrack]:
        return self.vessels.get(mmsi)


# ---------------------------------------------------------------------------
# AIS Data Processor
# ---------------------------------------------------------------------------

class AISDataProcessor:
    """Process AIS data: load, filter, and reconstruct vessel traffic.

    Usage:
        processor = AISDataProcessor()
        processor.load_csv("ais_data.csv")
        traffic = processor.build_traffic_window(
            center_lat=20.5, center_lon=60.3,
            radius_km=50, start_time=t1, end_time=t2,
        )
    """

    def __init__(self):
        self.records: list[VesselRecord] = []
        logger.info("AISDataProcessor initialised")

    # -----------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------

    def load_csv(self, path: str):
        """Load AIS records from a CSV file.

        Expected columns: mmsi, timestamp, lat, lon, sog, cog, ...
        Missing columns are filled with defaults.
        """
        import csv
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rec = self._parse_row(row)
                    self.records.append(rec)
                except Exception as e:
                    logger.warning("Skipping malformed AIS row: %s", e)
        logger.info("Loaded %d AIS records from %s", len(self.records), path)

    def load_json(self, path: str):
        """Load AIS records from a JSON file (list of dicts)."""
        import json
        with open(path) as f:
            data = json.load(f)
        for row in data:
            try:
                self.records.append(self._parse_row(row))
            except Exception as e:
                logger.warning("Skipping malformed AIS record: %s", e)
        logger.info("Loaded %d AIS records from %s", len(self.records), path)

    def load_records(self, records: list[VesselRecord]):
        """Load pre-parsed VesselRecord objects."""
        self.records.extend(records)

    def generate_synthetic(
        self,
        center_lat: float = 20.5,
        center_lon: float = 60.3,
        radius_km: float = 100,
        n_vessels: int = 25,
        n_hours: int = 48,
        suspect_mmsi: Optional[str] = None,
    ):
        """Generate synthetic AIS data for testing/demo.

        Creates a set of vessels with random tracks. If suspect_mmsi is
        provided, one vessel is created with a track that passes near
        the origin and exhibits anomalous behaviour (slow speed, course
        change near origin).
        """
        # Use location-based seed so each region gets unique vessel data
        import hashlib
        loc_seed = int(hashlib.md5(f'{center_lat:.2f}{center_lon:.2f}'.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(loc_seed)
        start_time = datetime(2025, 1, 1)

        lat_m = 111_320.0
        lon_m = 111_320.0 * max(np.cos(np.radians(center_lat)), 0.01)
        radius_lat = (radius_km * 1000) / lat_m
        radius_lon = (radius_km * 1000) / lon_m

        # Location-specific vessel names and types
        region_prefixes = {
            (15, 80): "MUMBAI", (25, 55): "DUBAI", (30, 50): "PERSIAN",
            (25, -90): "GULF_MEXICO", (1, 104): "MALACCA", (35, -5): "GIBRALTAR",
        }
        prefix = "OIL"
        for (lat_range, lon_range), name_prefix in region_prefixes.items():
            if abs(center_lat - lat_range) < 5 and abs(center_lon - lon_range) < 10:
                prefix = name_prefix
                break

        vessel_types = ["tanker", "cargo", "bulk_carrier", "container",
                        "fishing", "pleasure_craft", "sailing", "tug"]
        flags = ["PA", "LR", "HK", "SG", "MT", "MH", "GB", "NO"]

        vessels_created = []
        for i in range(n_vessels):
            mmsi = f"{suspect_mmsi}" if (i == 0 and suspect_mmsi) else f"200000{i:04d}"
            name = f"{prefix}_{vessel_types[i % len(vessel_types)].upper()}_{i:03d}" if i > 0 else f"{prefix}_TANKER"
            vtype = vessel_types[i % len(vessel_types)]
            flag = flags[i % len(flags)]

            # Random track
            start_lat = center_lat + rng.uniform(-radius_lat, radius_lat)
            start_lon = center_lon + rng.uniform(-radius_lon, radius_lon)
            speed = rng.uniform(3, 15)  # knots
            course = rng.uniform(0, 360)

            track = []
            lat, lon = start_lat, start_lon
            for h in range(n_hours):
                t = start_time + timedelta(hours=h)
                # Random walk with some trend
                course += rng.normal(0, 10)
                speed = max(0.5, speed + rng.normal(0, 0.5))
                dlat = speed * np.cos(np.radians(course)) * (3600 / lat_m) * 0.5144
                dlon = speed * np.sin(np.radians(course)) * (3600 / lon_m) * 0.5144
                lat += dlat
                lon += dlon

                track.append(VesselRecord(
                    mmsi=mmsi, timestamp=t, lat=lat, lon=lon,
                    sog=speed, cog=course,
                    vessel_name=name, vessel_type=vtype, imo=f"IMO{mmsi}",
                    flag=flag, draught=rng.uniform(5, 15),
                ))

            # Suspect vessel: passes near origin, slows down, changes course
            if i == 0 and suspect_mmsi:
                # Override with a suspicious track
                origin_lat = center_lat
                origin_lon = center_lon
                suspect_track = []
                lat = center_lat + radius_lat * 0.8
                lon = center_lon - radius_lon * 0.6
                for h in range(n_hours):
                    t = start_time + timedelta(hours=h)
                    # Steer toward origin
                    dlat = (origin_lat - lat) * 0.1 + rng.normal(0, 0.001)
                    dlon = (origin_lon - lon) * 0.1 + rng.normal(0, 0.001)
                    # Slow down near origin
                    dist_to_origin = np.sqrt((lat - origin_lat)**2 + (lon - origin_lon)**2)
                    speed_kts = max(0.5, 12.0 * (1 - np.exp(-dist_to_origin * 10)))
                    course = np.degrees(np.arctan2(dlon, dlat))

                    suspect_track.append(VesselRecord(
                        mmsi=mmsi, timestamp=t, lat=lat, lon=lon,
                        sog=speed_kts, cog=course,
                        vessel_name=f"{prefix}_TANKER", vessel_type="tanker",
                        imo=f"IMO{mmsi}", flag="PA", draught=12.0,
                    ))
                    lat += dlat
                    lon += dlon
                track = suspect_track

            vessels_created.append(mmsi)
            for rec in track:
                self.records.append(rec)

        logger.info(
            "Generated %d synthetic AIS vessels (%d records) around (%.2f, %.2f)",
            n_vessels, len(self.records), center_lat, center_lon,
        )
        return vessels_created

    # -----------------------------------------------------------------
    # Spatial & temporal filtering
    # -----------------------------------------------------------------

    def build_traffic_window(
        self,
        center_lat: float,
        center_lon: float,
        radius_km: float,
        start_time: datetime,
        end_time: datetime,
        vessel_types_filter: Optional[list[str]] = None,
    ) -> TrafficWindow:
        """Filter AIS records to a spatio-temporal window.

        Returns a TrafficWindow with all vessels active in the region
        during the time period.
        """
        radius_lat = (radius_km * 1000) / 111_320.0
        radius_lon = (radius_km * 1000) / (111_320.0 * max(np.cos(np.radians(center_lat)), 0.01))

        window = TrafficWindow(
            center_lat=center_lat,
            center_lon=center_lon,
            radius_km=radius_km,
            start_time=start_time,
            end_time=end_time,
        )

        for rec in self.records:
            # Temporal filter
            if not (start_time <= rec.timestamp <= end_time):
                continue
            # Spatial filter (simple bounding box — approximate)
            if abs(rec.lat - center_lat) > radius_lat * 1.5:
                continue
            if abs(rec.lon - center_lon) > radius_lon * 1.5:
                continue
            # Precise distance check
            dist = haversine_km(center_lat, center_lon, rec.lat, rec.lon)
            if dist > radius_km:
                continue
            # Vessel type filter
            if vessel_types_filter and rec.vessel_type not in vessel_types_filter:
                continue

            # Add to track
            if rec.mmsi not in window.vessels:
                window.vessels[rec.mmsi] = VesselTrack(
                    mmsi=rec.mmsi,
                    vessel_name=rec.vessel_name,
                    vessel_type=rec.vessel_type,
                    imo=rec.imo,
                    flag=rec.flag,
                )
            window.vessels[rec.mmsi].records.append(rec)
            window.total_records += 1

        logger.info(
            "Traffic window: %d vessels, %d records within %.0f km of (%.2f, %.2f) "
            "from %s to %s",
            window.num_vessels, window.total_records, radius_km,
            center_lat, center_lon, start_time, end_time,
        )
        return window

    # -----------------------------------------------------------------
    # Anomaly detection helpers
    # -----------------------------------------------------------------

    def detect_anomalous_speed(self, track: VesselTrack, low_speed_kts: float = 1.0) -> list[datetime]:
        """Return timestamps where vessel speed drops abnormally low."""
        anomalies = []
        sorted_recs = sorted(track.records, key=lambda r: r.timestamp)
        for rec in sorted_recs:
            if rec.sog < low_speed_kts:
                anomalies.append(rec.timestamp)
        return anomalies

    def detect_course_change(self, track: VesselTrack, threshold_deg: float = 30.0) -> list[datetime]:
        """Return timestamps with abrupt course changes."""
        anomalies = []
        sorted_recs = sorted(track.records, key=lambda r: r.timestamp)
        for i in range(1, len(sorted_recs)):
            delta_cog = abs(sorted_recs[i].cog - sorted_recs[i-1].cog)
            delta_cog = min(delta_cog, 360 - delta_cog)
            if delta_cog > threshold_deg:
                anomalies.append(sorted_recs[i].timestamp)
        return anomalies

    # -----------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------

    @staticmethod
    def _parse_row(row: dict) -> VesselRecord:
        """Parse a dict row into a VesselRecord."""
        def parse_time(s):
            for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
                try:
                    return datetime.strptime(s, fmt)
                except (ValueError, TypeError):
                    continue
            raise ValueError(f"Cannot parse time: {s}")

        return VesselRecord(
            mmsi=str(row.get("mmsi", "")),
            timestamp=parse_time(row.get("timestamp", row.get("time", ""))),
            lat=float(row.get("lat", row.get("latitude", 0))),
            lon=float(row.get("lon", row.get("longitude", 0))),
            sog=float(row.get("sog", row.get("speed", 0))),
            cog=float(row.get("cog", row.get("course", 0))),
            heading=float(row.get("heading", row.get("sog", 0))) if row.get("heading") else None,
            draught=float(row.get("draught", 0)),
            vessel_name=str(row.get("vessel_name", row.get("name", ""))),
            vessel_type=str(row.get("vessel_type", row.get("type", ""))),
            imo=str(row.get("imo", "")),
            flag=str(row.get("flag", "")),
            destination=str(row.get("destination", "")),
        )


# ---------------------------------------------------------------------------
# Distance helper
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two points."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon / 2) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
