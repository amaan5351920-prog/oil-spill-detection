"""
Oceanographic Data Manager — Handles loading, interpolation, and
retrieval of oceanographic and meteorological data for drift modelling.

Data sources:
  • Wind fields (speed, direction) — ERA5 / GFS reanalysis
  • Ocean currents (u, v) — HYCOM / Copernicus Marine
  • Wave fields (Hs, Tp, direction) — ERA5 wave product
  • SST — Copernicus Marine
  • Bathymetry — GEBCO

All data is accessed through a uniform spatio-temporal query interface.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WindField:
    """Wind field at a single time step."""
    time: datetime
    u: np.ndarray          # zonal wind (m/s), shape (lat, lon)
    v: np.ndarray          # meridional wind (m/s), shape (lat, lon)
    lat: np.ndarray        # latitude grid
    lon: np.ndarray        # longitude grid
    speed: Optional[np.ndarray] = None  # derived wind speed (m/s)
    direction: Optional[np.ndarray] = None  # meteorological direction (deg)

    def __post_init__(self):
        if self.speed is None:
            self.speed = np.sqrt(self.u ** 2 + self.v ** 2)
        if self.direction is None:
            self.direction = np.degrees(np.arctan2(-self.u, -self.v)) % 360


@dataclass
class CurrentField:
    """Ocean current field at a single time step."""
    time: datetime
    u: np.ndarray          # zonal current (m/s)
    v: np.ndarray          # meridional current (m/s)
    lat: np.ndarray
    lon: np.ndarray
    depth: float = 0.0     # surface current by default


@dataclass
class WaveField:
    """Wave field at a single time step."""
    time: datetime
    hs: np.ndarray         # significant wave height (m)
    tp: np.ndarray         # peak period (s)
    direction: np.ndarray  # wave direction (deg, from)
    lat: np.ndarray
    lon: np.ndarray


@dataclass
class OceanState:
    """Complete ocean state at a given time."""
    time: datetime
    wind: Optional[WindField] = None
    current: Optional[CurrentField] = None
    wave: Optional[WaveField] = None


class OceanographicDataManager:
    """Manages spatio-temporal queries for oceanographic data.

    In production this would interface with OPeNDAP/THREDDS servers,
    local NetCDF files, or APIs (Copernicus Marine, NOAA ERDDAP).

    This implementation provides:
    1. Synthetic data generation for testing
    2. Bilinear interpolation over the query grid
    3. Temporal interpolation between time steps
    """

    def __init__(self):
        self.wind_data: dict[datetime, WindField] = {}
        self.current_data: dict[datetime, CurrentField] = {}
        self.wave_data: dict[datetime, WaveField] = {}
        logger.info("OceanographicDataManager initialised")

    # -----------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------

    def load_wind(self, wind_field: WindField):
        """Store a wind field for its time step."""
        self.wind_data[wind_field.time] = wind_field

    def load_current(self, current_field: CurrentField):
        self.current_data[current_field.time] = current_field

    def load_wave(self, wave_field: WaveField):
        self.wave_data[wave_field.time] = wave_field

    def load_netcdf(self, path: str, variable_type: str = "wind"):
        """Load data from a NetCDF file (requires netCDF4)."""
        try:
            import netCDF4 as nc
        except ImportError:
            logger.warning("netCDF4 not installed — cannot load %s", path)
            return

        ds = nc.Dataset(path, "r")
        time_var = ds.variables.get("time") or ds.variables.get("valid_time")
        lat = np.array(ds.variables.get("latitude", ds.variables.get("lat")))
        lon = np.array(ds.variables.get("longitude", ds.variables.get("lon")))

        if variable_type == "wind":
            u = np.array(ds.variables.get("u10", ds.variables.get("u")))
            v = np.array(ds.variables.get("v10", ds.variables.get("v")))
            times = [datetime(1970, 1, 1) + timedelta(seconds=float(t)) for t in time_var]
            for i, t in enumerate(times):
                self.wind_data[t] = WindField(time=t, u=u[i], v=v[i], lat=lat, lon=lon)

        ds.close()
        logger.info("Loaded %s data from %s (%d time steps)", variable_type, path, len(self.wind_data))

    def generate_synthetic(
        self,
        center_lat: float = 20.0,
        center_lon: float = 60.0,
        n_lat: int = 50,
        n_lon: int = 50,
        start_time: Optional[datetime] = None,
        duration_hours: int = 72,
        time_step_hours: int = 6,
    ):
        """Generate synthetic oceanographic data for testing/demo.

        Creates realistic-looking wind, current, and wave fields
        with spatio-temporal variation.
        """
        if start_time is None:
            start_time = datetime.utcnow()

        lat = np.linspace(center_lat - 2, center_lat + 2, n_lat)
        lon = np.linspace(center_lon - 2, center_lon + 2, n_lon)
        lon_grid, lat_grid = np.meshgrid(lon, lat)

        np.random.seed(42)
        n_steps = duration_hours // time_step_hours + 1

        for i in range(n_steps):
            t = start_time + timedelta(hours=i * time_step_hours)

            # Wind: varying trade-wind pattern with noise
            base_u = -5.0 + 2.0 * np.sin(2 * np.pi * i / n_steps)
            base_v = -2.0 + 1.5 * np.cos(2 * np.pi * i / n_steps)
            u = base_u + 0.5 * np.random.randn(n_lat, n_lon) + \
                0.3 * np.sin(0.5 * lat_grid) * np.cos(0.5 * lon_grid)
            v = base_v + 0.5 * np.random.randn(n_lat, n_lon) + \
                0.2 * np.sin(0.3 * lon_grid)
            self.wind_data[t] = WindField(time=t, u=u, v=v, lat=lat, lon=lon)

            # Current: geostrophic-like flow
            cu = 0.3 * np.sin(np.pi * lat_grid / 20) + 0.1 * np.random.randn(n_lat, n_lon)
            cv = 0.2 * np.cos(np.pi * lon_grid / 30) + 0.1 * np.random.randn(n_lat, n_lon)
            self.current_data[t] = CurrentField(time=t, u=cu, v=cv, lat=lat, lon=lon)

            # Waves
            hs = np.abs(1.0 + 0.3 * np.sin(2 * np.pi * i / n_steps) +
                         0.1 * np.random.randn(n_lat, n_lon))
            tp = np.abs(5.0 + 1.0 * np.sin(2 * np.pi * i / n_steps) +
                         0.5 * np.random.randn(n_lat, n_lon))
            wd = (200 + 20 * np.sin(2 * np.pi * i / n_steps) +
                  10 * np.random.randn(n_lat, n_lon)) % 360
            self.wave_data[t] = WaveField(time=t, hs=hs, tp=tp, direction=wd, lat=lat, lon=lon)

        logger.info(
            "Generated synthetic ocean data: %d time steps, %d×%d grid, "
            "center=(%.1f, %.1f)",
            n_steps, n_lat, n_lon, center_lat, center_lon,
        )

    # -----------------------------------------------------------------
    # Spatio-temporal queries
    # -----------------------------------------------------------------

    def get_state(self, time: datetime, lat: float, lon: float) -> OceanState:
        """Get interpolated ocean state at a point and time."""
        return OceanState(
            time=time,
            wind=self._interpolate_field(self.wind_data, time, lat, lon, "wind"),
            current=self._interpolate_field(self.current_data, time, lat, lon, "current"),
            wave=self._interpolate_field(self.wave_data, time, lat, lon, "wave"),
        )

    def get_wind_vector(self, time: datetime, lat: float, lon: float) -> tuple[float, float]:
        """Get (u, v) wind vector at a point."""
        t_floor = self._time_floor(time, self.wind_data)
        if t_floor is None:
            return 0.0, 0.0
        wf = self.wind_data[t_floor]
        u = bilinear_interp(wf.lat, wf.lon, wf.u, lat, lon)
        v = bilinear_interp(wf.lat, wf.lon, wf.v, lat, lon)
        return float(u), float(v)

    def get_current_vector(self, time: datetime, lat: float, lon: float) -> tuple[float, float]:
        """Get (u, v) current vector at a point."""
        t_floor = self._time_floor(time, self.current_data)
        if t_floor is None:
            return 0.0, 0.0
        cf = self.current_data[t_floor]
        u = bilinear_interp(cf.lat, cf.lon, cf.u, lat, lon)
        v = bilinear_interp(cf.lat, cf.lon, cf.v, lat, lon)
        return float(u), float(v)

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _time_floor(target: datetime, data: dict[datetime, ...]) -> Optional[datetime]:
        """Find the largest time key ≤ target."""
        keys = sorted(data.keys())
        if not keys or keys[0] > target:
            return None
        for k in reversed(keys):
            if k <= target:
                return k
        return keys[0]

    def _interpolate_field(self, data, time, lat, lon, field_type):
        """Temporal + spatial interpolation."""
        t_floor = self._time_floor(time, data)
        if t_floor is None:
            return None
        return data[t_floor]  # Simplified: return nearest time step


# ---------------------------------------------------------------------------
# Bilinear interpolation helper
# ---------------------------------------------------------------------------

def bilinear_interp(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    values: np.ndarray,
    lat: float,
    lon: float,
) -> float:
    """Bilinear interpolation on a regular lat/lon grid."""
    lat = np.clip(lat, lat_grid.min(), lat_grid.max())
    lon = np.clip(lon, lon_grid.min(), lon_grid.max())

    # Find indices
    j = np.searchsorted(lon_grid, lon) - 1
    i = np.searchsorted(lat_grid, lat) - 1
    j = np.clip(j, 0, len(lon_grid) - 2)
    i = np.clip(i, 0, len(lat_grid) - 2)

    lon0, lon1 = lon_grid[j], lon_grid[j + 1]
    lat0, lat1 = lat_grid[i], lat_grid[i + 1]

    if lon1 == lon0:
        lon1 = lon0 + 1e-10
    if lat1 == lat0:
        lat1 = lat0 + 1e-10

    fy = (lat - lat0) / (lat1 - lat0)
    fx = (lon - lon0) / (lon1 - lon0)

    val = (
        values[i, j] * (1 - fx) * (1 - fy) +
        values[i, j + 1] * fx * (1 - fy) +
        values[i + 1, j] * (1 - fx) * fy +
        values[i + 1, j + 1] * fx * fy
    )
    return float(val)
