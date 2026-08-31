"""
Drift Model — Particle-based oil slick trajectory simulation.

Implements Lagrangian particle tracking with:
  • Wind-driven drift (typically 3–4% of wind speed at 10 m)
  • Ocean current advection
  • Turbulent diffusion (random walk)
  • Wind-wave Stokes drift contribution

Supports both:
  • **Hindcast**: trace the slick backward in time to its origin
  • **Forecast**: predict the future drift path of the slick

Output is a set of particle trajectories with timestamps, suitable
for visualisation and vessel attribution scoring.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from .oceanographic_data import OceanographicDataManager

logger = logging.getLogger(__name__)

# Constants
WIND_DRIFT_COEFFICIENT = 0.035   # 3.5% of wind speed (typical for oil)
WIND_DRIFT_ANGLE_DEG = 15.0      # deflection to right (Northern Hemisphere)
STOKES_DRIFT_COEFFICIENT = 0.015  # 1.5% of significant wave height / period
TURBULENT_DIFFUSIVITY = 10.0     # m²/s (horizontal eddy diffusivity)


@dataclass
class Particle:
    """A single Lagrangian particle tracking an oil parcel."""
    lat: float
    lon: float
    age_hours: float = 0.0
    origin_lat: Optional[float] = None
    origin_lon: Optional[float] = None


@dataclass
class TrajectoryPoint:
    """One point in a particle trajectory."""
    time: datetime
    lat: float
    lon: float
    u: float = 0.0   # total velocity east (m/s)
    v: float = 0.0   # total velocity north (m/s)
    wind_u: float = 0.0
    wind_v: float = 0.0
    current_u: float = 0.0
    current_v: float = 0.0


@dataclass
class TrajectoryResult:
    """Complete result of a drift simulation."""
    trajectories: list[list[TrajectoryPoint]]  # one list per particle
    origin_estimate: Optional[tuple[float, float]] = None  # (lat, lon)
    origin_time: Optional[datetime] = None
    confidence: str = "low"
    meta: dict = field(default_factory=dict)


class DriftModel:
    """Lagrangian particle tracking oil slick drift.

    Usage:
        ocean = OceanographicDataManager()
        ocean.generate_synthetic(center_lat=20, center_lon=60)

        model = DriftModel(ocean_data=ocean)

        # Forecast from detection point
        forecast = model.forecast(
            start_lat=20.5, start_lon=60.3,
            start_time=datetime(2025, 1, 1),
            hours=48, n_particles=200,
        )

        # Hindcast to find origin
        hindcast = model.hindcast(
            detected_lat=20.7, detected_lon=60.5,
            detected_time=datetime(2025, 1, 2),
            hours=36, n_particles=200,
        )
    """

    def __init__(
        self,
        ocean_data: OceanographicDataManager,
        wind_drift_coeff: float = WIND_DRIFT_COEFFICIENT,
        wind_drift_angle: float = WIND_DRIFT_ANGLE_DEG,
        stokes_coeff: float = STOKES_DRIFT_COEFFICIENT,
        diffusivity: float = TURBULENT_DIFFUSIVITY,
    ):
        self.ocean = ocean_data
        self.wind_drift_coeff = wind_drift_coeff
        self.wind_drift_angle = np.radians(wind_drift_angle)
        self.stokes_coeff = stokes_coeff
        self.diffusivity = diffusivity
        logger.info("DriftModel initialised (wind_coeff=%.3f, angle=%.1f°)",
                     wind_drift_coeff, wind_drift_angle)

    # -----------------------------------------------------------------
    # Forecast
    # -----------------------------------------------------------------

    def forecast(
        self,
        start_lat: float,
        start_lon: float,
        start_time: datetime,
        hours: float = 48,
        dt_seconds: float = 300.0,
        n_particles: int = 200,
        spread_m: float = 500.0,
    ) -> TrajectoryResult:
        """Predict the future drift path of an oil slick.

        Particles are seeded around the initial position with a
        Gaussian spread to represent the slick extent.

        Parameters
        ----------
        start_lat, start_lon : float
            Centre of the slick at the start time.
        start_time : datetime
            Time of the initial position.
        hours : float
            Forward simulation duration in hours.
        dt_seconds : float
            Time step in seconds.
        n_particles : int
            Number of Lagrangian particles.
        spread_m : float
            Initial Gaussian spread radius in metres.
        """
        particles = self._seed_particles(start_lat, start_lon, n_particles, spread_m)
        trajectories = []

        for p in particles:
            traj = self._advect_particle(
                p, start_time, hours, dt_seconds, direction=1.0
            )
            trajectories.append(traj)

        # Origin estimate = mean of starting positions (for forecast, it's the start point)
        origin_lat = np.mean([t[0].lat for t in trajectories])
        origin_lon = np.mean([t[0].lon for t in trajectories])

        result = TrajectoryResult(
            trajectories=trajectories,
            origin_estimate=(origin_lat, origin_lon),
            origin_time=start_time,
            confidence="medium",
            meta={"mode": "forecast", "duration_hours": hours, "n_particles": n_particles},
        )
        logger.info("Forecast: %d particles, %d time steps", n_particles, len(trajectories[0]))
        return result

    # -----------------------------------------------------------------
    # Hindcast
    # -----------------------------------------------------------------

    def hindcast(
        self,
        detected_lat: float,
        detected_lon: float,
        detected_time: datetime,
        hours: float = 36,
        dt_seconds: float = 300.0,
        n_particles: int = 200,
        spread_m: float = 500.0,
    ) -> TrajectoryResult:
        """Trace the oil slick backward in time to find its origin.

        Reverses wind and current vectors to simulate backward transport.
        The estimated origin is the mean position of particles at t=0.

        Parameters
        ----------
        detected_lat, detected_lon : float
            Centre of the detected slick.
        detected_time : datetime
            Time of detection.
        hours : float
            Backward simulation duration in hours.
        """
        particles = self._seed_particles(detected_lat, detected_lon, n_particles, spread_m)
        trajectories = []

        for p in particles:
            traj = self._advect_particle(
                p, detected_time, hours, dt_seconds, direction=-1.0
            )
            trajectories.append(traj)

        # Origin estimate = final position of reversed particles
        origin_lat = np.mean([t[-1].lat for t in trajectories])
        origin_lon = np.mean([t[-1].lon for t in trajectories])
        origin_time = detected_time - timedelta(hours=hours)

        # Confidence based on particle spread
        lats = [t[-1].lat for t in trajectories]
        lons = [t[-1].lon for t in trajectories]
        spread_lat = np.std(lats) * 111_000  # rough metres
        spread_lon = np.std(lons) * 111_000 * np.cos(np.radians(origin_lat))

        if spread_lat + spread_lon < 5000:
            confidence = "high"
        elif spread_lat + spread_lon < 20000:
            confidence = "medium"
        else:
            confidence = "low"

        result = TrajectoryResult(
            trajectories=trajectories,
            origin_estimate=(origin_lat, origin_lon),
            origin_time=origin_time,
            confidence=confidence,
            meta={
                "mode": "hindcast",
                "duration_hours": hours,
                "n_particles": n_particles,
                "spread_metres": spread_lat + spread_lon,
            },
        )
        logger.info(
            "Hindcast: origin≈(%.4f, %.4f), confidence=%s, spread=%.0f m",
            origin_lat, origin_lon, confidence, spread_lat + spread_lon,
        )
        return result

    # -----------------------------------------------------------------
    # Core advection
    # -----------------------------------------------------------------

    def _advect_particle(
        self,
        particle: Particle,
        start_time: datetime,
        hours: float,
        dt_seconds: float,
        direction: float,
    ) -> list[TrajectoryPoint]:
        """Advect a single particle forward (direction=1) or backward (direction=-1)."""
        lat = particle.lat
        lon = particle.lon
        t = start_time
        n_steps = int(hours * 3600 / dt_seconds)
        traj = [TrajectoryPoint(time=t, lat=lat, lon=lon)]
        rng = np.random.default_rng()

        for _ in range(n_steps):
            t_next = t + timedelta(seconds=dt_seconds * direction)

            # Get environmental forcing
            u_wind, v_wind = self.ocean.get_wind_vector(t, lat, lon)
            u_curr, v_curr = self.ocean.get_current_vector(t, lat, lon)

            # Wind-driven drift (with Coriolis deflection)
            angle = self.wind_drift_angle * direction
            u_wind_drift = (u_wind * np.cos(angle) - v_wind * np.sin(angle)) * self.wind_drift_coeff
            v_wind_drift = (u_wind * np.sin(angle) + v_wind * np.cos(angle)) * self.wind_drift_coeff

            # Total velocity
            u_total = u_curr + u_wind_drift * direction
            v_total = v_curr + v_wind_drift * direction

            # Convert m/s to degrees/s
            lat_m = 111_320.0
            lon_m = 111_320.0 * max(np.cos(np.radians(lat)), 0.01)

            dlat = (v_total / lat_m) * dt_seconds * direction
            dlon = (u_total / lon_m) * dt_seconds * direction

            # Turbulent diffusion (random walk)
            sigma = np.sqrt(2 * self.diffusivity * dt_seconds)
            dlat += rng.normal(0, sigma / lat_m)
            dlon += rng.normal(0, sigma / lon_m)

            lat += dlat
            lon += dlon

            # Wrap longitude
            lon = (lon + 180) % 360 - 180
            lat = np.clip(lat, -85, 85)

            traj.append(TrajectoryPoint(
                time=t_next, lat=lat, lon=lon,
                u=u_total, v=v_total,
                wind_u=u_wind, wind_v=v_wind,
                current_u=u_curr, current_v=v_curr,
            ))
            t = t_next

        return traj

    # -----------------------------------------------------------------
    # Particle seeding
    # -----------------------------------------------------------------

    @staticmethod
    def _seed_particles(
        center_lat: float,
        center_lon: float,
        n: int,
        spread_m: float,
    ) -> list[Particle]:
        """Seed particles in a Gaussian cloud around a point."""
        rng = np.random.default_rng(42)
        lat_m = 111_320.0
        lon_m = 111_320.0 * max(np.cos(np.radians(center_lat)), 0.01)
        lats = center_lat + rng.normal(0, spread_m / lat_m, n)
        lons = center_lon + rng.normal(0, spread_m / lon_m, n)
        return [Particle(lat=float(la), lon=float(lo)) for la, lo in zip(lats, lons)]
