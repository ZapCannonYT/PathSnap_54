"""Geographic helpers.

Coordinates in the raw files are (lon, lat) in degrees. Distances in degrees are
not metres, so every trip is projected to a local metric plane around its own
centroid (equirectangular projection). Trips cover a few km to tens of km, where
the scale error of this projection stays well below 1%.
"""
from __future__ import annotations

import numpy as np

EARTH_R = 6_371_000.0


class LocalProjection:
    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = float(lon0), float(lat0)
        self.kx = EARTH_R * np.cos(np.radians(self.lat0)) * np.pi / 180.0
        self.ky = EARTH_R * np.pi / 180.0

    @classmethod
    def for_points(cls, lonlat: np.ndarray) -> "LocalProjection":
        return cls(np.mean(lonlat[:, 0]), np.mean(lonlat[:, 1]))

    def to_xy(self, lonlat: np.ndarray) -> np.ndarray:
        x = (lonlat[:, 0] - self.lon0) * self.kx
        y = (lonlat[:, 1] - self.lat0) * self.ky
        return np.column_stack([x, y])

    def to_lonlat(self, xy: np.ndarray) -> np.ndarray:
        return np.column_stack([xy[:, 0] / self.kx + self.lon0, xy[:, 1] / self.ky + self.lat0])


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def bearing_deg(dx, dy):
    """Compass-style bearing in [0, 360) from a planar displacement."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_diff(a, b):
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)


def point_segment(px, py, ax, ay, bx, by):
    """Distance from points to segments (broadcasting) and projection fraction t in [0, 1]."""
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    t = np.where(L2 > 0, ((px - ax) * vx + (py - ay) * vy) / np.where(L2 > 0, L2, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    qx, qy = ax + t * vx, ay + t * vy
    return np.hypot(px - qx, py - qy), t


def lonlat_to_latlon(lonlat: np.ndarray) -> list[list[float]]:
    """Folium expects [lat, lon]; the raw files store lon, lat."""
    return [[float(p[1]), float(p[0])] for p in lonlat]
