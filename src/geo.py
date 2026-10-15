"""Great-circle distance helpers. Straight-line only; never described as road distance."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088


def valid_lat_lon(lat: float | None, lon: float | None) -> bool:
    if lat is None or lon is None:
        return False
    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except (TypeError, ValueError):
        return False
    if math.isnan(lat_f) or math.isnan(lon_f):
        return False
    return -90.0 <= lat_f <= 90.0 and -180.0 <= lon_f <= 180.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres. This is not road or carrier distance."""
    phi1, phi2 = math.radians(float(lat1)), math.radians(float(lat2))
    d_phi = math.radians(float(lat2) - float(lat1))
    d_lam = math.radians(float(lon2) - float(lon1))
    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2.0) ** 2
    return float(2.0 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a))))


def haversine_km_array(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    phi1 = np.radians(lat1.astype(float))
    phi2 = np.radians(lat2.astype(float))
    d_phi = np.radians(lat2.astype(float) - lat1.astype(float))
    d_lam = np.radians(lon2.astype(float) - lon1.astype(float))
    a = np.sin(d_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lam / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def lookup_centroid(centroids: pd.DataFrame, state: str | None) -> tuple[float | None, float | None]:
    if centroids is None or centroids.empty or not state:
        return None, None
    hit = centroids.loc[centroids["state"].astype(str) == str(state)]
    if hit.empty:
        return None, None
    return float(hit.iloc[0]["latitude"]), float(hit.iloc[0]["longitude"])
