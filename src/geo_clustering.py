# src/geo_clustering.py
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from typing import Tuple, Dict, Any

from .schema import LAT_COL, LON_COL, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
from .config import NOISE_LABEL

EARTH_RADIUS_KM = 6371.0

def km_to_radians(km: float) -> float:
    return km / EARTH_RADIUS_KM

def build_geo_matrix(df: pd.DataFrame) -> np.ndarray:
    """Return 2-col array [lat_rad, lon_rad] with lat/lon sanity checks."""
    if not {LAT_COL, LON_COL}.issubset(df.columns):
        missing = {LAT_COL, LON_COL} - set(df.columns)
        raise ValueError(f"Missing required geo columns: {missing}")

    lat_deg = df[LAT_COL].astype(float).to_numpy()
    lon_deg = df[LON_COL].astype(float).to_numpy()

    if np.any((lat_deg < LAT_MIN) | (lat_deg > LAT_MAX)):
        raise ValueError("Latitude values out of bounds [-90, 90].")
    if np.any((lon_deg < LON_MIN) | (lon_deg > LON_MAX)):
        raise ValueError("Longitude values out of bounds [-180, 180].")

    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)
    return np.c_[lat_rad, lon_rad]

def dbscan_haversine(X_latlon_rad: np.ndarray, eps_km: float, min_samples: int) -> np.ndarray:
    """DBSCAN over haversine distance. eps must be in km; we convert to radians."""
    if X_latlon_rad.ndim != 2 or X_latlon_rad.shape[1] != 2:
        raise ValueError("Expected X with shape (n_samples, 2) = [lat_rad, lon_rad].")

    eps_rad = km_to_radians(eps_km)
    model = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine")
    labels = model.fit_predict(X_latlon_rad)
    return labels

def compute_geo_centroids(df: pd.DataFrame, geo_labels: np.ndarray) -> pd.DataFrame:
    """Mean lat/lon per non-noise geo cluster for plotting."""
    out = (
        df.assign(_label=geo_labels)
        .loc[lambda d: d["_label"] != NOISE_LABEL, [LAT_COL, LON_COL, "_label"]]
        .groupby("_label", as_index=False)
        .agg({LAT_COL: "mean", LON_COL: "mean"})
        .rename(columns={"_label": "geo_label"})
    )
    return out
