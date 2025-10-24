# app.py
import os
import base64
import mimetypes
from pathlib import Path
from typing import Optional

import streamlit as st
import pandas as pd
import numpy as np
import requests
from calendar import month_name


# Maps & viz
import folium
from folium.plugins import Draw, MarkerCluster, FastMarkerCluster, HeatMap
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go

# ML
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN

# LightGBM (optional) with safe fallback
try:
    from lightgbm import LGBMRegressor as _LGBM
    GBM_AVAILABLE = True
except Exception:
    GBM_AVAILABLE = False
    from sklearn.ensemble import HistGradientBoostingRegressor as _HGBM

def make_gbm():
    """Return a GBM-like regressor; uses LightGBM if available, otherwise HistGBR."""
    if GBM_AVAILABLE:
        return _LGBM(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
    else:
        # HistGradientBoostingRegressor uses max_iter instead of n_estimators
        return _HGBM(max_iter=300, learning_rate=0.05, max_depth=5, random_state=42)



# --- Page + CSS (robust loader)
# Page config first
st.set_page_config(page_title="NowNow Wildlife Finder", layout="wide")

# Load CSS sitting next to app.py (src/app/style.css)
from pathlib import Path
css_path = Path(__file__).parent / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)
else:
    st.warning(f"Missing CSS at {css_path}")


# ==============================
# App header (logo left, title centered)
# ==============================
def _get_header_logo_b64():
    here = Path(__file__).parent / "assets"
    for name in ["logo-no-bg.png","logo-orange.png", "logo-brown.png", "logo-white.png", "logo.png"]:
        p = here / name if here.exists() else Path(name)
        if p.exists():
            return base64.b64encode(p.read_bytes()).decode()
    return None

header_logo_b64 = _get_header_logo_b64()
st.markdown(
    f"""
    <div class="app-header">
        {f'<img class="app-logo-left" src="data:image/png;base64,{header_logo_b64}" alt="logo">' if header_logo_b64 else ''}
        <h1 class="app-title">Now Now Wildlife Finder</h1>
        <div class="app-header-spacer"></div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================
# Load CSV (integrated)
# ==============================
DEFAULT_CSV = os.path.join("data", "animals_temp.csv")

@st.cache_data(show_spinner=False)
def load_and_normalize(path: str) -> pd.DataFrame:
    """
    Normalize columns:
      - animal_group: prefer 'animal group'; else fall back to species from 'common name' or 'name'
      - lat/lon: 'latitude'/'longitude' -> 'lat'/'lon'
      - ndvi: 'NDVI_mean' -> 'ndvi'
      - distance_to_water_km_comprehensive -> 'dist_to_water_km'
    Keep hour/month/temperature_C if present; derive hour/month from 'datetime' if needed.
    """
    df = pd.read_csv(path, low_memory=False)

    # Build rename map (keep original 'species' too if present)
    rename_map = {}
    if "animal group" in df.columns:
        rename_map["animal group"] = "animal_group"
    if "common name" in df.columns:
        rename_map["common name"] = "species"
    elif "name" in df.columns:
        rename_map["name"] = "species"
    if "latitude" in df.columns:  rename_map["latitude"] = "lat"
    if "longitude" in df.columns: rename_map["longitude"] = "lon"
    if "NDVI_mean" in df.columns: rename_map["NDVI_mean"] = "ndvi"
    if "distance_to_water_km_comprehensive" in df.columns:
        rename_map["distance_to_water_km_comprehensive"] = "dist_to_water_km"

    df = df.rename(columns=rename_map)

    # If animal_group missing, create from species if available
    if "animal_group" not in df.columns:
        if "species" in df.columns:
            df["animal_group"] = df["species"]
        else:
            raise ValueError("CSV needs 'animal group' or a species column ('common name'/'name').")

    # Required coords
    missing = {"animal_group", "lat", "lon"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns after mapping: {missing}")

    # Coerce numeric
    for c in ["lat", "lon", "ndvi", "dist_to_water_km", "temperature_C", "hour", "month"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Derive hour/month from datetime if not provided
    if "hour" not in df.columns and "datetime" in df.columns:
        try:
            dt = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
            df["hour"] = dt.dt.hour
            df["month"] = dt.dt.month
        except Exception:
            pass

    # Drop rows without coordinates
    df = df.dropna(subset=["lat", "lon"])

    # If NDVI present, drop rows missing NDVI (keeps models/maps clean)
    if "ndvi" in df.columns:
        before = len(df)
        df = df.dropna(subset=["ndvi"])
        dropped = before - len(df)
        if dropped > 0:
            st.sidebar.info(f"Dropped {dropped} rows with missing NDVI")

    # Keep tidy set
    keep = ["animal_group", "lat", "lon"]
    for c in ["species", "ndvi", "dist_to_water_km", "hour", "month", "temperature_C", "datetime"]:
        if c in df.columns:
            keep.append(c)
    return df[keep].copy()

if not os.path.exists(DEFAULT_CSV):
    st.error(f"CSV not found at {DEFAULT_CSV}. Please place your file there.")
    st.stop()

df = load_and_normalize(DEFAULT_CSV)

# ==============================
# Sidebar navigation
# ==============================
MODES = ["Animal Info", "Sightings Map", "Activity", "Planner"]

if "mode" not in st.session_state:
    st.session_state["mode"] = "Animal Info"
elif st.session_state["mode"] == "Travel":     # migrate old sessions
    st.session_state["mode"] = "Sightings Map"

def _set_mode(m: str):
    st.session_state["mode"] = m

def _get_sidebar_logo_b64(preferred: str | None = None):
    here = Path(__file__).parent / "assets"
    candidates = [preferred, "logo-brown.png", "logo-orange.png", "logo-white.png"] if preferred else \
                 ["logo-brown.png", "logo-orange.png", "logo-white.png"]
    for name in candidates:
        p = here / name
        if p.exists():
            return base64.b64encode(p.read_bytes()).decode()
    return None

with st.sidebar:
    b64 = _get_sidebar_logo_b64(preferred="logo-brown.png")
    if b64:
        st.markdown(
            f"""
            <div class="sidebar-brand">
                {'<img class="sidebar-logo" src="data:image/png;base64,'+b64+'" alt="NowNow logo" />' if b64 else ''}
                <div class="sidebar-brand-text">
                    <div class="sidebar-title">NowNow</div>
                    <div class="sidebar-desc">Plan sightings, explore maps, and build trips</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        st.markdown('<div class="sidebar-brand"><div class="sidebar-title">Now Now</div></div>', unsafe_allow_html=True)

    for m in MODES:
        active = (st.session_state["mode"] == m)
        st.markdown(f'<div class="nav-btn {"nav-active" if active else ""}" data-mode="{m}">', unsafe_allow_html=True)
        st.button(m, key=f"nav_{m}", on_click=_set_mode, args=(m,), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

mode = st.session_state["mode"]


# =========================================================
# -------------------- ACTIVITY HELPERS -------------------
# =========================================================
def have_activity_columns(_df: pd.DataFrame) -> bool:
    # Need ndvi, and either hour+month or datetime to derive them. temperature_C optional.
    has_ndvi = "ndvi" in _df.columns
    has_time = {"hour", "month"}.issubset(_df.columns) or ("datetime" in _df.columns)
    return has_ndvi and has_time

def _build_time_grid() -> pd.DataFrame:
    grid = pd.DataFrame([(m, h) for m in range(1, 13) for h in range(24)], columns=["month", "hour"])
    grid["sin_hour"]  = np.sin(2 * np.pi * grid["hour"] / 24)
    grid["cos_hour"]  = np.cos(2 * np.pi * grid["hour"] / 24)
    grid["sin_month"] = np.sin(2 * np.pi * grid["month"] / 12)
    grid["cos_month"] = np.cos(2 * np.pi * grid["month"] / 12)
    return grid

@st.cache_resource(show_spinner=False)
def train_lgbm(group_df: pd.DataFrame):
    # derive hour/month if only datetime exists
    if ("hour" not in group_df.columns or "month" not in group_df.columns) and "datetime" in group_df.columns:
        dt = pd.to_datetime(group_df["datetime"], errors="coerce", utc=True)
        group_df = group_df.assign(hour=dt.dt.hour, month=dt.dt.month)

    needed = {"hour", "month", "ndvi"}
    if not needed.issubset(group_df.columns) or len(group_df) < 100:
        return None

    # target: counts per (month,hour)
    counts = (
        group_df.groupby(["month","hour"]).size()
        .reindex(pd.MultiIndex.from_product([range(1,13), range(24)]), fill_value=0)
        .values
    )

    grid = _build_time_grid()
    grid["ndvi"] = group_df["ndvi"].mean()

    feature_cols = ["sin_hour","cos_hour","sin_month","cos_month","ndvi"]

    # temperature is optional
    use_temp = "temperature_C" in group_df.columns and group_df["temperature_C"].notna().any()
    if use_temp:
        grid["temperature_C"] = group_df["temperature_C"].mean()
        feature_cols.append("temperature_C")

    X_train, X_test, y_train, y_test = train_test_split(grid[feature_cols], counts, random_state=42)
    model = make_gbm()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    return {"model": model, "use_temp": use_temp, "r2": r2, "rmse": rmse, "feature_cols": feature_cols}

@st.cache_resource(show_spinner=False)
def train_gmm(group_df: pd.DataFrame):
    # derive hour/month if needed
    if ("hour" not in group_df.columns or "month" not in group_df.columns) and "datetime" in group_df.columns:
        dt = pd.to_datetime(group_df["datetime"], errors="coerce", utc=True)
        group_df = group_df.assign(hour=dt.dt.hour, month=dt.dt.month)

    needed = {"hour","month","ndvi"}
    if not needed.issubset(group_df.columns) or len(group_df) < 10:
        return None, False

    use_temp = "temperature_C" in group_df.columns and group_df["temperature_C"].notna().any()
    feat = ["hour","month","ndvi"] + (["temperature_C"] if use_temp else [])

    X = group_df[feat].dropna()
    if X.empty:
        return None, False

    gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42).fit(X)
    return gmm, use_temp

def predict_grid(bundle, group_df: pd.DataFrame):
    grid = _build_time_grid()
    grid["ndvi"] = group_df["ndvi"].mean()
    if bundle["use_temp"]:
        grid["temperature_C"] = group_df["temperature_C"].mean()

    grid["predicted_activity"] = bundle["model"].predict(grid[bundle["feature_cols"]])
    best = grid.loc[grid["predicted_activity"].idxmax()]
    return int(best.month), int(best.hour), grid

def gmm_activity_curve(gmm, month: int, group_df: pd.DataFrame, use_temp: bool):
    hours = np.arange(0, 24)
    ndvi = group_df["ndvi"].mean()
    if use_temp:
        temp = group_df["temperature_C"].mean()
        X_pred = np.column_stack([hours, np.repeat(month, 24), np.repeat(ndvi, 24), np.repeat(temp, 24)])
    else:
        X_pred = np.column_stack([hours, np.repeat(month, 24), np.repeat(ndvi, 24)])
    probs = np.exp(gmm.score_samples(X_pred))
    probs /= probs.sum()
    return pd.DataFrame({"hour": hours, "probability": probs})

def circular_plot(df_pred: pd.DataFrame, month: int):
    subset = df_pred[df_pred["month"] == month]
    if subset.empty:
        subset = df_pred.copy()
        subset["month"] = month
    hours = subset["hour"]
    vals = subset["predicted_activity"]
    fig = go.Figure(go.Barpolar(r=vals, theta=hours * 15, marker_color=vals, marker_colorscale="Viridis", opacity=0.85))
    fig.update_layout(
        title=f"Activity Cycle — Month {month}",
        polar=dict(
            angularaxis=dict(tickmode="array", tickvals=[0, 90, 180, 270],
                             ticktext=["Midnight", "6 AM", "Noon", "6 PM"], direction="clockwise"),
            radialaxis=dict(showticklabels=False, ticks=""),
        ),
        height=500, template="plotly_dark",
    )
    return fig

def daily_profile_plot(df_pred: pd.DataFrame):
    fig = px.line(
        df_pred, x="hour", y="predicted_activity",
        color=df_pred["month"].astype(str),
        color_discrete_sequence=px.colors.sequential.Viridis,
        labels={"hour": "Hour of Day", "predicted_activity": "Predicted Activity", "color": "Month"},
        title="Activity by Hour and Month",
    )
    fig.update_xaxes(tickvals=[0,3,6,9,12,15,18,21,24],
                     ticktext=["0 AM","3 AM","6 AM","9 AM","12 PM","3 PM","6 PM","9 PM","12 AM"])
    fig.update_traces(mode="lines+markers")
    fig.update_layout(height=500)
    return fig

# =========================================================
# ---------------------- MAP HELPERS ----------------------
# =========================================================
@st.cache_resource(show_spinner=False)
def precompute_clusters_global(df_all, eps=0.1, min_samples=10):
    """
    Compute DBSCAN clusters ONCE for the full dataset (degree-based).
    Returns a copy with a 'cluster' column (noise = -1). Cached.
    """
    coords = df_all[["lat","lon"]].dropna().to_numpy()
    out = df_all.copy()
    out["cluster"] = -1
    if len(coords) < min_samples:
        return out
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    out.loc[df_all[["lat","lon"]].dropna().index, "cluster"] = db.labels_
    return out

def downsample(df_in: pd.DataFrame, max_points: int = 5000) -> pd.DataFrame:
    """Sample to a reasonable cap for map rendering."""
    if len(df_in) <= max_points:
        return df_in
    return df_in.sample(max_points, random_state=42)



# =========================================================
# ----------------- Trip Planner helpers -----------------
# =========================================================
EARTH_RADIUS_KM = 6371.0

def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in kilometers."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))

def geocode_place(query: str):
    """
    Use Nominatim to geocode a free-text place name to (lat, lon).
    Returns (lat, lon) or None if not found.
    """
    if not query or not query.strip():
        return None
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": query.strip(), "format": "json", "limit": 1}
    headers = {"User-Agent": "NowNowWildlifeFinder/1.0 (Streamlit; user-requested geocode)"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        return None

def filter_within_radius(df_in, lat0, lon0, radius_km):
    """Return df rows within radius_km of (lat0, lon0)."""
    d = haversine_km(lat0, lon0, df_in["lat"].to_numpy(), df_in["lon"].to_numpy())
    return df_in.assign(_dist_km=d)[d <= radius_km].copy()

def densest_hotspot_center(df_subset):
    """
    Simple hotspot: DBSCAN (degree-based) to find clusters and
    return the center of the largest cluster (by count). Returns (lat, lon) or None.
    """
    if df_subset.empty:
        return None
    coords = df_subset[["lat","lon"]].dropna().to_numpy()
    if len(coords) < 10:
        return None
    db = DBSCAN(eps=0.1, min_samples=10)  # ~11km at equator
    labels = db.fit_predict(coords)
    if (labels >= 0).sum() == 0:
        return None
    dfc = df_subset.iloc[:len(labels)].copy()
    dfc["cluster"] = labels
    # pick largest non-noise cluster
    counts = dfc[dfc["cluster"] >= 0]["cluster"].value_counts()
    top = counts.index[0]
    c = dfc[dfc["cluster"] == top][["lat","lon"]].mean()
    return float(c["lat"]), float(c["lon"])

def best_hour_for_month(bundle, group_df: pd.DataFrame, month: int) -> int:
    """Within a specific month, find the hour that maximizes predicted activity."""
    hours = np.arange(24)
    grid = pd.DataFrame({"month": month, "hour": hours})
    grid["sin_hour"]  = np.sin(2*np.pi*grid["hour"]/24)
    grid["cos_hour"]  = np.cos(2*np.pi*grid["hour"]/24)
    grid["sin_month"] = np.sin(2*np.pi*month/12)
    grid["cos_month"] = np.cos(2*np.pi*month/12)
    grid["ndvi"] = group_df["ndvi"].mean()
    if bundle["use_temp"]:
        grid["temperature_C"] = group_df["temperature_C"].mean()
    preds = bundle["model"].predict(grid[bundle["feature_cols"]])
    return int(hours[np.argmax(preds)])

# ---------- Improved temporal scoring (percentile) ----------
def temporal_score_from_model_percentile(bundle, group_df, month, hour) -> float:
    """
    Robust 0–1 score using percentile rank of the model's score at (month,hour)
    against the full year/day grid distribution (less sensitive to outliers).
    """
    # Build full grid once
    grid = pd.DataFrame([(m,h) for m in range(1,13) for h in range(24)], columns=["month","hour"])
    grid["sin_hour"]  = np.sin(2*np.pi*grid["hour"]/24)
    grid["cos_hour"]  = np.cos(2*np.pi*grid["hour"]/24)
    grid["sin_month"] = np.sin(2*np.pi*grid["month"]/12)
    grid["cos_month"] = np.cos(2*np.pi*grid["month"]/12)
    grid["ndvi"] = group_df["ndvi"].mean()
    if bundle["use_temp"]:
        grid["temperature_C"] = group_df["temperature_C"].mean()

    scores = bundle["model"].predict(grid[bundle["feature_cols"]])

    row = {
        "sin_hour":  np.sin(2*np.pi*hour/24),
        "cos_hour":  np.cos(2*np.pi*hour/24),
        "sin_month": np.sin(2*np.pi*month/12),
        "cos_month": np.cos(2*np.pi*month/12),
        "ndvi":      grid["ndvi"].iloc[0],
    }
    if bundle["use_temp"]:
        row["temperature_C"] = grid["temperature_C"].iloc[0]

    score_at = float(bundle["model"].predict(pd.DataFrame([row])[bundle["feature_cols"]])[0])
    # Percentile rank (with tie-handling)
    less = np.mean(scores < score_at)
    equal = np.mean(scores == score_at)
    return float(less + 0.5 * equal)  # 0..1

def spatial_score(df_group, lat0, lon0, radius_km):
    """Simple spatial density score: fraction of this group's records within radius."""
    total = len(df_group)
    if total == 0:
        return 0.0
    nearby = filter_within_radius(df_group, lat0, lon0, radius_km)
    return float(len(nearby)) / float(total)

# ---------- Improved temporal scoring (percentile) ----------
def temporal_score_from_model_percentile(bundle, group_df, month, hour) -> float:
    """
    Robust 0–1 score using percentile rank of the model's score at (month,hour)
    against the full year/day grid distribution (less sensitive to outliers).
    """
    # Build full grid once
    grid = pd.DataFrame([(m,h) for m in range(1,13) for h in range(24)], columns=["month","hour"])
    grid["sin_hour"]  = np.sin(2*np.pi*grid["hour"]/24)
    grid["cos_hour"]  = np.cos(2*np.pi*grid["hour"]/24)
    grid["sin_month"] = np.sin(2*np.pi*grid["month"]/12)
    grid["cos_month"] = np.cos(2*np.pi*grid["month"]/12)
    grid["ndvi"] = group_df["ndvi"].mean()
    if bundle["use_temp"]:
        grid["temperature_C"] = group_df["temperature_C"].mean()

    scores = bundle["model"].predict(grid[bundle["feature_cols"]])

    row = {
        "sin_hour":  np.sin(2*np.pi*hour/24),
        "cos_hour":  np.cos(2*np.pi*hour/24),
        "sin_month": np.sin(2*np.pi*month/12),
        "cos_month": np.cos(2*np.pi*month/12),
        "ndvi":      grid["ndvi"].iloc[0],
    }
    if bundle["use_temp"]:
        row["temperature_C"] = grid["temperature_C"].iloc[0]

    score_at = float(bundle["model"].predict(pd.DataFrame([row])[bundle["feature_cols"]])[0])
    # Percentile rank (with tie-handling)
    less = np.mean(scores < score_at)
    equal = np.mean(scores == score_at)
    return float(less + 0.5 * equal)  # 0..1


# ---------- Haversine (km) DBSCAN for hotspot center ----------
def _dbscan_haversine_center(df_subset: pd.DataFrame, eps_km: float, min_samples: int = 10):
    """
    Haversine DBSCAN: eps specified in km (converted to radians).
    Returns (lat, lon) of largest non-noise cluster center, or None.
    """
    if df_subset.empty:
        return None
    coords_deg = df_subset[["lat","lon"]].dropna().to_numpy()
    if len(coords_deg) < min_samples:
        return None

    coords_rad = np.radians(coords_deg)
    eps_rad = eps_km / 6371.0  # Earth radius in km
    db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine")
    labels = db.fit_predict(coords_rad)

    mask = labels >= 0
    if not np.any(mask):
        return None

    # Pick largest cluster
    labels_non_noise = labels[mask]
    vals, counts = np.unique(labels_non_noise, return_counts=True)
    top_label = vals[np.argmax(counts)]
    cluster_pts = coords_deg[labels == top_label]
    center = cluster_pts.mean(axis=0)  # back in degrees
    return float(center[0]), float(center[1])


def propose_trip(
    df_all: pd.DataFrame,
    sel_group: str,
    lat0: float,
    lon0: float,
    radius_km: float,
    target_month: int | None = None,
    target_hour: int | None = None,
    choose_best_hour_if_none: bool = True,
    weight_temporal: float = 0.5,     # 0..1
    eps_km: float = 10.0,             # hotspot eps in km (haversine)
    min_samples: int = 10,
):
    """
    Returns:
      - 'where': (lat, lon) or None
      - 'when': (month, hour)
      - 'likelihood': 0–100 blended score
      - 'temporal_favor': 0..1
      - 'spatial_favor' : 0..1
      - 'weight_temporal', 'eps_km', 'min_samples'
      - 'notes': list[str]
    """
    notes = []
    group_df = df_all[df_all["animal_group"] == sel_group].copy()
    if group_df.empty:
        return {"where": None, "when": None, "likelihood": 0,
                "temporal_favor": 0.0, "spatial_favor": 0.0,
                "weight_temporal": weight_temporal, "eps_km": eps_km, "min_samples": min_samples,
                "notes": ["No records for this animal group."]}

    bundle = train_lgbm(group_df)

    # Decide when (month/hour)
    if target_month is not None:
        if bundle is not None and choose_best_hour_if_none and target_hour is None:
            month = int(target_month)
            hour  = best_hour_for_month(bundle, group_df, month)
            notes.append(f"Using model’s best hour for month {month}: {hour:02d}:00.")
        else:
            month = int(target_month)
            hour  = int(target_hour) if target_hour is not None else 6
            if target_hour is None:
                notes.append(f"No hour chosen; defaulting to {hour:02d}:00 for month {month}.")
    else:
        if bundle is None:
            notes.append("Insufficient data for model; using historical counts to suggest time.")
            if {"month","hour"}.issubset(group_df.columns):
                counts = group_df.groupby(["month","hour"]).size().reset_index(name="n")
                if counts.empty:
                    month, hour = 7, 6
                else:
                    top = counts.sort_values("n", ascending=False).iloc[0]
                    month, hour = int(top["month"]), int(top["hour"])
            else:
                month, hour = 7, 6
        else:
            month, hour, _ = predict_grid(bundle, group_df)

    # Spatial: within radius → haversine DBSCAN center
    nearby = filter_within_radius(group_df, lat0, lon0, radius_km)
    if nearby.empty:
        notes.append("No sightings within the chosen radius. Widen the radius or pick another start point.")
        where = None
        spatial_favor = 0.0
    else:
        center = _dbscan_haversine_center(nearby, eps_km=eps_km, min_samples=min_samples)
        if center is None:
            notes.append("Sightings too sparse for a hotspot; using mean of nearby sightings.")
            c = nearby[["lat","lon"]].mean()
            where = (float(c["lat"]), float(c["lon"]))
        else:
            where = center
        spatial_favor = spatial_score(group_df, lat0, lon0, radius_km)

    # Temporal favorability (percentile)
    temporal_favor = 0.6 if bundle is None else temporal_score_from_model_percentile(bundle, group_df, month, hour)

    # Weighted blend
    wt = float(np.clip(weight_temporal, 0.0, 1.0))
    likelihood = int(round(100 * (wt * temporal_favor + (1.0 - wt) * spatial_favor)))

    notes.append(f"Blend: {int(wt*100)}% temporal / {int((1-wt)*100)}% spatial. Hotspot eps ≈ {eps_km:.1f} km.")

    return {
        "where": where, "when": (month, hour), "likelihood": likelihood,
        "temporal_favor": float(temporal_favor), "spatial_favor": float(spatial_favor),
        "weight_temporal": wt, "eps_km": float(eps_km), "min_samples": int(min_samples),
        "notes": notes
    }




# =========================================================
# ---------------- HOTSPOTS (SIMPLE DBSCAN) ---------------
# =========================================================
def train_hotspots_simple(df_subset: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Simple spatial hotspots using degree-based DBSCAN.
    eps=0.1 degrees ≈ ~11 km at the equator (varies with latitude).
    Returns a copy with a 'cluster' column (-1 = noise).
    """
    coords = df_subset[["lat", "lon"]].dropna().to_numpy()
    if len(coords) < 10:
        return None
    db = DBSCAN(eps=0.1, min_samples=10)
    labels = db.fit_predict(coords)
    out = df_subset.copy()
    out["cluster"] = labels
    return out


# =========================================================
# --------------------- SIGHTINGS MAP ---------------------
# =========================================================
if mode == "Sightings Map":
    st.markdown('<h2 class="center" style="margin-bottom:.25rem;">Sightings Map</h2>', unsafe_allow_html=True)
    st.markdown("---")

    # --- Mode buttons ---
    if "sightings_render_mode" not in st.session_state:
        st.session_state["sightings_render_mode"] = "Heatmap"

    def mode_button(label: str, mode_name: str):
        active = st.session_state["sightings_render_mode"] == mode_name
        st.markdown(f'<div class="mode-btn {"active" if active else ""}">', unsafe_allow_html=True)
        clicked = st.button(label, key=f"mode_{mode_name}", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        if clicked:
            st.session_state["sightings_render_mode"] = mode_name

    c1, c2, c3 = st.columns(3)
    with c1: mode_button("Heatmap", "Heatmap")
    with c2: mode_button("Clustered", "Clustered")
    with c3: mode_button("Markers", "Markers")

    render_mode = st.session_state["sightings_render_mode"]
    st.markdown(f'<h4 class="center" style="margin:.25rem 0 .5rem;">{render_mode}</h4>', unsafe_allow_html=True)

    # --- Persist + filter by drawn polygon ---
    if "draw_shape" not in st.session_state:
        st.session_state["draw_shape"] = None
    shape = st.session_state["draw_shape"]

    visible = df.copy()
    if shape and shape.get("type") == "Polygon":
        coords = shape["coordinates"][0]
        lats = [pt[1] for pt in coords]; lons = [pt[0] for pt in coords]
        south, north = min(lats), max(lats); west, east = min(lons), max(lons)
        visible = visible[
            (visible["lat"] >= south) & (visible["lat"] <= north) &
            (visible["lon"] >= west)  & (visible["lon"] <= east)
        ]

    # --- Map base ---
    center = (
        [visible["lat"].mean(), visible["lon"].mean()]
        if not visible.empty else
        [df["lat"].mean(), df["lon"].mean()]
    )
    m = folium.Map(location=center, zoom_start=6, tiles="CartoDB Positron")
    Draw(export=True, draw_options={'rectangle': True, 'polygon': True}).add_to(m)

    if visible.empty:
        st.info("No points to show in the current area.")
    else:
        if render_mode == "Heatmap":
            vis = downsample(visible, max_points=20000)
            heat_data = vis[["lat","lon"]].dropna().values.tolist()
            if heat_data:
                HeatMap(heat_data, radius=12, blur=18, min_opacity=0.3).add_to(m)

        elif render_mode == "Clustered":
            with st.spinner("Computing clusters… (first run only)"):
                df_clustered = precompute_clusters_global(df, eps=0.1, min_samples=10)
            vis = df_clustered.loc[visible.index]
            vis = vis[(vis["cluster"] != -1)]
            pts = vis[["lat","lon"]].dropna().values.tolist()
            if pts:
                FastMarkerCluster(pts).add_to(m)
            else:
                st.warning("No dense clusters in this view. Try Heatmap or widen the area.")

        else:  # --- Markers (fast color circles) + popups ---
            vis = downsample(visible, max_points=4000)

            # Which column names your animal label
            group_col = "group_name" if "group_name" in vis.columns else (
                "animal_group" if "animal_group" in vis.columns else None
            )

            # Stable color map for groups currently visible
            PALETTE = [
                "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
                "#a65628", "#f781bf", "#999999", "#66c2a5", "#8da0cb",
                "#e78ac3", "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3"
            ]
            uniq_groups = (
                vis[group_col].dropna().astype(str).str.strip().unique().tolist()
                if group_col else []
            )
            color_map = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(sorted(uniq_groups))}

            cluster = MarkerCluster().add_to(m)

            def fmt_num(x, n=2):
                try: return f"{float(x):.{n}f}"
                except: return "N/A"

            for _, r in vis.dropna(subset=["lat", "lon"]).iterrows():
                animal  = (r.get(group_col) if group_col else "") or ""
                species = r.get("species", "")
                dt      = r.get("datetime", "")
                ndvi    = r.get("ndvi", None)
                temp    = r.get("temperature_C", None)
                water   = r.get("dist_to_water_km", None)

                popup_html = f"""
                <div style="font-size:13px; line-height:1.35;">
                  <b>Animal:</b> {animal or '—'}<br>
                  <b>Species:</b> {species or '—'}<br>
                  <b>Date:</b> {dt or '—'}<br>
                  <b>NDVI:</b> {fmt_num(ndvi)}<br>
                  <b>Temp (°C):</b> {fmt_num(temp)}<br>
                  <b>Dist. to water (km):</b> {fmt_num(water)}
                </div>
                """

                col = color_map.get(animal, "#377eb8")
                folium.CircleMarker(
                    location=[r["lat"], r["lon"]],
                    radius=5,
                    color=col,
                    fill=True,
                    fill_color=col,
                    fill_opacity=0.7,
                    popup=folium.Popup(popup_html, max_width=280),
                ).add_to(cluster)

            # Legend (bottom-right)
            if color_map:
                legend_items = "".join(
                    f'<div style="display:flex;align-items:center;margin:2px 0;">'
                    f'<span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:{c};margin-right:6px;"></span>'
                    f'<span style="font-size:12px">{g}</span></div>'
                    for g, c in sorted(color_map.items())
                )
                legend_html = f"""
                <div style="
                    position: fixed;
                    bottom: 20px; right: 20px; z-index: 9999;
                    background: rgba(255,255,255,.95);
                    padding: 8px 10px; border: 1px solid rgba(0,0,0,.15);
                    border-radius: 8px; box-shadow: 0 4px 14px rgba(0,0,0,.08);
                ">
                    <div style="font-size:12px;font-weight:700;margin-bottom:6px;">Animals</div>
                    {legend_items}
                </div>
                """
                m.get_root().html.add_child(folium.Element(legend_html))


    # --- Render + capture drawings ---
    st_data = st_folium(m, use_container_width=True, height=520)
    if st_data and st_data.get("last_active_drawing"):
        st.session_state["draw_shape"] = st_data["last_active_drawing"]["geometry"]
    elif st_data and not st_data.get("last_active_drawing"):
        if "all_drawings" in st_data and not st_data["all_drawings"]:
            st.session_state["draw_shape"] = None

# =========================================================
# ------------------------ ACTIVITY -----------------------
# =========================================================
elif mode == "Activity":
    st.markdown('<h2 class="center" style="margin-bottom:.25rem;">Activity Analysis</h2>', unsafe_allow_html=True)
    st.markdown("---")

    # Which column names your animal label
    group_col = "group_name" if "group_name" in df.columns else (
        "animal_group" if "animal_group" in df.columns else None
    )

    if not group_col or df[group_col].dropna().empty:
        st.info("No animal groups found in this dataset.")
    else:
        groups = sorted(df[group_col].dropna().astype(str).unique())
        sel_group = st.selectbox("Choose an animal group", groups, index=0)

        # centered selected animal header
        st.markdown(f'<h3 class="center">{sel_group}</h3>', unsafe_allow_html=True)

        group_df = df[df[group_col].astype(str) == sel_group]
        if group_df.empty:
            st.info("No rows for this group.")
        elif not have_activity_columns(group_df):
            st.info("This view needs NDVI and time (hour+month or datetime). Temperature is optional.")
        else:
            with st.spinner("Training models…"):
                bundle = train_lgbm(group_df)
                gmm, gmm_uses_temp = train_gmm(group_df)

            if bundle is None:
                st.warning("Not enough data to train the predictive model for this group.")
            else:
                best_m, best_h, grid = predict_grid(bundle, group_df)

                best_month_name = month_name[int(best_m)] if 1 <= int(best_m) <= 12 else str(best_m)

                # best cards (CSS styled)
                st.markdown(
                    f"""
                    <div class="best-cards">
                        <div class="best-card month">
                            <div class="label">Best Month</div>
                            <div class="value">{best_month_name}</div>
                        </div>
                        <div class="best-card hour">
                            <div class="label">Best Hour</div>
                            <div class="value">{best_h:02d}:00</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                tab1, tab2, tab3 = st.tabs(["24-Hour Clock", "Daily Profile (Model)", "Historical Pattern (GMM)"])
                with tab1:
                    st.plotly_chart(
                        circular_plot(grid.assign(month=grid["month"]), best_m),
                        use_container_width=True
                    )
                with tab2:
                    st.plotly_chart(
                        daily_profile_plot(grid.assign(month=grid["month"])),
                        use_container_width=True
                    )
                with tab3:
                    if gmm:
                        df_gmm = gmm_activity_curve(gmm, best_m, group_df, gmm_uses_temp)
                        fig = px.area(
                            df_gmm, x="hour", y="probability",
                            title="GMM — Historical Activity Distribution",
                            labels={"hour": "Hour of Day", "probability": "Relative Likelihood"},
                            template="plotly_dark"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Not enough data to fit the GMM model.")



# =========================================================
# ------------------------- PLANNER -----------------------
# =========================================================
elif mode == "Planner":
    st.subheader("Trip Planner")

    # Seed starting point once (single source of truth for lat/lon)
    if "lat0" not in st.session_state or "lon0" not in st.session_state:
        st.session_state["lat0"] = float(df["lat"].mean()) if not df.empty else 0.0
        st.session_state["lon0"] = float(df["lon"].mean()) if not df.empty else 0.0

    # --- Find your starting point ---
    with st.expander("Find your starting point", expanded=True):
        place_query = st.text_input(
            "Search for a place (city, park, coordinates, etc.)",
            placeholder="e.g., Kruger National Park"
        )

        # Center the button on the page without stretching it
        left, mid, right = st.columns([1, 0.6, 1])
        with mid:
            if st.button("Find on map", key="find_start_btn"):
                loc = geocode_place(place_query)
                if loc is None:
                    st.warning("Couldn’t find that place. Check spelling or try a more specific name.")
                else:
                    st.session_state["lat0"], st.session_state["lon0"] = loc
                    st.success(f"Found: {st.session_state['lat0']:.5f}, {st.session_state['lon0']:.5f}")

        # Map (full width, fixed height)
        m0 = folium.Map(
            location=[st.session_state["lat0"], st.session_state["lon0"]],
            zoom_start=6, tiles="CartoDB Positron"
        )
        folium.Marker(
            [st.session_state["lat0"], st.session_state["lon0"]],
            tooltip="Your location", icon=folium.Icon(color="blue")
        ).add_to(m0)
        st.markdown('<div class="map-shell">', unsafe_allow_html=True)
        st_folium(m0, use_container_width=True, height=275)
        st.markdown('</div>', unsafe_allow_html=True)

# --- Animal hotspot icon helpers (ONLY use *-icon.* files)
    def _icon_stems(name: str) -> list[str]:
        """Return candidate stems that all end with '-icon'."""
        base = (name or "").strip()
        low  = base.lower()
        # normalize spaces to hyphens/underscores, then append '-icon'
        variants = [
            low,
            low.replace(" ", "-"),
            low.replace(" ", "_"),
            low.replace(" ", ""),
        ]
        # ensure each ends with '-icon'
        stems = []
        seen = set()
        for v in variants:
            s = f"{v}-icon" if not v.endswith("-icon") else v
            if s and s not in seen:
                seen.add(s); stems.append(s)
        return stems

    def _icon_data_url(animal_name: str) -> Optional[str]:
        """Look for assets/<normalized>-icon.(png|webp), prefer PNG."""
        assets = Path(__file__).parent / "assets"
        if not assets.exists():
            return None
        exts_preference = (".png", ".webp")  # don't match .jpg/.jpeg to avoid photo picks
        for stem in _icon_stems(animal_name):
            for ext in exts_preference:
                p = assets / f"{stem}{ext}"
                if p.exists():
                    mime = mimetypes.guess_type(p.name)[0] or "image/png"
                    b64  = base64.b64encode(p.read_bytes()).decode()
                    return f"data:{mime};base64,{b64}"
        return None

    @st.cache_data(ttl=86400, show_spinner=False)
    def reverse_geocode(lat: float, lon: float) -> Optional[str]:
        """Reverse geocode (lat, lon) -> display address using Nominatim."""
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {"lat": lat, "lon": lon, "format": "json", "zoom": 14}
        headers = {"User-Agent": "NowNowWildFinder/1.0 (Streamlit; reverse geocode)"}
        try:
            r = requests.get(url, params=params, headers=headers, timeout=10)
            r.raise_for_status()
            data = r.json()
            return data.get("display_name")
        except Exception:
            return None


    # --- Planner form (submit computes) ---
    with st.form("planner_form", clear_on_submit=False):
        st.markdown("### Preferences")

        # Radius + animal selection
        radius_km = st.slider("Travel radius (km)", 5, 500, 100, 5, key="radius_km")
        group_list = sorted(df["animal_group"].dropna().unique()) if "animal_group" in df.columns else []
        sel_group_plan = st.selectbox("Animal group you want to see", group_list, key="sel_group_plan")

        # Fixed temporal weight (no slider in UI)
        weight_temporal = 0.70

        # Target date/hour
        c1, c2 = st.columns(2)
        with c1:
            target_date = st.date_input("Target date (optional)", key="target_date")
            target_month = int(target_date.month) if target_date else None
        with c2:
            hour_mode = st.radio("Hour selection", ["Best hour for that month", "Pick an exact hour"], index=0, key="hour_mode")
            if hour_mode == "Pick an exact hour":
                target_time = st.time_input("Pick a time", value=pd.to_datetime("06:00").time(), key="target_time")
                target_hour = int(target_time.hour)
            else:
                target_hour = None

        # Fixed clustering params (hidden from UI)
        eps_km = 10.0
        min_samples = 10

        # Center the submit button
        b1, b2, b3 = st.columns([1, 0.6, 1])
        with b2:
            submitted = st.form_submit_button("Suggest a plan", use_container_width=False)

    # Persist & compute proposal
    if "proposal" not in st.session_state:
        st.session_state["proposal"] = None

    if submitted and sel_group_plan:
        st.session_state["proposal"] = propose_trip(
            df, sel_group_plan,
            st.session_state["lat0"], st.session_state["lon0"],
            radius_km=st.session_state["radius_km"],
            target_month=(target_month if st.session_state.get("target_date") else None),
            target_hour=target_hour,
            choose_best_hour_if_none=True,
            weight_temporal=weight_temporal,
            eps_km=eps_km,
            min_samples=min_samples,
        )

# --- Render proposal ---
    proposal = st.session_state.get("proposal")
    if proposal:
        st.markdown("### Recommendation")
        if proposal["where"] is None:
            st.error("I couldn't find a viable hotspot within your radius.")
        else:
            lat_rec, lon_rec = proposal["where"]
            when = proposal["when"]
            like = proposal["likelihood"]

            # Pretty month name (e.g., January)
            if when:
                month_name = month_name[int(when[0])]
                when_text = f"{month_name}, around {when[1]:02d}:00"
            else:
                when_text = "N/A"

            # Reverse geocode for a readable address (fallback to coords)
            address = reverse_geocode(lat_rec, lon_rec)
            where_text = address or f"{lat_rec:.4f}, {lon_rec:.4f}"

            st.success(
                f"**Go to:** {where_text}  \n"
                f"**When:** {when_text}  \n"
                f"**Likelihood (approx):** {like}%"
            )

            # Context map showing current point, radius, and hotspot
            m = folium.Map(
                location=[st.session_state["lat0"], st.session_state["lon0"]],
                zoom_start=7, tiles="CartoDB Positron"
            )
            folium.Marker(
                [st.session_state["lat0"], st.session_state["lon0"]],
                tooltip="Your location", icon=folium.Icon(color="blue")
            ).add_to(m)
            folium.Circle(
                [st.session_state["lat0"], st.session_state["lon0"]],
                radius=st.session_state["radius_km"] * 1000, color="blue", fill=False
            ).add_to(m)

            # Hotspot marker with animal-specific icon (fallback to red marker)
            sel_animal = st.session_state.get("sel_group_plan", "")
            icon_url = _icon_data_url(sel_animal)
            if icon_url:
                folium.Marker(
                    [lat_rec, lon_rec],
                    tooltip=f"Suggested hotspot: {sel_animal}",
                    icon=folium.features.CustomIcon(
                        icon_image=icon_url,
                        icon_size=(40, 40),
                        icon_anchor=(20, 38)
                    )
                ).add_to(m)
            else:
                folium.Marker(
                    [lat_rec, lon_rec],
                    tooltip=f"Suggested hotspot: {sel_animal}",
                    icon=folium.Icon(color="red")
                ).add_to(m)

            # Nearby sightings for context
            nearby = filter_within_radius(
                df[df["animal_group"] == st.session_state.get("sel_group_plan","")],
                st.session_state["lat0"], st.session_state["lon0"],
                st.session_state["radius_km"]
            )
            layer = MarkerCluster().add_to(m)
            for _, r in nearby.iterrows():
                folium.CircleMarker(
                    [r["lat"], r["lon"]], radius=4, color="crimson",
                    fill=True, fill_opacity=0.5
                ).add_to(layer)

            st_folium(m, use_container_width=True, height=300)
        # Reset proposal (centered button)
        cb1, cb2, cb3 = st.columns([1, 0.51, 1])
        with cb2:
            if st.button("Clear plan", key="clear_btn"):
                st.session_state["proposal"] = None



# =========================================================
# ---------------------- ANIMAL INFO ----------------------
# =========================================================
else:
    st.markdown('<h2 class="center" style="margin-bottom:.25rem;">Animal Info</h2>', unsafe_allow_html=True)
    st.markdown("---")

    # ---------- Helpers ----------
    def detect_group_col(frame):
        for c in ["animal_group", "group_name", "species", "common_name"]:
            if c in frame.columns:
                return c
        return None

    # (Keep these in case you reuse later; not used for facts now)
    def month_from_datetime(series):
        if "month" in df.columns:
            try:
                return series["month"].astype(int).clip(1, 12)
            except Exception:
                pass
        if "datetime" in df.columns:
            try:
                return pd.to_datetime(df["datetime"], errors="coerce").dt.month
            except Exception:
                pass
        return None

    def hour_from_datetime(series):
        if "hour" in df.columns:
            try:
                return series["hour"].astype(int).clip(0, 23)
            except Exception:
                pass
        if "datetime" in df.columns:
            try:
                return pd.to_datetime(df["datetime"], errors="coerce").dt.hour
            except Exception:
                pass
        return None

    # --- image helpers (assets/<slug>.jpg|.jpeg|.png|.webp) -> data URL ---
    ASSETS_DIR = Path(__file__).parent / "assets"

    def _slug_candidates(name: str) -> list[str]:
        base = name.strip()
        cands = [
            base,
            base.lower(),
            base.replace(" ", ""),
            base.lower().replace(" ", "_"),
            base.lower().replace(" ", "-"),
        ]
        seen, out = set(), []
        for c in cands:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def image_src_for(name: str) -> Optional[str]:
        if not ASSETS_DIR.exists():
            return None
        for stem in _slug_candidates(name):
            for ext in (".jpg", ".jpeg", ".png", ".webp"):
                p = ASSETS_DIR / f"{stem}{ext}"
                if p.exists():
                    mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
                    b64 = base64.b64encode(p.read_bytes()).decode()
                    return f"data:{mime};base64,{b64}"
        return None

    # --- Animal Info blurbs for your exact groups ---
    ANIMAL_KNOWLEDGE = {
        "lion": {
            "habitat": "Open savanna and light woodland; often near water in dry periods.",
            "diet": "Carnivore—antelope, zebra, buffalo; will scavenge.",
            "note": "Social cats in prides; most hunting at dusk/dawn."
        },
        "warthog": {
            "habitat": "Open savanna and short-grass plains with access to burrows.",
            "diet": "Grazer/rooter—short grasses, rhizomes; kneels to feed.",
            "note": "Day-active; uses burrows for refuge at night and in heat."
        },
        "hippopotamus": {
            "habitat": "Rivers, lakes and backwaters with grassy banks.",
            "diet": "Grazer—night feeding on short, lush grasses away from water.",
            "note": "Rests in water by day; well-worn ‘hippo trails’ to feeding lawns."
        },
        "elephant": {
            "habitat": "Savanna, woodland and riparian corridors.",
            "diet": "Grazer/browser—grasses, bark, pods; drinks daily when possible.",
            "note": "Large daily movements; often concentrates near permanent water."
        },
        "zebra": {
            "habitat": "Grasslands and open savannas; edges of woodland.",
            "diet": "Grazer—short to mid-height grasses; regular water visits.",
            "note": "Herd-forming; often mixes with wildebeest for shared vigilance."
        },
        "giraffe": {
            "habitat": "Open woodland and savanna with tall acacia/combretum.",
            "diet": "Browser—favours acacia and other thorny trees; high canopy.",
            "note": "Active mornings/evenings; long vigilance near open water."
        },
        "cheetah": {
            "habitat": "Open grasslands and semi-arid savannas.",
            "diet": "Carnivore—hunts small/mid-sized antelope by daytime sprint.",
            "note": "Seeks open sight lines and low cover; avoids dense thickets."
        },
    }

    def _make_bullets(name: str) -> list[str]:
        """Return 3 short bullet points: habitat, diet, behavior (no name prefix)."""
        info = ANIMAL_KNOWLEDGE.get(name.strip().lower())
        if info:
            return [
                f"Habitat: {info['habitat']}",
                f"Diet: {info['diet']}",
                f"Notes: {info['note']}",
            ]
        # Fallback bullets if not found
        return [
            "Habitat: Savanna and mixed woodland.",
            "Diet: Varies seasonally; often near greener vegetation.",
            "Notes: Sightings cluster near water during dry periods.",
        ]

    def summarize_group(name, sub):
        """Return (bullets, facts) where facts = Records + Range span."""
        facts = [f"Records: {len(sub):,}"]

        if {"lat", "lon"}.issubset(sub.columns):
            try:
                lat_span = float(sub["lat"].max() - sub["lat"].min())
                lon_span = float(sub["lon"].max() - sub["lon"].min())
                approx_km = int((lat_span + lon_span) * 55)  # simple span heuristic
                if approx_km > 0:
                    facts.append(f"Range span (approx): ~{approx_km} km")
            except Exception:
                pass

        bullets = _make_bullets(name)
        return bullets, facts[:2]

    # ---------- Build animal list from df ----------
    group_col = next((c for c in ["animal_group", "group_name", "species", "common_name"] if c in df.columns), None)
    if not group_col or df[group_col].dropna().empty:
        st.info("No animal groups found in this dataset.")
    else:
        groups = sorted(df[group_col].dropna().astype(str).unique())

        animals = []
        for g in groups:
            sub = df[df[group_col].astype(str) == g]
            blurb, facts = summarize_group(g, sub)
            img_src = image_src_for(g)  # data URL or None
            animals.append({"name": g, "blurb": blurb, "facts": facts, "img": img_src})

        # ---------- Render horizontal cards (no search) ----------
        rows = ['<div class="scroll-row">']  # IMPORTANT: no leading spaces
        for an in animals:
            name = an["name"]
            bullets = an.get("blurb", [])
            facts = an.get("facts", [])
            img = an.get("img")

            tags_html = "".join(f'<span class="tag">{f}</span>' for f in facts)
            # image block on top (fallback placeholder if missing)
            if img:
                img_block = f'<img class="card-img" src="{img}" alt="{name}"/>'
            else:
                img_block = '<div class="card-img placeholder"></div>'

            blurb_html = "<ul class='blurb-list'>" + "".join(f"<li>{b}</li>" for b in bullets) + "</ul>"

            rows.append(
                '<div class="card">'
                f'{img_block}'
                '<div class="card-body">'
                f'<h3 class="card-title">{name}</h3>'
                f'{blurb_html}'
                f'<div class="tags">{tags_html}</div>'
                '</div>'
                '</div>'
            )
        rows.append('</div>')
        st.markdown("".join(rows), unsafe_allow_html=True)
