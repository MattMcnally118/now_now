# app.py
import os
import base64
from typing import Optional

import streamlit as st
import pandas as pd
import numpy as np

# Networking for geocoding
import requests

# Maps & viz
import folium
from folium.plugins import Draw, MarkerCluster
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go

# ML
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
try:
    from lightgbm import LGBMRegressor
    GBM_AVAILABLE = True
except Exception:
    GBM_AVAILABLE = False
    from sklearn.ensemble import HistGradientBoostingRegressor as LGBMRegressor


# ==============================
# Header with logo (if present)
# ==============================
logo_base64 = None
if os.path.exists("logo.png"):
    with open("logo.png", "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode()

st.set_page_config(page_title="NowNow WildFinder", layout="wide")
st.markdown(
    f"""
    <div style="
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: #c09153;
        padding: 15px 25px;
        border-radius: 8px;
        margin-bottom: 20px;">
        <h1 style="color: white; font-family: 'Helvetica Neue', sans-serif; font-weight: 600; margin: 0; font-size: 32px;">
            NowNow WildFinder
        </h1>
        {f'<img src="data:image/png;base64,{logo_base64}" width="120" style="border-radius:8px;">' if logo_base64 else ''}
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================
# Sidebar styling
# ==============================
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { background-color: #c09153 !important; }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] label { color: white !important; font-weight: 500 !important; }
    [data-testid="stSidebar"] input, [data-testid="stSidebar"] select {
        background-color: #f8f5f1 !important; color: #333 !important; border-radius: 6px !important;
    }
    </style>
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
st.sidebar.success(f"Loaded dataset: {DEFAULT_CSV}  •  {len(df)} rows")

# ==============================
# Sidebar controls (global)
# ==============================
st.sidebar.title("Explore Wildlife")
mode = st.sidebar.radio("Select Mode", ["Travel", "Activity", "Performance", "Planner"], index=0)
start_date = st.sidebar.date_input("Start Date")
end_date = st.sidebar.date_input("End Date")

group_list = sorted(df["animal_group"].dropna().unique()) if "animal_group" in df.columns else []
sel_group = st.sidebar.selectbox("Animal group (for Activity view)", group_list) if group_list else None


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
    model = LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
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
    headers = {"User-Agent": "NowNowWildFinder/1.0 (Streamlit; user-requested geocode)"}
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
# ------------------------ TRAVEL -------------------------
# =========================================================
if mode == "Travel":
    st.subheader("WildFinder Travel View – Dataset")
    st.write(f"Dates: {start_date} → {end_date}")
    st.markdown("---")

    view_option = st.sidebar.radio("View as:", ["Animal List", "Interactive Map"], index=1)

    # Simple cards (placeholder)
    cards_data = [
        {"name": "Cheetah group", "likelihood": 0.85, "active_months": "Year-round",
         "image": "https://upload.wikimedia.org/wikipedia/commons/4/4e/African_Elephant.jpg"},
        {"name": "Lion group", "likelihood": 0.78, "active_months": "Year-round",
         "image": "https://upload.wikimedia.org/wikipedia/commons/1/1b/Lion_au_repos_parc_pendjari.jpg"},
        {"name": "Zebra group", "likelihood": 0.90, "active_months": "Year-round",
         "image": "https://upload.wikimedia.org/wikipedia/commons/2/2d/Plains_Zebra_Equus_quagga_cropped.jpg"},
    ]
    cards_df = pd.DataFrame(cards_data)

    if view_option == "Animal List":
        for _, row in cards_df.iterrows():
            st.markdown(
                f"""
                <div style="display:flex; background:#f9f9f9; border-radius:12px; padding:12px;
                box-shadow:2px 2px 12px rgba(0,0,0,0.1); margin-bottom:15px; align-items:center;">
                    <img src="{row['image']}" width="180" style="border-radius:10px; margin-right:15px;">
                    <div style="flex:1; font-family:sans-serif;">
                        <h3 style="margin:0; color:#264653;">{row['name']}</h3>
                        <p style="margin:4px 0; font-size:14px;"><strong>Likelihood of Sighting:</strong> {row['likelihood']*100:.0f}%</p>
                        <p style="margin:2px 0; font-size:14px;"><strong>Active Months:</strong> {row['active_months']}</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    else:
        # Persist last drawn polygon
        if "draw_shape" not in st.session_state:
            st.session_state["draw_shape"] = None

        shape = st.session_state["draw_shape"]

        # Filter by polygon (if any)
        visible = df.copy()
        if shape and shape.get("type") == "Polygon":
            coords = shape["coordinates"][0]
            lats = [pt[1] for pt in coords]; lons = [pt[0] for pt in coords]
            south, north = min(lats), max(lats); west, east = min(lons), max(lons)
            visible = visible[(visible["lat"]>=south)&(visible["lat"]<=north)&(visible["lon"]>=west)&(visible["lon"]<=east)]

        # Center map
        center = [visible["lat"].mean(), visible["lon"].mean()] if not visible.empty else [df["lat"].mean(), df["lon"].mean()]
        m = folium.Map(location=center, zoom_start=7, tiles="CartoDB Positron")
        Draw(export=True, draw_options={'rectangle': True, 'polygon': True}).add_to(m)

        if not visible.empty:
            hot = train_hotspots_simple(visible)
            if hot is None or hot["cluster"].nunique() <= 1:
                st.warning("Not enough spatial data to detect hotspots.")
            else:
                layer = MarkerCluster().add_to(m)
                for _, r in hot.iterrows():
                    if r["cluster"] != -1:
                        folium.CircleMarker(
                            location=[r["lat"], r["lon"]],
                            radius=4,
                            color="crimson",
                            fill=True,
                            fill_opacity=0.5,
                            popup=str(r.get("animal_group", "")),
                        ).add_to(layer)

        # Render map + capture drawings
        st_data = st_folium(m, width=900, height=520)
        if st_data and st_data.get("last_active_drawing"):
            st.session_state["draw_shape"] = st_data["last_active_drawing"]["geometry"]
        elif st_data and not st_data.get("last_active_drawing"):
            if "all_drawings" in st_data and not st_data["all_drawings"]:
                st.session_state["draw_shape"] = None


# =========================================================
# ----------------------- ACTIVITY ------------------------
# =========================================================
elif mode == "Activity":
    st.subheader(f"Activity Analysis — {sel_group}")
    group_df = df[df["animal_group"] == sel_group] if sel_group else pd.DataFrame()

    if group_df.empty:
        st.warning("No data for the selected animal group.")
        st.stop()
    if not have_activity_columns(group_df):
        st.info("Activity view needs ndvi and time (hour+month or datetime). Temperature is optional.")
        st.stop()

    bundle = train_lgbm(group_df)
    gmm, gmm_uses_temp = train_gmm(group_df)

    if bundle is None:
        st.warning("Not enough data to train the predictive model for this group.")
        st.stop()

    best_m, best_h, grid = predict_grid(bundle, group_df)
    st.success(f"Best Month: {best_m} • Best Hour: {best_h:02d}:00")
    st.caption(f"{'LightGBM' if GBM_AVAILABLE else 'HistGBR'} performance — R²: {bundle['r2']:.3f} • RMSE: {bundle['rmse']:.4f}")

    tab1, tab2, tab3 = st.tabs(["24-Hour Clock", "Daily Profile (Model)", "Historical Pattern (GMM)"])
    with tab1:
        st.plotly_chart(circular_plot(grid.assign(month=grid["month"]), best_m), use_container_width=True)
    with tab2:
        st.plotly_chart(daily_profile_plot(grid.assign(month=grid["month"])), use_container_width=True)
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

    # --- Geocoder / starting point (single source of truth for lat/lon) ---
    if "lat0" not in st.session_state or "lon0" not in st.session_state:
        st.session_state["lat0"] = float(df["lat"].mean()) if not df.empty else 0.0
        st.session_state["lon0"] = float(df["lon"].mean()) if not df.empty else 0.0

    with st.expander("Find your starting point", expanded=True):
        place_query = st.text_input("Search for a place (city, park, coordinates, etc.)",
                                    placeholder="e.g., Kruger National Park")
        if st.button("Find on map"):
            loc = geocode_place(place_query)
            if loc is None:
                st.warning("Couldn’t find that place. Check spelling or try a more specific name.")
            else:
                st.session_state["lat0"], st.session_state["lon0"] = loc
                st.success(f"Found: {st.session_state['lat0']:.5f}, {st.session_state['lon0']:.5f}")

        m0 = folium.Map(location=[st.session_state["lat0"], st.session_state["lon0"]],
                        zoom_start=6, tiles="CartoDB Positron")
        folium.Marker([st.session_state["lat0"], st.session_state["lon0"]],
                      tooltip="Your location", icon=folium.Icon(color="blue")).add_to(m0)
        st_folium(m0, width=800, height=250)

    # ---- Planner form (submit computes) ----
    with st.form("planner_form", clear_on_submit=False):
        st.markdown("### Preferences")

        # No lat/lon inputs here—use geocoder above
        radius_km = st.slider("Travel radius (km)", 5, 500, 100, 5, key="radius_km")

        group_list = sorted(df["animal_group"].dropna().unique()) if "animal_group" in df.columns else []
        sel_group_plan = st.selectbox("Animal group you want to see", group_list, key="sel_group_plan")

        weight_temporal = st.slider("Weight: temporal vs spatial", 0, 100, 50, 5,
                                    help="0 = spatial only, 100 = temporal only") / 100.0

        # Target date/hour
        t1, t2 = st.columns(2)
        with t1:
            target_date = st.date_input("Target date (optional)", key="target_date")
            target_month = int(target_date.month) if target_date else None
        with t2:
            hour_mode = st.radio("Hour selection", ["Best hour for that month", "Pick an exact hour"], index=0, key="hour_mode")
            if hour_mode == "Pick an exact hour":
                target_time = st.time_input("Pick a time", value=pd.to_datetime("06:00").time(), key="target_time")
                target_hour = int(target_time.hour)
            else:
                target_hour = None

        # Hotspot parameters (keep sliders for tuning)
        st.markdown("#### Hotspot detection")
        h1, h2 = st.columns(2)
        with h1:
            eps_km = st.slider("Cluster neighborhood (eps, km)", 2, 50, 10, 1,
                               help="DBSCAN radius in km (haversine metric)")
        with h2:
            min_samples = st.slider("Min points per cluster", 5, 50, 10, 1)

        submitted = st.form_submit_button("Suggest a plan")

    # Persist & compute
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

    # Render proposal if present
    proposal = st.session_state.get("proposal")
    if proposal:
        st.markdown("### Recommendation")
        if proposal["where"] is None:
            st.error("I couldn't find a viable hotspot within your radius.")
        else:
            lat_rec, lon_rec = proposal["where"]
            when = proposal["when"]
            like = proposal["likelihood"]
            when_text = f"Month {when[0]}, around {when[1]:02d}:00" if when else "N/A"

            st.success(
                f"**Go to:** {lat_rec:.4f}, {lon_rec:.4f}  \n"
                f"**When:** {when_text}  \n"
                f"**Likelihood (approx):** {like}%"
            )

            # Map context
            m = folium.Map(location=[st.session_state["lat0"], st.session_state["lon0"]],
                        zoom_start=7, tiles="CartoDB Positron")
            folium.Marker([st.session_state["lat0"], st.session_state["lon0"]],
                        tooltip="Your location", icon=folium.Icon(color="blue")).add_to(m)
            folium.Circle([st.session_state["lat0"], st.session_state["lon0"]],
                        radius=st.session_state["radius_km"]*1000, color="blue", fill=False).add_to(m)
            folium.Marker([lat_rec, lon_rec],
                        tooltip=f"Suggested hotspot: {st.session_state.get('sel_group_plan','')}",
                        icon=folium.Icon(color="red")).add_to(m)

            nearby = filter_within_radius(df[df["animal_group"] == st.session_state.get("sel_group_plan","")],
                                        st.session_state["lat0"], st.session_state["lon0"],
                                        st.session_state["radius_km"])
            layer = MarkerCluster().add_to(m)
            for _, r in nearby.iterrows():
                folium.CircleMarker([r["lat"], r["lon"]], radius=4, color="crimson",
                                    fill=True, fill_opacity=0.5).add_to(layer)

            st_folium(m, width=900, height=520)

        # Debug panel to understand the % breakdown
        with st.expander("How this % was computed", expanded=False):
            st.write(f"Temporal favorability: **{proposal['temporal_favor']:.2f}** (0–1)")
            st.write(f"Spatial density: **{proposal['spatial_favor']:.2f}** (0–1)")
            st.write(f"Weight (temporal): **{int(proposal['weight_temporal']*100)}%**")
            st.write(f"Hotspot: eps≈**{proposal['eps_km']:.1f} km**, min_samples=**{proposal['min_samples']}**")
            st.caption("Likelihood = 100 × [ weight*temporal + (1-weight)*spatial ]")

        if proposal["notes"]:
            st.caption(" • " + "\n • ".join(proposal["notes"]))

    if st.button("Clear plan"):
        st.session_state["proposal"] = None





# =========================================================
# ---------------------- PERFORMANCE ----------------------
# =========================================================
else:
    st.subheader("Model Evaluation for All Animal Groups")
    st.write(f"Dates: {start_date} → {end_date}")
    st.markdown("---")

    total_groups = df["animal_group"].nunique()
    total_rows   = len(df)
    c1, c2 = st.columns(2)
    c1.metric("Unique animal groups", total_groups)
    c2.metric("Total records", total_rows)

    results = []
    for grp, sub in df.groupby("animal_group"):
        if not have_activity_columns(sub) or len(sub) < 100:
            continue
        t = train_lgbm(sub)
        if t is None:
            continue
        results.append({"animal_group": grp, "R²": t["r2"], "RMSE": t["rmse"]})

    if results:
        st.dataframe(pd.DataFrame(results).sort_values("R²", ascending=False), use_container_width=True)
    else:
        st.info("No animal groups with sufficient activity data to evaluate.")
