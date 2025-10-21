# app.py
import os, sys
import base64

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
from PIL import Image
import plotly.express as px

# --- import from src/
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.append(SRC)

from src.schema import LAT_COL, LON_COL, SPECIES_COL
from src.config import (
    EPS_KM_DEFAULT, MIN_SAMPLES_GEO,
    USE_DBSCAN_ATTR, EPS_ATTR_DEFAULT, MIN_SAMPLES_ATTR, KMEANS_K_DEFAULT,
    NOISE_LABEL, PALETTE
)
from src.geo_clustering import build_geo_matrix, dbscan_haversine
from src.attr_clustering import make_attr_preprocessor, transform_attributes, cluster_attributes_matrix

# ==============================
# Header with logo (if present)
# ==============================
logo_base64 = None
if os.path.exists("logo.png"):
    with open("logo.png", "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode()

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
# Load your CSV from disk (no uploader, no dummy data)
# ==============================
DEFAULT_CSV = os.path.join("data", "animals_with_ndvi_temperature_updated_water.csv")

def load_and_normalize(path: str) -> pd.DataFrame:
    """
    Map your CSV columns to the app's schema:
      - species: prefer 'common name', then 'name'
      - lat/lon: 'latitude'/'longitude'
      - ndvi: 'NDVI_mean' (drop rows where it's NaN)
      - distance: 'distance_to_water_km_comprehensive'
      - drop 'temperature_C' entirely
    """
    df = pd.read_csv(path, low_memory=False)

    # Drop temperature
    if "temperature_C" in df.columns:
        df = df.drop(columns=["temperature_C"])

    # Build rename map
    rename_map = {}
    if "common name" in df.columns:
        rename_map["common name"] = "species"
    elif "name" in df.columns:
        rename_map["name"] = "species"

    if "latitude" in df.columns:
        rename_map["latitude"] = "lat"
    if "longitude" in df.columns:
        rename_map["longitude"] = "lon"

    if "NDVI_mean" in df.columns:
        rename_map["NDVI_mean"] = "ndvi"
    if "distance_to_water_km_comprehensive" in df.columns:
        rename_map["distance_to_water_km_comprehensive"] = "dist_to_water_km"

    df = df.rename(columns=rename_map)

    # Basic requireds
    missing = {"species", "lat", "lon"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns after mapping: {missing}")

    # Coerce numeric
    for c in ["lat", "lon", "ndvi", "dist_to_water_km"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows without coordinates
    df = df.dropna(subset=["lat", "lon"])

    # Drop rows with NaN NDVI if present
    if "ndvi" in df.columns:
        before = len(df)
        df = df.dropna(subset=["ndvi"])
        dropped = before - len(df)
        if dropped > 0:
            st.sidebar.info(f"Dropped {dropped} rows with missing NDVI")

    # Keep only columns used downstream
    keep_cols = ["species", "lat", "lon"]
    if "ndvi" in df.columns: keep_cols.append("ndvi")
    if "dist_to_water_km" in df.columns: keep_cols.append("dist_to_water_km")
    df = df[keep_cols].copy()

    return df

if not os.path.exists(DEFAULT_CSV):
    st.error(f"CSV not found at {DEFAULT_CSV}. Please place your file there.")
    st.stop()

animal_occurrences = load_and_normalize(DEFAULT_CSV)
st.sidebar.success(f"Loaded dataset: {DEFAULT_CSV}  •  {len(animal_occurrences)} rows")

# ==============================
# Sidebar controls (basic)
# ==============================
st.sidebar.title("Explore Wildlife")
mode = st.sidebar.radio("Select Mode", ["Dashboard", "Travel"], index=1)  # default to Travel
start_date = st.sidebar.date_input("Start Date")
end_date = st.sidebar.date_input("End Date")

# ==============================
# Simple cards data (optional eye-candy for Animal List)
# ==============================
cards_data = [
    {"species": "Elephant", "likelihood": 0.85, "active_months": "Year-round",
     "image": "https://upload.wikimedia.org/wikipedia/commons/4/4e/African_Elephant.jpg"},
    {"species": "Lion", "likelihood": 0.78, "active_months": "Year-round",
     "image": "https://upload.wikimedia.org/wikipedia/commons/1/1b/Lion_au_repos_parc_pendjari.jpg"},
    {"species": "Zebra", "likelihood": 0.90, "active_months": "Year-round",
     "image": "https://upload.wikimedia.org/wikipedia/commons/2/2d/Plains_Zebra_Equus_quagga_cropped.jpg"},
]
cards_df = pd.DataFrame(cards_data)

# ==============================
# Travel Mode
# ==============================
if mode == "Travel":
    st.subheader("WildFinder Travel View – Dataset")
    st.write(f"Dates: {start_date} → {end_date}")
    st.markdown("---")

    view_option = st.sidebar.radio("View as:", ["Animal List", "Interactive Map"], index=1)

    # --- Animal List (optional) ---
    if view_option == "Animal List":
        for _, row in cards_df.iterrows():
            st.markdown(
                f"""
                <div style="display:flex; background:#f9f9f9; border-radius:12px; padding:12px;
                box-shadow:2px 2px 12px rgba(0,0,0,0.1); margin-bottom:15px; align-items:center;">
                    <img src="{row['image']}" width="180" style="border-radius:10px; margin-right:15px;">
                    <div style="flex:1; font-family:sans-serif;">
                        <h3 style="margin:0; color:#264653;">{row['species']}</h3>
                        <p style="margin:4px 0; font-size:14px;"><strong>Likelihood of Sighting:</strong> {row['likelihood']*100:.0f}%</p>
                        <p style="margin:2px 0; font-size:14px;"><strong>Active Months:</strong> {row['active_months']}</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --- Interactive Map (MAP ONLY) ---
    else:
        # Remember last drawn polygon across reruns
        if "draw_shape" not in st.session_state:
            st.session_state["draw_shape"] = None

        # Determine filter polygon (from previous interaction)
        shape = st.session_state["draw_shape"]

        # Filter by stored polygon (if any)
        visible_animals = animal_occurrences.copy()
        if shape and shape.get("type") == "Polygon":
            coords = shape["coordinates"][0]
            lats = [pt[1] for pt in coords]
            lons = [pt[0] for pt in coords]
            south, north = min(lats), max(lats)
            west, east = min(lons), max(lons)
            visible_animals = visible_animals[
                (visible_animals["lat"] >= south)
                & (visible_animals["lat"] <= north)
                & (visible_animals["lon"] >= west)
                & (visible_animals["lon"] <= east)
            ]

        # Center on filtered (or full) dataset
        if not visible_animals.empty:
            center = [visible_animals["lat"].mean(), visible_animals["lon"].mean()]
        else:
            center = [animal_occurrences["lat"].mean(), animal_occurrences["lon"].mean()]
        zoom = 7

        # Build map + drawing tools
        m = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB Positron")
        Draw(export=True, draw_options={'rectangle': True, 'polygon': True}).add_to(m)

        # Only render markers if we have points
        if not visible_animals.empty:
            # --- Optional: quick Stage-1 tuning (map-only; hidden by default)
            with st.sidebar.expander("Tuning (Stage-1 Geo)", expanded=False):
                try:
                    from sklearn.neighbors import NearestNeighbors
                    # Use current visible points to suggest a scale
                    tmp = visible_animals.rename(columns={"lat": "lat_deg", "lon": "lon_deg"})
                    X_tmp = build_geo_matrix(tmp)
                    k_default = max(2, MIN_SAMPLES_GEO)
                    if len(tmp) >= k_default:
                        nn = NearestNeighbors(n_neighbors=k_default, metric="haversine").fit(X_tmp)
                        dists, _ = nn.kneighbors(X_tmp)
                        kth_rad = np.sort(dists[:, -1])
                        med_km = float(np.median(kth_rad)) * 6371.0
                        st.caption(f"Median {k_default}-NN ≈ {med_km:.2f} km • try eps ≈ 1–2× this")
                    else:
                        med_km = None
                        st.caption("Not enough points to estimate a good eps")
                except Exception:
                    med_km = None

                # Sliders (fall back to config defaults if we couldn't compute med_km)
                eps_init = float(EPS_KM_DEFAULT if med_km is None else max(0.5, round(1.5 * med_km, 1)))
                eps_km = st.slider("Geo eps (km)", 0.5, 100.0, eps_init, 0.5)
                min_samples_geo = st.slider("Geo min_samples", 2, 30, int(MIN_SAMPLES_GEO), 1)

            # Rename to schema expected by geo builder
            visible_animals = visible_animals.rename(columns={"lat": "lat_deg", "lon": "lon_deg"})

            # Ensure attribute cols exist
            if "ndvi" not in visible_animals.columns:
                visible_animals["ndvi"] = 0.0
            if "dist_to_water_km" not in visible_animals.columns:
                visible_animals["dist_to_water_km"] = 1.0

            # ---- Two-stage clustering: Geo (haversine) → Attribute ----
            X_geo = build_geo_matrix(visible_animals)
            # USE the tuned values from the sidebar:
            geo_labels = dbscan_haversine(X_geo, eps_km=eps_km, min_samples=min_samples_geo)
            visible_animals["geo_label"] = geo_labels

            pre = make_attr_preprocessor()
            compound_labels = np.full(len(visible_animals), fill_value=-1, dtype=int)
            compound_map = {}
            next_id = 0

            for g in sorted(visible_animals["geo_label"].unique()):
                if g == NOISE_LABEL:
                    continue
                mask = visible_animals["geo_label"] == g
                df_g = visible_animals.loc[mask, [SPECIES_COL, "ndvi", "dist_to_water_km"]]

                X_attr = transform_attributes(df_g, pre)
                attr_labels = cluster_attributes_matrix(
                    X_attr,
                    use_dbscan=USE_DBSCAN_ATTR,
                    eps_attr=EPS_ATTR_DEFAULT,
                    min_samples_attr=MIN_SAMPLES_ATTR,
                    kmeans_k=KMEANS_K_DEFAULT,
                )

                for idx, a in zip(df_g.index, attr_labels):
                    if a == NOISE_LABEL:
                        compound_labels[visible_animals.index.get_loc(idx)] = -1
                    else:
                        key = (int(g), int(a))
                        if key not in compound_map:
                            compound_map[key] = next_id
                            next_id += 1
                        compound_labels[visible_animals.index.get_loc(idx)] = compound_map[key]

            visible_animals["cluster"] = compound_labels

            def color_for(label: int) -> str:
                if label == -1:
                    return "#9AA0A6"  # grey noise
                return PALETTE[label % len(PALETTE)]

            # Add clustered markers (MAP ONLY, no tables)
            for _, row in visible_animals.iterrows():
                folium.CircleMarker(
                    location=[row[LAT_COL], row[LON_COL]],
                    radius=10,
                    color=color_for(row["cluster"]),
                    fill=True,
                    fill_color=color_for(row["cluster"]),
                    fill_opacity=0.7,
                    popup=f"{row[SPECIES_COL]} (cluster {row['cluster']})",
                ).add_to(m)

        # Render the map and capture new drawing (if any)
        st_data = st_folium(m, width=900, height=520)
        if st_data and st_data.get("last_active_drawing"):
            st.session_state["draw_shape"] = st_data["last_active_drawing"]["geometry"]
        elif st_data and not st_data.get("last_active_drawing"):
            if "all_drawings" in st_data and not st_data["all_drawings"]:
                st.session_state["draw_shape"] = None

# ==============================
# Dashboard Mode
# ==============================
else:
    st.subheader("Wildlife Dashboard Overview")
    st.write(f"Dates: {start_date} → {end_date}")
    st.markdown("---")

    total_species = animal_occurrences["species"].nunique()
    total_rows = len(animal_occurrences)

    c1, c2 = st.columns(2)
    c1.metric("Unique species", total_species)
    c2.metric("Total records", total_rows)

    # Top 15 species by count
    counts = animal_occurrences["species"].value_counts().head(15).sort_values(ascending=True)
    if not counts.empty:
        fig = px.bar(
            counts,
            orientation="h",
            labels={"value": "Records", "index": "Species"},
            title="Top species by record count",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data to show. Check your CSV load.")
