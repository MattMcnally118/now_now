# src/config.py

# ---------- Stage 1 (Geo / haversine) ----------
EPS_KM_DEFAULT = 3.0       # neighborhood radius in kilometers
MIN_SAMPLES_GEO = 10       # points needed to form a core point

# ---------- Stage 2 (Attributes) ----------
USE_DBSCAN_ATTR = True     # if False, we use KMeans
EPS_ATTR_DEFAULT = 0.8     # after scaling; tune per dataset
MIN_SAMPLES_ATTR = 10
KMEANS_K_DEFAULT = 3

# Label used by DBSCAN for noise
NOISE_LABEL = -1

# Fixed palette for clusters (feel free to expand)
PALETTE = [
    "#2A9D8F", "#E76F51", "#264653", "#F4A261", "#8AB17D",
    "#577590", "#FF9F1C", "#3D5A80", "#43AA8B", "#B56576"
]

