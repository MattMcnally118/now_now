# src/schema.py

# Column names expected by the clustering helpers
LAT_COL = "lat_deg"          # degrees
LON_COL = "lon_deg"          # degrees
SPECIES_COL = "species"      # categorical (string)

# Numeric, non-geo features used in attribute clustering
NUMERIC_COLS = ["ndvi", "dist_to_water_km"]

# Categorical columns used in attribute clustering
CATEGORICAL_COLS = [SPECIES_COL]

# Basic sanity bounds (used for validation)
LAT_MIN, LAT_MAX = -90.0, 90.0
LON_MIN, LON_MAX = -180.0, 180.0

