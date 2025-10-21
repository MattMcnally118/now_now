# src/attr_clustering.py
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import DBSCAN, KMeans

from .schema import NUMERIC_COLS, CATEGORICAL_COLS
from .config import (
    USE_DBSCAN_ATTR, EPS_ATTR_DEFAULT, MIN_SAMPLES_ATTR, KMEANS_K_DEFAULT, NOISE_LABEL
)

def make_attr_preprocessor() -> ColumnTransformer:
    """Impute + scale numeric, impute + one-hot categorical."""
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer(transformers=[
        ("num", num_pipe, NUMERIC_COLS),
        ("cat", cat_pipe, CATEGORICAL_COLS),
    ])
    return pre

def transform_attributes(df: pd.DataFrame, pre: ColumnTransformer) -> np.ndarray:
    """Transform attributes into a numeric feature matrix for attribute clustering."""
    missing = (set(NUMERIC_COLS) | set(CATEGORICAL_COLS)) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for attribute clustering: {missing}")
    X = pre.fit_transform(df)  # fit per subset is OK; small and fast
    return X

def cluster_attributes_matrix(
    X: np.ndarray,
    use_dbscan: bool = USE_DBSCAN_ATTR,
    eps_attr: float = EPS_ATTR_DEFAULT,
    min_samples_attr: int = MIN_SAMPLES_ATTR,
    kmeans_k: int = KMEANS_K_DEFAULT,
) -> np.ndarray:
    """Cluster the transformed attributes. Returns 1D labels array."""
    if use_dbscan:
        model = DBSCAN(eps=eps_attr, min_samples=min_samples_attr, metric="euclidean")
        labels = model.fit_predict(X)
    else:
        model = KMeans(n_clusters=kmeans_k, n_init="auto", random_state=42)
        labels = model.fit_predict(X)
    return labels

def summarize_attribute_cluster(
    df: pd.DataFrame, labels: np.ndarray
) -> pd.DataFrame:
    """Simple summaries per attribute cluster within a geo-group."""
    tmp = df.copy()
    tmp["_attr_label"] = labels
    # ignore noise for summaries if DBSCAN is used
    grp = tmp[tmp["_attr_label"] != NOISE_LABEL].groupby("_attr_label", as_index=False)
    out = grp.agg({
        "species": lambda s: s.value_counts().idxmax() if len(s) else None,
        "ndvi": "mean",
        "dist_to_water_km": "mean"
    }).rename(columns={"species": "dominant_species", "ndvi": "avg_ndvi",
                    "dist_to_water_km": "avg_dist_to_water_km"})
    return out
