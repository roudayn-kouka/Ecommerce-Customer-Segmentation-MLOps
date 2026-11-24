"""
Scale customer features: standardisation (z-score) for most columns and
min-max normalisation (0-1) for heavily skewed count columns.

Columns are configured in ``config/feature_processing.json`` (section ``scaler``).
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.utils import get_logger, load_config, load_frame, resolve, save_parquet

logger = get_logger(__name__)

_CONFIG = load_config("scaler")
DEFAULT_IN_PATH = resolve(_CONFIG["ingest_path"])
DEFAULT_OUT_PATH = resolve(_CONFIG["output_path"])
COLS = (_CONFIG["standardize_columns"], _CONFIG["normalize_columns"])


def scale_frame(data: pd.DataFrame, cols=COLS) -> pd.DataFrame:
    """
    Return a scaled copy of ``data``.

    :param cols: ``(standardize_columns, normalize_columns)``.
    :raises KeyError: If a configured column is missing.
    """
    std_cols, norm_cols = list(cols[0]), list(cols[1])
    missing = [c for c in std_cols + norm_cols if c not in data.columns]
    if missing:
        raise KeyError(f"Columns to scale not found in DataFrame: {missing}")

    out = data.copy()
    if std_cols:
        out[std_cols] = StandardScaler().fit_transform(out[std_cols].astype(float))
    if norm_cols:
        # MinMaxScaler scales each *column* to [0, 1]. The previous sklearn
        # ``Normalizer`` scaled each *row* to unit norm, which is not what was intended.
        out[norm_cols] = MinMaxScaler().fit_transform(out[norm_cols].astype(float))
    return out


def scaler(in_path: str = DEFAULT_IN_PATH, out_path: str = DEFAULT_OUT_PATH, cols=COLS) -> str:
    """Load customer features, scale them and save as parquet."""
    if not isinstance(out_path, str):
        raise TypeError("Save Path should be a String!")
    data = load_frame(in_path)
    scaled = scale_frame(data, cols)
    logger.info("Scaled %d rows x %d columns", *scaled.shape)
    return save_parquet(scaled, out_path)
