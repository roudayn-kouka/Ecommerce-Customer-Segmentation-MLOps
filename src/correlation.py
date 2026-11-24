"""
Correlation analysis of the scaled customer features.

Outputs a lower-triangle heatmap (PNG), the correlation matrix (parquet) and
the list of highly correlated feature pairs (text file).
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless backend: works in Airflow / CI containers

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from src.utils import get_logger, load_config, load_frame, resolve, save_parquet  # noqa: E402

logger = get_logger(__name__)

_CONFIG = load_config("correlation")
DEFAULT_IN_PATH = resolve(_CONFIG["ingest_path"])
IMG_PATH = resolve(_CONFIG["image_path"])
PARQUET_PATH = resolve(_CONFIG["correlation_matrix_path"])
LIST_PATH = resolve(_CONFIG["list_path"])
CORR_THRESHOLD = _CONFIG["correlation_threshold"]


def high_correlations(corr: pd.DataFrame, threshold: float) -> list[tuple[str, str, float]]:
    """Feature pairs whose absolute correlation is >= ``threshold`` (sorted, desc)."""
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack()
    pairs = pairs[pairs.abs() >= threshold].sort_values(key=np.abs, ascending=False)
    return [(a, b, round(float(v), 3)) for (a, b), v in pairs.items()]


def correlation_check(
    in_path: str = DEFAULT_IN_PATH,
    out_path=(IMG_PATH, PARQUET_PATH),
    correlation_threshold: float = CORR_THRESHOLD,
    list_path: str = LIST_PATH,
) -> list[tuple[str, str, float]]:
    """
    Compute and persist the correlation matrix of the numeric features.

    :return: Highly correlated feature pairs.
    """
    if not all(isinstance(p, str) for p in out_path):
        raise TypeError("Save paths should be strings!")
    if not 0 < correlation_threshold < 1:
        raise ValueError("correlation_threshold should lie between 0 and 1.")

    data = load_frame(in_path)
    corr = data.select_dtypes("number").corr()

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    cmap = LinearSegmentedColormap.from_list(
        "custom_map", ["#ff6200", "#ffcaa8", "white", "#ffcaa8", "#ff6200"], N=256
    )
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(12, 10))
    plt.title(f"Correlation Matrix (threshold {correlation_threshold})", fontsize=14)
    sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, center=0, fmt=".2f", linewidths=2)
    os.makedirs(os.path.dirname(out_path[0]), exist_ok=True)
    fig.savefig(out_path[0], bbox_inches="tight")
    plt.close(fig)

    save_parquet(corr, out_path[1])

    pairs = high_correlations(corr, correlation_threshold)
    os.makedirs(os.path.dirname(list_path), exist_ok=True)
    with open(list_path, "w", encoding="utf-8") as file:
        file.writelines(f"{a}\t{b}\t{v}\n" for a, b, v in pairs)
    logger.info("%d feature pairs above |r| >= %s", len(pairs), correlation_threshold)
    return pairs
