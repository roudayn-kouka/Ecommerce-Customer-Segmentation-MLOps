"""
Cluster interpretation plots: radar chart of standardised centroids and
per-cluster feature histograms.

Usage::

    python -m src.plots   # reads data/processed/customer_clusters.parquet
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.clustering import ASSIGNMENTS_PATH, FEATURES  # noqa: E402
from src.utils import PLOTS_DIR, get_logger, load_frame  # noqa: E402

logger = get_logger(__name__)
PALETTE = plt.get_cmap("tab10").colors


def radar_chart(customers: pd.DataFrame, out_path: str) -> str:
    """Radar chart of z-scored feature means per cluster."""
    feats = customers[FEATURES].astype(float)
    z = (feats - feats.mean()) / feats.std(ddof=0).replace(0, 1)
    centroids = z.groupby(customers["cluster"]).mean()

    angles = np.linspace(0, 2 * np.pi, len(FEATURES), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"polar": True})
    for i, (cluster, row) in enumerate(centroids.iterrows()):
        values = row.tolist() + row.tolist()[:1]
        color = PALETTE[i % len(PALETTE)]
        ax.plot(angles, values, color=color, linewidth=2, label=f"Cluster {cluster}")
        ax.fill(angles, values, color=color, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(FEATURES, fontsize=8)
    ax.set_title("Cluster profiles (standardised means)", pad=30)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def histograms(customers: pd.DataFrame, out_path: str) -> str:
    """Grid of histograms: one row per feature, one column per cluster."""
    clusters = sorted(customers["cluster"].unique())
    fig, axes = plt.subplots(
        len(FEATURES),
        len(clusters),
        squeeze=False,
        figsize=(4 * len(clusters), 2.5 * len(FEATURES)),
    )
    for i, feature in enumerate(FEATURES):
        for j, cluster in enumerate(clusters):
            data = customers.loc[customers["cluster"] == cluster, feature].astype(float)
            axes[i, j].hist(
                data, bins=20, color=PALETTE[j % len(PALETTE)], edgecolor="white", alpha=0.8
            )
            axes[i, j].set_title(f"Cluster {cluster} - {feature}", fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_plots(in_path: str = ASSIGNMENTS_PATH, out_dir: str = str(PLOTS_DIR)) -> list[str]:
    """Generate every cluster plot and return their paths."""
    customers = load_frame(in_path)
    paths = [
        radar_chart(customers, os.path.join(out_dir, "cluster_radar.png")),
        histograms(customers, os.path.join(out_dir, "histogram_analysis.png")),
    ]
    logger.info("Plots saved: %s", paths)
    return paths


if __name__ == "__main__":
    make_plots()
