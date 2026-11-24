"""
Data drift monitoring.

Compares a *current* batch of customer features against the *reference*
(training) data with two complementary tests per feature:

* the two-sample Kolmogorov-Smirnov test (p-value), and
* the Population Stability Index (PSI): < 0.1 stable, 0.1-0.25 moderate, > 0.25 drift.

Usage::

    python -m src.drift reference.parquet current.parquet [--report drift.json]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

from src.utils import get_logger, load_frame

logger = get_logger(__name__)

PSI_THRESHOLD = 0.25
P_VALUE_THRESHOLD = 0.05


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-6
) -> float:
    """PSI computed on quantile bins of the reference distribution."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:  # constant reference feature
        return 0.0 if np.allclose(current, reference[0]) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, edges)[0] / len(reference) + eps
    cur_pct = np.histogram(current, edges)[0] / len(current) + eps
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns=None,
    psi_threshold: float = PSI_THRESHOLD,
    p_value_threshold: float = P_VALUE_THRESHOLD,
) -> dict:
    """
    Per-feature drift report.

    A feature is flagged as drifted when ``PSI > psi_threshold`` **and**
    the KS p-value is below ``p_value_threshold``.
    """
    columns = columns or [
        c for c in reference.select_dtypes("number").columns if c in current.columns
    ]
    features = {}
    for col in columns:
        ref = reference[col].dropna().astype(float).to_numpy()
        cur = current[col].dropna().astype(float).to_numpy()
        ks = stats.ks_2samp(ref, cur)
        psi = population_stability_index(ref, cur)
        features[col] = {
            "psi": round(psi, 4),
            "ks_statistic": round(float(ks.statistic), 4),
            "p_value": round(float(ks.pvalue), 6),
            "drift": bool(psi > psi_threshold and ks.pvalue < p_value_threshold),
        }
    drifted = [c for c, r in features.items() if r["drift"]]
    share = len(drifted) / len(features) if features else 0.0
    return {
        "n_features": len(features),
        "drifted_features": drifted,
        "drift_share": round(share, 3),
        "dataset_drift": share >= 0.5,
        "features": features,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Data drift report")
    parser.add_argument("reference")
    parser.add_argument("current")
    parser.add_argument("--report", help="write the JSON report to this path")
    args = parser.parse_args()
    report = detect_drift(load_frame(args.reference), load_frame(args.current))
    text = json.dumps(report, indent=2)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as file:
            file.write(text)
    print(text)
    if report["dataset_drift"]:
        logger.warning("Dataset drift detected: %s", report["drifted_features"])


if __name__ == "__main__":
    main()
