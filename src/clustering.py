"""
Train the customer segmentation model.

A single scikit-learn ``Pipeline`` (scaling -> PCA -> clustering) is fitted on
the customer features, so the exact same preprocessing is reused at inference
time by the API. The number of clusters and the algorithm (KMeans or Gaussian
Mixture) are selected with the silhouette score; runs are logged to MLflow when
it is installed.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.utils import get_logger, load_config, load_frame, resolve

logger = get_logger(__name__)

_SCALER = load_config("scaler")
_PCA = load_config("pca")
_CONFIG = load_config("clustering")

STD_COLS = _SCALER["standardize_columns"]
NORM_COLS = _SCALER["normalize_columns"]
PASSTHROUGH_COLS = _CONFIG["passthrough_columns"]
FEATURES = STD_COLS + NORM_COLS + PASSTHROUGH_COLS
DEFAULT_IN_PATH = resolve(_CONFIG["ingest_path"])
MODEL_PATH = resolve(_CONFIG["model_path"])
METRICS_PATH = resolve(_CONFIG["metrics_path"])
ASSIGNMENTS_PATH = resolve(_CONFIG["assignments_path"])


@dataclass
class Candidate:
    """Score of one (algorithm, k) combination."""

    algorithm: str
    n_clusters: int
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float


def build_preprocessor(n_components: float | int = _PCA["cvr_threshold"]) -> Pipeline:
    """Scaling + PCA keeping ``n_components`` (variance ratio or integer)."""
    scaling = ColumnTransformer(
        [
            ("std", StandardScaler(), STD_COLS),
            ("norm", MinMaxScaler(), NORM_COLS),
            ("pass", "passthrough", PASSTHROUGH_COLS),
        ]
    )
    return Pipeline(
        [("scale", scaling), ("pca", PCA(n_components=n_components, svd_solver="full"))]
    )


def _make_estimator(algorithm: str, k: int, random_state: int):
    if algorithm == "kmeans":
        return KMeans(n_clusters=k, n_init=10, random_state=random_state)
    if algorithm == "gmm":
        return GaussianMixture(n_components=k, covariance_type="full", random_state=random_state)
    raise ValueError(f"Unknown algorithm: {algorithm}")


def evaluate_candidates(
    reduced: np.ndarray, k_range, algorithms, random_state: int = 42, sample_size: int | None = 5000
) -> list[Candidate]:
    """Fit every (algorithm, k) on the reduced data and compute clustering metrics."""
    results = []
    for algorithm in algorithms:
        for k in k_range:
            labels = _make_estimator(algorithm, k, random_state).fit_predict(reduced)
            if len(set(labels)) < 2:
                continue
            sil = silhouette_score(
                reduced,
                labels,
                random_state=random_state,
                sample_size=min(sample_size or len(reduced), len(reduced)),
            )
            results.append(
                Candidate(
                    algorithm,
                    k,
                    round(float(sil), 4),
                    round(float(davies_bouldin_score(reduced, labels)), 4),
                    round(float(calinski_harabasz_score(reduced, labels)), 2),
                )
            )
            logger.info("%s k=%d silhouette=%.3f", algorithm, k, sil)
    if not results:
        raise ValueError("No valid clustering candidate could be evaluated.")
    return results


def train(
    customers: pd.DataFrame, k_range=None, algorithms=None, random_state: int | None = None
) -> tuple[Pipeline, Candidate, list[Candidate]]:
    """
    Select the best (algorithm, k) by silhouette and fit the full pipeline.

    :return: ``(fitted_pipeline, best_candidate, all_candidates)``.
    """
    missing = [c for c in FEATURES if c not in customers.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    k_range = k_range or range(_CONFIG["k_min"], _CONFIG["k_max"] + 1)
    algorithms = algorithms or _CONFIG["algorithms"]
    random_state = _CONFIG["random_state"] if random_state is None else random_state

    X = customers[FEATURES].astype(float)
    preprocessor = build_preprocessor()
    reduced = preprocessor.fit_transform(X)

    candidates = evaluate_candidates(reduced, k_range, algorithms, random_state)
    best = max(candidates, key=lambda c: c.silhouette)
    logger.info(
        "Best model: %s with k=%d (silhouette=%.3f)",
        best.algorithm,
        best.n_clusters,
        best.silhouette,
    )

    model = Pipeline(
        preprocessor.steps
        + [("cluster", _make_estimator(best.algorithm, best.n_clusters, random_state))]
    )
    model.fit(X)
    return model, best, candidates


def profile_clusters(customers: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Mean of each raw feature per cluster (used to interpret the segments)."""
    profile = customers[FEATURES].astype(float).assign(cluster=labels)
    summary = profile.groupby("cluster").mean().round(2)
    summary.insert(0, "size", profile.groupby("cluster").size())
    return summary


def _log_mlflow(best: Candidate, candidates: list[Candidate], model_path: str) -> None:
    try:
        import mlflow
    except ImportError:
        logger.info("MLflow not installed: skipping experiment tracking.")
        return
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", "customer-segmentation"))
    with mlflow.start_run(run_name=f"{best.algorithm}-k{best.n_clusters}"):
        mlflow.log_params(
            {
                "algorithm": best.algorithm,
                "n_clusters": best.n_clusters,
                "pca_variance": _PCA["cvr_threshold"],
            }
        )
        mlflow.log_metrics(
            {
                "silhouette": best.silhouette,
                "davies_bouldin": best.davies_bouldin,
                "calinski_harabasz": best.calinski_harabasz,
            }
        )
        mlflow.log_artifact(model_path)
        for cand in candidates:
            mlflow.log_metric(f"silhouette_{cand.algorithm}", cand.silhouette, step=cand.n_clusters)


def train_and_save(
    in_path: str = DEFAULT_IN_PATH,
    model_path: str = MODEL_PATH,
    metrics_path: str = METRICS_PATH,
    assignments_path: str = ASSIGNMENTS_PATH,
    **kwargs,
) -> dict:
    """Train, persist the model + metrics + cluster assignments and return a summary."""
    customers = load_frame(in_path)
    model, best, candidates = train(customers, **kwargs)
    labels = model.predict(customers[FEATURES].astype(float))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "best": asdict(best)}, model_path)

    profile = profile_clusters(customers, labels)
    summary = {
        "best": asdict(best),
        "candidates": [asdict(c) for c in candidates],
        "cluster_sizes": profile["size"].to_dict(),
        "profile": profile.drop(columns="size").to_dict(orient="index"),
    }
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, default=str)

    os.makedirs(os.path.dirname(assignments_path), exist_ok=True)
    customers.assign(cluster=labels).to_parquet(assignments_path)

    _log_mlflow(best, candidates, model_path)
    logger.info("Model saved to %s", model_path)
    return summary


if __name__ == "__main__":
    print(json.dumps(train_and_save()["best"], indent=2))
