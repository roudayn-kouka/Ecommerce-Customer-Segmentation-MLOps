"""Tests for clustering, drift detection, plots and the REST API."""

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import clustering, drift
from src.plots import make_plots


@pytest.fixture
def trained(tmp_path, blob_customers):
    in_path = tmp_path / "customers.pkl"
    blob_customers.to_pickle(in_path)
    paths = {
        "model_path": str(tmp_path / "model.joblib"),
        "metrics_path": str(tmp_path / "metrics.json"),
        "assignments_path": str(tmp_path / "clusters.parquet"),
    }
    summary = clustering.train_and_save(
        str(in_path), k_range=range(2, 5), algorithms=["kmeans", "gmm"], **paths
    )
    return summary, paths


def test_train_selects_best_by_silhouette(trained):
    summary, _ = trained
    best = summary["best"]
    assert best["silhouette"] == max(c["silhouette"] for c in summary["candidates"])
    assert best["algorithm"] in {"kmeans", "gmm"}
    assert sum(summary["cluster_sizes"].values()) > 0


def test_train_outputs_files(trained):
    _, paths = trained
    with open(paths["metrics_path"], encoding="utf-8") as file:
        assert "best" in json.load(file)
    assert "cluster" in pd.read_parquet(paths["assignments_path"]).columns


def test_train_missing_feature(blob_customers):
    with pytest.raises(KeyError):
        clustering.train(blob_customers.drop(columns=["Hour"]), k_range=range(2, 3))


def test_plots(tmp_path, trained):
    _, paths = trained
    for path in make_plots(paths["assignments_path"], str(tmp_path / "plots")):
        assert (tmp_path / "plots" / path.split("plots")[-1].strip("\\/")).exists()


def test_psi_identical_distribution_is_zero():
    x = np.random.default_rng(0).normal(size=2000)
    assert drift.population_stability_index(x, x) == pytest.approx(0, abs=1e-6)


def test_drift_detected_on_shift():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 2000), "b": rng.normal(0, 1, 2000)})
    cur = pd.DataFrame({"a": rng.normal(3, 1, 2000), "b": rng.normal(0, 1, 2000)})
    report = drift.detect_drift(ref, cur)
    assert report["drifted_features"] == ["a"]
    assert report["drift_share"] == 0.5


def test_no_drift_on_same_distribution():
    rng = np.random.default_rng(2)
    ref = pd.DataFrame({"a": rng.normal(size=3000)})
    cur = pd.DataFrame({"a": rng.normal(size=3000)})
    assert drift.detect_drift(ref, cur)["dataset_drift"] is False


@pytest.fixture
def client(monkeypatch, trained):
    from api import main

    _, paths = trained
    monkeypatch.setattr(main, "MODEL_PATH", paths["model_path"])
    main.load_bundle.cache_clear()
    yield TestClient(main.app)
    main.load_bundle.cache_clear()


def test_api_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_api_predict(client, blob_customers):
    payload = blob_customers[clustering.FEATURES].head(3).astype(float)
    payload[["Day_Of_Week", "Hour", "Is_UK"]] = payload[["Day_Of_Week", "Hour", "Is_UK"]].astype(
        int
    )
    response = client.post("/predict", json=payload.to_dict(orient="records"))
    assert response.status_code == 200
    assert len(response.json()) == 3
    assert all(isinstance(p["cluster"], int) for p in response.json())


def test_api_validation_error(client):
    assert client.post("/predict", json=[{"Hour": 99}]).status_code == 422


def test_api_without_model(monkeypatch, tmp_path):
    from api import main

    monkeypatch.setattr(main, "MODEL_PATH", str(tmp_path / "none.joblib"))
    main.load_bundle.cache_clear()
    client = TestClient(main.app)
    assert client.get("/health").json()["model_loaded"] is False
    main.load_bundle.cache_clear()
