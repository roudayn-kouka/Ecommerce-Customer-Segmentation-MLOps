"""Tests for scaling, PCA and correlation analysis."""

import os

import pandas as pd
import pytest

from src.correlation import correlation_check, high_correlations
from src.pca import pc_analyzer
from src.scaler import COLS, scale_frame, scaler


@pytest.fixture
def customers_pkl(tmp_path, customers):
    path = tmp_path / "customers.pkl"
    customers.to_pickle(path)
    return str(path)


@pytest.fixture
def scaled_parquet(tmp_path, customers_pkl):
    return scaler(customers_pkl, str(tmp_path / "scaled.parquet"))


def test_scale_frame_ranges(customers):
    out = scale_frame(customers)
    std_cols, norm_cols = COLS
    assert out[std_cols].mean().abs().max() < 1e-6
    assert out[norm_cols].min().min() >= 0
    assert out[norm_cols].max().max() <= 1 + 1e-9
    assert list(out.columns) == list(customers.columns)  # no MultiIndex columns


def test_scale_frame_missing_column(customers):
    with pytest.raises(KeyError):
        scale_frame(customers.drop(columns=["Total_Spend"]))


def test_scaler_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        scaler(str(tmp_path / "missing.pkl"), str(tmp_path / "out.parquet"))


def test_scaler_bad_output_path(customers_pkl):
    with pytest.raises(TypeError):
        scaler(customers_pkl, 123)


def test_pca_output(tmp_path, scaled_parquet):
    out = pc_analyzer(scaled_parquet, str(tmp_path / "pca.parquet"), drop_cols=[])
    reduced = pd.read_parquet(out)
    assert reduced.shape[1] >= 2
    assert all(c.startswith("PC") for c in reduced.columns)


def test_pca_errors(tmp_path, scaled_parquet, customers):
    out = str(tmp_path / "pca.parquet")
    with pytest.raises(FileNotFoundError):
        pc_analyzer("nonexistent.parquet", out)
    with pytest.raises(KeyError):
        pc_analyzer(scaled_parquet, out, drop_cols=["Not_A_Column"])
    with pytest.raises(ValueError):
        pc_analyzer(scaled_parquet, out, drop_cols=[], cvr_thresh=80)
    keep_three = [c for c in customers.columns if c != "CustomerID"][3:]
    with pytest.raises(ValueError):
        pc_analyzer(scaled_parquet, out, drop_cols=keep_three)


def test_correlation_outputs(tmp_path, scaled_parquet):
    img, mat, lst = (str(tmp_path / n) for n in ("corr.png", "corr.parquet", "pairs.txt"))
    pairs = correlation_check(scaled_parquet, (img, mat), 0.5, lst)
    assert os.path.exists(img) and os.path.exists(mat) and os.path.exists(lst)
    assert all(abs(v) >= 0.5 for _, _, v in pairs)


def test_correlation_threshold_validation(tmp_path, scaled_parquet):
    with pytest.raises(ValueError):
        correlation_check(scaled_parquet, (str(tmp_path / "a.png"), str(tmp_path / "b.pq")), 5)


def test_high_correlations():
    corr = pd.DataFrame(
        [[1, 0.9, 0.1], [0.9, 1, -0.7], [0.1, -0.7, 1]], index=list("abc"), columns=list("abc")
    )
    assert high_correlations(corr, 0.5) == [("a", "b", 0.9), ("b", "c", -0.7)]
