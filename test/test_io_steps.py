"""Tests for download / unzip / load and the file-based pipeline steps."""

import os
import pickle
import zipfile

import pandas as pd
import pytest
import requests

from src import pipeline
from src.data_loader import load_data
from src.download_data import ingest_data
from src.unzip_data import unzip_file


def test_ingest_data_writes_zip(tmp_path, requests_mock):
    url = "https://example.org/data.zip"
    requests_mock.get(url, content=b"zip-bytes")
    path = ingest_data(url, output_dir=str(tmp_path))
    assert path == str(tmp_path / "data.zip")
    assert (tmp_path / "data.zip").read_bytes() == b"zip-bytes"


def test_ingest_data_raises_on_http_error(tmp_path, requests_mock):
    url = "https://example.org/missing.zip"
    requests_mock.get(url, status_code=404)
    with pytest.raises(requests.HTTPError):
        ingest_data(url, output_dir=str(tmp_path))


def test_unzip_file(tmp_path):
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("Online Retail.xlsx", "fake")
    result = unzip_file(str(archive), str(tmp_path / "out"))
    assert os.path.exists(result)


def test_unzip_bad_file(tmp_path):
    bad = tmp_path / "bad.zip"
    bad.write_text("not a zip")
    with pytest.raises(zipfile.BadZipFile):
        unzip_file(str(bad), str(tmp_path))


def test_unzip_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        unzip_file(str(tmp_path / "nope.zip"), str(tmp_path))


def test_load_data_from_pickle(tmp_path, transactions):
    pkl = tmp_path / "raw.pkl"
    transactions.to_pickle(pkl)
    assert load_data(str(pkl), str(tmp_path / "missing.xlsx")) == str(pkl)


def test_load_data_no_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(str(tmp_path / "a.pkl"), str(tmp_path / "b.xlsx"))


def test_step_missing_input_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        pipeline.handle_missing(str(tmp_path / "nope.pkl"), str(tmp_path / "out.pkl"))


def test_file_steps_chain(tmp_path, transactions):
    """Run every cleaning + feature step through files, as Airflow does."""
    path = str(tmp_path / "raw.pkl")
    transactions.to_pickle(path)
    for i, step in enumerate(pipeline.CLEANING_STEPS):
        path = step(path, str(tmp_path / f"clean_{i}.pkl"))
    clean = path

    customer = pipeline.rfm(clean, str(tmp_path / "rfm.pkl"))
    for i, step in enumerate(
        [
            pipeline.unique_products,
            pipeline.customers_behavior,
            pipeline.geographic_features,
            pipeline.cancellation_details,
            pipeline.seasonality_impacts,
        ]
    ):
        customer = step(customer, clean, str(tmp_path / f"feat_{i}.pkl"))
    final = pipeline.removing_outlier(customer, str(tmp_path / "final.pkl"))

    with open(final, "rb") as file:
        result = pickle.load(file)
    assert isinstance(result, pd.DataFrame)
    assert {"CustomerID", "Cancellation_Rate", "Spending_Trend"} <= set(result.columns)
    assert len(result) > 0
