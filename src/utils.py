"""
Shared helpers for the customer segmentation pipeline.

Centralises project paths, logging and DataFrame (de)serialisation so that
every pipeline step behaves the same way and fails loudly on missing inputs.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PLOTS_DIR = DATA_DIR / "plots"
MODELS_DIR = PROJECT_DIR / "models"
CONFIG_PATH = PROJECT_DIR / "config" / "feature_processing.json"


def get_logger(name: str) -> logging.Logger:
    """Return a module logger with a consistent format."""
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers and not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
    return logger


def processed_path(filename: str) -> str:
    """Absolute path of a file inside ``data/processed``."""
    return str(PROCESSED_DIR / filename)


def load_config(section: str | None = None) -> dict[str, Any]:
    """Load the feature-processing JSON config (optionally a single section)."""
    with open(CONFIG_PATH, encoding="utf-8") as file:
        config = json.load(file)
    if section is None:
        return config
    if section not in config:
        raise KeyError(f"Section '{section}' not found in {CONFIG_PATH}")
    return config[section]


def resolve(path: str) -> str:
    """Resolve a path relative to the project root (absolute paths unchanged)."""
    return path if os.path.isabs(path) else str(PROJECT_DIR / path)


def load_pickle(path: str) -> pd.DataFrame:
    """Load a pickled object, raising ``FileNotFoundError`` if it is missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No data found at the specified path: {path}")
    with open(path, "rb") as file:
        return pickle.load(file)


def save_pickle(obj: Any, path: str) -> str:
    """Pickle ``obj`` to ``path`` (creating parent folders) and return the path."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(obj, file)
    get_logger(__name__).info("Data saved to %s", path)
    return path


def load_frame(path: str) -> pd.DataFrame:
    """Load a DataFrame from a pickle or parquet file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found at {path}.")
    data = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_pickle(path)
    if not isinstance(data, pd.DataFrame):
        raise TypeError("File did not load DataFrame correctly.")
    return data


def save_parquet(data: pd.DataFrame, path: str) -> str:
    """Save a DataFrame as parquet (creating parent folders) and return the path."""
    if not isinstance(path, str):
        raise TypeError("Save Path should be a String!")
    if not isinstance(data, pd.DataFrame):
        raise AttributeError("to_parquet could not be executed as object not DataFrame.")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data.to_parquet(path)
    get_logger(__name__).info("File saved successfully at Path: %s", path)
    return path
