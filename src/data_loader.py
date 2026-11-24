"""
Load the e-commerce dataset from a pickle cache or from the original Excel file.
"""

from __future__ import annotations

import os

import pandas as pd

from src.utils import DATA_DIR, get_logger, load_pickle, processed_path, save_pickle

logger = get_logger(__name__)

DEFAULT_PICKLE_PATH = processed_path("raw_data.pkl")
DEFAULT_EXCEL_PATH = str(DATA_DIR / "Online Retail.xlsx")


def load_data(pickle_path: str = DEFAULT_PICKLE_PATH, excel_path: str = DEFAULT_EXCEL_PATH) -> str:
    """
    Load the dataset, preferring the pickle cache over the Excel file, and
    (re)write the pickle cache.

    :return: Path to the pickle file.
    :raises FileNotFoundError: If neither file exists.
    """
    if os.path.exists(pickle_path):
        df = load_pickle(pickle_path)
        logger.info("Data loaded successfully from %s", pickle_path)
    elif os.path.exists(excel_path):
        df = pd.read_excel(excel_path)
        logger.info("Data loaded from %s", excel_path)
    else:
        message = f"No data found in the specified paths: {pickle_path} or {excel_path}"
        logger.error(message)
        raise FileNotFoundError(message)
    return save_pickle(df, pickle_path)
