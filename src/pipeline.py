"""
File-based pipeline steps (one function = one Airflow task) and a CLI runner.

Every step reads its inputs from ``data/processed``, applies a pure function
from :mod:`src.cleaning`, writes its output and returns the output path, so the
steps can be chained through Airflow XComs or called directly.

Usage::

    python -m src.pipeline                 # full run (download included)
    python -m src.pipeline --skip-download # reuse data/Online Retail.xlsx
"""

from __future__ import annotations

import argparse
import time

from src import cleaning
from src.data_loader import load_data
from src.download_data import ingest_data
from src.unzip_data import unzip_file
from src.utils import get_logger, load_pickle, processed_path, save_pickle

logger = get_logger(__name__)

RAW = processed_path("raw_data.pkl")
CLEAN_TRANSACTIONS = processed_path("after_removing_zero_unitprice.pkl")
CUSTOMER_FEATURES = processed_path("seasonality.pkl")
CUSTOMERS_NO_OUTLIERS = processed_path("after_outlier_treatment.pkl")


def _transform(func, input_path: str, output_path: str) -> str:
    """Load -> transform -> save helper for single-input steps."""
    df = load_pickle(input_path)
    before = len(df)
    out = func(df)
    logger.info("%s: %d -> %d rows", func.__name__, before, len(out))
    return save_pickle(out, output_path)


# --- transaction cleaning ---------------------------------------------------
def handle_missing(
    input_pickle_path: str = RAW,
    output_pickle_path: str = processed_path("after_missing_values.pkl"),
) -> str:
    """Drop rows without CustomerID / Description."""
    return _transform(cleaning.drop_missing, input_pickle_path, output_pickle_path)


def remove_duplicates(
    input_pickle_path: str = processed_path("after_missing_values.pkl"),
    output_pickle_path: str = processed_path("after_duplicates.pkl"),
) -> str:
    """Drop duplicated transaction lines."""
    return _transform(cleaning.drop_duplicates, input_pickle_path, output_pickle_path)


def handle_transaction_status(
    input_pickle_path: str = processed_path("after_duplicates.pkl"),
    output_pickle_path: str = processed_path("after_transaction_status.pkl"),
) -> str:
    """Flag cancelled invoices."""
    return _transform(cleaning.add_transaction_status, input_pickle_path, output_pickle_path)


def handle_anomalous_codes(
    input_pickle_path: str = processed_path("after_transaction_status.pkl"),
    output_pickle_path: str = processed_path("after_anomaly_code.pkl"),
) -> str:
    """Drop anomalous stock codes."""
    return _transform(cleaning.drop_anomalous_codes, input_pickle_path, output_pickle_path)


def cleaning_description(
    input_pickle_path: str = processed_path("after_anomaly_code.pkl"),
    output_pickle_path: str = processed_path("after_cleaning_description.pkl"),
) -> str:
    """Remove service lines and normalise descriptions."""
    return _transform(cleaning.clean_description, input_pickle_path, output_pickle_path)


def removing_zero(
    input_pickle_path: str = processed_path("after_cleaning_description.pkl"),
    output_pickle_path: str = CLEAN_TRANSACTIONS,
) -> str:
    """Remove non-positive unit prices."""
    return _transform(cleaning.drop_zero_unitprice, input_pickle_path, output_pickle_path)


# --- customer features ------------------------------------------------------
def rfm(
    transactions_path: str = CLEAN_TRANSACTIONS,
    output_pickle_path: str = processed_path("after_RFM.pkl"),
) -> str:
    """Build the customer table with RFM features."""
    return save_pickle(cleaning.rfm_features(load_pickle(transactions_path)), output_pickle_path)


def _enrich(func, customer_path: str, transactions_path: str, output_path: str) -> str:
    customer = load_pickle(customer_path)
    transactions = load_pickle(transactions_path)
    return save_pickle(func(transactions, customer), output_path)


def unique_products(
    customer_path: str = processed_path("after_RFM.pkl"),
    transactions_path: str = CLEAN_TRANSACTIONS,
    output_pickle_path: str = processed_path("unique_products.pkl"),
) -> str:
    """Add the number of distinct products bought."""
    return _enrich(
        cleaning.unique_products_features, customer_path, transactions_path, output_pickle_path
    )


def customers_behavior(
    customer_path: str = processed_path("unique_products.pkl"),
    transactions_path: str = CLEAN_TRANSACTIONS,
    output_pickle_path: str = processed_path("customers_behavior.pkl"),
) -> str:
    """Add behavioural features (purchase gaps, favourite day/hour)."""
    return _enrich(cleaning.behavior_features, customer_path, transactions_path, output_pickle_path)


def geographic_features(
    customer_path: str = processed_path("customers_behavior.pkl"),
    transactions_path: str = CLEAN_TRANSACTIONS,
    output_pickle_path: str = processed_path("geographic_features.pkl"),
) -> str:
    """Add the Is_UK flag."""
    return _enrich(
        cleaning.geographic_features, customer_path, transactions_path, output_pickle_path
    )


def cancellation_details(
    customer_path: str = processed_path("geographic_features.pkl"),
    transactions_path: str = CLEAN_TRANSACTIONS,
    output_pickle_path: str = processed_path("cancellation_details.pkl"),
) -> str:
    """Add cancellation frequency and rate."""
    return _enrich(
        cleaning.cancellation_features, customer_path, transactions_path, output_pickle_path
    )


def seasonality_impacts(
    customer_path: str = processed_path("cancellation_details.pkl"),
    transactions_path: str = CLEAN_TRANSACTIONS,
    output_pickle_path: str = CUSTOMER_FEATURES,
) -> str:
    """Add monthly spending statistics and trend."""
    return _enrich(
        cleaning.seasonality_features, customer_path, transactions_path, output_pickle_path
    )


def removing_outlier(
    input_pickle_path: str = CUSTOMER_FEATURES, output_pickle_path: str = CUSTOMERS_NO_OUTLIERS
) -> str:
    """Remove outlying customers (Isolation Forest)."""
    return _transform(cleaning.remove_outliers, input_pickle_path, output_pickle_path)


CLEANING_STEPS = [
    handle_missing,
    remove_duplicates,
    handle_transaction_status,
    handle_anomalous_codes,
    cleaning_description,
    removing_zero,
]
FEATURE_STEPS = [
    rfm,
    unique_products,
    customers_behavior,
    geographic_features,
    cancellation_details,
    seasonality_impacts,
    removing_outlier,
]


def run_pipeline(skip_download: bool = False, train: bool = True) -> dict:
    """Run every step sequentially and return the training summary (if any)."""
    # Imported lazily: heavy plotting / ML deps are only needed at this point.
    from src.clustering import train_and_save
    from src.correlation import correlation_check
    from src.pca import pc_analyzer
    from src.scaler import scaler

    start = time.perf_counter()
    if not skip_download:
        unzip_file(ingest_data())
    load_data()
    for step in CLEANING_STEPS + FEATURE_STEPS:
        logger.info("Running step %s", step.__name__)
        step()
    scaled = scaler()
    correlation_check(in_path=scaled)
    pc_analyzer(in_path=scaled)
    summary = train_and_save() if train else {}
    logger.info("Pipeline finished in %.1fs", time.perf_counter() - start)
    return summary


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Customer segmentation pipeline")
    parser.add_argument(
        "--skip-download", action="store_true", help="reuse the local Excel file / pickle cache"
    )
    parser.add_argument("--no-train", action="store_true", help="stop after PCA")
    args = parser.parse_args()
    summary = run_pipeline(skip_download=args.skip_download, train=not args.no_train)
    if summary:
        logger.info("Training summary: %s", summary)


if __name__ == "__main__":
    main()
