"""
Shared fixtures: small synthetic datasets so the test-suite runs offline,
in a few seconds, without the 45 MB UCI file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import cleaning


def make_transactions(n_customers: int = 60, seed: int = 0) -> pd.DataFrame:
    """Synthetic Online Retail-like transactions with a few known anomalies."""
    rng = np.random.default_rng(seed)
    rows = []
    invoice = 500000
    for cid in range(n_customers):
        country = "United Kingdom" if cid % 3 else "France"
        n_invoices = int(rng.integers(1, 6))
        for _ in range(n_invoices):
            invoice += 1
            cancelled = rng.random() < 0.1
            date = pd.Timestamp("2011-01-01") + pd.Timedelta(
                days=int(rng.integers(0, 330)), hours=int(rng.integers(8, 19))
            )
            for _ in range(int(rng.integers(1, 4))):
                code = str(85000 + int(rng.integers(0, 40)))
                qty = int(rng.integers(1, 24))
                rows.append(
                    {
                        "InvoiceNo": f"C{invoice}" if cancelled else str(invoice),
                        "StockCode": code,
                        "Description": f"product {code}",
                        "Quantity": -qty if cancelled else qty,
                        "InvoiceDate": date,
                        "UnitPrice": round(float(rng.uniform(0.5, 15)), 2),
                        "CustomerID": float(12000 + cid),
                        "Country": country,
                    }
                )
    df = pd.DataFrame(rows)
    anomalies = pd.DataFrame(
        [
            {**rows[0], "CustomerID": np.nan},  # missing customer
            {**rows[1], "Description": None},  # missing description
            {**rows[2], "StockCode": "POST"},  # anomalous code
            {**rows[3], "Description": "Next Day Carriage", "StockCode": "84001"},
            {**rows[4], "UnitPrice": 0.0, "StockCode": "84002"},  # zero price
        ]
    )
    return pd.concat([df, anomalies, df.iloc[[5]]], ignore_index=True)  # + 1 duplicate


@pytest.fixture
def transactions() -> pd.DataFrame:
    """Raw synthetic transactions."""
    return make_transactions()


@pytest.fixture
def clean_transactions(transactions) -> pd.DataFrame:
    """Transactions after every cleaning rule."""
    df = cleaning.drop_missing(transactions)
    df = cleaning.drop_duplicates(df)
    df = cleaning.add_transaction_status(df)
    df = cleaning.drop_anomalous_codes(df)
    df = cleaning.clean_description(df)
    return cleaning.drop_zero_unitprice(df)


@pytest.fixture
def customers(clean_transactions) -> pd.DataFrame:
    """Customer-level feature table."""
    df = clean_transactions
    out = cleaning.rfm_features(df)
    for func in (
        cleaning.unique_products_features,
        cleaning.behavior_features,
        cleaning.geographic_features,
        cleaning.cancellation_features,
        cleaning.seasonality_features,
    ):
        out = func(df, out)
    return out


@pytest.fixture
def blob_customers(customers) -> pd.DataFrame:
    """Customers with two clearly separated spending groups (for clustering tests)."""
    out = customers.copy()
    big = out.index % 2 == 0
    for col in ("Total_Spend", "Monthly_Spending_Mean", "Average_Transaction_Value"):
        out[col] = out[col].astype(float)
        out.loc[big, col] = out.loc[big, col] * 20 + 5000
    return out
