"""Unit tests for the pure cleaning / feature-engineering functions."""

import numpy as np
import pandas as pd
import pytest

from src import cleaning


def test_drop_missing_removes_rows_without_customer_or_description(transactions):
    out = cleaning.drop_missing(transactions)
    assert out["CustomerID"].notna().all()
    assert out["Description"].notna().all()
    assert len(out) == len(transactions) - 2


def test_drop_missing_raises_when_other_nans_remain(transactions):
    df = transactions.copy()
    df.loc[10, "Country"] = np.nan
    with pytest.raises(ValueError):
        cleaning.drop_missing(df)


def test_drop_duplicates(transactions):
    out = cleaning.drop_duplicates(transactions)
    assert not out.duplicated(subset=cleaning.DUPLICATE_COLUMNS).any()


def test_transaction_status(transactions):
    out = cleaning.add_transaction_status(transactions)
    cancelled = out["InvoiceNo"].str.startswith("C")
    assert (out.loc[cancelled, "transaction_status"] == "Cancelled").all()
    assert (out.loc[~cancelled, "transaction_status"] == "Completed").all()
    assert "transaction_status" not in transactions.columns  # input not mutated


def test_transaction_status_requires_invoice_column(transactions):
    with pytest.raises(KeyError):
        cleaning.add_transaction_status(transactions.drop(columns="InvoiceNo"))


@pytest.mark.parametrize(
    "code, expected",
    [("POST", True), ("M", True), ("D", True), ("C2", True), ("85123A", False), ("22423", False)],
)
def test_is_anomalous_code(code, expected):
    assert cleaning.is_anomalous_code(code) is expected


def test_clean_description(transactions):
    out = cleaning.clean_description(transactions.dropna(subset=["Description"]))
    assert "NEXT DAY CARRIAGE" not in out["Description"].values
    assert (out["Description"] == out["Description"].str.upper()).all()


def test_drop_zero_unitprice(transactions):
    assert (cleaning.drop_zero_unitprice(transactions)["UnitPrice"] > 0).all()


def test_rfm_features_values():
    df = pd.DataFrame(
        {
            "CustomerID": [1, 1, 2],
            "InvoiceNo": ["a", "b", "c"],
            "InvoiceDate": pd.to_datetime(["2011-01-01", "2011-01-11", "2011-01-11"]),
            "Quantity": [2, 3, 10],
            "UnitPrice": [5.0, 10.0, 1.0],
        }
    )
    out = cleaning.rfm_features(df).set_index("CustomerID")
    assert out.loc[1, "Total_Transactions"] == 2
    assert out.loc[1, "Total_Spend"] == pytest.approx(40.0)
    assert out.loc[1, "Average_Transaction_Value"] == pytest.approx(20.0)
    assert out.loc[2, "Days_Since_Last_Purchase"] == 0


def test_cancellation_rate_is_aligned_per_customer():
    """Regression test: the rate must be computed per CustomerID, not per row position."""
    df = pd.DataFrame(
        {
            "CustomerID": [1, 1, 1, 1, 2, 2],
            "InvoiceNo": ["C1", "2", "3", "4", "C5", "6"],
        }
    )
    customer = pd.DataFrame({"CustomerID": [2, 1]})  # deliberately reversed order
    out = cleaning.cancellation_features(df, customer).set_index("CustomerID")
    assert out.loc[1, "Cancellation_Rate"] == pytest.approx(0.25)
    assert out.loc[2, "Cancellation_Rate"] == pytest.approx(0.5)


def test_behavior_features_gap():
    df = pd.DataFrame(
        {
            "CustomerID": [1, 1, 1],
            "InvoiceDate": pd.to_datetime(
                ["2011-01-10 10:00", "2011-01-01 10:00", "2011-01-04 15:00"]
            ),
        }
    )
    out = cleaning.behavior_features(df, pd.DataFrame({"CustomerID": [1]}))
    assert out.loc[0, "Average_Days_Between_Purchases"] == pytest.approx(4.5)
    assert out.loc[0, "Hour"] == 10


def test_geographic_flag():
    df = pd.DataFrame(
        {"CustomerID": [1, 1, 2], "Country": ["United Kingdom", "United Kingdom", "France"]}
    )
    out = cleaning.geographic_features(df, pd.DataFrame({"CustomerID": [1, 2]}))
    assert out.set_index("CustomerID")["Is_UK"].to_dict() == {1: 1, 2: 0}


def test_seasonality_trend_positive_for_growing_spend():
    df = pd.DataFrame(
        {
            "CustomerID": [1, 1, 1],
            "InvoiceDate": pd.to_datetime(["2011-01-05", "2011-02-05", "2011-03-05"]),
            "Quantity": [1, 2, 3],
            "UnitPrice": [10.0, 10.0, 10.0],
        }
    )
    out = cleaning.seasonality_features(df, pd.DataFrame({"CustomerID": [1]}))
    assert out.loc[0, "Spending_Trend"] == pytest.approx(10.0)
    assert out.loc[0, "Monthly_Spending_Mean"] == pytest.approx(20.0)


def test_full_feature_table(customers):
    expected = {
        "Days_Since_Last_Purchase",
        "Total_Transactions",
        "Unique_Products_Purchased",
        "Average_Days_Between_Purchases",
        "Is_UK",
        "Cancellation_Rate",
        "Spending_Trend",
    }
    assert expected <= set(customers.columns)
    assert customers["CustomerID"].is_unique
    assert not customers.isna().any().any()


def test_remove_outliers(customers):
    out = cleaning.remove_outliers(customers, contamination=0.1)
    assert len(out) < len(customers)
    assert list(out.index) == list(range(len(out)))
