"""
Pure (in-memory) transformation functions used by the pipeline steps.

Keeping the business logic free of any file I/O makes every rule easy to unit
test with small synthetic DataFrames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest

DUPLICATE_COLUMNS = ["InvoiceNo", "StockCode", "Description", "CustomerID", "Quantity"]
SERVICE_DESCRIPTIONS = ["Next Day Carriage", "High Resolution Image"]


# --------------------------------------------------------------------------- #
# Transaction-level cleaning
# --------------------------------------------------------------------------- #
def drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows without ``CustomerID`` or ``Description``."""
    out = df.dropna(subset=["CustomerID", "Description"])
    missing = int(out.isna().sum().sum())
    if missing:
        raise ValueError(f"There are {missing} missing values left in the dataframe.")
    return out


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicated transaction lines."""
    return df.drop_duplicates(subset=DUPLICATE_COLUMNS)


def is_cancelled(invoice_no: pd.Series) -> pd.Series:
    """Boolean mask of cancelled invoices (``InvoiceNo`` starting with ``C``)."""
    return invoice_no.astype(str).str.startswith("C")


def add_transaction_status(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``transaction_status`` column (``Cancelled`` / ``Completed``)."""
    if "InvoiceNo" not in df.columns:
        raise KeyError("The input dataframe does not contain an 'InvoiceNo' column.")
    out = df.copy()
    out["transaction_status"] = np.where(is_cancelled(out["InvoiceNo"]), "Cancelled", "Completed")
    return out


def is_anomalous_code(code) -> bool:
    """Regular stock codes have 5-6 digits; codes with 0 or 1 digit are anomalies."""
    return sum(c.isdigit() for c in str(code)) in (0, 1)


def drop_anomalous_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose ``StockCode`` is anomalous (``POST``, ``M``, ``D``...)."""
    anomalous = [code for code in df["StockCode"].unique() if is_anomalous_code(code)]
    return df[~df["StockCode"].isin(anomalous)]


def clean_description(df: pd.DataFrame) -> pd.DataFrame:
    """Drop service lines and upper-case descriptions."""
    out = df[~df["Description"].isin(SERVICE_DESCRIPTIONS)].copy()
    out["Description"] = out["Description"].astype(str).str.strip().str.upper()
    return out


def drop_zero_unitprice(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with a strictly positive ``UnitPrice``."""
    return df[df["UnitPrice"] > 0]


# --------------------------------------------------------------------------- #
# Customer-level feature engineering
# --------------------------------------------------------------------------- #
def rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recency / Frequency / Monetary features per customer."""
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["InvoiceDay"] = df["InvoiceDate"].dt.normalize()

    most_recent = df["InvoiceDay"].max()
    customer = df.groupby("CustomerID")["InvoiceDay"].max().reset_index()
    customer["Days_Since_Last_Purchase"] = (most_recent - customer["InvoiceDay"]).dt.days
    customer = customer.drop(columns=["InvoiceDay"])

    transactions = (
        df.groupby("CustomerID")["InvoiceNo"].nunique().rename("Total_Transactions").reset_index()
    )
    products = (
        df.groupby("CustomerID")["Quantity"].sum().rename("Total_Products_Purchased").reset_index()
    )
    df["Total_Spend"] = df["UnitPrice"] * df["Quantity"]
    spend = df.groupby("CustomerID")["Total_Spend"].sum().reset_index()

    customer = (
        customer.merge(transactions, on="CustomerID")
        .merge(products, on="CustomerID")
        .merge(spend, on="CustomerID")
    )
    customer["Average_Transaction_Value"] = customer["Total_Spend"] / customer["Total_Transactions"]
    return customer


def unique_products_features(df: pd.DataFrame, customer: pd.DataFrame) -> pd.DataFrame:
    """Number of distinct products bought by each customer."""
    uniques = (
        df.groupby("CustomerID")["StockCode"]
        .nunique()
        .rename("Unique_Products_Purchased")
        .reset_index()
    )
    return customer.merge(uniques, on="CustomerID")


def behavior_features(df: pd.DataFrame, customer: pd.DataFrame) -> pd.DataFrame:
    """Average days between purchases, favourite weekday and favourite hour."""
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["InvoiceDay"] = df["InvoiceDate"].dt.normalize()
    df["Day_Of_Week"] = df["InvoiceDate"].dt.dayofweek
    df["Hour"] = df["InvoiceDate"].dt.hour

    # Gaps are measured between distinct purchase days (several lines of the
    # same invoice would otherwise count as 0-day gaps and bias the mean).
    days = (
        df[["CustomerID", "InvoiceDay"]].drop_duplicates().sort_values(["CustomerID", "InvoiceDay"])
    )
    gaps = days.groupby("CustomerID")["InvoiceDay"].diff().dt.days
    avg_gap = (
        gaps.groupby(days["CustomerID"])
        .mean()
        .fillna(0)
        .rename("Average_Days_Between_Purchases")
        .reset_index()
    )

    def _mode(column: str) -> pd.DataFrame:
        counts = df.groupby(["CustomerID", column]).size().reset_index(name="Count")
        idx = counts.groupby("CustomerID")["Count"].idxmax()
        return counts.loc[idx, ["CustomerID", column]]

    return (
        customer.merge(avg_gap, on="CustomerID")
        .merge(_mode("Day_Of_Week"), on="CustomerID")
        .merge(_mode("Hour"), on="CustomerID")
    )


def geographic_features(df: pd.DataFrame, customer: pd.DataFrame) -> pd.DataFrame:
    """``Is_UK`` flag based on the customer's main country."""
    country = (
        df.groupby(["CustomerID", "Country"]).size().reset_index(name="Number_of_Transactions")
    )
    main = country.sort_values("Number_of_Transactions", ascending=False).drop_duplicates(
        "CustomerID"
    )
    main["Is_UK"] = (main["Country"] == "United Kingdom").astype(int)
    return customer.merge(main[["CustomerID", "Is_UK"]], on="CustomerID", how="left")


def cancellation_features(df: pd.DataFrame, customer: pd.DataFrame) -> pd.DataFrame:
    """
    Number of cancelled invoices and cancellation rate per customer.

    The rate is computed by joining on ``CustomerID`` (the original code divided
    two Series aligned on their positional index, which mixed customers up).
    """
    total = df.groupby("CustomerID")["InvoiceNo"].nunique().rename("_total").reset_index()
    cancelled = (
        df[is_cancelled(df["InvoiceNo"])]
        .groupby("CustomerID")["InvoiceNo"]
        .nunique()
        .rename("Cancellation_Frequency")
        .reset_index()
    )
    out = customer.merge(cancelled, on="CustomerID", how="left").merge(
        total, on="CustomerID", how="left"
    )
    out["Cancellation_Frequency"] = out["Cancellation_Frequency"].fillna(0)
    out["Cancellation_Rate"] = (out["Cancellation_Frequency"] / out["_total"]).fillna(0)
    return out.drop(columns=["_total"])


def _trend(values: pd.Series) -> float:
    """Slope of a linear regression over the monthly spend series."""
    if len(values) < 2:
        return 0.0
    return float(stats.linregress(np.arange(len(values)), values.to_numpy()).slope)


def seasonality_features(df: pd.DataFrame, customer: pd.DataFrame) -> pd.DataFrame:
    """Monthly spending mean, standard deviation and trend per customer."""
    df = df.copy()
    df["Total_Spend"] = df["UnitPrice"] * df["Quantity"]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month

    monthly = df.groupby(["CustomerID", "Year", "Month"])["Total_Spend"].sum().reset_index()
    patterns = (
        monthly.groupby("CustomerID")["Total_Spend"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "Monthly_Spending_Mean", "std": "Monthly_Spending_Std"})
        .reset_index()
    )
    patterns["Monthly_Spending_Std"] = patterns["Monthly_Spending_Std"].fillna(0)
    trends = (
        monthly.groupby("CustomerID")["Total_Spend"]
        .apply(_trend)
        .rename("Spending_Trend")
        .reset_index()
    )

    out = customer.merge(patterns, on="CustomerID").merge(trends, on="CustomerID")
    out["CustomerID"] = out["CustomerID"].astype(str)
    return out.convert_dtypes()


def remove_outliers(
    customer: pd.DataFrame, contamination: float = 0.05, random_state: int = 0
) -> pd.DataFrame:
    """Remove outlying customers with an Isolation Forest (all columns but the ID)."""
    features = customer.drop(columns=["CustomerID"]).astype(float).to_numpy()
    model = IsolationForest(contamination=contamination, random_state=random_state)
    keep = model.fit_predict(features) == 1
    return customer.loc[keep].reset_index(drop=True)
