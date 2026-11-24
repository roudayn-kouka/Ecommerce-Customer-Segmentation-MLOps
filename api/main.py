"""
REST API serving the customer segmentation model.

Run locally::

    uvicorn api.main:app --reload --port 8000

Then open http://localhost:8000/docs for the interactive Swagger UI.
"""

from __future__ import annotations

import os
from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models",
        "segmentation_model.joblib",
    ),
)

SEGMENT_NAMES = os.environ.get("SEGMENT_NAMES")  # optional "0:Loyal,1:At risk,..."


class Customer(BaseModel):
    """Customer features produced by the data pipeline."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Days_Since_Last_Purchase": 12,
                "Total_Transactions": 8,
                "Total_Products_Purchased": 640,
                "Total_Spend": 1450.5,
                "Average_Transaction_Value": 181.3,
                "Unique_Products_Purchased": 57,
                "Average_Days_Between_Purchases": 3.4,
                "Day_Of_Week": 2,
                "Hour": 12,
                "Is_UK": 1,
                "Cancellation_Frequency": 1,
                "Cancellation_Rate": 0.125,
                "Monthly_Spending_Mean": 181.3,
                "Monthly_Spending_Std": 60.2,
                "Spending_Trend": 4.1,
            }
        }
    )

    Days_Since_Last_Purchase: float = Field(ge=0)
    Total_Transactions: float = Field(ge=0)
    Total_Products_Purchased: float
    Total_Spend: float
    Average_Transaction_Value: float
    Unique_Products_Purchased: float = Field(ge=0)
    Average_Days_Between_Purchases: float = Field(ge=0)
    Day_Of_Week: int = Field(ge=0, le=6)
    Hour: int = Field(ge=0, le=23)
    Is_UK: int = Field(ge=0, le=1)
    Cancellation_Frequency: float = Field(ge=0)
    Cancellation_Rate: float = Field(ge=0, le=1)
    Monthly_Spending_Mean: float
    Monthly_Spending_Std: float = Field(ge=0)
    Spending_Trend: float


class Prediction(BaseModel):
    """Predicted segment for one customer."""

    cluster: int
    segment: str


@lru_cache(maxsize=1)
def load_bundle() -> dict:
    """Load the model bundle once (cached)."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run `make train` first.")
    return joblib.load(MODEL_PATH)


def _segment_name(cluster: int) -> str:
    if SEGMENT_NAMES:
        names = dict(item.split(":", 1) for item in SEGMENT_NAMES.split(","))
        return names.get(str(cluster), f"Segment {cluster}")
    return f"Segment {cluster}"


app = FastAPI(
    title="Customer Segmentation API",
    version="2.0.0",
    description="Assigns e-commerce customers to behavioural segments.",
)


@app.get("/health")
def health() -> dict:
    """Liveness / readiness probe."""
    try:
        bundle = load_bundle()
    except FileNotFoundError as exc:
        return {"status": "degraded", "model_loaded": False, "detail": str(exc)}
    return {"status": "ok", "model_loaded": True, "model": bundle["best"]}


@app.post("/predict", response_model=list[Prediction])
def predict(customers: list[Customer]) -> list[Prediction]:
    """Predict the segment of one or many customers."""
    if not customers:
        raise HTTPException(status_code=422, detail="Empty payload.")
    try:
        bundle = load_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    frame = pd.DataFrame([c.model_dump() for c in customers])[bundle["features"]].astype(float)
    labels = bundle["model"].predict(frame)
    return [Prediction(cluster=int(lbl), segment=_segment_name(int(lbl))) for lbl in labels]
