"""FastAPI app that serves the promoted (staging) price-premium model.

This is the online counterpart to the batch scoring stage. It loads the same
``model.pt`` (weights plus the preprocessing stats fit on the training split)
through the same ``stages.inference`` module, so an online request and a batch
row go through byte-identical preprocessing. That's the no-training/serving-skew
guarantee made concrete.

The endpoint takes *engineered* features, the same columns the model was trained
on, because feature computation is an upstream (Spark) concern. Nullable
features (rolling average, pre-drop search) are allowed and imputed with the
saved training means, exactly as in training.

Run it:
    uvicorn serving.app:app --port 8000
    # By default it loads the model at the MLflow `staging` alias. To pin a
    # specific artifact instead, set MODEL_URI=/path/to/model.pt.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from stages import inference

app = FastAPI(
    title="sneaker price-premium model",
    description="Serves the model promoted to the MLflow 'staging' alias.",
    version="1.0.0",
)

_bundle: inference.ModelBundle | None = None


def get_bundle() -> inference.ModelBundle:
    """Load the model once and cache it. MODEL_URI overrides the registry."""
    global _bundle
    if _bundle is None:
        model_uri = os.environ.get("MODEL_URI")
        _bundle = (
            inference.load_bundle_from_uri(model_uri)
            if model_uri
            else inference.load_staging_bundle()
        )
    return _bundle


class SaleFeatures(BaseModel):
    """Engineered features for one sale (the model's input columns)."""

    days_since_release: float
    size_us: float = Field(gt=0)
    retail_price: float = Field(gt=0)
    size_premium: float
    release_type_encoded: float
    brand_avg_premium: float
    # Nullable: imputed with the saved training mean, same as in training.
    rolling_7d_avg_premium: float | None = None
    search_index_7d_pre_drop: float | None = None


class Prediction(BaseModel):
    predicted_premium: float
    model_version: str | None = None


class BatchPrediction(BaseModel):
    predictions: list[float]
    model_version: str | None = None


@app.get("/")
def root() -> dict:
    return {"service": "sneaker price-premium model", "docs": "/docs"}


@app.get("/health")
def health() -> dict:
    """Report whether a model is loaded and which version it is."""
    try:
        bundle = get_bundle()
    except Exception as exc:  # surface load failures without crashing the app
        raise HTTPException(status_code=503, detail=f"model not available: {exc}") from exc
    return {
        "status": "ok",
        "model_name": inference.MODEL_NAME,
        "model_version": bundle.model_version,
        "run_id": bundle.run_id,
        "run_date": bundle.run_date,
        "feature_columns": bundle.feature_columns,
    }


@app.post("/predict", response_model=Prediction)
def predict_one(features: SaleFeatures) -> Prediction:
    bundle = get_bundle()
    preds = inference.predict_records(bundle, [features.model_dump()])
    return Prediction(predicted_premium=preds[0], model_version=bundle.model_version)


@app.post("/predict/batch", response_model=BatchPrediction)
def predict_batch(features: list[SaleFeatures]) -> BatchPrediction:
    if not features:
        raise HTTPException(status_code=422, detail="empty batch")
    bundle = get_bundle()
    preds = inference.predict_records(bundle, [f.model_dump() for f in features])
    return BatchPrediction(predictions=preds, model_version=bundle.model_version)
