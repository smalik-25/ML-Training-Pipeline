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

Model lifecycle:
  * The model is loaded at startup (a FastAPI lifespan handler). If that load
    fails, the app stays up but reports unready via /health, so an orchestrator's
    readiness probe holds traffic until a model is available.
  * The registry alias is the source of truth for what's live. When you promote
    a new version or roll back by moving the ``staging`` alias, call POST /reload
    to pick it up without restarting the process.

Run it:
    uvicorn serving.app:app --port 8000
    # Loads the model at the MLflow `staging` alias by default. To pin a specific
    # artifact instead, set MODEL_URI=/path/to/model.pt.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from stages import inference

log = logging.getLogger("serving.app")

_bundle: inference.ModelBundle | None = None


def _load() -> inference.ModelBundle:
    """Load from MODEL_URI if set, otherwise from the registry staging alias."""
    model_uri = os.environ.get("MODEL_URI")
    if model_uri:
        return inference.load_bundle_from_uri(model_uri)
    return inference.load_staging_bundle()


def reload_bundle() -> inference.ModelBundle:
    """(Re)load the model and replace the cache. Picks up a moved staging alias."""
    global _bundle
    _bundle = _load()
    log.info(
        "model loaded: version=%s run_id=%s", _bundle.model_version, _bundle.run_id
    )
    return _bundle


def get_bundle() -> inference.ModelBundle:
    """Return the loaded model or 503 if none is loaded (not ready)."""
    if _bundle is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return _bundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load at startup so a broken/unavailable model shows up as an unready
    # readiness check, not a 500 on the first user request. Stay up on failure
    # so /reload can recover without a restart.
    try:
        reload_bundle()
    except Exception:
        log.exception("startup model load failed; unready until /reload succeeds")
    yield


app = FastAPI(
    title="sneaker price-premium model",
    description="Serves the model promoted to the MLflow 'staging' alias.",
    version="1.0.0",
    lifespan=lifespan,
)


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


class ReloadResult(BaseModel):
    status: str
    model_version: str | None = None
    run_id: str | None = None


@app.get("/")
def root() -> dict:
    return {"service": "sneaker price-premium model", "docs": "/docs"}


@app.get("/health")
def health() -> dict:
    """Readiness: 200 with the loaded version, or 503 if no model is loaded."""
    bundle = get_bundle()
    return {
        "status": "ok",
        "model_name": inference.MODEL_NAME,
        "model_version": bundle.model_version,
        "run_id": bundle.run_id,
        "run_date": bundle.run_date,
        "feature_columns": bundle.feature_columns,
    }


@app.post("/reload", response_model=ReloadResult)
def reload() -> ReloadResult:
    """Reload the model from the registry (or MODEL_URI). Use after promotion."""
    try:
        bundle = reload_bundle()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"reload failed: {exc}") from exc
    return ReloadResult(
        status="reloaded",
        model_version=bundle.model_version,
        run_id=bundle.run_id,
    )


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
