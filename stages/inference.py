"""Shared inference: load the promoted model and score with it.

Both the batch scoring stage (``stages/score.py``) and the serving app
(``serving/app.py``) go through this one module, so there's a single code path
for loading the model and applying preprocessing. That's the whole
no-training/serving-skew point: the imputation and standardization applied at
inference are the exact values fit on the training split and saved inside
``model.pt`` by the train stage, not a second implementation that can drift.

``build_matrix`` is deliberately torch-free (numpy only) so the transform can be
unit-tested without installing torch. The model load and forward pass import
torch and mlflow lazily.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pandas as pd

from stages.io import read_bytes
from stages.train import EXPERIMENT_NAME, resolve_tracking_uri

STAGING_ALIAS = "staging"
MODEL_NAME = EXPERIMENT_NAME  # "sneaker-price-model"


@dataclass
class ModelBundle:
    """A loaded model plus the preprocessing stats saved with it."""

    model: object  # torch.nn.Module in eval mode
    feature_columns: list[str]
    impute_values: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    run_id: str | None = None
    run_date: str | None = None
    model_version: str | None = None


def build_matrix(
    df: pd.DataFrame,
    feature_columns: list[str],
    impute_values: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> np.ndarray:
    """Apply the saved impute + standardize transform. Torch-free.

    Mirrors exactly what ``stages.train.prepare_arrays`` does at fit time, but
    using the persisted stats instead of recomputing them, so training and
    serving preprocessing cannot diverge.
    """
    # copy=True: to_numpy() can return a read-only view on newer pandas/numpy,
    # and we assign into it below.
    x = df[feature_columns].to_numpy(dtype=float, copy=True)
    rows, cols = np.where(np.isnan(x))
    x[rows, cols] = np.take(impute_values, cols)
    return (x - feature_mean) / feature_std


def _bundle_from_bytes(raw: bytes, **meta) -> ModelBundle:
    """Reconstruct a ModelBundle from serialized model.pt bytes."""
    import torch

    from models.net import ModelConfig, SneakerPriceNet

    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    model = SneakerPriceNet(payload["input_dim"], ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return ModelBundle(
        model=model,
        feature_columns=list(payload["feature_columns"]),
        impute_values=np.asarray(payload["impute_values"], dtype=float),
        feature_mean=np.asarray(payload["feature_mean"], dtype=float),
        feature_std=np.asarray(payload["feature_std"], dtype=float),
        run_date=payload.get("run_date"),
        **meta,
    )


def load_bundle_from_uri(model_uri: str) -> ModelBundle:
    """Load a bundle directly from a ``model.pt`` URI (S3 or local)."""
    return _bundle_from_bytes(read_bytes(model_uri))


def load_staging_bundle() -> ModelBundle:
    """Load the model currently at the ``staging`` alias in the MLflow registry.

    The registry alias is the source of truth for "which model is deployed". We
    resolve it to a run, read the ``model_s3_uri`` the train stage logged, and
    load the artifact through the same fsspec I/O layer the rest of the pipeline
    uses.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    tracking_uri = resolve_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    version = client.get_model_version_by_alias(MODEL_NAME, STAGING_ALIAS)
    run = client.get_run(version.run_id)
    model_uri = run.data.params.get("model_s3_uri")
    if not model_uri:
        raise RuntimeError(
            f"staging model (version {version.version}, run {version.run_id}) has "
            "no 'model_s3_uri' param -- was it produced by the train stage?"
        )

    return _bundle_from_bytes(
        read_bytes(model_uri),
        run_id=version.run_id,
        model_version=str(version.version),
    )


def predict(bundle: ModelBundle, df: pd.DataFrame) -> np.ndarray:
    """Predict price_premium for a DataFrame of engineered features."""
    import torch

    x = build_matrix(
        df,
        bundle.feature_columns,
        bundle.impute_values,
        bundle.feature_mean,
        bundle.feature_std,
    )
    with torch.no_grad():
        preds = bundle.model(torch.tensor(x, dtype=torch.float32))
    return preds.numpy()


def predict_records(bundle: ModelBundle, records: list[dict]) -> list[float]:
    """Predict for a list of feature dicts (used by the serving API)."""
    df = pd.DataFrame(records)
    return [float(p) for p in predict(bundle, df)]
