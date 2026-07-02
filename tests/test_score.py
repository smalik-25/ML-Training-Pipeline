"""End-to-end test for the batch scoring stage.

Seeds an MLflow run + staging alias pointing at a real model.pt, writes a
validated-features batch, runs score.run, and checks the predictions and report.
Guarded on torch + mlflow (uses a SQLite registry).
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest


def test_score_run_writes_predictions_and_report(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("mlflow")

    import mlflow
    import torch
    from mlflow.tracking import MlflowClient

    from models.net import ModelConfig, SneakerPriceNet
    from stages import score
    from stages.config import PipelineConfig
    from stages.inference import MODEL_NAME, STAGING_ALIAS
    from stages.io import path_join, read_json, read_parquet, write_bytes, write_parquet
    from stages.train import FEATURE_COLUMNS, TARGET

    tracking = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking)
    # Pin the global tracking URI so seeding lands in THIS test's DB. A prior
    # test's set_tracking_uri() would otherwise take precedence over the env var.
    mlflow.set_tracking_uri(tracking)
    monkeypatch.chdir(tmp_path)

    config = PipelineConfig(
        storage_root=str(tmp_path), raw_prefix="raw", features_prefix="features",
        validated_prefix="validated", models_prefix="models",
        failures_prefix="failures", run_date="run",
    )

    # 1) write a real model.pt into the lake at the model_uri the run will point to.
    n = len(FEATURE_COLUMNS)
    model = SneakerPriceNet(n, ModelConfig(hidden_dim=8))
    model.eval()
    payload = {
        "state_dict": model.state_dict(),
        "input_dim": n,
        "model_config": {
            "hidden_dim": 8, "dropout_rate": 0.1, "learning_rate": 1e-3,
            "batch_size": 16, "num_epochs": 2,
        },
        "feature_columns": FEATURE_COLUMNS,
        "impute_values": [0.0] * n,
        "feature_mean": [0.0] * n,
        "feature_std": [1.0] * n,
        "run_date": "run",
    }
    model_uri = path_join(config.model_uri("run123"), "model.pt")
    buf = io.BytesIO()
    torch.save(payload, buf)
    write_bytes(buf.getvalue(), model_uri)

    # 2) seed an MLflow run that logs model_s3_uri, then register + alias staging.
    mlflow.set_experiment(MODEL_NAME)
    with mlflow.start_run() as active:
        mlflow.log_param("model_s3_uri", model_uri)
        mlflow.log_param("run_date", "run")
        run_id = active.info.run_id
        artifact_uri = active.info.artifact_uri
    client = MlflowClient(tracking)
    client.create_registered_model(MODEL_NAME)
    version = client.create_model_version(MODEL_NAME, source=artifact_uri, run_id=run_id)
    client.set_registered_model_alias(MODEL_NAME, STAGING_ALIAS, version.version)

    # 3) write a validated-features batch (with ground truth so RMSE is reported).
    rng = np.random.default_rng(0)
    features = pd.DataFrame({c: rng.normal(size=6) for c in FEATURE_COLUMNS})
    features["sale_id"] = range(6)
    features["shoe_id"] = 1
    features["brand"] = "Off-White"
    features["sale_date"] = pd.Timestamp("2019-01-01")
    features[TARGET] = rng.uniform(0, 2, size=6)
    write_parquet(features, config.validated_uri())

    # 4) score.
    out_uri = score.run(config)
    preds = read_parquet(out_uri)
    assert len(preds) == 6
    assert "predicted_premium" in preds.columns
    assert "actual_premium" in preds.columns

    report = read_json(config.scoring_report_uri())
    assert report["n_scored"] == 6
    assert report["model_version"] == str(version.version)
    assert "rmse_vs_actual" in report
