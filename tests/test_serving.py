"""Tests for the FastAPI serving app.

Points the app at a specific model.pt via MODEL_URI (bypassing the registry) and
exercises the lifecycle: startup load (via the TestClient context manager, which
runs the lifespan), /health readiness, /predict, and /reload. Guarded on
torch + fastapi.
"""

from __future__ import annotations

import pytest


def _write_model_pt(tmp_path):
    import torch

    from models.net import ModelConfig, SneakerPriceNet
    from stages.train import FEATURE_COLUMNS

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
    path = tmp_path / "model.pt"
    torch.save(payload, path)
    return str(path)


def _prepared_app(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_URI", _write_model_pt(tmp_path))
    from serving import app as appmod

    appmod._bundle = None  # reset the module cache between tests
    return appmod


def test_health_reports_loaded_model(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    appmod = _prepared_app(tmp_path, monkeypatch)
    with TestClient(appmod.app) as client:  # context manager runs the lifespan
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["feature_columns"]


def test_predict_single_and_batch(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    appmod = _prepared_app(tmp_path, monkeypatch)
    one = {
        "days_since_release": 100.0,
        "size_us": 9.0,
        "retail_price": 180.0,
        "size_premium": 0.1,
        "release_type_encoded": 2,
        "brand_avg_premium": 0.5,
        # nullable features omitted -> imputed with the saved training means
    }
    with TestClient(appmod.app) as client:
        resp = client.post("/predict", json=one)
        assert resp.status_code == 200
        assert "predicted_premium" in resp.json()

        resp = client.post("/predict/batch", json=[one, {**one, "size_us": 11.0}])
        assert resp.status_code == 200
        assert len(resp.json()["predictions"]) == 2


def test_reload_returns_version(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    appmod = _prepared_app(tmp_path, monkeypatch)
    with TestClient(appmod.app) as client:
        resp = client.post("/reload")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "reloaded"


def test_unready_returns_503_when_no_model(tmp_path, monkeypatch) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("mlflow")
    from fastapi.testclient import TestClient

    # No MODEL_URI and an empty registry -> startup load fails -> not ready.
    monkeypatch.delenv("MODEL_URI", raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/empty.db")
    monkeypatch.chdir(tmp_path)
    from serving import app as appmod

    appmod._bundle = None
    with TestClient(appmod.app) as client:
        assert client.get("/health").status_code == 503
        assert client.post("/predict", json={
            "days_since_release": 1.0, "size_us": 9.0, "retail_price": 180.0,
            "size_premium": 0.0, "release_type_encoded": 0, "brand_avg_premium": 0.0,
        }).status_code == 503
