"""Tests for the MLflow registry/promotion stage.

MLflow-only (no torch/ray). Uses a SQLite-backed tracking store in a temp dir
because the model registry requires a database backend. Each test seeds runs
directly via MLflow rather than running training, then exercises the promotion
decision.
"""

from __future__ import annotations

import os
import tempfile

import pytest

pytest.importorskip("mlflow")

import mlflow  # noqa: E402

from stages.config import PipelineConfig  # noqa: E402
from stages.io import read_json  # noqa: E402
from stages.register import run  # noqa: E402
from stages.train import EXPERIMENT_NAME  # noqa: E402


def _config(root: str, run_date: str) -> PipelineConfig:
    return PipelineConfig(
        storage_root=root, raw_prefix="raw", features_prefix="features",
        validated_prefix="validated", models_prefix="models",
        failures_prefix="failures", run_date=run_date,
    )


def _seed_run(run_date: str, rmse: float) -> str:
    """Create an MLflow run with a run_date param, a val RMSE, and a model artifact."""
    # Pin the tracking URI from env so seeds land in THIS test's SQLite DB.
    # (mlflow's global tracking URI, once set by a prior test's run(), otherwise
    # takes precedence over the env var.)
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run() as r, tempfile.TemporaryDirectory() as d:
        mlflow.log_param("run_date", run_date)
        mlflow.log_metric("final_val_rmse", rmse)
        path = os.path.join(d, "model.pt")
        with open(path, "wb") as fh:
            fh.write(b"dummy-weights")
        mlflow.log_artifact(path, artifact_path="model")
        return r.info.run_id


@pytest.fixture
def tracking(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_first_run_promotes(tracking) -> None:
    _seed_run("d1", rmse=0.50)
    report = read_json(run(_config(str(tracking), "d1")))
    assert report["promoted"] is True
    assert report["beat_metric"] == "no existing model"
    assert report["val_rmse"] == 0.50

    client = mlflow.tracking.MlflowClient()
    staged = client.get_model_version_by_alias(EXPERIMENT_NAME, "staging")
    assert int(staged.version) == report["model_version"]


def test_better_model_promotes_and_beats_prior(tracking) -> None:
    _seed_run("d1", rmse=0.50)
    run(_config(str(tracking), "d1"))           # promotes 0.50
    _seed_run("d2", rmse=0.30)
    report = read_json(run(_config(str(tracking), "d2")))  # 0.30 beats 0.50
    assert report["promoted"] is True
    assert report["beat_metric"] == 0.50
    assert report["val_rmse"] == 0.30


def test_worse_model_not_promoted(tracking) -> None:
    _seed_run("d1", rmse=0.30)
    run(_config(str(tracking), "d1"))           # promotes 0.30
    _seed_run("d2", rmse=0.90)
    report = read_json(run(_config(str(tracking), "d2")))  # 0.90 worse
    assert report["promoted"] is False
    assert report["beat_metric"] == 0.30

    # The staging alias still points at the better (0.30) model.
    client = mlflow.tracking.MlflowClient()
    staged = client.get_model_version_by_alias(EXPERIMENT_NAME, "staging")
    staged_rmse = client.get_run(staged.run_id).data.metrics["final_val_rmse"]
    assert staged_rmse == 0.30


def test_best_of_multiple_runs_for_run_date(tracking) -> None:
    """Among several runs for one run_date, the lowest-RMSE run is registered."""
    _seed_run("d1", rmse=0.70)
    _seed_run("d1", rmse=0.40)  # best for d1
    _seed_run("d1", rmse=0.55)
    report = read_json(run(_config(str(tracking), "d1")))
    assert report["val_rmse"] == 0.40
    assert report["promoted"] is True


def test_no_runs_for_run_date_raises(tracking) -> None:
    _seed_run("d1", rmse=0.50)
    with pytest.raises(RuntimeError, match="no training runs"):
        run(_config(str(tracking), "d2"))
