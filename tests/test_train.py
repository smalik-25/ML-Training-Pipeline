"""Tests for the training stage.

The pure data-prep logic (temporal split, imputation, standardization) is
torch-free and tested directly here, so it runs in the light CI job. The model
forward pass and the full Ray training run are guarded with ``importorskip`` and
execute only where torch/ray/mlflow are installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stages.train import (
    FEATURE_COLUMNS,
    SPLIT_YEAR,
    TARGET,
    prepare_arrays,
    temporal_split,
)


def _toy_validated() -> pd.DataFrame:
    """A small validated-features frame straddling the 2023 split boundary."""
    dates = [
        "2021-06-01", "2021-09-01", "2022-03-01", "2022-08-01",  # train (<2023)
        "2022-11-01", "2023-02-01", "2023-07-01", "2024-01-01",  # val (>=2023)
    ]
    n = len(dates)
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "sale_date": pd.to_datetime(dates),
            "days_since_release": rng.integers(1, 500, n).astype(float),
            "size_us": rng.choice([8.0, 9.0, 10.0, 11.0], n),
            "retail_price": rng.choice([100.0, 180.0, 220.0], n),
            "size_premium": rng.normal(0, 0.1, n),
            "release_type_encoded": rng.integers(0, 4, n).astype(float),
            "rolling_7d_avg_premium": rng.normal(0.5, 0.2, n),
            "search_index_7d_pre_drop": rng.uniform(20, 95, n),
            "brand_avg_premium": rng.uniform(0.2, 1.1, n),
            TARGET: rng.uniform(0.0, 2.0, n),
        }
    )
    # Two insufficient-history rows carry a null rolling average.
    df.loc[0, "rolling_7d_avg_premium"] = np.nan
    df.loc[5, "rolling_7d_avg_premium"] = np.nan
    return df


def test_temporal_split_boundary() -> None:
    train_df, val_df = temporal_split(_toy_validated(), SPLIT_YEAR)
    assert pd.to_datetime(train_df["sale_date"]).dt.year.max() < SPLIT_YEAR
    assert pd.to_datetime(val_df["sale_date"]).dt.year.min() >= SPLIT_YEAR
    assert len(train_df) == 5 and len(val_df) == 3


def test_prepare_arrays_imputes_and_standardizes() -> None:
    arrays = prepare_arrays(_toy_validated(), SPLIT_YEAR)

    # No NaNs survive imputation.
    assert not np.isnan(arrays.X_train).any()
    assert not np.isnan(arrays.X_val).any()
    # Shapes line up with feature count.
    assert arrays.X_train.shape == (5, len(FEATURE_COLUMNS))
    assert arrays.X_val.shape == (3, len(FEATURE_COLUMNS))
    # Standardized on train stats -> train columns ~ mean 0, std 1.
    assert np.allclose(arrays.X_train.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(arrays.X_train.std(axis=0), 1.0, atol=1e-9)
    # Stats are saved for inference reproducibility.
    assert len(arrays.impute_values) == len(FEATURE_COLUMNS)
    assert len(arrays.feature_mean) == len(FEATURE_COLUMNS)


def test_prepare_arrays_no_val_leakage() -> None:
    """Val is standardized with TRAIN stats, so its mean is generally not 0."""
    arrays = prepare_arrays(_toy_validated(), SPLIT_YEAR)
    assert not np.allclose(arrays.X_val.mean(axis=0), 0.0, atol=1e-6)


def test_prepare_arrays_raises_on_empty_side() -> None:
    df = _toy_validated()
    df["sale_date"] = pd.to_datetime("2021-01-01")  # all train, no val
    with pytest.raises(ValueError, match="temporal split"):
        prepare_arrays(df, SPLIT_YEAR)


def test_net_forward_shape() -> None:
    torch = pytest.importorskip("torch")
    from models.net import ModelConfig, SneakerPriceNet

    model = SneakerPriceNet(input_dim=len(FEATURE_COLUMNS), config=ModelConfig())
    model.eval()
    x = torch.randn(4, len(FEATURE_COLUMNS))
    out = model(x)
    assert out.shape == (4,)


def test_run_trains_and_saves_model(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("ray")
    pytest.importorskip("mlflow")

    from stages.config import PipelineConfig
    from stages.io import read_bytes, write_parquet

    config = PipelineConfig(
        storage_root=str(tmp_path), raw_prefix="raw", features_prefix="features",
        validated_prefix="validated", models_prefix="models",
        failures_prefix="failures", run_date="run",
    )
    write_parquet(_toy_validated(), config.validated_uri())
    # SQLite backend (the file store is deprecated in MLflow 3.x and can't back
    # the registry). chdir so artifacts land under the tmp dir.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.chdir(tmp_path)

    run_id = __import__("stages.train", fromlist=["run"]).run(
        config,
        num_workers=1,
        model_config={"hidden_dim": 8, "batch_size": 16, "num_epochs": 2},
    )
    assert isinstance(run_id, str) and run_id

    model_bytes = read_bytes(
        __import__("stages.io", fromlist=["path_join"]).path_join(
            config.model_uri(run_id), "model.pt"
        )
    )
    assert len(model_bytes) > 0
