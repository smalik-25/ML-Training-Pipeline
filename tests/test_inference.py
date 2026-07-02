"""Tests for the shared inference module.

``build_matrix`` (the preprocessing transform) is torch-free and runs anywhere.
The load + predict round-trip is guarded on torch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stages.inference import build_matrix


def test_build_matrix_imputes_then_standardizes() -> None:
    cols = ["a", "b"]
    df = pd.DataFrame({"a": [1.0, np.nan], "b": [10.0, 20.0]})
    impute = np.array([5.0, 0.0])
    mean = np.array([2.0, 15.0])
    std = np.array([2.0, 5.0])

    x = build_matrix(df, cols, impute, mean, std)

    # row 0: a=(1-2)/2=-0.5, b=(10-15)/5=-1.0
    # row 1: a imputed to 5 -> (5-2)/2=1.5, b=(20-15)/5=1.0
    assert np.allclose(x, [[-0.5, -1.0], [1.5, 1.0]])
    assert not np.isnan(x).any()


def _make_model_pt(tmp_path):
    """Build a real (untrained) model.pt payload for the round-trip test."""
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
    return str(path), FEATURE_COLUMNS


def test_load_bundle_and_predict_roundtrip(tmp_path) -> None:
    pytest.importorskip("torch")
    from stages.inference import load_bundle_from_uri, predict

    model_uri, feature_columns = _make_model_pt(tmp_path)
    bundle = load_bundle_from_uri(model_uri)
    assert bundle.feature_columns == feature_columns

    df = pd.DataFrame({c: [1.0, 2.0] for c in feature_columns})
    df.loc[0, "rolling_7d_avg_premium"] = np.nan  # nullable feature -> imputed

    preds = predict(bundle, df)
    assert preds.shape == (2,)
    assert not np.isnan(preds).any()  # imputation kept NaNs out of the model
