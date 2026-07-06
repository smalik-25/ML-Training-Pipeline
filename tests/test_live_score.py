"""Tests for live KicksDB scoring through the shared inference path.

The feature builder and the uncomputable-SKU logic are torch-free; the scoring
round-trip is guarded on torch (it runs in the train/serve CI job).
"""

from __future__ import annotations

import numpy as np
import pytest

from stages import kicksdb, live_score

_ORDER = {"limited": 2, "collab": 1, "general": 0}


def test_snapshot_features_cover_the_model_inputs() -> None:
    from stages.train import FEATURE_COLUMNS

    frames = kicksdb.read_all("2026-07-04", api_key=None)
    feats = live_score.build_snapshot_features(frames, _ORDER)

    for col in FEATURE_COLUMNS:
        assert col in feats.columns, f"missing model input {col}"
    # Snapshot-honest values: one size, no history, no pre-drop signal.
    assert (feats["size_premium"] == 0.0).all()
    assert feats["rolling_7d_avg_premium"].isna().all()
    assert feats["search_index_7d_pre_drop"].isna().all()
    assert (feats["days_since_release"] > 0).all()


def test_uncomputable_skus_reports_missing_retail() -> None:
    ref = kicksdb.load_retail_reference()
    requested = ["DD1391-100", "TOTALLY-UNKNOWN-1"]
    # A SKU with no retail reference is reported, never scored on a guess.
    assert live_score.uncomputable_skus(requested, ref) == ["TOTALLY-UNKNOWN-1"]


def _untrained_bundle(tmp_path):
    import torch

    from models.net import ModelConfig, SneakerPriceNet
    from stages.inference import load_bundle_from_uri
    from stages.train import FEATURE_COLUMNS

    n = len(FEATURE_COLUMNS)
    model = SneakerPriceNet(n, ModelConfig(hidden_dim=8))
    model.eval()
    payload = {
        "state_dict": model.state_dict(), "input_dim": n,
        "model_config": {
            "hidden_dim": 8, "dropout_rate": 0.1, "learning_rate": 1e-3,
            "batch_size": 16, "num_epochs": 2,
        },
        "feature_columns": FEATURE_COLUMNS, "impute_values": [0.0] * n,
        "feature_mean": [0.0] * n, "feature_std": [1.0] * n, "run_date": "run",
    }
    path = tmp_path / "model.pt"
    torch.save(payload, path)
    return load_bundle_from_uri(str(path))


def test_score_snapshot_uses_the_one_shared_transform(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")
    from stages import inference

    bundle = _untrained_bundle(tmp_path)
    frames = kicksdb.read_all("2026-07-04", api_key=None)

    # The transform must be stages.inference.build_matrix, not a copy that can
    # drift from training preprocessing.
    calls = {"n": 0}
    real = inference.build_matrix

    def spy(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(inference, "build_matrix", spy)
    out = live_score.score_snapshot(frames, bundle, _ORDER)

    assert calls["n"] == 1  # scored through the single shared transform
    assert len(out) == len(frames["shoes"])
    # Untrained weights here, so predictions are arbitrary but must be real
    # numbers (no NaN leaking through the imputation of the null features).
    assert np.isfinite(out["predicted_premium"].to_numpy()).all()
    assert "implied_resale" in out.columns


def test_score_current_reports_uncomputable_requested_skus(tmp_path, monkeypatch) -> None:
    """A requested SKU with no retail reference is reported via
    results.attrs['uncomputable_skus'] and never scored on a guessed retail."""
    pytest.importorskip("torch")
    monkeypatch.delenv("KICKSDB_API_KEY", raising=False)

    from stages.config import PipelineConfig, load_feature_config

    bundle = _untrained_bundle(tmp_path)
    config = PipelineConfig(
        storage_root=str(tmp_path), raw_prefix="raw", features_prefix="features",
        validated_prefix="validated", models_prefix="models",
        failures_prefix="failures", run_date="2026-07-04",
    )
    fcfg = load_feature_config("config/feature_config.yaml")

    reference = kicksdb.load_retail_reference()
    requested = reference.skus() + ["TOTALLY-UNKNOWN-1"]
    results = live_score.score_current(
        config, fcfg, bundle=bundle, requested_skus=requested,
    )

    # The unknown SKU is reported as uncomputable, not scored on a fabricated retail.
    assert results.attrs["uncomputable_skus"] == ["TOTALLY-UNKNOWN-1"]
    # The computable requested sneakers still score to real numbers.
    assert len(results) > 0
    assert np.isfinite(results["predicted_premium"].to_numpy()).all()


def test_score_current_narrows_to_requested_subset(tmp_path, monkeypatch) -> None:
    """Requesting a single SKU scores only that shoe. Guards the narrowing filter
    against being a no-op: it must DROP the landed shoes that weren't requested."""
    pytest.importorskip("torch")
    monkeypatch.delenv("KICKSDB_API_KEY", raising=False)

    from stages.config import PipelineConfig, load_feature_config

    bundle = _untrained_bundle(tmp_path)
    config = PipelineConfig(
        storage_root=str(tmp_path), raw_prefix="raw", features_prefix="features",
        validated_prefix="validated", models_prefix="models",
        failures_prefix="failures", run_date="2026-07-04",
    )
    fcfg = load_feature_config("config/feature_config.yaml")

    # The full snapshot scores several shoes; pick one that actually lands so the
    # test doesn't depend on which fixture SKUs are coverable.
    full = live_score.score_current(config, fcfg, bundle=bundle)
    assert len(full) > 1
    target_id = int(full["shoe_id"].iloc[0])
    reference = kicksdb.load_retail_reference()
    target_sku = next(
        s for s in reference.skus()
        if kicksdb.stable_shoe_id(kicksdb.normalize_sku(s)) == target_id
    )

    narrowed = live_score.score_current(
        config, fcfg, bundle=bundle, requested_skus=[target_sku],
    )
    assert len(narrowed) == 1
    assert int(narrowed["shoe_id"].iloc[0]) == target_id
