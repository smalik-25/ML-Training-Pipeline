"""Tests for the drift-monitoring stage. Torch-free (numpy/pandas only)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from stages.config import PipelineConfig
from stages.io import read_json, write_parquet
from stages.monitor import compute_psi, drift_report, run
from stages.train import FEATURE_COLUMNS


def test_psi_low_for_same_distribution() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 2000)
    b = rng.normal(0, 1, 2000)
    assert compute_psi(a, b) < 0.1


def test_psi_high_for_shifted_distribution() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 2000)
    shifted = rng.normal(3, 1, 2000)
    assert compute_psi(a, shifted) > 0.2


def test_psi_zero_for_constant_reference() -> None:
    ref = np.ones(100)
    cur = np.random.default_rng(0).normal(0, 1, 100)
    assert compute_psi(ref, cur) == 0.0


def _frame(rng, shift: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame({c: rng.normal(shift, 1, 500) for c in FEATURE_COLUMNS})


def test_drift_report_flags_no_drift_when_similar() -> None:
    rng = np.random.default_rng(1)
    report = drift_report(_frame(rng), _frame(rng))
    assert report["drifted"] is False
    assert report["n_drifted_features"] == 0


def test_drift_report_flags_drift_when_shifted() -> None:
    rng = np.random.default_rng(1)
    reference = _frame(rng)
    current = _frame(rng)
    current["retail_price"] = current["retail_price"] + 5  # shift one feature hard
    report = drift_report(reference, current)
    assert report["drifted"] is True
    drifted_features = [f["feature"] for f in report["features"] if f["drifted"]]
    assert "retail_price" in drifted_features


def _config(root: str, run_date: str) -> PipelineConfig:
    return PipelineConfig(
        storage_root=root, raw_prefix="raw", features_prefix="features",
        validated_prefix="validated", models_prefix="models",
        failures_prefix="failures", run_date=run_date,
    )


def test_run_writes_report_and_returns_drift(tmp_path) -> None:
    rng = np.random.default_rng(2)
    config = _config(str(tmp_path), "current")

    # reference batch
    reference = _frame(rng)
    write_parquet(reference, replace(config, run_date="baseline").validated_uri())
    # current batch: shift a feature so drift is detected
    current = _frame(rng)
    current["days_since_release"] = current["days_since_release"] + 6
    write_parquet(current, config.validated_uri())

    report_uri, drifted = run(config, reference_run_date="baseline")
    assert drifted is True

    report = read_json(report_uri)
    assert report["reference_run_date"] == "baseline"
    assert report["run_date"] == "current"
    assert report["n_drifted_features"] >= 1


def test_run_reports_no_drift_for_similar_data(tmp_path) -> None:
    rng = np.random.default_rng(3)
    config = _config(str(tmp_path), "current")
    write_parquet(_frame(rng), replace(config, run_date="baseline").validated_uri())
    write_parquet(_frame(rng), config.validated_uri())

    _, drifted = run(config, reference_run_date="baseline")
    assert drifted is False
