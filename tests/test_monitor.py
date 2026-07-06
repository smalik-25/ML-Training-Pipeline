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


def test_drift_report_respects_feature_subset() -> None:
    rng = np.random.default_rng(4)
    subset = ["retail_price", "brand_avg_premium"]
    report = drift_report(_frame(rng), _frame(rng), feature_columns=subset)
    assert [f["feature"] for f in report["features"]] == subset


def test_run_records_excluded_features_for_kicksdb_subset(tmp_path) -> None:
    """The KicksDB drift path PSI's only snapshot-computable features and records
    the rest as excluded, instead of measuring drift on degenerate columns."""
    from stages.kicksdb import SNAPSHOT_DRIFT_FEATURES

    rng = np.random.default_rng(5)
    config = _config(str(tmp_path), "current")
    write_parquet(_frame(rng), replace(config, run_date="baseline").validated_uri())
    write_parquet(_frame(rng), config.validated_uri())

    report_uri, _ = run(
        config, reference_run_date="baseline",
        feature_columns=SNAPSHOT_DRIFT_FEATURES,
    )
    report = read_json(report_uri)
    reported = {f["feature"] for f in report["features"]}
    assert reported == set(SNAPSHOT_DRIFT_FEATURES)
    assert set(report["excluded_features"]) == {
        "size_us", "size_premium", "rolling_7d_avg_premium",
        "search_index_7d_pre_drop",
    }


def test_run_records_exclusion_reasons_for_kicksdb_subset(tmp_path) -> None:
    """The KicksDB drift path records not just which features are excluded but
    WHY, so the report explains the coverage gap instead of only counting it."""
    from stages.kicksdb import SNAPSHOT_DRIFT_FEATURES, SNAPSHOT_EXCLUDED_FEATURES

    rng = np.random.default_rng(6)
    config = _config(str(tmp_path), "current")
    write_parquet(_frame(rng), replace(config, run_date="baseline").validated_uri())
    write_parquet(_frame(rng), config.validated_uri())

    report_uri, _ = run(
        config, reference_run_date="baseline",
        feature_columns=SNAPSHOT_DRIFT_FEATURES,
        excluded_reasons=SNAPSHOT_EXCLUDED_FEATURES,
    )
    excluded = read_json(report_uri)["excluded_features"]
    assert isinstance(excluded, dict)
    assert set(excluded) == set(SNAPSHOT_EXCLUDED_FEATURES)
    # the "why" is carried through to the report, not just the names
    assert excluded["rolling_7d_avg_premium"] == (
        SNAPSHOT_EXCLUDED_FEATURES["rolling_7d_avg_premium"]
    )
    assert all(reason for reason in excluded.values())


def test_kicksdb_subset_trips_on_shift_and_not_on_match(tmp_path) -> None:
    """A shift in a snapshot-computable feature trips the KicksDB-subset drift
    check; matching current data does not. This is the Phase 2 trip/no-trip
    guarantee on the exact subset the drift DAG runs."""
    from stages.kicksdb import SNAPSHOT_DRIFT_FEATURES

    rng = np.random.default_rng(7)
    match = _config(str(tmp_path / "match"), "current")
    write_parquet(_frame(rng), replace(match, run_date="baseline").validated_uri())
    write_parquet(_frame(rng), match.validated_uri())
    _, drifted = run(
        match, reference_run_date="baseline",
        feature_columns=SNAPSHOT_DRIFT_FEATURES,
    )
    assert drifted is False

    shift = _config(str(tmp_path / "shift"), "current")
    current = _frame(rng)
    current["retail_price"] = current["retail_price"] + 6  # a subset feature
    write_parquet(_frame(rng), replace(shift, run_date="baseline").validated_uri())
    write_parquet(current, shift.validated_uri())
    _, drifted = run(
        shift, reference_run_date="baseline",
        feature_columns=SNAPSHOT_DRIFT_FEATURES,
    )
    assert drifted is True


def test_kicksdb_subset_ignores_shift_in_excluded_feature(tmp_path) -> None:
    """A hard shift in an EXCLUDED feature must not trip the subset drift check,
    proving the excluded features are genuinely left out of the measurement and
    not silently folded back in."""
    from stages.kicksdb import SNAPSHOT_DRIFT_FEATURES

    rng = np.random.default_rng(8)
    config = _config(str(tmp_path), "current")
    current = _frame(rng)
    current["size_premium"] = current["size_premium"] + 8  # an excluded feature
    write_parquet(_frame(rng), replace(config, run_date="baseline").validated_uri())
    write_parquet(current, config.validated_uri())
    _, drifted = run(
        config, reference_run_date="baseline",
        feature_columns=SNAPSHOT_DRIFT_FEATURES,
    )
    assert drifted is False
