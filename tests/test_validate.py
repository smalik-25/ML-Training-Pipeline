"""Tests for the Pandera validation stage.

Deliberately Spark-free: validation operates on a features DataFrame, so we
hand-build small clean and broken frames. This keeps the tests fast and runnable
in the light CI job (no Spark needed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stages.config import FeatureConfig, PipelineConfig
from stages.io import read_json, read_parquet, write_parquet
from stages.validate import FeatureValidationError, run, validate_features

NAN = float("nan")


def _fcfg() -> FeatureConfig:
    return FeatureConfig(
        rolling_window_days=7,
        search_signal_window_days=7,
        release_type_order={"limited": 2, "collab": 1, "general": 0},
        partition_by="brand",
        price_premium_min=-1.0,
        price_premium_max=50.0,
    )


def _clean_features() -> pd.DataFrame:
    """A small, fully-valid features frame.

    rolling_7d_avg_premium is null exactly on the rows whose history (days since
    the shoe's first sale) is < 7, matching the contract the validator enforces.
    """
    rows = [
        # shoe 1 (Jordan, limited, release 2021-12-01)
        dict(shoe_id=1, brand="Jordan", release_type="limited", rte=2,
             retail=180.0, sale_date="2022-01-01", roll=NAN),   # hist 0
        dict(shoe_id=1, brand="Jordan", release_type="limited", rte=2,
             retail=180.0, sale_date="2022-01-04", roll=NAN),   # hist 3
        dict(shoe_id=1, brand="Jordan", release_type="limited", rte=2,
             retail=180.0, sale_date="2022-01-12", roll=1.30),  # hist 11
        # shoe 2 (Nike, general, release 2022-02-01)
        dict(shoe_id=2, brand="Nike", release_type="general", rte=0,
             retail=100.0, sale_date="2022-03-01", roll=NAN),   # hist 0
        dict(shoe_id=2, brand="Nike", release_type="general", rte=0,
             retail=100.0, sale_date="2022-03-15", roll=0.10),  # hist 14
    ]
    release = {1: "2021-12-01", 2: "2022-02-01"}
    premium = {1: 1.30, 2: 0.10}
    sizes = [9.0, 9.5, 10.0, 9.0, 10.5]

    recs = []
    for i, r in enumerate(rows):
        sale_date = pd.Timestamp(r["sale_date"])
        rel = pd.Timestamp(release[r["shoe_id"]])
        prem = premium[r["shoe_id"]]
        recs.append(
            {
                "sale_id": i + 1,
                "shoe_id": r["shoe_id"],
                "size_us": sizes[i],
                "sale_date": sale_date,
                "release_date": rel,
                "retail_price": r["retail"],
                "sale_price": round(r["retail"] * (1 + prem), 2),
                "release_type": r["release_type"],
                "price_premium": prem,
                "days_since_release": int((sale_date - rel).days),
                "size_premium": 0.0,
                "release_type_encoded": r["rte"],
                "rolling_7d_avg_premium": r["roll"],
                "search_index_7d_pre_drop": 80.0,
                "brand_avg_premium": prem,
                "brand": r["brand"],
            }
        )
    return pd.DataFrame(recs)


def test_clean_features_pass() -> None:
    validated, checks = validate_features(_clean_features(), _fcfg())
    assert len(validated) == 5
    assert checks  # named checks recorded for the report


def test_negative_days_since_release_fails() -> None:
    bad = _clean_features()
    bad.loc[0, "days_since_release"] = -5
    with pytest.raises(FeatureValidationError) as exc:
        validate_features(bad, _fcfg())
    assert "days_since_release" in str(exc.value)


def test_price_premium_out_of_range_fails() -> None:
    bad = _clean_features()
    bad.loc[2, "price_premium"] = 99.0  # above the 20.0 ceiling
    with pytest.raises(FeatureValidationError) as exc:
        validate_features(bad, _fcfg())
    assert "price_premium" in str(exc.value)


def test_rolling_null_inconsistency_fails() -> None:
    """A non-null rolling value on an insufficient-history row must fail."""
    bad = _clean_features()
    bad.loc[0, "rolling_7d_avg_premium"] = 0.9  # row 0 has 0 days of history
    with pytest.raises(FeatureValidationError) as exc:
        validate_features(bad, _fcfg())
    assert "rolling_7d_avg_premium" in str(exc.value)


def test_duplicate_sale_id_fails() -> None:
    """sale_id is the real unique key; a repeated sale_id must fail."""
    clean = _clean_features()
    dup = pd.concat([clean, clean.iloc[[0]]], ignore_index=True)  # repeats sale_id 1
    with pytest.raises(FeatureValidationError):
        validate_features(dup, _fcfg())


def test_repeated_shoe_size_date_is_allowed() -> None:
    """Multiple sales of the same shoe/size/day are valid (real resale grain)."""
    clean = _clean_features()
    extra = clean.iloc[[0]].copy()
    extra["sale_id"] = 999  # distinct sale, same (shoe_id, sale_date, size_us)
    combined = pd.concat([clean, extra], ignore_index=True)
    validated, _ = validate_features(combined, _fcfg())
    assert len(validated) == 6


def test_run_writes_validated_and_report(tmp_path) -> None:
    config = PipelineConfig(
        storage_root=str(tmp_path),
        raw_prefix="raw",
        features_prefix="features",
        validated_prefix="validated",
        models_prefix="models",
        failures_prefix="failures",
        run_date="run",
    )
    clean = _clean_features()
    write_parquet(clean, config.features_uri())
    # Raw sales row count drives the retention check; equal length -> 100%.
    write_parquet(clean, config.raw_uri("sales"))

    out_uri = run(config, _fcfg())
    validated = read_parquet(out_uri)
    assert len(validated) == 5

    report = read_json(config.validation_report_uri())
    assert report["status"] == "passed"
    assert report["row_retention_pct"] == 100.0
    assert report["rolling_null_rows"] == 3


def test_run_fails_on_low_row_retention(tmp_path) -> None:
    config = PipelineConfig(
        storage_root=str(tmp_path),
        raw_prefix="raw",
        features_prefix="features",
        validated_prefix="validated",
        models_prefix="models",
        failures_prefix="failures",
        run_date="run",
    )
    clean = _clean_features()
    write_parquet(clean, config.features_uri())
    # 100 raw rows but only 5 features -> 5% retention, below the 90% floor.
    raw = pd.DataFrame({"x": np.arange(100)})
    write_parquet(raw, config.raw_uri("sales"))

    with pytest.raises(FeatureValidationError) as exc:
        run(config, _fcfg())
    assert "retention" in str(exc.value)
