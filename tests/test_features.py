"""Tests for the PySpark feature engineering stage, against fixtures.

Skipped automatically where PySpark is not installed (e.g. the light CI
lint+test job), so it never breaks collection there; the full local/dev
environment and a dedicated Spark CI job run it for real.
"""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("pyspark")

from pyspark.sql import SparkSession  # noqa: E402

from generate_fixtures import build_fixtures  # noqa: E402
from stages.config import FeatureConfig, PipelineConfig  # noqa: E402
from stages.features import OUTPUT_COLUMNS, compute_features, run  # noqa: E402


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[1]")
        .appName("test-features")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture(scope="module")
def fcfg() -> FeatureConfig:
    return FeatureConfig(
        rolling_window_days=7,
        search_signal_window_days=7,
        release_type_order={"limited": 2, "collab": 1, "general": 0},
        partition_by="brand",
        price_premium_min=-1.0,
        price_premium_max=50.0,
    )


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("fixtures"))
    build_fixtures(out)
    return out


def _features(spark, raw_dir, fcfg):
    rd = lambda name: spark.read.parquet(f"{raw_dir}/{name}.parquet")  # noqa: E731
    return compute_features(
        rd("sales"), rd("shoes"), rd("drops"), rd("search_interest"), fcfg
    )


def test_price_premium_is_correct(spark, raw, fcfg) -> None:
    feats = _features(spark, raw, fcfg).toPandas()
    expected = (feats["sale_price"] - feats["retail_price"]) / feats["retail_price"]
    pd.testing.assert_series_equal(
        feats["price_premium"], expected, check_names=False, rtol=1e-9
    )


def test_rolling_null_semantics(spark, raw, fcfg) -> None:
    """rolling_7d_avg_premium is null IFF the shoe has < 7 days of history."""
    feats = _features(spark, raw, fcfg).toPandas()
    feats["sale_date"] = pd.to_datetime(feats["sale_date"])
    first_sale = feats.groupby("shoe_id")["sale_date"].transform("min")
    days_of_history = (feats["sale_date"] - first_sale).dt.days

    is_null = feats["rolling_7d_avg_premium"].isna()
    insufficient = days_of_history < fcfg.rolling_window_days
    assert (is_null == insufficient).all()
    # The fixtures guarantee signal on both sides.
    assert is_null.any() and (~is_null).any()


def test_output_schema_matches_contract(spark, raw, fcfg) -> None:
    feats = _features(spark, raw, fcfg)
    assert feats.columns == OUTPUT_COLUMNS


def test_run_writes_partitioned_by_brand(spark, raw, fcfg) -> None:
    import os
    import shutil

    # Lay the fixtures out where raw_uri() expects them, then run the stage.
    config = PipelineConfig(
        storage_root=raw,
        raw_prefix="",
        features_prefix="features",
        validated_prefix="validated",
        models_prefix="models",
        failures_prefix="failures",
        run_date="run",
    )
    raw_run = os.path.join(raw, "run")
    os.makedirs(raw_run, exist_ok=True)
    for name in ("sales", "shoes", "drops", "search_interest"):
        shutil.copy(f"{raw}/{name}.parquet", f"{raw_run}/{name}.parquet")

    out_uri = run(config, fcfg)
    out_path = out_uri.replace("file://", "")

    entries = os.listdir(out_path)
    assert any(e.startswith("brand=") for e in entries)

    back = pd.read_parquet(out_path)
    assert "brand" in back.columns
    assert len(back) > 0
