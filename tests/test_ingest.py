"""Tests for the ingest stage in fixture mode (no Postgres, no S3).

Generates fixtures into a temp dir, lands them into a temp raw zone via the real
ingest code path, and checks the contract every downstream stage relies on:
four tables landed, row counts preserved, schemas intact.
"""

from __future__ import annotations

import pandas as pd

from generate_fixtures import build_fixtures
from stages.config import PipelineConfig
from stages.ingest import LOGICAL_TABLES, run
from stages.io import read_parquet

EXPECTED_COLUMNS = {
    "sales": {
        "sale_id", "shoe_id", "platform", "sale_price", "sale_date",
        "size_us", "condition", "source_item_id",
    },
    "shoes": {
        "shoe_id", "brand", "model", "colorway", "retail_price",
        "release_date", "release_type", "silhouette",
    },
    "drops": {"drop_id", "shoe_id", "drop_date", "drop_type", "announced_at"},
    "search_interest": {"shoe_id", "signal_date", "platform", "search_index"},
}


def _config(root: str) -> PipelineConfig:
    return PipelineConfig(
        storage_root=root,
        raw_prefix="raw",
        features_prefix="features",
        validated_prefix="validated",
        models_prefix="models",
        failures_prefix="failures",
        run_date="2025-01-01",
    )


def test_ingest_lands_all_tables_from_fixtures(tmp_path) -> None:
    fixtures_dir = str(tmp_path / "fixtures")
    built = build_fixtures(fixtures_dir)

    config = _config(str(tmp_path / "lake"))
    written = run(config, dsn=None, fixtures_dir=fixtures_dir)

    assert set(written) == set(LOGICAL_TABLES)

    for table, uri in written.items():
        landed = read_parquet(uri)
        source = read_parquet(built[table])
        # Row counts preserved through the landing.
        assert len(landed) == len(source), f"{table} row count drifted"
        # Schema intact.
        assert set(landed.columns) == EXPECTED_COLUMNS[table], f"{table} schema"


def test_ingest_path_layout_partitions_by_run_date(tmp_path) -> None:
    fixtures_dir = str(tmp_path / "fixtures")
    build_fixtures(fixtures_dir)
    config = _config(str(tmp_path / "lake"))
    written = run(config, dsn=None, fixtures_dir=fixtures_dir)
    # Every raw file lives under raw/{run_date}/.
    for uri in written.values():
        assert "/raw/2025-01-01/" in uri


def test_fixtures_are_plausible(tmp_path) -> None:
    """Guard the properties downstream statistical checks depend on."""
    fixtures_dir = str(tmp_path / "fixtures")
    build_fixtures(fixtures_dir)
    sales = read_parquet(str(tmp_path / "fixtures" / "sales.parquet"))
    shoes = read_parquet(str(tmp_path / "fixtures" / "shoes.parquet"))

    # No duplicate (shoe_id, sale_date, size_us) -- the Phase 3 uniqueness check.
    assert not sales.duplicated(subset=["shoe_id", "sale_date", "size_us"]).any()

    # Sales straddle the 2023 temporal split boundary (Phase 4).
    years = pd.to_datetime(sales["sale_date"]).dt.year
    assert (years < 2023).any() and (years >= 2023).any()

    # Every sale happens on or after its shoe's release (days_since_release >= 0).
    merged = sales.merge(shoes[["shoe_id", "release_date"]], on="shoe_id")
    assert (
        pd.to_datetime(merged["sale_date"]) >= pd.to_datetime(merged["release_date"])
    ).all()
