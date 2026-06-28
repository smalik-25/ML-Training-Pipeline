"""Stage 1 -- Ingest: land the four source tables into the raw Parquet zone.

Two source modes, one output contract. Either way the stage writes
``sales.parquet``, ``shoes.parquet``, ``drops.parquet`` and
``search_interest.parquet`` under ``raw/{run_date}/`` and logs row counts,
schema, and the URIs written (which Airflow pushes to XCom).

  * Real export -- if ``SNEAKER_INTEL_DSN`` is set, read each table from the
    live sneaker-intel Postgres warehouse.
  * Fixture mode (default for dev/CI) -- if the DSN is unset, land the synthetic
    fixtures from ``data/fixtures/`` instead. The pipeline runs end-to-end with
    no database; the warehouse is one env var away.

The dbt mart tables (mart_shoe_performance, mart_price_trajectory) are
deliberately NOT exported -- the Phase 2 Spark stage re-derives and extends
those metrics from these raw tables. Same economics, different compute layer.

"Partitioned by run_date" is satisfied by the path layout: every raw file lives
under a ``raw/{run_date}/`` prefix, so re-running for a new date never clobbers
a previous landing.

CLI (independently runnable, no Airflow required):
    python -m stages.ingest --run-date 2025-01-01 --config config/pipeline_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import os

import pandas as pd

from stages.config import PipelineConfig, load_config
from stages.io import path_join, read_parquet, write_parquet

logger = logging.getLogger("stages.ingest")

# Logical table name -> sneaker-intel Postgres table name. The logical name is
# what every downstream stage and the raw layout use.
PG_TABLE_NAMES: dict[str, str] = {
    "sales": "fact_sales",
    "shoes": "dim_shoes",
    "drops": "dim_drops",
    "search_interest": "fact_search_interest",
}


def _read_from_postgres(dsn: str, pg_table: str) -> pd.DataFrame:
    """Read a full table from Postgres.

    SQLAlchemy + a driver are only needed in this mode, so they are imported
    lazily -- fixture-mode runs and CI never have to install them.
    """
    try:
        from sqlalchemy import create_engine
    except ImportError as exc:  # fail loudly with an actionable message
        raise RuntimeError(
            "Postgres ingest requires the 'ingest' extra: "
            'pip install -e ".[ingest]"'
        ) from exc

    engine = create_engine(dsn)
    try:
        return pd.read_sql(f"SELECT * FROM {pg_table}", engine)
    finally:
        engine.dispose()


def _read_from_fixtures(fixtures_dir: str, table: str) -> pd.DataFrame:
    uri = path_join(fixtures_dir, f"{table}.parquet")
    return read_parquet(uri)


def run(
    config: PipelineConfig,
    dsn: str | None = None,
    fixtures_dir: str = "data/fixtures",
) -> dict[str, str]:
    """Land all four raw tables. Returns ``{table: written_uri}`` for XCom.

    Args:
        config: resolved pipeline config (provides raw_uri + run_date).
        dsn: Postgres connection string; if falsy, fixture mode is used.
        fixtures_dir: where fixture Parquet lives when in fixture mode.
    """
    source = "postgres" if dsn else "fixtures"
    logger.info("ingest source=%s run_date=%s", source, config.run_date)

    written: dict[str, str] = {}
    for table, pg_table in PG_TABLE_NAMES.items():
        if dsn:
            df = _read_from_postgres(dsn, pg_table)
        else:
            df = _read_from_fixtures(fixtures_dir, table)

        if df.empty:
            raise ValueError(
                f"Source table '{table}' produced 0 rows -- refusing to land an "
                f"empty raw partition (source={source})."
            )

        uri = config.raw_uri(table)
        write_parquet(df, uri)
        written[table] = uri
        schema = ", ".join(f"{c}:{t}" for c, t in df.dtypes.astype(str).items())
        logger.info("landed %-16s rows=%-5d -> %s", table, len(df), uri)
        logger.info("  schema[%s]: %s", table, schema)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest raw tables to Parquet.")
    parser.add_argument("--run-date", required=True, help="ISO run date (YYYY-MM-DD).")
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    parser.add_argument(
        "--fixtures-dir",
        default="data/fixtures",
        help="Fixture directory used when SNEAKER_INTEL_DSN is unset.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    config = load_config(args.config, run_date=args.run_date)
    dsn = os.environ.get("SNEAKER_INTEL_DSN") or None
    written = run(config, dsn=dsn, fixtures_dir=args.fixtures_dir)
    logger.info("ingest complete: %d tables landed under run_date=%s",
                len(written), config.run_date)


if __name__ == "__main__":
    main()
