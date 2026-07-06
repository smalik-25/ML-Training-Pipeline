"""Stage 1 -- Ingest: land the four source tables into the raw Parquet zone.

Three source modes, one output contract. Any of them writes ``sales.parquet``,
``shoes.parquet``, ``drops.parquet`` and ``search_interest.parquet`` under
``raw/{run_date}/`` and logs row counts, schema, and the URIs written (which
Airflow pushes to XCom).

  * Real Postgres export -- if ``SNEAKER_INTEL_DSN`` is set, read from the live
    sneaker-intel Postgres warehouse.
  * KicksDB export -- with ``source="kicksdb"``, pull the live kicks.dev market
    API (StockX + GOAT), gated behind ``KICKSDB_API_KEY``. No key falls back to
    canned KicksDB responses. The reshape lives in ``stages/kicksdb.py``.
  * Fixture mode (default for dev/CI) -- otherwise land the synthetic fixtures
    from ``data/fixtures/``. The pipeline runs end-to-end with no database and no
    API; both real sources are one env var away.

Anti-corruption layer. The real sneaker-intel star schema does not match our
canonical raw schema 1:1 -- it uses ``shoe_key``/``sold_price``/``sold_date``/
``size``/``interest``/``point_date``, and the per-release ``retail_price`` /
``release_date`` / ``release_type`` live in ``dim_drops``, not ``dim_shoes``.
The SQL in ``TABLE_QUERIES`` is the mapping seam: it aliases columns, casts
numerics to float, and rolls each shoe's canonical (earliest) drop up onto the
shoe row, so every downstream stage stays oblivious to the source schema. This
is the one place that knows what the warehouse actually looks like.
``stages/kicksdb.py`` is the same seam for the KicksDB API -- a second source
with a completely different shape, absorbed into the identical canonical layout.

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

# The canonical logical tables every downstream stage reads, in landing order.
LOGICAL_TABLES = ["sales", "shoes", "drops", "search_interest"]

# Each shoe's canonical drop = its earliest release (DISTINCT ON ... ORDER BY
# release_date). A shoe can have several drops; the earliest defines the retail
# baseline used for premium math. Shoes with no drop are excluded (their sales
# can't get a retail_price), which the validate stage's row-retention check will
# surface if it ever loses meaningful volume.
_CANONICAL_DROP_CTE = """
WITH canonical_drop AS (
    SELECT DISTINCT ON (shoe_key)
        shoe_key, release_date, retail_price, release_type
    FROM dim_drops
    ORDER BY shoe_key, release_date ASC
)
"""

# SQL mapping the real star schema -> our canonical raw schema. Column aliases
# and numeric casts (Postgres numeric -> Decimal otherwise) live here so the
# Parquet we land is byte-compatible with the synthetic fixtures.
TABLE_QUERIES: dict[str, str] = {
    "sales": """
        SELECT
            sale_key                AS sale_id,
            shoe_key                AS shoe_id,
            source                  AS platform,
            sold_price::float8      AS sale_price,
            sold_date               AS sale_date,
            size::float8            AS size_us,
            condition,
            source_item_id
        FROM fact_sales
    """,
    "shoes": _CANONICAL_DROP_CTE + """
        SELECT
            s.shoe_key              AS shoe_id,
            s.brand,
            s.model_name            AS model,
            s.colorway,
            cd.retail_price::float8 AS retail_price,
            cd.release_date         AS release_date,
            cd.release_type         AS release_type,
            s.silhouette
        FROM dim_shoes s
        JOIN canonical_drop cd ON cd.shoe_key = s.shoe_key
    """,
    "drops": """
        SELECT
            drop_key                AS drop_id,
            shoe_key                AS shoe_id,
            release_date            AS drop_date,
            release_type            AS drop_type,
            NULL::timestamp         AS announced_at
        FROM dim_drops
    """,
    "search_interest": """
        SELECT
            shoe_key                AS shoe_id,
            point_date              AS signal_date,
            geo                     AS platform,
            interest::int           AS search_index
        FROM fact_search_interest
    """,
}


def _normalize_dsn(dsn: str) -> str:
    """Force the psycopg3 SQLAlchemy dialect.

    The ``[ingest]`` extra installs psycopg (v3), but a bare ``postgresql://``
    URL makes SQLAlchemy default to psycopg2. Rewrite the scheme so the DSN from
    sneaker-intel's ``.env`` works unchanged.
    """
    for already_qualified in ("postgresql+psycopg://", "postgresql+psycopg2://"):
        if dsn.startswith(already_qualified):
            return dsn
    for prefix in ("postgresql://", "postgres://"):
        if dsn.startswith(prefix):
            return "postgresql+psycopg://" + dsn[len(prefix):]
    return dsn


def _read_all_from_postgres(dsn: str) -> dict[str, pd.DataFrame]:
    """Run every canonical query against Postgres. Returns {table: DataFrame}.

    SQLAlchemy + the driver are only needed in this mode, so they are imported
    lazily -- fixture-mode runs and CI never have to install them.
    """
    try:
        from sqlalchemy import create_engine, text
    except ImportError as exc:  # fail loudly with an actionable message
        raise RuntimeError(
            "Postgres ingest requires the 'ingest' extra: "
            'pip install -e ".[ingest]"'
        ) from exc

    engine = create_engine(_normalize_dsn(dsn))
    try:
        return {
            table: pd.read_sql(text(query), engine)
            for table, query in TABLE_QUERIES.items()
        }
    finally:
        engine.dispose()


def _read_from_fixtures(fixtures_dir: str, table: str) -> pd.DataFrame:
    return read_parquet(path_join(fixtures_dir, f"{table}.parquet"))


def _read_all_from_kicksdb(
    run_date: str,
    api_key: str | None,
    reference_path: str | None,
    fixtures_dir: str | None,
) -> dict[str, pd.DataFrame]:
    """Reshape KicksDB (live if a key is present, else canned) into the frames.

    The adapter lives in its own module; imported lazily so fixture/Postgres runs
    and CI never depend on it. The live client's ``requests`` dependency is only
    imported inside the client, so this mode still works from fixtures with the
    core install alone.
    """
    from stages import kicksdb

    kwargs: dict[str, str] = {}
    if reference_path:
        kwargs["reference_path"] = reference_path
    if fixtures_dir:
        kwargs["fixtures_dir"] = fixtures_dir
    return kicksdb.read_all(run_date, api_key=api_key, **kwargs)


def run(
    config: PipelineConfig,
    dsn: str | None = None,
    fixtures_dir: str = "data/fixtures",
    source: str = "auto",
    kicksdb_reference_path: str | None = None,
    kicksdb_fixtures_dir: str | None = None,
) -> dict[str, str]:
    """Land all four raw tables. Returns ``{table: written_uri}`` for XCom.

    Args:
        config: resolved pipeline config (provides raw_uri + run_date).
        dsn: Postgres connection string; if falsy (and source is "auto"),
            fixture mode is used.
        fixtures_dir: where fixture Parquet lives when in fixture mode.
        source: "auto" (Postgres if ``dsn`` else fixtures) or "kicksdb" (the
            kicks.dev adapter, live if ``KICKSDB_API_KEY`` is set else canned
            KicksDB fixtures). "auto" preserves the original two-mode behavior.
        kicksdb_reference_path: override for the SKU->retail reference JSON.
        kicksdb_fixtures_dir: override for the canned KicksDB responses dir.
    """
    if source == "kicksdb":
        api_key = os.environ.get("KICKSDB_API_KEY") or None
        source_label = "kicksdb-live" if api_key else "kicksdb-fixtures"
        frames = _read_all_from_kicksdb(
            config.run_date, api_key, kicksdb_reference_path, kicksdb_fixtures_dir
        )
    elif dsn:
        source_label = "postgres"
        frames = _read_all_from_postgres(dsn)
    else:
        source_label = "fixtures"
        frames = {t: _read_from_fixtures(fixtures_dir, t) for t in LOGICAL_TABLES}
    logger.info("ingest source=%s run_date=%s", source_label, config.run_date)

    written: dict[str, str] = {}
    for table in LOGICAL_TABLES:
        df = frames[table]
        if df.empty:
            raise ValueError(
                f"Source table '{table}' produced 0 rows -- refusing to land an "
                f"empty raw partition (source={source_label})."
            )

        uri = config.raw_uri(table)
        write_parquet(df, uri)
        written[table] = uri
        schema = ", ".join(f"{c}:{t}" for c, t in df.dtypes.astype(str).items())
        logger.info("landed %-16s rows=%-6d -> %s", table, len(df), uri)
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
    parser.add_argument(
        "--source",
        choices=["auto", "kicksdb"],
        default="auto",
        help=(
            "auto: Postgres if SNEAKER_INTEL_DSN is set, else synthetic fixtures. "
            "kicksdb: the kicks.dev adapter (live if KICKSDB_API_KEY is set, else "
            "canned KicksDB fixtures)."
        ),
    )
    parser.add_argument(
        "--kicksdb-fixtures-dir",
        default=None,
        help="Override dir for canned KicksDB responses (source=kicksdb).",
    )
    parser.add_argument(
        "--kicksdb-reference-path",
        default=None,
        help="Override the SKU->retail reference JSON (source=kicksdb).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    config = load_config(args.config, run_date=args.run_date)
    dsn = os.environ.get("SNEAKER_INTEL_DSN") or None
    written = run(
        config,
        dsn=dsn,
        fixtures_dir=args.fixtures_dir,
        source=args.source,
        kicksdb_reference_path=args.kicksdb_reference_path,
        kicksdb_fixtures_dir=args.kicksdb_fixtures_dir,
    )
    logger.info("ingest complete: %d tables landed under run_date=%s",
                len(written), config.run_date)


if __name__ == "__main__":
    main()
