"""Stage 1 -- Ingest: export sneaker-intel Postgres tables to raw Parquet.

Reachability note: the live sneaker-intel Postgres does NOT need to be
reachable for the pipeline to run end-to-end. If ``SNEAKER_INTEL_DSN`` is set,
this stage does a real export. Otherwise the working dataset is the synthetic
fixtures produced by ``generate_fixtures.py`` (see data/fixtures/), which every
downstream stage and all of CI consume. The 99K real StockX rows are the
production story; fixtures are the default development path.

CLI (independently runnable, no Airflow required):
    python -m stages.ingest --run-date 2025-01-01 --config config/pipeline_config.yaml

Implemented in Phase 1.
"""

from __future__ import annotations

import argparse
import logging

from stages.config import PipelineConfig, load_config

logger = logging.getLogger("stages.ingest")


def run(config: PipelineConfig) -> dict[str, str]:
    """Export raw tables to Parquet under the raw prefix.

    Returns a mapping of table name -> written URI (pushed to XCom by Airflow).
    """
    raise NotImplementedError("Ingest stage is implemented in Phase 1.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest raw tables to Parquet.")
    parser.add_argument("--run-date", required=True, help="ISO run date (YYYY-MM-DD).")
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    config = load_config(args.config, run_date=args.run_date)
    written = run(config)
    for table, uri in written.items():
        logger.info("ingested %s -> %s", table, uri)


if __name__ == "__main__":
    main()
