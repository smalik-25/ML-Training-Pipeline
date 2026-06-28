"""Stage 2 -- Feature engineering (PySpark).

Reads raw Parquet, computes premium economics and temporal/demand features,
writes a flat features Parquet partitioned by brand. Extends the sneaker-intel
dbt int_sales_enriched logic on the Spark compute layer.

CLI:
    python -m stages.features --run-date 2025-01-01 --config config/pipeline_config.yaml

Implemented in Phase 2.
"""

from __future__ import annotations

import argparse
import logging

from stages.config import PipelineConfig, load_config

logger = logging.getLogger("stages.features")


def run(config: PipelineConfig) -> str:
    """Compute features and write them. Returns the features URI."""
    raise NotImplementedError("Feature engineering stage is implemented in Phase 2.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute features with PySpark.")
    parser.add_argument("--run-date", required=True, help="ISO run date (YYYY-MM-DD).")
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    config = load_config(args.config, run_date=args.run_date)
    uri = run(config)
    logger.info("features written -> %s", uri)


if __name__ == "__main__":
    main()
