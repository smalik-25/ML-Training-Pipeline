"""Stage 3 -- Dataset validation (Pandera).

Schema + statistical checks on the features Parquet. Fails loudly on breach;
on success writes validated Parquet plus a validation_report.json.

CLI:
    python -m stages.validate --run-date 2025-01-01 --config config/pipeline_config.yaml

Implemented in Phase 3.
"""

from __future__ import annotations

import argparse
import logging

from stages.config import PipelineConfig, load_config

logger = logging.getLogger("stages.validate")


def run(config: PipelineConfig) -> str:
    """Validate features; write validated Parquet + report. Returns validated URI."""
    raise NotImplementedError("Validation stage is implemented in Phase 3.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate features with Pandera.")
    parser.add_argument("--run-date", required=True, help="ISO run date (YYYY-MM-DD).")
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    config = load_config(args.config, run_date=args.run_date)
    uri = run(config)
    logger.info("validated features written -> %s", uri)


if __name__ == "__main__":
    main()
