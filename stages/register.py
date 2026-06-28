"""Stage 5 -- Model registry (MLflow).

Finds the best run for the run_date, registers it as "sneaker-price-model",
and promotes to Staging only if it beats the current Staging model's val RMSE.
Logs a promotion_report.json. A worse model does not promote and exits with a
warning, not a failure.

CLI:
    python -m stages.register --run-date 2025-01-01 --config config/pipeline_config.yaml

Implemented in Phase 5.
"""

from __future__ import annotations

import argparse
import logging

from stages.config import PipelineConfig, load_config

logger = logging.getLogger("stages.register")


def run(config: PipelineConfig) -> str:
    """Register/promote the best model. Returns the promotion report URI."""
    raise NotImplementedError("Registration stage is implemented in Phase 5.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Register/promote model in MLflow.")
    parser.add_argument("--run-date", required=True, help="ISO run date (YYYY-MM-DD).")
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    config = load_config(args.config, run_date=args.run_date)
    uri = run(config)
    logger.info("promotion report written -> %s", uri)


if __name__ == "__main__":
    main()
