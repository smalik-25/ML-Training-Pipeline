"""Stage 4 -- Training (Ray Train + PyTorch).

Reads validated Parquet, does a temporal train/val split, trains
SneakerPriceNet via ray.train.torch.TorchTrainer, logs params/metrics to MLflow,
and saves the model artifact to MLflow and S3.

CLI:
    python -m stages.train --run-date 2025-01-01 \
        --config config/pipeline_config.yaml --num-workers 2

Implemented in Phase 4.
"""

from __future__ import annotations

import argparse
import logging

from stages.config import PipelineConfig, load_config

logger = logging.getLogger("stages.train")


def run(config: PipelineConfig, num_workers: int) -> str:
    """Train the model. Returns the MLflow run_id."""
    raise NotImplementedError("Training stage is implemented in Phase 4.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train with Ray Train + PyTorch.")
    parser.add_argument("--run-date", required=True, help="ISO run date (YYYY-MM-DD).")
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    parser.add_argument("--num-workers", type=int, default=1, help="Ray workers.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    config = load_config(args.config, run_date=args.run_date)
    run_id = run(config, num_workers=args.num_workers)
    logger.info("training complete, mlflow run_id=%s", run_id)


if __name__ == "__main__":
    main()
