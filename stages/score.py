"""Stage 6 -- Batch scoring.

Loads the model currently promoted to the ``staging`` alias in the MLflow
registry and scores a batch of sales with it, writing predictions to
``predictions/{run_date}/``. This is what makes the registry mean something:
training and promotion produce a *deployed* model, and this stage consumes it.

For the demo it scores the validated features for a run_date, which still carry
the true ``price_premium``, so the scoring report can report an RMSE against
ground truth. In production you'd point this at unlabeled features (new sales
with no known premium); the code path is identical.

CLI:
    python -m stages.score --run-date 2026-06-28 --config config/pipeline_config.yaml
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

from stages.config import PipelineConfig, load_config
from stages.inference import MODEL_NAME, load_staging_bundle, predict
from stages.io import read_parquet, write_json, write_parquet
from stages.train import TARGET

logger = logging.getLogger("stages.score")

# Row keys carried through to the predictions output for traceability.
KEY_COLUMNS = ["sale_id", "shoe_id", "brand", "sale_date"]


def run(config: PipelineConfig) -> str:
    """Score the run_date's validated features with the staging model.

    Returns the predictions URI.
    """
    bundle = load_staging_bundle()
    logger.info(
        "loaded staging model: name=%s version=%s run_id=%s",
        MODEL_NAME, bundle.model_version, bundle.run_id,
    )

    features = read_parquet(config.validated_uri())
    preds = predict(bundle, features)

    keys = [c for c in KEY_COLUMNS if c in features.columns]
    out = features[keys].copy()
    out["predicted_premium"] = preds

    report = {
        "run_date": config.run_date,
        "scored_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "n_scored": int(len(out)),
        "model_name": MODEL_NAME,
        "model_version": bundle.model_version,
        "model_run_id": bundle.run_id,
    }

    # If the batch carries ground truth, report accuracy alongside it.
    if TARGET in features.columns:
        actual = features[TARGET].to_numpy(dtype=float)
        out["actual_premium"] = actual
        rmse = float((((preds - actual) ** 2).mean()) ** 0.5)
        report["rmse_vs_actual"] = round(rmse, 5)
        logger.info("scored %d rows, RMSE vs actual = %.5f", len(out), rmse)
    else:
        logger.info("scored %d rows (no ground truth in batch)", len(out))

    out_uri = write_parquet(out, config.predictions_uri())
    write_json(report, config.scoring_report_uri())
    logger.info("predictions -> %s", out_uri)
    logger.info("scoring report -> %s", config.scoring_report_uri())
    return out_uri


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-score with the staging model.")
    parser.add_argument("--run-date", required=True, help="ISO run date (YYYY-MM-DD).")
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    config = load_config(args.config, run_date=args.run_date)
    uri = run(config)
    logger.info("scoring stage complete -> %s", uri)


if __name__ == "__main__":
    main()
