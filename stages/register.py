"""Stage 5 -- Model registry & promotion (MLflow).

This is the reproducibility story: every trained model becomes a versioned
registry entry, and every promotion decision is logged, so the registry always
has a clean lineage from run -> model version -> alias.

For the given run_date the stage:
  1. finds the best run in the experiment (lowest final_val_rmse);
  2. registers it under "sneaker-price-model";
  3. promotes it to the ``staging`` alias *only if* it beats the current
     staging model's val RMSE;
  4. writes a ``promotion_report.json`` recording the decision.

A worse model does not promote and exits with a warning, not a failure -- bad
models must never silently overwrite good ones, but a routine "this run wasn't
an improvement" is not a pipeline error.

Note on aliases vs stages: the original plan said "transition to Staging".
MLflow 3.x deprecated stage transitions (`Staging`/`Production`) in favour of
*aliases*. We use a ``staging`` alias, which is the current-API equivalent of a
staged pointer and the supported way to mark a promoted version.

CLI:
    python -m stages.register --run-date 2025-01-01 \\
        --config config/pipeline_config.yaml
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

from stages.config import PipelineConfig, load_config
from stages.io import path_join, write_json
from stages.train import EXPERIMENT_NAME, resolve_tracking_uri

logger = logging.getLogger("stages.register")

STAGING_ALIAS = "staging"


def run(config: PipelineConfig, model_name: str = EXPERIMENT_NAME) -> str:
    """Register and conditionally promote the best model. Returns report URI."""
    import mlflow
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    tracking_uri = resolve_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment '{EXPERIMENT_NAME}' not found -- run training first."
        )

    # Best run for this run_date = lowest validation RMSE.
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"params.run_date = '{config.run_date}'",
        order_by=["metrics.final_val_rmse ASC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(
            f"no training runs for run_date={config.run_date} in "
            f"'{EXPERIMENT_NAME}' -- nothing to register."
        )
    best = runs[0]
    best_run_id = best.info.run_id
    if "final_val_rmse" not in best.data.metrics:
        raise RuntimeError(
            f"best run {best_run_id} has no final_val_rmse metric to compare on."
        )
    best_rmse = float(best.data.metrics["final_val_rmse"])

    # Read the incumbent staging model's metric (if any) BEFORE registering.
    current_rmse: float | None = None
    try:
        current = client.get_model_version_by_alias(model_name, STAGING_ALIAS)
        current_rmse = float(
            client.get_run(current.run_id).data.metrics["final_val_rmse"]
        )
    except MlflowException:
        current_rmse = None  # no registered model / no staging alias yet

    # Always register the new candidate; promotion is a separate decision.
    # Register via the low-level client against the run's model artifact dir.
    # MLflow 3.x's high-level register_model requires a logged MLmodel flavor;
    # we register the artifact directory directly so the registry lineage stays
    # decoupled from any specific model flavor (our artifact is a plain
    # state_dict + scaler model.pt).
    try:
        client.create_registered_model(model_name)
    except MlflowException:
        pass  # registered model already exists
    model_version = client.create_model_version(
        name=model_name,
        source=f"{best.info.artifact_uri}/model",
        run_id=best_run_id,
    )

    promoted = current_rmse is None or best_rmse < current_rmse
    if promoted:
        client.set_registered_model_alias(
            model_name, STAGING_ALIAS, model_version.version
        )
        logger.info(
            "promoted %s v%s (val_rmse=%.5f) to @%s",
            model_name, model_version.version, best_rmse, STAGING_ALIAS,
        )
    else:
        logger.warning(
            "NOT promoting %s v%s: val_rmse=%.5f does not beat current @%s "
            "(%.5f). Registered but left unpromoted.",
            model_name, model_version.version, best_rmse, STAGING_ALIAS, current_rmse,
        )

    report = {
        "run_date": config.run_date,
        "registered_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "model_name": model_name,
        "model_version": int(model_version.version),
        "run_id": best_run_id,
        "val_rmse": best_rmse,
        "promoted": promoted,
        "alias": STAGING_ALIAS,
        "beat_metric": current_rmse if current_rmse is not None else "no existing model",
    }
    report_uri = path_join(config.model_uri(best_run_id), "promotion_report.json")
    write_json(report, report_uri)
    logger.info("promotion report -> %s", report_uri)
    return report_uri


def main() -> None:
    parser = argparse.ArgumentParser(description="Register/promote model in MLflow.")
    parser.add_argument("--run-date", required=True, help="ISO run date (YYYY-MM-DD).")
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    config = load_config(args.config, run_date=args.run_date)
    uri = run(config)
    logger.info("registration stage complete -> %s", uri)


if __name__ == "__main__":
    main()
