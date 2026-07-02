"""Airflow DAG -- sneaker training pipeline orchestration.

    ingest -> feature_engineering -> validate -> train -> register

The DAG is the source of truth for pipeline *topology*; it does not contain
business logic. Each task is a thin wrapper that builds the run's
``PipelineConfig`` and calls the corresponding stage's ``run()`` -- all the real
work lives in ``stages/``.

Design notes:

  * Lazy stage imports. The heavy stage modules (PySpark, Torch, Ray, MLflow)
    are imported *inside* the task callables, not at module top level. DAG files
    are re-parsed constantly by the scheduler, so keeping top-level imports light
    makes parsing fast and lets the DAG load (and its structure be tested) in an
    environment that doesn't have Spark/Torch/Ray installed.
  * XCom for lineage. Config is the single source of truth for paths, but each
    task still pushes the concrete artifact URI(s) it produced to XCom, and the
    next task pulls and logs the upstream artifact it's consuming. This threads a
    visible run -> artifact lineage through the Airflow UI without hardcoding any
    path in the DAG.
  * run_date = the run's logical date (``ds``), so each DAG run partitions its
    data under its own date prefix.
  * A validation failure raises out of the validate task, which fails the task
    and therefore the whole DAG run -- bad data never reaches training.
  * Every task carries an on_failure_callback that writes a failure summary to
    the ``failures/`` prefix on S3.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

# Make the repo root importable for the stage packages, both in the Airflow
# container and when tests import this DAG.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

log = logging.getLogger("dags.sneaker_training_pipeline")

CONFIG_PATH = os.environ.get(
    "PIPELINE_CONFIG", str(_REPO_ROOT / "config" / "pipeline_config.yaml")
)
FEATURE_CONFIG_PATH = os.environ.get(
    "FEATURE_CONFIG", str(_REPO_ROOT / "config" / "feature_config.yaml")
)

# Human-readable description of what each task is supposed to produce -- recorded
# in the failure report so an on-call engineer knows what's missing.
TASK_EXPECTED_OUTPUT = {
    "ingest": "raw/{run_date}/{sales,shoes,drops,search_interest}.parquet",
    "feature_engineering": "features/{run_date}/features.parquet (partitioned by brand)",
    "validate": "validated/{run_date}/features_validated.parquet + validation_report.json",
    "train": "models/{run_date}/{run_id}/model.pt + tracked MLflow run",
    "register": "models/{run_date}/{run_id}/promotion_report.json + staging alias",
    "score": "predictions/{run_date}/predictions.parquet + scoring_report.json",
}


def _build_config(run_date: str):
    from stages.config import load_config

    return load_config(CONFIG_PATH, run_date=run_date)


def _on_failure(context) -> None:
    """Write a failure summary to the failures/ prefix. Never raises."""
    from stages.io import write_json

    task_instance = context["ti"]
    task_id = task_instance.task_id
    run_date = context["ds"]
    exc = context.get("exception")
    report = {
        "failed_task": task_id,
        "run_date": run_date,
        "expected_output": TASK_EXPECTED_OUTPUT.get(task_id, "unknown"),
        "exception_type": type(exc).__name__ if exc else None,
        "exception": str(exc) if exc else None,
        "try_number": task_instance.try_number,
        "failed_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
    }
    try:
        config = _build_config(run_date)
        uri = write_json(report, config.failure_uri(task_id))
        log.error("failure report for task '%s' written -> %s", task_id, uri)
    except Exception:  # a failing callback must not mask the original failure
        log.exception("could not write failure report for task '%s'", task_id)


# --- task callables: thin wrappers around stage run() functions ---------------

def _ingest(**context) -> dict[str, str]:
    from stages import ingest

    config = _build_config(context["ds"])
    dsn = os.environ.get("SNEAKER_INTEL_DSN") or None
    return ingest.run(config, dsn=dsn)  # {table: uri} -> XCom


def _feature_engineering(**context) -> str:
    from stages import features
    from stages.config import load_feature_config

    config = _build_config(context["ds"])
    raw = context["ti"].xcom_pull(task_ids="ingest")
    log.info("feature stage consuming raw artifacts from XCom: %s", raw)
    return features.run(config, load_feature_config(FEATURE_CONFIG_PATH))


def _validate(**context) -> str:
    from stages import validate
    from stages.config import load_feature_config

    config = _build_config(context["ds"])
    features_uri = context["ti"].xcom_pull(task_ids="feature_engineering")
    log.info("validate stage consuming features artifact from XCom: %s", features_uri)
    # A FeatureValidationError here propagates, failing the task and the DAG run.
    return validate.run(config, load_feature_config(FEATURE_CONFIG_PATH))


def _train(**context) -> str:
    from stages import train

    config = _build_config(context["ds"])
    validated_uri = context["ti"].xcom_pull(task_ids="validate")
    log.info("train stage consuming validated artifact from XCom: %s", validated_uri)
    num_workers = int(os.environ.get("RAY_NUM_WORKERS", "1"))
    return train.run(config, num_workers=num_workers)  # mlflow run_id -> XCom


def _register(**context) -> str:
    from stages import register

    config = _build_config(context["ds"])
    run_id = context["ti"].xcom_pull(task_ids="train")
    log.info("register stage consuming train run_id from XCom: %s", run_id)
    return register.run(config)  # promotion_report uri -> XCom


def _score(**context) -> str:
    from stages import score

    config = _build_config(context["ds"])
    report_uri = context["ti"].xcom_pull(task_ids="register")
    log.info("score stage consuming register report from XCom: %s", report_uri)
    return score.run(config)  # predictions uri -> XCom


default_args = {
    "owner": "ml-platform",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    # Pipeline-level expectation: a full run should finish within 2 hours.
    "sla": timedelta(hours=2),
    "on_failure_callback": _on_failure,
}

with DAG(
    dag_id="sneaker_training_pipeline",
    description="Offline ML training pipeline for sneaker resale price-premium.",
    default_args=default_args,
    schedule=None,           # trigger manually; set a cron to run on a cadence
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "sneaker-intel"],
) as dag:
    ingest_task = PythonOperator(task_id="ingest", python_callable=_ingest)
    feature_task = PythonOperator(
        task_id="feature_engineering", python_callable=_feature_engineering
    )
    validate_task = PythonOperator(task_id="validate", python_callable=_validate)
    train_task = PythonOperator(task_id="train", python_callable=_train)
    register_task = PythonOperator(task_id="register", python_callable=_register)
    score_task = PythonOperator(task_id="score", python_callable=_score)

    (
        ingest_task
        >> feature_task
        >> validate_task
        >> train_task
        >> register_task
        >> score_task
    )
