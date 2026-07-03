"""Airflow DAG -- scheduled drift monitoring with retrain-on-drift.

    ingest -> feature_engineering -> validate -> detect_drift -> trigger_training

This runs on a schedule, lands and validates the latest data, and checks whether
its feature distributions have drifted from what the deployed (staging) model was
trained on. Only when drift is detected does it trigger the training pipeline.
That's the monitor-and-retrain half of the loop: we don't retrain on a blind
cron, we retrain when the data says the model is going stale.

It reuses the training DAG's thin task callables for ingest/features/validate, so
there's no duplicated stage-wiring. The drift check is a `ShortCircuitOperator`:
if there's no drift it short-circuits and the trigger is skipped; if there is, it
fires `TriggerDagRunOperator` against ``sneaker_training_pipeline``.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import ShortCircuitOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from airflow.operators.python import PythonOperator  # noqa: E402

from dags.sneaker_training_pipeline import (  # noqa: E402
    _build_config,
    _feature_engineering,
    _ingest,
    _validate,
)

log = logging.getLogger("dags.drift_monitor")


def _detect_drift(**context) -> bool:
    """Run the drift stage and short-circuit unless the data has drifted."""
    from stages import monitor

    config = _build_config(context["ds"])
    _, drifted = monitor.run(config)  # reference = the staging model's training run
    log.info("drift=%s -> %s", drifted, "trigger retrain" if drifted else "skip")
    return drifted


default_args = {
    "owner": "ml-platform",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="sneaker_drift_monitor",
    description="Scheduled drift check; retrains only when the data has drifted.",
    default_args=default_args,
    schedule="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "sneaker-intel", "monitoring"],
) as dag:
    ingest_task = PythonOperator(task_id="ingest", python_callable=_ingest)
    feature_task = PythonOperator(
        task_id="feature_engineering", python_callable=_feature_engineering
    )
    validate_task = PythonOperator(task_id="validate", python_callable=_validate)
    detect_drift_task = ShortCircuitOperator(
        task_id="detect_drift", python_callable=_detect_drift
    )
    trigger_training = TriggerDagRunOperator(
        task_id="trigger_training",
        trigger_dag_id="sneaker_training_pipeline",
    )

    (
        ingest_task
        >> feature_task
        >> validate_task
        >> detect_drift_task
        >> trigger_training
    )
