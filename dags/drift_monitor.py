"""Airflow DAG -- scheduled drift monitoring with retrain-on-drift.

    ingest -> feature_engineering -> validate -> detect_drift -> trigger_training

This runs on a schedule, lands the current KicksDB market snapshot, validates it,
and checks whether its feature distributions have drifted from what the deployed
(staging) model was trained on. Only when drift is detected does it trigger the
training pipeline. That's the monitor-and-retrain half of the loop: we don't
retrain on a blind cron, we retrain when the data says the model is going stale.

The ``current`` distribution is live KicksDB (the model was trained on 2017-2019
StockX data, so this is the honest "has the market moved since 2019" measurement).
The ``reference`` stays the staging model's training distribution, resolved from
the registry -- drift always means "moved from what the live model learned". A
current-only snapshot can't compute every feature, so ``detect_drift`` PSI's only
the snapshot-computable subset (``kicksdb.SNAPSHOT_DRIFT_FEATURES``) and records
the rest as excluded rather than measuring the grain.

It reuses the training DAG's ``feature_engineering``/``validate`` callables and
runs its own KicksDB ingest. The drift check is a `ShortCircuitOperator`: no drift
short-circuits and the trigger is skipped; drift fires `TriggerDagRunOperator`
against ``sneaker_training_pipeline``.
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
    _validate,
)

log = logging.getLogger("dags.drift_monitor")


def _ingest_kicksdb(**context) -> dict[str, str]:
    """Land the current KicksDB market snapshot as this run's 'current' data.

    Live when KICKSDB_API_KEY is set, else canned KicksDB fixtures. Budget: the
    adapter pulls two requests per reference SKU, so a daily schedule over ~13
    SKUs is ~26 requests/run (~780/month), well under the 50,000/month Starter
    cap. This is the KicksDB counterpart to the training DAG's Postgres ingest.
    """
    from stages import ingest

    config = _build_config(context["ds"])
    return ingest.run(config, source="kicksdb")  # {table: uri} -> XCom


def _detect_drift(**context) -> bool:
    """Run the drift stage and short-circuit unless the data has drifted.

    Only the features a current KicksDB snapshot can compute honestly are PSI'd;
    the rest are recorded as excluded (see kicksdb.SNAPSHOT_DRIFT_FEATURES).
    """
    from stages import kicksdb, monitor

    config = _build_config(context["ds"])
    # reference = the staging model's training run; current = the KicksDB snapshot.
    _, drifted = monitor.run(
        config,
        feature_columns=kicksdb.SNAPSHOT_DRIFT_FEATURES,
        excluded_reasons=kicksdb.SNAPSHOT_EXCLUDED_FEATURES,
    )
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
    ingest_task = PythonOperator(task_id="ingest", python_callable=_ingest_kicksdb)
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
