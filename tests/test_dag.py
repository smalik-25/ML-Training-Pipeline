"""DAG-structure tests.

These assert the *shape* of the DAG (task set, dependency edges, retry/SLA/
failure-callback config) without executing any task -- the standard Airflow unit
test. Skipped where Airflow isn't installed (the light CI lint+test job); the
dedicated DAG-integrity CI job installs Airflow and runs them.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

pytest.importorskip("airflow")

from dags.drift_monitor import dag as drift_dag  # noqa: E402
from dags.sneaker_training_pipeline import dag  # noqa: E402

EXPECTED_TASKS = {
    "ingest",
    "feature_engineering",
    "validate",
    "train",
    "register",
    "score",
}


def test_dag_loads_with_expected_tasks() -> None:
    assert dag.dag_id == "sneaker_training_pipeline"
    assert set(dag.task_ids) == EXPECTED_TASKS
    assert len(dag.tasks) == 6


def test_linear_dependency_chain() -> None:
    expected_downstream = {
        "ingest": {"feature_engineering"},
        "feature_engineering": {"validate"},
        "validate": {"train"},
        "train": {"register"},
        "register": {"score"},
        "score": set(),
    }
    for task_id, downstream in expected_downstream.items():
        assert dag.get_task(task_id).downstream_task_ids == downstream


def test_retries_and_retry_delay_configured() -> None:
    for task in dag.tasks:
        assert task.retries == 2
        assert task.retry_delay == timedelta(minutes=2)


def test_sla_and_failure_callback_configured() -> None:
    for task in dag.tasks:
        assert task.sla == timedelta(hours=2)
        # on_failure_callback may be normalised to a list by Airflow.
        assert task.on_failure_callback is not None


def test_ingest_is_root_score_is_leaf() -> None:
    roots = {t.task_id for t in dag.roots}
    leaves = {t.task_id for t in dag.leaves}
    assert roots == {"ingest"}
    assert leaves == {"score"}


# --- drift monitor DAG ---------------------------------------------------------

def test_drift_dag_structure() -> None:
    assert drift_dag.dag_id == "sneaker_drift_monitor"
    assert set(drift_dag.task_ids) == {
        "ingest",
        "feature_engineering",
        "validate",
        "detect_drift",
        "trigger_training",
    }
    expected_downstream = {
        "ingest": {"feature_engineering"},
        "feature_engineering": {"validate"},
        "validate": {"detect_drift"},
        "detect_drift": {"trigger_training"},
        "trigger_training": set(),
    }
    for task_id, downstream in expected_downstream.items():
        assert drift_dag.get_task(task_id).downstream_task_ids == downstream


def test_drift_dag_gate_and_trigger() -> None:
    from airflow.operators.python import ShortCircuitOperator
    from airflow.operators.trigger_dagrun import TriggerDagRunOperator

    assert isinstance(drift_dag.get_task("detect_drift"), ShortCircuitOperator)
    trigger = drift_dag.get_task("trigger_training")
    assert isinstance(trigger, TriggerDagRunOperator)
    assert trigger.trigger_dag_id == "sneaker_training_pipeline"
