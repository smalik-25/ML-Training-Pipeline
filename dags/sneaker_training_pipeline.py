"""Airflow DAG -- sneaker training pipeline orchestration.

ingest -> feature_engineering -> validate -> train -> register

Task logic stays THIN: each PythonOperator calls the corresponding stage
function; business logic lives in the stages, not here. S3 paths pass between
tasks via XCom -- no hardcoded paths in the DAG.

Implemented in Phase 6.
"""

from __future__ import annotations
