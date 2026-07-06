# ml-pipeline -- developer entrypoints.
#
# Every stage is independently runnable here without Airflow. RUN_DATE is
# required for stage targets; CONFIG defaults to the committed pipeline config.

RUN_DATE ?= 2025-01-01
CONFIG   ?= config/pipeline_config.yaml
PYTHON   ?= python

.PHONY: help fixtures ingest ingest-kicksdb features validate train register \
        score serve monitor live-score kicksdb-fixtures pipeline \
        airflow-up airflow-down mlflow-up mlflow-down \
        minio-up minio-down test lint format install

help:
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  %-14s %s\n", $$1, $$2}'

install: ## Install dev extras (lint + tests, no heavy frameworks)
	pip install -e ".[dev]"

fixtures: ## Generate synthetic Parquet fixtures into data/fixtures/
	$(PYTHON) generate_fixtures.py

ingest: ## Run the ingest stage
	$(PYTHON) -m stages.ingest --run-date $(RUN_DATE) --config $(CONFIG)

ingest-kicksdb: ## Ingest from the KicksDB API (canned fixtures if no KICKSDB_API_KEY)
	$(PYTHON) -m stages.ingest --run-date $(RUN_DATE) --config $(CONFIG) --source kicksdb

features: ## Run the feature engineering stage
	$(PYTHON) -m stages.features --run-date $(RUN_DATE) --config $(CONFIG)

validate: ## Run the validation stage
	$(PYTHON) -m stages.validate --run-date $(RUN_DATE) --config $(CONFIG)

train: ## Run the training stage
	$(PYTHON) -m stages.train --run-date $(RUN_DATE) --config $(CONFIG) --num-workers 2

register: ## Run the model registration stage
	$(PYTHON) -m stages.register --run-date $(RUN_DATE) --config $(CONFIG)

score: ## Batch-score with the staging model
	$(PYTHON) -m stages.score --run-date $(RUN_DATE) --config $(CONFIG)

monitor: ## Check feature drift vs the staging model's training data
	$(PYTHON) -m stages.monitor --run-date $(RUN_DATE) --config $(CONFIG)

live-score: ## Score current KicksDB sneakers with the staging model (MODEL_URI works)
	$(PYTHON) -m stages.live_score --run-date $(RUN_DATE) --config $(CONFIG)

kicksdb-fixtures: ## Refresh the canned KicksDB fixtures (needs KICKSDB_API_KEY)
	$(PYTHON) generate_kicksdb_fixtures.py

serve: ## Serve the staging model (FastAPI at http://localhost:8000)
	uvicorn serving.app:app --host 0.0.0.0 --port 8000

pipeline: ingest features validate train register score ## Run all stages in order

airflow-up: ## Start local Airflow (LocalExecutor)
	docker compose -f infra/docker-compose.airflow.yml up -d

airflow-down:
	docker compose -f infra/docker-compose.airflow.yml down

mlflow-up: ## Start local MLflow (Postgres backend + S3 artifacts)
	docker compose -f infra/docker-compose.mlflow.yml up -d

mlflow-down:
	docker compose -f infra/docker-compose.mlflow.yml down

minio-up: ## Optional: faithful local S3 API for testing s3:// paths
	docker compose -f infra/docker-compose.minio.yml up -d

minio-down:
	docker compose -f infra/docker-compose.minio.yml down

test: ## Run the test suite against fixtures
	pytest

lint: ## Lint with ruff
	ruff check .

format: ## Auto-fix lint + format
	ruff check --fix .
	ruff format .
