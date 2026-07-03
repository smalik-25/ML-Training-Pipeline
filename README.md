# ml-pipeline

I built this to predict sneaker resale **price premium** from StockX sales. It's
a portfolio project about ML *infrastructure*, not modeling. The model is a plain
feedforward net on purpose; what I cared about is everything around it: the data
lake layout, Spark feature engineering, dataset validation, distributed training,
experiment tracking, model promotion, and Airflow orchestration. My goal was for
every architectural choice to hold up to a "why did you do it that way" question,
in terms of reliability, reproducibility, and scale.

It's the Phase 2 of [sneaker-intel](https://github.com/smalik-25/sneaker-intel)
that I deferred there on purpose. sneaker-intel built the data engineering
foundation: a hand-written Postgres star schema over ~99K StockX sales with a
full dbt transformation layer. This repo builds the ML layer on top of it.

## Live demo

- **This project · interactive dashboard:** https://huggingface.co/spaces/smalik25/sneaker-ml-platform
- **sneaker-intel (Phase 1) · dashboard:** https://sneaker-intel-2.streamlit.app/
- **Portfolio:** https://sam-malik.com

## Architecture

```mermaid
flowchart TD
    A[S3 raw data lake<br/>Parquet, versioned by run_date] --> B[PySpark feature engineering<br/>premiums, rolling signals, demand]
    B --> C[Pandera validation<br/>schema + statistical checks]
    C --> D[Ray Train + PyTorch<br/>TorchTrainer regression net]
    D --> E[MLflow registry<br/>experiment tracking + staged promotion]
    F[Airflow DAG] -. orchestrates .-> A
    F -. orchestrates .-> B
    F -. orchestrates .-> C
    F -. orchestrates .-> D
    F -. orchestrates .-> E
```

Each stage runs on its own from the CLI and talks to the next one only through
Parquet at an S3 (or local) path. The Airflow DAG is the source of truth for the
topology. It orchestrates the stages but holds none of their logic.

## Tech stack

| Concern | Tool |
|---|---|
| Storage / data lake | AWS S3, Parquet (local filesystem in dev/CI via the same I/O layer) |
| Feature engineering | PySpark |
| Dataset validation | Pandera |
| Model | PyTorch (simple feedforward regression) |
| Distributed training | Ray Train (`TorchTrainer`) |
| Experiment tracking / registry | MLflow (Postgres backend, S3 artifacts) |
| Orchestration | Apache Airflow (LocalExecutor) |
| Packaging / CI | Docker, GitHub Actions, ruff, pytest |
| Language | Python 3.11 |

## Storage abstraction (dev vs prod)

Every stage reads and writes through one helper, `stages/io.py`, which picks the
filesystem from the URI scheme: `s3://…` in production, a local path in dev/CI.
Both run through the same code, so the fixtures exercise the real I/O path
instead of a mock. Switching environments is a one-line change to `storage_root`
in `config/pipeline_config.yaml` (or the `STORAGE_ROOT` env var). There's a MinIO
compose file if you want a faithful local S3 API, but it's optional.

## Running a stage on its own (no Airflow)

```bash
make install                      # pip install -e ".[dev]"
make fixtures                     # generate synthetic data into data/fixtures/
make ingest   RUN_DATE=2025-01-01
make features RUN_DATE=2025-01-01
make validate RUN_DATE=2025-01-01
make train    RUN_DATE=2025-01-01
make register RUN_DATE=2025-01-01
make score    RUN_DATE=2025-01-01   # batch-score with the @staging model
make pipeline RUN_DATE=2025-01-01   # all stages in order
```

Each stage is also a module entrypoint, e.g.
`python -m stages.features --run-date 2025-01-01 --config config/pipeline_config.yaml`.

## Serving the model

The batch scoring stage and a FastAPI app both load the model at the MLflow
`staging` alias and reuse the preprocessing saved inside `model.pt`, so online and
batch predictions run the exact same transform (no training/serving skew).

```bash
pip install -e ".[train,serving]"
make serve          # FastAPI at http://localhost:8000, docs at /docs
# or point it at a specific artifact instead of the registry:
MODEL_URI=data/models/2026-06-28/<run_id>/model.pt make serve
```

`GET /health` reports the loaded model version (503 until a model is loaded);
`POST /predict` scores one sale's engineered features; `POST /predict/batch`
scores a list. The model loads at startup, so a broken or missing model shows up
as a failed readiness check rather than a 500 on the first request. After you
promote a new version or roll back by moving the `staging` alias, `POST /reload`
picks it up without a restart.

## Deploying to AWS

There's a full runbook in [`DEPLOY.md`](./DEPLOY.md): Terraform under
`infra/terraform/` provisions an S3 data-lake bucket, an ECR repo, and an App
Runner service that serves the model from S3 with least-privilege IAM. The
serving image is `serving/Dockerfile` (CPU-only torch), and
`.github/workflows/deploy-serving.yml` builds and pushes it. Switching the whole
pipeline between local and S3 is one env var, `STORAGE_ROOT=s3://<bucket>`, and
that `s3://` path is exercised in CI against a moto S3 server so it's not an
untested claim.

## Running the full DAG

```bash
make mlflow-up      # local MLflow (Postgres backend + S3 artifacts)
make airflow-up     # local Airflow (LocalExecutor); UI at http://localhost:8080
# trigger the `sneaker_training_pipeline` DAG from the Airflow UI
```

On MLflow tracking: leave `MLFLOW_TRACKING_URI` unset for a zero-setup local run.
The train and register stages default to a SQLite store (`sqlite:///mlflow.db`),
which backs both experiment tracking and the registry. Point it at the compose
server (`http://localhost:5000`) for the production-pattern Postgres + S3 setup.
I don't use the legacy file store because MLflow 3.x deprecated it and it can't
back the registry.

## Source data

The production source is the sneaker-intel Postgres warehouse (`fact_sales`,
`dim_shoes`, `dim_drops`, `fact_search_interest`). You don't need it reachable to
run the pipeline. Set `SNEAKER_INTEL_DSN` to do a real export, or leave it unset
and run on synthetic fixtures (the default for dev and CI). The 99K real rows are
the production story; the fixtures make the whole pipeline runnable and testable
with one command.

To run on real data, bring up the sneaker-intel Postgres (`make db-up && make
db-init && make load` in that repo), then in this repo run `pip install -e
".[ingest]"`, set `export SNEAKER_INTEL_DSN=postgresql://sneaker:sneaker@localhost:5432/sneaker_intel`,
and run the stages with `--run-date <date>`. Pass `--split-year 2019` at train,
since the StockX data is 2017–2019.

## Results on real data

I ran it end-to-end on the live warehouse: **99,956 StockX sales** (Off-White and
Yeezy, 2017–2019). Validation passed at **100% row retention**, and the model
trained to **val RMSE ≈ 0.21** on a 0–20 premium scale, using an 84/16 temporal
split at 2019, then registered and promoted to `@staging` as v1. Connecting real
data turned up four contract mismatches the synthetic fixtures had hidden: the
release-type vocabulary, premium outliers (max 20.3× retail, over my old
ceiling), the transaction grain, and pre-release sales. The validator caught each
one, and I fixed each as a documented config change. The full story is in
[`DEVLOG.md`](./DEVLOG.md) and [`docs/architecture.md`](./docs/architecture.md).

## Build progress

- [x] **Phase 0**: Project scaffold, shared I/O layer, config, packaging, CI skeleton
- [x] **Phase 1**: Data lake layer & ingest stage (+ fixtures)
- [x] **Phase 2**: PySpark feature engineering
- [x] **Phase 3**: Pandera dataset validation
- [x] **Phase 4**: PyTorch model & Ray Train + MLflow
- [x] **Phase 5**: MLflow model registry & promotion logic
- [x] **Phase 6**: Airflow DAG orchestration
- [x] **Phase 7**: Docker, CI/CD, and docs polish
- [x] **Real-data integration**: runs end-to-end on the live sneaker-intel warehouse (99,956 StockX sales)
- [x] **Closing the loop**: batch scoring stage + FastAPI serving through one inference path (no train/serve skew)
- [x] **Drift monitoring**: PSI drift stage + a scheduled retrain-on-drift DAG
- [x] **AWS deployment**: Terraform (S3 · ECR · App Runner) + serving image + deploy workflow (see [`DEPLOY.md`](./DEPLOY.md))
- [x] **Live demo**: Streamlit dashboard on Hugging Face Spaces

## Design decisions

I recorded these as I made them. [`DEVLOG.md`](./DEVLOG.md) has the full
narrative; this is the short version.

- **Parquet at every stage boundary.** Columnar, typed, splittable. Never CSV
  between stages.
- **One fsspec-resolved I/O layer.** Stage code is identical against S3 and
  local, the fixtures exercise the real path, and CI needs zero S3 setup.
- **Stages run independently of Airflow.** Airflow orchestrates the stages, it
  doesn't define them. State passes between stages only as S3 paths (XCom in the
  DAG), never as shared local state.
- **Fixture-first, real-but-gated Postgres ingest.** The pipeline runs
  end-to-end on synthetic data, and the live warehouse is one env var away.
- **Python 3.11 with split dependency extras.** It's the highest version all
  four heavy frameworks support cleanly, and the extras keep the lint and
  DAG-import CI jobs fast.
- **Broadcast joins for the small dimensions (Phase 2).** `dim_shoes` and the
  per-brand average premium are broadcast instead of shuffled. I compute
  `brand_avg_premium` as a standalone aggregate and join it back, which avoids a
  window over the whole fact table.
- **Partition features by brand (Phase 2).** Brand-filtered training reads one
  partition instead of the whole dataset.
- **Google Trends over Reddit as the demand signal (Phase 2).**
  `fact_social_posts` is too sparse in sneaker-intel, so `search_index` is the
  signal I trust.
- **Microsecond Parquet timestamps (Phase 2).** I coerce them at the I/O layer
  so the lake reads cleanly in both pyarrow and Spark 3.5.
- **Pandera over ad-hoc validation (Phase 3).** Declarative checks double as a
  data contract, and lazy validation reports every failing column, check, and
  row count at once. The rolling-null rule is a custom cross-row check, outliers
  are flagged rather than dropped, and a ≥90% row-retention check guards the
  stage boundary.
- **Temporal train/val split (Phase 4).** I train on earlier sales and validate
  on later ones, so the model is always judged predicting forward in time. A
  random split would leak the future.
- **Ray Train even on a single machine (Phase 4).** `TorchTrainer` with
  `prepare_model`, `prepare_data_loader`, and checkpointing, so scaling to more
  workers or GPUs is a `ScalingConfig` change, not a rewrite. Preprocessing is
  fit on the train split only and saved inside `model.pt`, so inference
  reproduces it with no leakage.
- **Promote only on a strict improvement (Phase 5).** Every run is registered for
  lineage, but the `staging` alias only moves if the candidate beats the current
  model's val RMSE. A worse model logs a warning instead of failing. I use MLflow
  3.x aliases (stages are deprecated), and every decision leaves a
  `promotion_report.json`.
- **Thin Airflow tasks with lazy imports (Phase 6).** Each task just calls a
  stage's `run()`, and the heavy libraries import inside the callables, so DAG
  parsing stays fast and the graph is testable without Spark or Torch installed.
  XCom threads the artifact lineage (config stays the source of truth), a
  validation failure fails the whole run, and an `on_failure_callback` writes to
  the `failures/` prefix.
- **One inference path for batch and online.** The scoring stage and the serving
  API both load the `@staging` model through the same `stages/inference.py` and
  apply the preprocessing saved in `model.pt`, so a batch row and an API request
  get identical treatment. The registry alias is the source of truth for which
  model is live, so a rollback is an alias move.
- **Retrain on drift, not on a cron.** A scheduled monitoring DAG
  (`dags/drift_monitor.py`) computes per-feature PSI against the distribution the
  `@staging` model was trained on. It triggers the training pipeline only when a
  feature clears the drift threshold, and short-circuits otherwise. `make monitor`
  runs the check on its own.
