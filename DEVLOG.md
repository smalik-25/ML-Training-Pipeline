# DEVLOG

Notes on what I built and why, newest entries first.

---

## Monitoring and retrain-on-drift

**What I built**

`stages/monitor.py` (a drift-detection stage) and `dags/drift_monitor.py` (a
scheduled DAG that retrains only when the data has drifted). Plus a `drift/`
prefix, a `monitor` Makefile/entrypoint target, and `tests/test_monitor.py`. The
drift tests are numpy-only, so they run in the light CI job.

**Why it's the piece worth adding.** The training pipeline already produced a
deployed model, and scoring and serving consumed it, but nothing watched whether
the model was going stale. This is that watcher, and it closes the loop:
train, register, serve, monitor, retrain.

**Design decisions**

- **PSI, not a heavyweight drift library.** I compute Population Stability Index
  per feature by hand (numpy histograms over reference quantiles). PSI is the
  standard drift measure (< 0.1 stable, 0.1–0.2 moderate, > 0.2 significant), and
  hand-rolling it keeps the stage torch-free, dependency-light, and fully
  testable without installing Evidently or scipy. Constant features return 0
  rather than dividing by zero.
- **The reference is the deployed model's training data.** By default the stage
  resolves the reference distribution from the registry: the run that produced
  the current `staging` model. So "has the data drifted" always means "drifted
  from what the live model actually learned", not from some arbitrary snapshot.
  If there's no staging model yet, there's no baseline, so it reports drift and
  the first model gets trained.
- **Retrain on evidence, not on a schedule.** The monitoring DAG runs on a cron,
  but the retrain doesn't. A `ShortCircuitOperator` runs the drift check and lets
  the run proceed only if a feature clears the threshold; then a
  `TriggerDagRunOperator` fires the training pipeline. No drift means no retrain.
  This is the difference between "retrain nightly and hope" and "retrain when the
  data says to".
- **No duplicated wiring.** The monitoring DAG imports the training DAG's thin
  ingest/features/validate callables instead of redefining them, so the two DAGs
  share one definition of each stage's orchestration.

**Next up**

The honest remaining pre-deploy items are all infrastructure: one real run
against S3 and the Postgres+S3 MLflow server to prove the storage abstraction,
secrets management, and hosting the serving container. Those demonstrate cloud
ops rather than ML platform design, so I'm treating them as optional.

---

## Serving correctness: startup load and alias reload

Two fixes to the serving app before it could honestly be called deployable, both
found by thinking through what happens in production rather than in a demo.

- **The bundle was cached forever.** The app loaded the model once and never
  invalidated it, so moving the `staging` alias (a promotion or a rollback)
  wouldn't take effect until the process restarted. That quietly breaks the
  alias-as-source-of-truth story. I added a `POST /reload` endpoint that reloads
  from the registry, so an alias move is picked up without a restart.
- **The model loaded lazily on the first request.** A broken or missing model
  would surface as a 500 to whoever hit the endpoint first. I moved the load into
  a FastAPI lifespan handler so it happens at startup; if it fails, the app stays
  up but `/health` returns 503, which is what a readiness probe should see. Bad
  model means "hold traffic," not "serve an error." Added a test for the unready
  path (it runs without torch, since the failure happens before the model is
  even deserialized).

Neither is exotic, but both are the difference between "works in a demo" and
"survives a promotion in production."

---

## Closing the loop: batch scoring and online serving

**What I built**

`stages/score.py` (a batch scoring stage that runs after register in the DAG),
`serving/app.py` (a FastAPI app), and `stages/inference.py`, which both of them
share. Plus a `predictions/{run_date}/` prefix in the config, a `serving` extra
in pyproject, `score` and `serve` Makefile targets, and tests for all three.
Suite is now 39 passed / 4 skipped locally, ruff clean. The 4 skips are the
torch/fastapi paths that run on my machine, not in the build sandbox.

**Why this was the right next thing.** Before this, the pipeline trained and
registered a model that nothing consumed. The registry alias existed but didn't
mean anything. Scoring and serving are what make "promoted to staging" a real
event: the batch stage reads whatever version sits at the `staging` alias and
scores the run_date's features, and the API serves that same model online.

**Design decisions**

- **One inference module for both consumers.** Batch and online both go through
  `stages/inference.py`. That's deliberate. The train stage saves the imputation
  and standardization stats (fit on the training split) inside `model.pt`, and
  inference applies those saved stats rather than recomputing them. There is no
  second implementation of the preprocessing to drift, so a batch row and an API
  request get byte-identical treatment. This is the no-training/serving-skew
  guarantee made concrete instead of asserted.
- **The registry alias is the source of truth for what's live.** Neither consumer
  hardcodes a model path. They resolve the `staging` alias to a run, read the
  `model_s3_uri` the train stage logged, and load through the same fsspec I/O
  layer everything else uses. Rolling back a bad model is an alias move, and both
  consumers pick it up on their next load.
- **`build_matrix` is torch-free.** The transform is numpy-only so it's unit
  tested without installing torch; the model load and forward pass import torch
  lazily. Same pattern as the rest of the project: keep the pure logic testable
  in the light path, gate the heavy frameworks.
- **The API takes engineered features, and nullable ones are imputed.** Feature
  computation is an upstream Spark concern, so the endpoint accepts the same
  columns the model trained on. The nullable features (rolling average, pre-drop
  search) can be omitted and get the saved training means, exactly as in
  training.

**Next up**

Natural follow-ons if I keep going: a drift-detection stage comparing incoming
feature distributions against training, and a retrain trigger that only fires
when drift crosses a threshold. That would close the monitor-and-retrain half of
the loop.

---

## Phase 7: Docker, CI/CD, docs, and a build retrospective

**What I built**

I closed out the build. A clean multi-stage `Dockerfile` (one image runs any
stage via `entrypoint.sh`, installs `[spark,train,ingest]`, and includes a JRE
for Spark), a CI workflow split into focused jobs (lint, core unit tests, a Spark
job with a JRE, a Torch/Ray/MLflow job, and a DAG-integrity job with Airflow
pinned to its constraints file), and a finalized `docs/architecture.md`
(topology, per-prefix data flow, the anti-corruption ingest layer, and the full
design-decisions section including the real-data reconciliation). The README
checklist is complete and carries the real-data results.

**Why the CI is split this way.** This is where the Phase 0 dependency-extra split
pays off. Each job installs only what its tests need, so the light jobs stay fast
and the heavy frameworks are quarantined to the jobs that actually exercise them.
The `importorskip` guards mean nothing breaks at collection when a framework is
absent, so the Spark and train tests just run in their own jobs. CI never touches
real S3, the real Postgres, or the committed `data/` tree.

**Dockerfile fix worth noting.** The Phase 0 Dockerfile copied only
`pyproject.toml` before `pip install ".[...]"`, so the build silently fell back to
`requirements.txt` and dragged in Airflow plus dev. Now it copies the package
sources and installs the stage-runner extras properly. The package lands in the
venv and imports at runtime with no PYTHONPATH gymnastics.

**Retrospective: what this project demonstrates.** Seven phases plus a real-data
integration, each shipped as one reviewable unit with a DEVLOG rationale:

- A data lake with an fsspec-resolved I/O seam (`s3://` vs local), Parquet
  everywhere, and timestamps coerced for cross-engine (pyarrow and Spark) reads.
- PySpark feature engineering with broadcast joins, a brand partition, and a
  time-range rolling window with explicit null semantics.
- A Pandera data contract that fails loudly, and proved its worth by catching
  four real-data surprises the synthetic fixtures had hidden.
- Distributed training via Ray `TorchTrainer` with leakage-safe,
  checkpoint-saved preprocessing and a temporal split.
- An MLflow registry with strict, logged promotion using aliases rather than the
  deprecated stages.
- An Airflow DAG with thin tasks, lazy imports, XCom lineage, retries/SLA, and a
  failure callback.
- And the seam that ties it to reality: ingest as an anti-corruption layer over
  the actual sneaker-intel star schema, run end-to-end on 99,956 real sales.

The same idea runs through every phase. Stages are independent and CLI-runnable,
config is the single source of truth, the data contract is explicit and loud, and
every non-obvious decision is written down. The model is simple on purpose; the
reliability, reproducibility, and scale of the pipeline around it are the actual
deliverable.

**Build complete.**

---

## Real-data integration: reconciling with the sneaker-intel schema

**What happened**

Up to this point every stage ran on synthetic fixtures. Wiring up the actual
sneaker-intel Postgres showed me that the warehouse's real schema differs a lot
from the idealized one the project plan assumed. Instead of rewriting five
stages, I made ingest the anti-corruption layer between the source schema and my
canonical pipeline schema, and reconciled three data-contract assumptions the
synthetic data had quietly gotten wrong.

**Schema gaps (real vs. what I assumed)**

- Keys are `shoe_key`/`sale_key`/etc., not `shoe_id`/`sale_id`.
- `fact_sales` uses `sold_price`/`sold_date`/`size`/`source`, not
  `sale_price`/`sale_date`/`size_us`/`platform`.
- `retail_price`, `release_date`, and `release_type` live in `dim_drops`, not
  `dim_shoes`, and a shoe can have several drops.
- `fact_search_interest` uses `interest`/`point_date`/`geo`, not
  `search_index`/`signal_date`/`platform`.
- `release_type` is `('general','limited','collab')`, not the synthetic
  `limited/raffle/fcfs/general`.

**Decisions**

- **Ingest as an anti-corruption layer.** `stages/ingest.py` now holds explicit
  SQL (`TABLE_QUERIES`) that aliases every column, casts Postgres `numeric` to
  float, and rolls each shoe's canonical drop (earliest release via `DISTINCT
  ON`) up onto the shoe row to supply retail and release. The Parquet it lands is
  byte-compatible with the fixtures, so features, validate, train, and register
  never learn the source schema exists. This is the one seam that knows what the
  warehouse looks like.
- **`release_type` vocabulary fixed** to `{limited, collab, general}` in
  `feature_config.yaml`, with the fixtures updated to match. Because validate
  builds its `isin` set from the config, that one change propagates to the schema
  check.
- **Premium ceiling raised from 20 to 50.** Real StockX resale of hyped Off-White
  and Yeezy pairs reaches roughly 20–25× retail, so a ceiling of 20 would reject
  legitimate sales. Anything over 50× is still a data error. (Fixtures stay under
  2×.)
- **Dropped the `(shoe_id, sale_date, size_us)` uniqueness check.** This was an
  idealization that doesn't survive contact with transaction data, since real
  resale has many sales of the same shoe and size on the same day. The true
  unique key is `sale_id`, which is already enforced as a column constraint.
  Keeping the composite check would have rejected valid data. The validator's
  contract has to match the real grain of the data, not a tidy assumption.
- **DSN normalized to psycopg3.** The `[ingest]` extra ships psycopg v3, but a
  bare `postgresql://` DSN makes SQLAlchemy reach for psycopg2, so ingest rewrites
  the scheme to `postgresql+psycopg://` and sneaker-intel's `.env` DSN works
  unchanged.

**Verification.** The fixture-mode pipeline (ingest, features, validate) is green
on the new vocabulary, the full suite is 33 passed / 2 skipped, and ruff is
clean. The real Postgres export runs on the host (the DB isn't reachable from the
build sandbox), so its first real exercise was on my machine, with the Pandera
stage as the safety net for whatever real-data quirks showed up.

**The real run (99,956 StockX sales).** Ingest pulled 99,956 sales, 55 shoes, 55
drops, and 552 trends rows. Every shoe had a drop, so nothing was orphaned. Spark
computed features on all 99,956 rows in about 15 seconds. Validation then earned
its place by catching two things the synthetic data never would have:

- **5,601 sales (~5.6%) with negative `days_since_release`**, sold before the
  recorded release date (min −69 days, none beyond −90). This is real: hyped
  pairs trade pre-release. I made `days_since_release_min` a config value and set
  it to −90, a bounded pre-release window that still flags grossly wrong dates.
- **`price_premium` max of 20.32**, which would have failed the original 20.0
  ceiling. The earlier bump to 50, made on reasoning about Off-White resale,
  turned out to be right; the real max landed just over the old line.

With a temporal split at 2019 (train 2017–2018, about 83.8K rows; validate 2019,
about 16.2K), the model trained to **val RMSE 0.209** on the 0–20 premium scale
and was registered and promoted to `staging` as v1. The brands in the StockX data
are Off-White and Yeezy. `search_index_7d_pre_drop` is mostly null because the
trends pulls don't overlap the 2017–2018 release dates, and it gets imputed at
train time, as designed.

So the pipeline runs end-to-end on real data, and everywhere my synthetic
assumptions were wrong, the validator caught it loudly and the fix was a
documented config change.

---

## Phase 6: Airflow DAG orchestration

**What I built**

`dags/sneaker_training_pipeline.py`, `tests/test_dag.py`, and a fleshed-out
`infra/docker-compose.airflow.yml` (LocalExecutor, Postgres metadata DB, and
init/webserver/scheduler). The DAG chains ingest, feature_engineering, validate,
train, and register. I verified it against real Airflow 2.9.3 in the sandbox: 5
structure tests green (task set, linear dependency edges, retries/retry_delay,
SLA plus failure callback, root/leaf).

**Design decisions**

- **Thin tasks; logic stays in the stages.** Each PythonOperator is a wrapper
  that builds the run's `PipelineConfig` and calls the stage's `run()`. The DAG
  owns the topology, not the business logic, so the same stage code runs
  identically from the CLI, from a test, or from Airflow.
- **Lazy stage imports for fast, robust parsing.** PySpark, Torch, Ray, and
  MLflow are imported inside the task callables, never at module top. The
  scheduler re-parses DAG files constantly, so heavy top-level imports would tax
  every parse. It also lets the DAG load (and its structure be unit-tested) in an
  environment without Spark or Torch installed, which is what makes
  `tests/test_dag.py` runnable in the light CI job.
- **XCom carries lineage; config stays the source of truth.** Paths come from
  config, but each task still pushes the concrete artifact URI it produced, and
  the next task pulls and logs what it's consuming. No path is hardcoded in the
  DAG, and the run-to-artifact lineage is visible in the UI. I chose not to make
  the stages re-read their input path from XCom (they derive it from config).
  Threading it for observability rather than control keeps the
  single-source-of-truth invariant while still showing the XCom pattern.
- **A validation failure fails the whole run.** The validate task lets
  `FeatureValidationError` propagate, so a Pandera breach fails the task and
  therefore the DAG run. Bad data can't reach training.
- **Failure callback writes to `failures/`.** Every task carries an
  `on_failure_callback` that records the failed task, run_date, what it was
  supposed to produce, and the exception, to `failures/{run_date}/{task}.json`.
  The callback swallows its own errors (logged, not raised) so it can never mask
  the original failure.
- **retries=2, 2-minute delay, 2-hour SLA** in `default_args`, applied uniformly.

**On the compose file.** Running the full DAG needs the heavy runtime libs in the
Airflow image, which I install via `_PIP_ADDITIONAL_REQUIREMENTS` (`-e
.[spark,train]`). That makes first boot slow and the image large; in a real
deployment you'd bake a custom image. The DAG's correctness as a graph is verified
independently by the structure tests.

**Next up**

Phase 7: the Docker stage-runner image, Makefile/CI polish (including a Spark and
a train CI job), and finalizing `docs/architecture.md`.

---

## Phase 5: MLflow model registry and promotion

**What I built**

`stages/register.py` and `tests/test_register.py`. The stage finds the best run
for a run_date (lowest `final_val_rmse`), registers it under
`sneaker-price-model`, and promotes it to the `staging` alias only if it beats the
current model's val RMSE. Otherwise it logs a warning and leaves the alias alone.
Every run writes a `promotion_report.json`. Unlike Phases 1–4, I could install
MLflow (3.14) in the sandbox, so this stage is verified end to end rather than
just statically: 5 tests covering first-run promotion, a better model beating the
incumbent, a worse model being refused, best-of-N selection within a run_date, and
the no-runs error path, all green against real MLflow 3.x.

**Design decisions**

- **Aliases, not stages.** The plan said "transition to Staging", but MLflow 3.x
  deprecated stage transitions (`Staging`/`Production`) in favor of aliases. I use
  a `staging` alias, the current-API equivalent of a staged pointer. I documented
  it in the module so the divergence from the plan is an explained choice.
- **Promote only on a strict improvement.** A new version is always registered, so
  lineage is preserved for every run, but the `staging` alias only moves if the
  candidate's val RMSE is strictly lower than the current one. A worse model exits
  with a warning, not a failure. "This run wasn't better" is a normal outcome, and
  a bad model must never silently overwrite a good one.
- **Read the incumbent metric before registering.** I resolve the current
  `staging` model's RMSE up front (via its run), so the comparison is against the
  truly-deployed model and the report records exactly what was beaten (or "no
  existing model" on the first run).
- **Registry decoupled from model flavor.** MLflow 3.x's high-level
  `register_model("runs:/…/model")` requires a logged MLmodel flavor, but my
  training stage logs a plain `state_dict + scaler` `model.pt`. Instead of
  coupling the registry to a PyTorch flavor, I register the run's artifact
  directory directly via the low-level client. Clean lineage, no flavor lock-in.

**Bugs caught by running it.** Two real issues only surfaced because MLflow was
actually executing: the `runs:/…/model` MLmodel requirement above, and a
test-isolation trap where, once `run()` calls `mlflow.set_tracking_uri()`, that
global overrides the env var so a later test's seed writes to the previous test's
DB. Pinning the URI in the seed helper fixed it. The stage itself was correct;
this was a test-harness subtlety worth recording.

**Next up**

Phase 6: `dags/sneaker_training_pipeline.py` (Airflow), orchestrating ingest,
features, validate, train, and register with XCom path passing, retries, SLA, and
a failure callback.

---

## Phase 4: PyTorch model & Ray Train + MLflow

**What I built**

`models/net.py` (SneakerPriceNet + ModelConfig), `stages/train.py` (Ray Train,
PyTorch, MLflow), `tests/test_train.py`, a fleshed-out
`infra/docker-compose.mlflow.yml` (Postgres backend, S3 artifacts), and
`io.write_bytes`/`read_bytes` for saving the model artifact. The full suite is now
22 passing, 2 skipped (the torch/ray/mlflow tests, see below).

**Design decisions**

- **Temporal train/val split.** Train on sales before 2023, validate on 2023+. A
  random split leaks the future: the model could see a shoe's later sales while
  predicting its earlier ones, inflating validation metrics that wouldn't hold in
  production where we always predict forward. The split also has to actually
  divide the data, so `prepare_arrays` raises if either side is empty rather than
  training on a degenerate split.
- **Ray Train even on one machine.** The loop runs inside
  `ray.train.torch.TorchTrainer` with `prepare_model`, `prepare_data_loader`, and
  `report`/`Checkpoint`, rather than a raw loop or `DataParallel`. On a single
  machine it behaves like a normal run, but scaling to more workers or GPUs
  becomes a `ScalingConfig` change instead of a rewrite. `prepare_data_loader`
  also shards the data via a DistributedSampler, so multi-worker training is
  genuinely distributed.
- **Preprocessing fit on train only, saved with the model.** Imputation
  (rolling-null rows get the train column means) and standardization (train
  mean/std) are fit on the training split and travel inside `model.pt` next to the
  weights. Inference reproduces the exact pipeline with no separate scaler to
  lose, and val is standardized with the train stats, so there's no leakage.
- **MLflow is the reproducibility ledger.** Every run logs all ModelConfig
  hyperparameters, per-epoch train/val loss, final val RMSE, the feature-column
  list, and run_date. The model artifact goes to both MLflow and
  `s3://.../models/{run_date}/{run_id}/model.pt`. The driver logs metrics from
  Ray's `result.metrics_dataframe`, which keeps MLflow calls out of the worker.
- **Local SQLite MLflow by default, server when you want it.** The stage honors
  `MLFLOW_TRACKING_URI` and otherwise falls back to a local SQLite store
  (`sqlite:///mlflow.db`), so a quick run needs no infrastructure. The compose
  file provides the production-pattern Postgres+S3 server for when you do.
  (Correction after a local run: I initially defaulted to the file store, but
  MLflow 3.x put the file backend in maintenance mode and hard-errors on it, and
  it never supported the model registry that Phase 5 needs anyway. SQLite is the
  right zero-setup default since it backs both tracking and the registry.)

**Sandbox limitation: what was and wasn't executed here.** The sandbox's network
blocks the PyTorch CPU wheel index, and the default wheel is too large to install
within the time limit, so I couldn't run the torch/ray/mlflow path in that
environment. What I did verify there: the pure data-prep logic (temporal split,
leakage-safe imputation and standardization, the empty-split guard) runs green,
and the whole tree is ruff-clean. The model forward pass and the full Ray training
run are real tests, `importorskip`-guarded so they skip where the heavy deps are
absent. They should run on a machine with the `train` extra installed (`pip
install -e ".[train,dev]" && pytest tests/test_train.py`). Phase 7 CI should add a
job that installs the `train` extra and runs them.

**Next up**

Phase 5: `stages/register.py` (MLflow Model Registry), picking the best run,
registering, promoting to staging only if it beats the current staging val RMSE,
and logging a promotion report.

---

## Phase 3: Pandera dataset validation

**What I built**

`stages/validate.py` and `tests/test_validate.py`. The stage builds a Pandera
`DataFrameSchema` over the features Parquet, validates lazily, and on success
writes the validated Parquet plus a `validation_report.json`. Verified end to end:
68 rows, 100% retention, 16 rolling-nulls, report written. Tests cover the clean
pass and each failure mode, 7 cases. The full suite is now 18 passing.

**Design decisions**

- **Pandera over ad-hoc `if/else`.** The checks are declarative, so the schema
  doubles as documentation of the data contract, and lazy validation
  (`lazy=True`) collects all violations in one pass and reports which column,
  which check, and how many rows, not just the first failure. The failure summary
  groups by (column, check) with affected row counts, which is what you actually
  want when a run breaks.
- **The rolling-null rule is a custom cross-row check, not a null check.**
  `rolling_7d_avg_premium` may be null, but only on rows where the shoe has under
  7 days of history. I wrote it as a wide check returning a row-aligned boolean
  Series (`is_null == insufficient_history`), so Pandera pinpoints any row that's
  wrongly nulled or wrongly populated. A plain `nullable=True` would have let a
  silently-broken feature stage through.
- **Outliers are flagged, never dropped.** `price_premium ∈ [-1.0, 20.0]` is a
  hard check that fails the run rather than clipping or filtering. Bad data should
  stop the pipeline, not get quietly laundered into the training set.
- **A row-retention check spans the stage boundary.** The validator reads the raw
  sales count and asserts the features output kept at least 90% of it. This is the
  one check that can't live inside the feature DataFrame; it guards against silent
  row loss in an upstream join or filter, and it lands in the report regardless.
- **`coerce=True` to absorb engine dtype drift.** Spark and pyarrow hand back
  `int32`, `datetime64[us]`, and a `category` brand column. Rather than make the
  schema brittle to those, I coerce to the declared types, which also means the
  validated Parquet I write downstream has normalized, predictable dtypes.

**Noted**

The validation report is the reproducibility artifact for this stage. Every run
leaves a timestamped record of row counts, retention, and which checks ran. Phase
5's promotion logic and the Airflow failure callback lean on the same "every
decision leaves a JSON" pattern.

**Next up**

Phase 4: `models/net.py` (SneakerPriceNet) and `stages/train.py` (Ray Train,
PyTorch, MLflow), with a temporal train/val split.

---

## Phase 2: PySpark feature engineering

**What I built**

`stages/features.py` (PySpark, `local[*]`), a `FeatureConfig` loader in
`stages/config.py`, `tests/test_features.py`, and `tests/conftest.py`. The stage
reads the four raw tables, computes seven features, and writes a flat features
Parquet partitioned by brand. Verified end to end on fixtures: 68 in / 68 out,
four brand partitions, premiums 0.04–1.71, 16 rolling nulls (insufficient
history), ordinal encoding ordered correctly, and sensible brand averages (Jordan
highest at 1.14).

**Design decisions**

- **Broadcast joins for the small dimensions.** `dim_shoes` and the per-brand
  average premium are tiny relative to the sales fact table, so I attach both with
  `F.broadcast(...)` instead of a correlated subquery or a shuffle join. I compute
  `brand_avg_premium` as a standalone group-by aggregate and broadcast it back,
  which avoids a window over the whole fact table and keeps the brand aggregate
  from forcing a wide shuffle.
- **Partition output by brand.** Downstream training that filters to one brand
  reads a single partition instead of scanning the full dataset. The tradeoff is
  the many-small-files risk on a tiny dataset, which I mitigate with `coalesce(1)`
  (one file per brand partition).
- **Rolling-average null semantics are explicit, not incidental.**
  `rolling_7d_avg_premium` is a time-range window (`rangeBetween` over seconds,
  not a row-count window, so it spans real days regardless of sale density). I
  null it out exactly when the shoe has fewer than 7 days of history
  (`days_since_first_sale < 7`), which is the precise contract Phase 3 validates.
  Encoding the "insufficient history" rule here, rather than letting the window
  silently average a partial window, means the null carries meaning.
- **Google Trends over Reddit as the demand signal.** `search_index_7d_pre_drop`
  averages `fact_search_interest` in the 7 days before each drop.
  `fact_social_posts` (Reddit) is too sparse in sneaker-intel to be a reliable
  signal. I documented this as a data-quality call so it reads as a deliberate
  choice rather than an omission.
- **Microsecond timestamps at the I/O layer.** pandas and pyarrow default to
  nanosecond Parquet timestamps, which Spark 3.5 refuses to read ("Illegal Parquet
  type: INT64 TIMESTAMP(NANOS)"). I coerce timestamps to microseconds in
  `io.write_parquet` so the raw lake reads in both the pyarrow stages and the
  Spark stage. This is the cross-engine compatibility seam that bites people in
  real lakes.
- **`to_spark_path` mirrors the pyarrow I/O seam.** Spark reads through Hadoop,
  not pyarrow, so S3 has to be `s3a://` and local paths absolute. I added a small
  translator alongside `_resolve` so the Spark stage shares the same URI
  conventions as everything else, and I pin `SPARK_LOCAL_IP` to loopback in
  `build_spark` so an unresolvable container hostname can't crash driver startup.

**Noted for later**

The Spark tests are `importorskip`-guarded so the light CI lint+test job (no Spark
installed) doesn't break on collection. Phase 7 CI polish should add a dedicated
job that installs the `spark` extra and runs them for real.

**Next up**

Phase 3: `stages/validate.py` (Pandera), with schema and statistical checks on the
features Parquet, including the rolling-null custom check, failing loudly.

---

## Phase 1: data lake layer and ingest

**What I built**

`stages/ingest.py` (real, not a stub) and `generate_fixtures.py`, plus
`tests/test_ingest.py`. Ingest lands the four source tables (`sales`, `shoes`,
`drops`, `search_interest`) as separate Parquet files under `raw/{run_date}/`,
logging row counts, schema, and the written URIs. I don't export the dbt mart
tables; the Phase 2 Spark stage re-derives those metrics from these raw tables.

**Why I made these decisions**

- **One output contract, two source modes.** Ingest reads from the live
  sneaker-intel Postgres when `SNEAKER_INTEL_DSN` is set, and otherwise lands the
  synthetic fixtures from `data/fixtures/`. Both modes write byte-identical raw
  layouts, so every downstream stage is oblivious to where the data came from.
  SQLAlchemy and psycopg are imported lazily and live in an optional `[ingest]`
  extra, so fixture-mode runs and CI never install them.
- **"Partitioned by run_date" via path layout, not a partition column.** Each raw
  file sits under a `raw/{run_date}/` prefix. Re-running for a new date never
  clobbers a prior landing, and downstream stages address a run by its date.
  Simpler and more S3-idiomatic than a Hive partition column on an already-dated
  export.
- **Fail loudly on empty sources.** Ingest refuses to land a zero-row partition
  rather than silently producing an empty lake that breaks training three stages
  later.

**Fixtures: built to be plausible, not just present.** Deterministic (seed 42) so
tests assert on exact values. Eight shoes across four brands and the release
types; 68 sales clustered per shoe so rolling windows populate; demand (Google
Trends `search_index`) points within 7 days before each drop so the Phase 2
pre-drop join has rows. Verified properties: premium range 0.04–1.71 (inside the
Pandera bounds), mean premium ordered by scarcity, sale dates spanning 2020–2024
(both sides of the 2023 train/val split), and a unique `sale_id`. The
deliberately-bad fixture for the validation-failure test is deferred to Phase 3,
where it's a features-level mutation (negative `days_since_release`) rather than a
raw-table concern.

**Next up**

Phase 2: `stages/features.py` (PySpark), with price/size premiums, the rolling
7-day premium, the pre-drop search signal, the brand broadcast join, and output
partitioned by brand.

---

## Phase 0: project scaffold

**What I built**

The full repo skeleton for ml-pipeline: `stages/` (ingest, features, validate,
train, register as CLI stubs to fill in per phase), `dags/`, `config/`, `models/`,
`tests/`, `infra/`, `docs/`, and a GitHub Actions workflow. Two pieces are real,
not stubs: the shared I/O helper (`stages/io.py`) and the config loader
(`stages/config.py`), plus a real round-trip test for the I/O layer
(`tests/test_io.py`). Packaging (`pyproject.toml` as source of truth, generated
`requirements.txt`), `.env.example`, `.gitignore`, `Makefile`, a multi-stage
`Dockerfile`, and infra compose scaffolds for Airflow, MLflow, and (optional)
MinIO.

**Why I made these decisions**

I settled three things up front because they ripple across every later phase and
retrofitting them would be expensive.

1. **One fsspec-resolved I/O seam instead of per-stage boto3.** Every stage reads
   and writes through `stages/io.py`, which picks the filesystem from the URI
   scheme (`s3://…` in prod, a local path in dev/CI) using `pyarrow.fs`. The
   payoff is that two of the project's own constraints (stages runnable
   independently from the CLI, and CI never touching real S3) are satisfied by the
   same code path that runs in production. Fixtures aren't a special case; they
   flow through the real I/O layer. The honest tradeoff is that the local path
   doesn't exercise true S3 semantics (eventual consistency, multipart, IAM). I
   accept that for zero-setup, fast CI, and I provide an optional MinIO compose
   file for anyone who wants a faithful local S3 API. `storage_root` is the single
   knob that switches environments, so no stage hardcodes a path.

2. **Python 3.11, `pyproject.toml` as source of truth, split dependency extras.**
   3.11 is the highest version Spark, Ray, Airflow, and PyTorch all support
   without edge cases. The dependencies split into `spark`, `train`, `airflow`,
   and `dev` extras, so the CI lint and DAG-import job installs only `airflow,dev`
   and never has to resolve Spark/Ray/Torch just to confirm the DAG imports.
   Airflow is pinned with its official constraints file, the one dependency that
   fights back otherwise. `requirements.txt` is a generated lockfile to satisfy
   the scaffold spec, but pyproject is authoritative.

3. **Fixture-first, real-but-gated Postgres ingest.** The ingest stage is real
   Postgres-to-Parquet code, gated behind `SNEAKER_INTEL_DSN`. With the env var
   unset, the working dataset is synthetic fixtures, and that's the default for
   dev and all of CI. "The pipeline runs end-to-end on synthetic data with one
   command, and points at the real warehouse via one env var" is both more honest
   and a stronger demo than requiring live DB credentials to see anything work.

**What I noted for later**

Fixtures have to be schema-faithful and statistically plausible (enough sales
history per shoe that `rolling_7d_avg_premium` populates, realistic premium
distributions) or the Phase 3 Pandera statistical checks become theater. I'll also
produce one deliberately-bad fixture for the validation-failure test.

**Next up**

Phase 1: implement `stages/ingest.py` (real Postgres export plus S3 raw landing)
and `generate_fixtures.py`.
