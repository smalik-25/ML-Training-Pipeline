# DEVLOG

Running narrative of what was built and — more importantly — *why*. Newest
entries at the top.

---

## 2026-06-28 — Real-data integration: reconciling with the sneaker-intel schema

**What happened**

Up to now every stage ran on synthetic fixtures. Wiring up the actual
sneaker-intel Postgres surfaced that the warehouse's real schema differs
materially from the idealized one the project plan assumed. Rather than rewrite
five stages, I made **ingest the anti-corruption layer** between the source
schema and our canonical pipeline schema, and reconciled three data-contract
assumptions that the synthetic data had quietly gotten wrong.

**Schema gaps found (real vs. what we assumed)**

- Keys are `shoe_key`/`sale_key`/etc., not `shoe_id`/`sale_id`.
- `fact_sales` uses `sold_price`/`sold_date`/`size`/`source`, not
  `sale_price`/`sale_date`/`size_us`/`platform`.
- **`retail_price` / `release_date` / `release_type` live in `dim_drops`, not
  `dim_shoes`.** A shoe can have several drops.
- `fact_search_interest` uses `interest`/`point_date`/`geo`, not
  `search_index`/`signal_date`/`platform`.
- `release_type` is `('general','limited','collab')` — not the synthetic
  `limited/raffle/fcfs/general`.

**Decisions**

- **Ingest as an anti-corruption layer.** `stages/ingest.py` now holds explicit
  SQL (`TABLE_QUERIES`) that aliases every column, casts Postgres `numeric` to
  float, and rolls each shoe's *canonical drop* (earliest release via
  `DISTINCT ON`) up onto the shoe row to supply retail/release. The Parquet it
  lands is byte-compatible with the fixtures, so features/validate/train/register
  never learn the source schema exists. This is the one seam that knows what the
  warehouse looks like — a clean, interview-legible pattern.
- **`release_type` vocabulary fixed** to `{limited, collab, general}` in
  `feature_config.yaml`; fixtures updated to match. Because validate builds its
  `isin` set from the config, that one change propagates to the schema check.
- **Premium ceiling raised 20 → 50.** Real StockX resale of hyped Off-White /
  Yeezy pairs reaches ~20–25× retail; a ceiling of 20 would reject legitimate
  sales. >50× is still treated as a data error. (Fixtures stay under 2×.)
- **Dropped the `(shoe_id, sale_date, size_us)` uniqueness check.** This was an
  idealization that does not survive contact with transaction data: real resale
  has many sales of the same shoe/size on the same day. The true unique key is
  `sale_id` (already enforced as a column constraint). Keeping the composite
  check would have rejected valid data — a good example of the validator's
  contract needing to match the real *grain* of the data, not a tidy assumption.
- **DSN normalized to psycopg3.** The `[ingest]` extra ships psycopg v3, but a
  bare `postgresql://` DSN makes SQLAlchemy reach for psycopg2; ingest rewrites
  the scheme to `postgresql+psycopg://` so sneaker-intel's `.env` DSN works
  unchanged.

**Verification.** Fixture-mode pipeline (ingest→features→validate) is green on
the new vocabulary, full suite 33 passed / 2 skipped, ruff clean. The real
Postgres export itself runs on the host (the DB isn't reachable from the build
sandbox), so its first real exercise is on Sam's machine — and the Pandera stage
is deliberately the safety net for whatever real-data quirks surface.

**The real run (99,956 StockX sales).** Ingest pulled 99,956 sales / 55 shoes /
55 drops / 552 trends rows; every shoe had a drop, so nothing was orphaned.
Spark computed features on all 99,956 rows in ~15s. Validation then earned its
place by catching two things the synthetic data never would have:

- **5,601 sales (~5.6%) with negative `days_since_release`** — sold *before* the
  recorded release date (min −69 days, none beyond −90). This is real: hyped
  pairs trade pre-release. I made `days_since_release_min` a config value and set
  it to −90 — a bounded pre-release window that still flags grossly wrong dates.
- **`price_premium` max of 20.32** — which would have *failed* the original 20.0
  ceiling. The earlier bump to 50 (made on reasoning about Off-White resale) was
  exactly right; the real max landed just over the old line.

With a temporal split at 2019 (train 2017–2018 ≈ 83.8K rows, validate 2019 ≈
16.2K), the model trained to **val RMSE 0.209** (premium scale 0–20) and was
registered + promoted to `staging` as v1. Brands in the StockX data are
Off-White and Yeezy; `search_index_7d_pre_drop` is mostly null (the trends pulls
don't overlap 2017–2018 release dates) and is imputed at train time, as designed.

Net: the pipeline runs end-to-end on real data, and every place the synthetic
assumptions were wrong, the validator caught it loudly and the fix was a
documented config/contract change — not a silent patch.

---

## 2026-06-27 — Phase 6: Airflow DAG orchestration

**What I built**

`dags/sneaker_training_pipeline.py`, `tests/test_dag.py`, and a fleshed-out
`infra/docker-compose.airflow.yml` (LocalExecutor + Postgres metadata DB +
init/webserver/scheduler). The DAG chains ingest → feature_engineering →
validate → train → register. Verified against real Airflow 2.9.3 in the sandbox:
5 structure tests green (task set, linear dependency edges, retries/retry_delay,
SLA + failure callback, root/leaf).

**Design decisions**

- **Thin tasks; logic stays in the stages.** Each PythonOperator is a wrapper
  that builds the run's `PipelineConfig` and calls the stage's `run()`. The DAG
  owns *topology*, not business logic — so the same stage code runs identically
  from the CLI, from a test, or from Airflow.
- **Lazy stage imports for fast, robust parsing.** PySpark/Torch/Ray/MLflow are
  imported *inside* the task callables, never at module top. The scheduler
  re-parses DAG files constantly, so heavy top-level imports would tax every
  parse — and, just as importantly, it lets the DAG load (and its structure be
  unit-tested) in an environment without Spark/Torch installed. That's exactly
  what makes `tests/test_dag.py` runnable in the light CI job and the DAG-import
  CI check meaningful.
- **XCom carries lineage; config stays the source of truth.** Paths are derived
  from config, but each task still pushes the concrete artifact URI(s) it
  produced and the next task pulls and logs what it's consuming. No path is
  hardcoded in the DAG, and the run → artifact lineage is visible in the UI.
  I chose not to make the stages *re-read* their input path from XCom (they
  derive it from config) — threading it for observability rather than control
  keeps the single-source-of-truth invariant intact while still demonstrating
  the XCom pattern.
- **Validation failure fails the whole run.** The validate task lets
  `FeatureValidationError` propagate, so a Pandera breach fails the task and
  therefore the DAG run — bad data can't reach training.
- **Failure callback writes to `failures/`.** Every task carries an
  `on_failure_callback` that records the failed task, run_date, what it was
  supposed to produce, and the exception, to `failures/{run_date}/{task}.json`.
  The callback swallows its *own* errors (logged, not raised) so it can never
  mask the original failure.
- **retries=2 / 2-min delay / 2-hour SLA** in `default_args` — the resilience
  knobs an interviewer expects to see, applied uniformly.

**On the compose file.** Running the *full* DAG needs the heavy runtime libs in
the Airflow image, which I install via `_PIP_ADDITIONAL_REQUIREMENTS` (`-e
.[spark,train]`). That makes first boot slow and the image large; in a real
deployment you'd bake a custom image. The DAG's correctness as a graph is
verified independently of all that by the structure tests.

**Next up**

Phase 7 — Docker stage-runner image, Makefile/CI polish (incl. a Spark + a
train CI job), and finalizing `docs/architecture.md`.

---

## 2026-06-27 — Phase 5: MLflow model registry & promotion

**What I built**

`stages/register.py` and `tests/test_register.py`. The stage finds the best run
for a run_date (lowest `final_val_rmse`), registers it under
`sneaker-price-model`, and promotes it to the `staging` alias only if it beats
the incumbent's val RMSE — otherwise it logs a warning and leaves the alias
alone. Every run writes a `promotion_report.json`. Unlike Phases 1–4, I was able
to install MLflow (3.14) in the sandbox, so this stage is **verified end to
end**, not just statically: 5 tests covering first-run promotion, a better model
beating the incumbent, a worse model being refused, best-of-N selection within a
run_date, and the no-runs error path — all green against real MLflow 3.x.

**Design decisions**

- **Aliases, not stages.** The plan said "transition to Staging", but MLflow 3.x
  deprecated stage transitions (`Staging`/`Production`) in favour of aliases. I
  use a `staging` alias, which is the current-API equivalent of a staged
  pointer. Documented in the module so the divergence from the plan is a
  deliberate, explained choice.
- **Promote only on strict improvement.** A new version is always *registered*
  (lineage is preserved for every run), but the `staging` alias only moves if the
  candidate's val RMSE is strictly lower than the incumbent's. A worse model
  exits with a warning, not a failure — "this run wasn't better" is a normal
  outcome, not a pipeline error, and bad models must never silently overwrite
  good ones.
- **Read the incumbent metric before registering.** We resolve the current
  `staging` model's RMSE up front (via its run), so the comparison is against the
  truly-deployed model, and the report records exactly what was beaten (or
  "no existing model" on the first run).
- **Registry decoupled from model flavor.** MLflow 3.x's high-level
  `register_model("runs:/…/model")` requires a logged MLmodel flavor; our
  training stage logs a plain `state_dict + scaler` `model.pt`. Rather than
  couple the registry to a PyTorch flavor, I register the run's artifact
  *directory* directly via the low-level client. Clean lineage, no flavor lock-in.

**Bug caught by running it.** Two real issues only surfaced because MLflow was
actually executing: (1) the `runs:/…/model` MLmodel requirement above, and (2) a
test-isolation trap — once `run()` calls `mlflow.set_tracking_uri()`, that global
overrides the env var, so a later test's seed writes to the previous test's DB.
Pinning the URI in the seed helper fixed it. The stage itself was correct; this
was a test-harness subtlety worth recording.

**Next up**

Phase 6 — `dags/sneaker_training_pipeline.py` (Airflow): orchestrate
ingest → features → validate → train → register with XCom path passing, retries,
SLA, and a failure callback.

---

## 2026-06-27 — Phase 4: PyTorch model & Ray Train + MLflow

**What I built**

`models/net.py` (SneakerPriceNet + ModelConfig), `stages/train.py` (Ray Train
+ PyTorch + MLflow), `tests/test_train.py`, a fleshed-out
`infra/docker-compose.mlflow.yml` (Postgres backend + S3 artifacts), and
`io.write_bytes`/`read_bytes` for saving the model artifact. Full suite is now
22 passing, 2 skipped (the torch/ray/mlflow tests, see below).

**Design decisions**

- **Temporal train/val split, not random.** Train on sales before 2023,
  validate on 2023+. A random split leaks the future: the model could see a
  shoe's later sales while predicting its earlier ones, inflating validation
  metrics that wouldn't hold in production where we always predict forward. The
  split must actually divide the data — `prepare_arrays` raises if either side
  is empty rather than training on a degenerate split.
- **Ray Train even on one machine.** The loop runs inside
  `ray.train.torch.TorchTrainer` with `prepare_model` + `prepare_data_loader` +
  `report`/`Checkpoint`, not a raw loop or `DataParallel`. On a single machine
  it behaves like a normal run, but scaling to N workers/GPUs becomes a
  `ScalingConfig` change, not a rewrite. `prepare_data_loader` also shards the
  data via a DistributedSampler, so multi-worker training is genuinely
  distributed, not duplicated.
- **Preprocessing fit on train only, saved with the model.** Imputation
  (rolling-null rows → train column means) and standardization (train mean/std)
  are fit on the training split and travel inside `model.pt` alongside the
  weights. Inference reproduces the exact pipeline; no separate scaler to lose.
  This is the leakage-safe version — val is standardized with *train* stats.
- **MLflow is the reproducibility ledger.** Every run logs all ModelConfig
  hyperparameters, per-epoch train/val loss, final val RMSE, the feature-column
  list, and run_date; the model artifact goes to both MLflow and
  `s3://.../models/{run_date}/{run_id}/model.pt`. The driver logs metrics from
  Ray's `result.metrics_dataframe`, keeping MLflow calls out of the worker.
- **Local SQLite MLflow by default, server when you want it.** The stage honours
  `MLFLOW_TRACKING_URI` and otherwise falls back to a local SQLite store
  (`sqlite:///mlflow.db`), so a quick run needs no infrastructure; the compose
  file provides the production-pattern Postgres+S3 server for when you do.
  _(Correction after a local run: I initially defaulted to the file store, but
  MLflow 3.x put the file backend in maintenance mode and hard-errors on it —
  and it never supported the model registry Phase 5 needs anyway. SQLite is the
  right zero-setup default; it backs both tracking and the registry.)_

**Sandbox limitation — what was and wasn't executed here.** The sandbox's
network blocks the PyTorch CPU wheel index and the default wheel is too large to
install within the time limit, so I could not run the torch/ray/mlflow path in
this environment. What I *did* verify: the pure data-prep logic (temporal split,
leakage-safe imputation/standardization, empty-split guard) runs green, and the
whole tree is ruff-clean. The model forward pass and the full Ray training run
are real tests, `importorskip`-guarded so they skip where the heavy deps are
absent — they should be run on a machine with the `train` extra installed
(`pip install -e ".[train,dev]" && pytest tests/test_train.py`). Phase 7 CI
should add a job that installs the `train` extra and runs them.

**Next up**

Phase 5 — `stages/register.py` (MLflow Model Registry): pick the best run,
register, promote to Staging only if it beats the current Staging val RMSE, log
a promotion report.

---

## 2026-06-27 — Phase 3: Pandera dataset validation

**What I built**

`stages/validate.py` and `tests/test_validate.py`. The stage builds a Pandera
`DataFrameSchema` over the features Parquet, validates lazily, and on success
writes the validated Parquet plus a `validation_report.json`. Verified end to
end: 68 rows, 100% retention, 16 rolling-nulls, report written. Tests: 7 cases
covering the clean pass and each failure mode. Full suite now 18 passing.

**Design decisions**

- **Pandera over ad-hoc `if/else`.** The checks are declarative, so the schema
  doubles as documentation of the data contract, and lazy validation
  (`lazy=True`) collects *all* violations in one pass and reports which column,
  which check, and how many rows — not just the first failure. The failure
  summary groups by (column, check) with affected row counts, which is what you
  actually want at 3am when a run breaks.
- **The rolling-null rule is a custom cross-row check, not a null check.**
  `rolling_7d_avg_premium` may be null, but *only* on rows where the shoe has
  < 7 days of history. I implemented this as a wide check returning a row-aligned
  boolean Series (`is_null == insufficient_history`), so Pandera pinpoints any
  row that is wrongly nulled *or* wrongly populated. A plain `nullable=True`
  would have let a silently-broken feature stage through.
- **Outliers are flagged, never dropped.** `price_premium ∈ [-1.0, 20.0]` is a
  hard check that fails the run, rather than clipping or filtering — bad data
  should stop the pipeline, not be quietly laundered into the training set.
- **Row-retention check spans the stage boundary.** The validator reads the raw
  sales count and asserts the features output kept ≥ 90% of it. This is the one
  check that can't live inside the feature DataFrame — it guards against silent
  row loss in a join or filter upstream, and lands in the report regardless.
- **`coerce=True` to absorb engine dtype drift.** Spark/pyarrow hand back
  `int32`, `datetime64[us]`, and a `category` brand column. Rather than make the
  schema brittle to those, I coerce to the declared types — which also means the
  validated Parquet we write downstream has normalised, predictable dtypes.

**Noted**

The validation report is the reproducibility artifact for this stage: every run
leaves a timestamped record of row counts, retention, and which checks ran.
Phase 5's promotion logic and the Airflow failure callback will lean on the same
"every decision leaves a JSON" pattern.

**Next up**

Phase 4 — `models/net.py` (SneakerPriceNet) and `stages/train.py` (Ray Train +
PyTorch + MLflow), with a temporal train/val split.

---

## 2026-06-27 — Phase 2: PySpark feature engineering

**What I built**

`stages/features.py` (PySpark, `local[*]`), a `FeatureConfig` loader in
`stages/config.py`, `tests/test_features.py`, and `tests/conftest.py`. The stage
reads the four raw tables, computes seven features, and writes a flat features
Parquet partitioned by brand. Verified end to end on fixtures: 68 in / 68 out,
four brand partitions, premiums 0.04–1.71, 16 rolling nulls (insufficient
history), ordinal encoding correctly ordered, brand averages sensible
(Jordan 1.14 highest).

**Design decisions**

- **Broadcast joins for the small dimensions.** `dim_shoes` and the per-brand
  average premium are tiny relative to the sales fact table, so both are
  attached with `F.broadcast(...)` rather than a correlated subquery or a
  shuffle join. `brand_avg_premium` is computed as a standalone group-by
  aggregate and broadcast back — this avoids a window over the whole fact table
  and keeps the brand aggregate from forcing a wide shuffle. This is the kind of
  Spark-physical-plan choice worth being able to explain.
- **Partition output by brand.** Downstream training that filters to one brand
  reads a single partition instead of scanning the full dataset. The tradeoff is
  many-small-files risk on a tiny dataset, which I mitigate with `coalesce(1)`
  (one file per brand partition).
- **Rolling-average null semantics are explicit, not incidental.**
  `rolling_7d_avg_premium` is a time-range window (`rangeBetween` over seconds,
  not a row-count window, so it spans real days regardless of sale density). I
  null it out exactly when the shoe has fewer than 7 days of history
  (`days_since_first_sale < 7`), which is the precise contract Phase 3 will
  validate. Encoding the "insufficient history" rule here, rather than letting
  the window silently average a partial window, means the null carries meaning.
- **Google Trends over Reddit as the demand signal.** `search_index_7d_pre_drop`
  averages `fact_search_interest` in the 7 days before each drop.
  `fact_social_posts` (Reddit) is too sparse in sneaker-intel to be a reliable
  signal — a data-quality call, documented so it's a deliberate choice rather
  than an omission.
- **Microsecond timestamps at the I/O layer.** pandas/pyarrow default to
  nanosecond Parquet timestamps, which Spark 3.5 refuses to read ("Illegal
  Parquet type: INT64 TIMESTAMP(NANOS)"). I coerce timestamps to microseconds in
  `io.write_parquet`, so the raw lake is readable by both the pyarrow stages and
  the Spark stage. This is exactly the cross-engine compatibility seam that bites
  people in real lakes.
- **`to_spark_path` mirrors the pyarrow I/O seam.** Spark reads through Hadoop,
  not pyarrow, so S3 must be `s3a://` and local paths absolute. Added a small
  translator alongside `_resolve` so the Spark stage shares the same URI
  conventions as everything else. Also pin `SPARK_LOCAL_IP` to loopback in
  `build_spark` so an unresolvable container hostname can't crash driver startup.

**Noted for later**

The Spark tests are `importorskip`-guarded so the light CI lint+test job (no
Spark installed) doesn't break on collection. Phase 7 CI polish should add a
dedicated job that installs the `spark` extra and runs them for real.

**Next up**

Phase 3 — `stages/validate.py` (Pandera): schema + statistical checks on the
features Parquet, including the rolling-null custom check, with loud failure.

---

## 2026-06-27 — Phase 1: data lake layer & ingest

**What I built**

`stages/ingest.py` (real, not a stub) and `generate_fixtures.py`, plus
`tests/test_ingest.py`. Ingest lands the four source tables — `sales`, `shoes`,
`drops`, `search_interest` — as separate Parquet files under
`raw/{run_date}/`, logging row counts, schema, and the written URIs. The dbt
mart tables are deliberately not exported; the Phase 2 Spark stage re-derives
those metrics from these raw tables.

**Why I made these decisions**

- **One output contract, two source modes.** Ingest reads from the live
  sneaker-intel Postgres when `SNEAKER_INTEL_DSN` is set, and otherwise lands
  the synthetic fixtures from `data/fixtures/`. Both modes write byte-identical
  raw layouts, so every downstream stage is oblivious to where the data came
  from. SQLAlchemy/psycopg are imported lazily and live in an optional
  `[ingest]` extra — fixture-mode runs and CI never install them.
- **"Partitioned by run_date" via path layout, not a partition column.** Each
  raw file sits under a `raw/{run_date}/` prefix. Re-running for a new date
  never clobbers a prior landing, and downstream stages address a run by its
  date. Simpler and more S3-idiomatic than a Hive partition column on an
  already-dated export.
- **Fail loudly on empty sources.** Ingest refuses to land a zero-row partition
  rather than silently producing an empty lake that breaks training three stages
  later.

**Fixtures: built to be plausible, not just present.** Deterministic (seed 42)
so tests assert on exact values. Eight shoes across four brands and all four
release types; 68 sales clustered per shoe so rolling windows populate; demand
(Google Trends `search_index`) points within 7 days before each drop so the
Phase 2 pre-drop join has rows. Verified properties: premium range 0.04–1.71
(inside the Pandera −1…20 bounds), mean premium strictly ordered by scarcity
(limited 1.42 > raffle 0.92 > fcfs 0.40 > general 0.05), sale dates span
2020–2024 (both sides of the 2023 train/val split), and
`(shoe_id, sale_date, size_us)` is unique. The deliberately-bad fixture for the
validation-failure test is deferred to Phase 3, where it's a *features*-level
mutation (negative `days_since_release`) rather than a raw-table concern.

**Next up**

Phase 2 — `stages/features.py` (PySpark): price/size premiums, rolling 7-day
premium, pre-drop search signal, brand broadcast join, output partitioned by
brand.

---

## 2026-06-27 — Phase 0: project scaffold

**What I built**

The full repo skeleton for ml-pipeline: `stages/` (ingest, features, validate,
train, register — CLI stubs to be filled in per phase), `dags/`, `config/`,
`models/`, `tests/`, `infra/`, `docs/`, and a GitHub Actions workflow. Two
pieces are real, not stubs: the shared I/O helper (`stages/io.py`) and the
config loader (`stages/config.py`), plus a real round-trip test for the I/O
layer (`tests/test_io.py`). Packaging (`pyproject.toml` as source of truth,
generated `requirements.txt`), `.env.example`, `.gitignore`, `Makefile`,
multi-stage `Dockerfile`, and infra compose scaffolds for Airflow, MLflow, and
(optional) MinIO.

**Why I made these decisions**

Three decisions were settled up front because they ripple across every later
phase, and retrofitting them would be expensive:

1. **One fsspec-resolved I/O seam instead of per-stage boto3.** Every stage
   reads/writes through `stages/io.py`, which resolves the filesystem from the
   URI scheme — `s3://…` in prod, a local path in dev/CI — using `pyarrow.fs`.
   The payoff is that two of the project's own constraints ("stages runnable
   independently from the CLI" and "CI must never touch real S3") are satisfied
   by the *same* code path that runs in production. Fixtures aren't a special
   case; they flow through the real I/O layer. The honest tradeoff: the local
   path doesn't exercise true S3 semantics (eventual consistency, multipart,
   IAM). I accept that for zero-setup, fast CI, and provide an optional MinIO
   compose file for anyone who wants a faithful local S3 API. `storage_root` is
   the single knob that switches environments, so no stage hardcodes a path.

2. **Python 3.11, `pyproject.toml` as source of truth, split dependency
   extras.** 3.11 is the highest version Spark, Ray, Airflow, and PyTorch all
   support without edge cases. The dependencies are split into `spark`,
   `train`, `airflow`, and `dev` extras so that, for example, the CI lint +
   DAG-import job installs only `airflow,dev` and never has to resolve
   Spark/Ray/Torch just to confirm the DAG imports. Airflow is pinned with its
   official constraints file (the one dependency that fights back otherwise).
   `requirements.txt` is kept as a generated lockfile to satisfy the scaffold
   spec, but pyproject is authoritative.

3. **Fixture-first, real-but-gated Postgres ingest.** The ingest stage will be
   real Postgres→Parquet code, but it's gated behind `SNEAKER_INTEL_DSN`. With
   the env var unset, the working dataset is synthetic fixtures, and that's the
   default for dev and all of CI. "The pipeline runs end-to-end on synthetic
   data with one command, and points at the real warehouse via one env var" is
   both more honest and a stronger demo than requiring live DB credentials to
   see anything work.

**What I learned / noted for later**

Fixtures have to be schema-faithful *and* statistically plausible — enough sales
history per shoe that `rolling_7d_avg_premium` populates, realistic premium
distributions — or the Phase 3 Pandera statistical checks become theater. I'll
also produce one deliberately-bad fixture for the validation-failure test.

**Next up**

Phase 1 — implement `stages/ingest.py` (real Postgres export + S3 raw landing)
and `generate_fixtures.py`.
