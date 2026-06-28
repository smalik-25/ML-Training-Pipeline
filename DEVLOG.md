# DEVLOG

Running narrative of what was built and — more importantly — *why*. Newest
entries at the top.

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
