# DEVLOG

Running narrative of what was built and — more importantly — *why*. Newest
entries at the top.

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
