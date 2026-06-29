# Architecture

An offline ML training pipeline that predicts sneaker resale **price premium**
from StockX sales. This document covers the topology, what the data looks like at
each stage boundary, and — most importantly — *why* the design is the way it is.
The model is deliberately simple; the pipeline patterns are the point.

## Pipeline topology

```mermaid
flowchart TD
    subgraph src[Source]
      PG[(sneaker-intel Postgres<br/>star schema)]
      FX[synthetic fixtures<br/>data/fixtures/]
    end
    PG -- "SNEAKER_INTEL_DSN set" --> I
    FX -- "default dev/CI" --> I
    I[ingest<br/>anti-corruption layer] --> R[(S3/local raw<br/>raw/{run_date}/*.parquet)]
    R --> F[features<br/>PySpark]
    F --> FE[(features/{run_date}/<br/>partitioned by brand)]
    FE --> V[validate<br/>Pandera]
    V --> VD[(validated/{run_date}/<br/>+ validation_report.json)]
    VD --> T[train<br/>Ray Train + PyTorch]
    T --> M[(models/{run_date}/{run_id}/<br/>model.pt)]
    T --> ML[(MLflow tracking)]
    M --> RG[register<br/>MLflow registry]
    ML --> RG
    RG --> AL[sneaker-price-model<br/>@staging alias]
    AIR{{Airflow DAG}} -. "orchestrates; XCom path passing" .-> I
    AIR -.-> F
    AIR -.-> V
    AIR -.-> T
    AIR -.-> RG
```

Each stage is independently runnable from the CLI and communicates with the next
only through Parquet at an S3 (or local) path. The Airflow DAG is the source of
truth for pipeline topology; it orchestrates the stages but contains none of
their logic.

## Data at each prefix

All paths derive from `storage_root` in `config/pipeline_config.yaml` via
`PipelineConfig` (`stages/config.py`); nothing hardcodes a path.

| Prefix | Written by | Contents |
|---|---|---|
| `raw/{run_date}/` | ingest | One canonical Parquet per source table: `sales`, `shoes`, `drops`, `search_interest` |
| `features/{run_date}/` | features | Flat engineered feature table, Hive-partitioned by `brand` |
| `validated/{run_date}/` | validate | Validated features + `validation_report.json` |
| `models/{run_date}/{run_id}/` | train, register | `model.pt` (weights + preprocessing) + `promotion_report.json` |
| `failures/{run_date}/` | DAG callback | Per-task failure summaries on any task failure |

The model artifact also goes to the MLflow tracking store (SQLite locally, or the
Postgres+S3 server via `MLFLOW_TRACKING_URI`), and the registry holds the
versioned model and the `staging` alias.

### Features computed

Target is `price_premium = (sale_price - retail_price) / retail_price`. The
feature stage extends the sneaker-intel dbt `int_sales_enriched` logic on the
Spark compute layer: `days_since_release`, `size_premium` (a size's premium
relative to its shoe's baseline), `release_type_encoded` (ordinal scarcity),
`rolling_7d_avg_premium` (time-range window, null when a shoe has < 7 days of
history), `search_index_7d_pre_drop` (Google Trends demand before the drop), and
`brand_avg_premium` (broadcast-joined per-brand mean).

## Ingest as an anti-corruption layer

The real sneaker-intel warehouse does not match the pipeline's canonical schema
1:1 — it uses `shoe_key`/`sold_price`/`sold_date`/`size`/`interest`/`point_date`,
and per-release `retail_price`/`release_date`/`release_type` live in `dim_drops`,
not `dim_shoes`. Rather than leak those names into five stages, **ingest is the
single seam that knows the source schema.** Its SQL aliases every column, casts
Postgres `numeric` to float, and rolls each shoe's canonical (earliest) drop up
onto the shoe row. The Parquet it lands is byte-compatible with the synthetic
fixtures, so features/validate/train/register are oblivious to where the data
came from. Fixtures flow through the identical downstream code, which is what
makes dev/CI faithful without a database.

## Design decisions

**Parquet at every stage boundary.** Columnar, typed, splittable; supports
predicate pushdown and partition pruning. Never CSV. Timestamps are coerced to
microseconds at the I/O layer so the lake is readable by both pyarrow (the
Python stages) and Spark 3.5 (which rejects nanosecond Parquet).

**One fsspec-resolved I/O layer.** Every stage reads/writes through `stages/io.py`,
which resolves the filesystem from the URI scheme — `s3://` in prod, a local path
in dev/CI. Identical code in both, so fixtures exercise the real I/O path and CI
needs zero S3 setup. `storage_root` is the one knob that switches environments.

**Stages CLI-callable independently of Airflow.** Airflow orchestrates stages;
it does not define them. Inter-stage state passes only via paths (XCom in the
DAG), never shared local state. The DAG's tasks are thin wrappers that call each
stage's `run()`, and they import the heavy libraries lazily so DAG parsing stays
fast and the graph is testable without Spark/Torch installed.

**Config is the single source of truth for paths.** No stage hardcodes a path;
each asks `PipelineConfig`. The DAG threads concrete artifact URIs through XCom
for lineage/observability, but the stages derive their paths from config — so the
single-source-of-truth invariant holds while the XCom pattern is still
demonstrated.

**Temporal train/val split, not random.** Train on sales before the split year,
validate on/after it. A random split leaks the future (the model would see a
shoe's later sales while predicting its earlier ones), inflating metrics that
wouldn't survive production. The split year is configurable (`--split-year`)
because it depends on the data's date range — 2023 for the synthetic 2021–2024
fixtures, 2019 for the real 2017–2019 StockX data.

**Ray Train even on a single machine.** The loop runs inside
`ray.train.torch.TorchTrainer` with `prepare_model`/`prepare_data_loader`/
checkpointing, not a raw loop or `DataParallel`. On one machine it behaves
normally, but scaling to N workers/GPUs is a `ScalingConfig` change, not a
rewrite. Preprocessing (imputation + standardization) is fit on the train split
only and saved inside `model.pt`, so inference reproduces the exact pipeline with
no leakage.

**Pandera over ad-hoc validation.** Declarative checks double as a data contract;
lazy validation reports every failing column/check/row count in one pass. The
rolling-null rule is a custom cross-row check; outliers are flagged, not dropped;
a ≥90% row-retention check guards the stage boundary. This is the seam that
caught the real-data surprises (see below).

**MLflow: SQLite locally, registry via aliases.** The legacy file store is
deprecated in MLflow 3.x and can't back the registry, so the stages default to a
SQLite store (which backs both tracking and the registry) and use a `staging`
*alias* (stages are deprecated) for the promoted version. Promotion is strict: a
model is always registered for lineage, but the alias moves only if the candidate
beats the incumbent's val RMSE; a worse model warns instead of failing.

## What real data changed (and what caught it)

Connecting the live warehouse surfaced four assumptions the synthetic data had
quietly gotten wrong — each caught loudly by validation and fixed with a
documented config/contract change, never a silent patch:

- **`release_type` vocabulary** is `{general, limited, collab}`, not the synthetic
  `raffle/fcfs/...`. Fixed in `feature_config.yaml`; validation's `isin` set
  derives from it.
- **Premium outliers**: real Off-White resale reached ~20.3× retail, which would
  have failed the original `[-1, 20]` bound. Ceiling raised to 50 (>50× = data
  error).
- **Transaction grain**: real resale legitimately has many sales of the same
  shoe/size/day, so the `(shoe, date, size)` uniqueness check was dropped in
  favor of `sale_id` (the true key).
- **Pre-release sales**: ~5.6% of sales traded before the official drop (min −69
  days). `days_since_release_min` is now a config value set to −90 — a bounded
  pre-release window that still flags grossly wrong dates.

On the full 99,956-row StockX dataset the pipeline ran end to end: validation
passed at 100% retention, and the model trained to val RMSE ≈ 0.21 with an
84/16 temporal split at 2019, registered and promoted as v1.
