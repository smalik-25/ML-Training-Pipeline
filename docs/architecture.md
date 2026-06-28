# Architecture

> Scaffold. The data-flow narrative and full design-decisions write-up are
> completed in Phase 7; the structure and the decisions settled so far are
> recorded here and in [`../DEVLOG.md`](../DEVLOG.md).

## Pipeline topology

```mermaid
flowchart TD
    A[S3 raw data lake<br/>raw/{run_date}/*.parquet] --> B[PySpark feature engineering<br/>features/{run_date}/features.parquet]
    B --> C[Pandera validation<br/>validated/{run_date}/features_validated.parquet]
    C --> D[Ray Train + PyTorch<br/>models/{run_date}/{run_id}/model.pt]
    D --> E[MLflow registry<br/>sneaker-price-model -> Staging]
    F[Airflow DAG] -. orchestrates, passes S3 paths via XCom .-> A
    F -.-> B
    F -.-> C
    F -.-> D
    F -.-> E
```

## Data at each S3 prefix

| Prefix | Written by | Contents |
|---|---|---|
| `raw/{run_date}/` | ingest | One Parquet per source table: `sales`, `shoes`, `drops`, `search_interest` |
| `features/{run_date}/` | features | Flat engineered feature table, partitioned by `brand` |
| `validated/{run_date}/` | validate | Validated features + `validation_report.json` |
| `models/{run_date}/{run_id}/` | train | `model.pt` + MLflow-tracked artifacts |
| `failures/{run_date}/` | DAG callback | Per-task failure summaries |

All paths are derived from `storage_root` in `config/pipeline_config.yaml` via
`PipelineConfig` (`stages/config.py`); nothing hardcodes a path.

## Design decisions (running)

- **Parquet between stages** — columnar, typed, splittable; supports predicate
  pushdown and partition pruning. Never CSV.
- **Single fsspec-resolved I/O layer** — `s3://` vs local resolved by URI scheme;
  identical code in prod and CI; fixtures exercise the real path.
- **CLI-callable stages independent of Airflow** — orchestration and stage logic
  are decoupled; the DAG calls thin tasks that invoke stage functions.
- **Python 3.11 + split dependency extras** — broad framework compatibility; CI
  jobs install only what they need.

_To be completed in Phase 7: why temporal train/val split, why partition by
brand, why Ray Train on a single machine, why Pandera over ad-hoc validation,
why Google Trends over Reddit as the demand signal._
