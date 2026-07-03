# sneaker-intel · Phase 2: an offline ML training platform

**A project report.** This is the second half of a two-part project. The first
half, [sneaker-intel](https://github.com/smalik-25/sneaker-intel), built the data
engineering foundation: a hand-written Postgres star schema over ~99K StockX
resale sales, with a full dbt transformation layer and a live dashboard. It
shipped with the ML layer explicitly deferred, called out in its own roadmap as a
Phase 2 to come back to. This repo is that Phase 2. It builds the ML
infrastructure on top of the warehouse: ingest, feature engineering, a data
contract, distributed training, a model registry, batch and online serving, and
drift-driven retraining, all orchestrated and run end to end on the real data.

- **Live demo:** https://huggingface.co/spaces/smalik25/sneaker-ml-platform
- **Code:** https://github.com/smalik-25/ML-Training-Pipeline
- **Phase 1 (sneaker-intel) dashboard:** https://sneaker-intel-2.streamlit.app/

## The thesis

The model predicts one number: a sneaker's resale **price premium** over retail.
It's a plain feedforward network, and it's simple on purpose. The point of the
project was never the model. It was everything around it, the parts that decide
whether an ML system is reliable, reproducible, and able to scale. I wanted every
architectural choice to survive a "why did you do it that way" question, so I
wrote the reasons down as I made them (`DEVLOG.md`), and I built the pipeline so
that each decision could be pointed at and defended.

The through-line, repeated in every stage: stages are independent and
CLI-runnable, config is the single source of truth for paths, the data contract
is explicit and fails loudly, and the same code runs against a laptop and against
S3.

## Why two repos

sneaker-intel and this project are one story told in two parts, and I think the
sequencing is the most honest thing about it. In sneaker-intel I built the
warehouse first and shipped it, and I deliberately left the ML layer out of
scope, saying so in the roadmap rather than half-building it. Coming back to build
that layer as a separate, documented phase is the discipline the arc demonstrates:
knowing what to defer, and then actually returning to finish it. On a resume that
reads as one multi-phase project, not two disconnected ones.

## Architecture

```
sneaker-intel Postgres  ─┐
synthetic fixtures       ├─▶ ingest ─▶ raw ─▶ features ─▶ validated ─▶ train ─▶ models
(default dev/CI)         ┘   (anti-      (Parquet)  (PySpark)   (Pandera)   (Ray +   (+ MLflow
                             corruption)                                    PyTorch)  registry)
                                                                               │
                        register ◀───────────────────────────────────────────┘
                          │  promotes to @staging on a strict improvement
                          ├─▶ score   (batch, reads @staging)   ─▶ predictions
                          └─▶ serving (FastAPI, reads @staging)  ─▶ /predict

Airflow orchestrates the training DAG; a scheduled monitor DAG computes drift
and triggers a retrain only when the data has moved.
```

Every stage talks to the next only through Parquet at a path. Nothing is passed
in memory between stages, and nothing shares local state. That is what lets each
stage run on its own from the CLI, in a test, or as an Airflow task, with the same
code.

**Tech stack:** PySpark, Pandera, PyTorch, Ray Train, MLflow, Apache Airflow, AWS
S3 + Parquet, Docker, GitHub Actions, FastAPI, Streamlit, Terraform. Python 3.11.

## The seams that make it work

A few design decisions do most of the load-bearing work, and they were settled
early because retrofitting them would have been expensive.

**One fsspec-resolved I/O layer.** Every stage reads and writes through
`stages/io.py`, which picks the filesystem from the URI scheme: `s3://` in
production, a local path in dev and CI. The same code runs in both, so the
synthetic fixtures exercise the real I/O path rather than a mock, and CI never
touches real S3. Switching the whole pipeline to the cloud is one variable,
`STORAGE_ROOT=s3://<bucket>`. I later proved this against the actual S3 API in CI
using an in-process moto server, so it isn't an untested claim.

**Config is the single source of truth.** No stage hardcodes a path. Each asks
`PipelineConfig` for the URI it needs, and every URI is built from one
`storage_root`. The Airflow DAG threads the concrete artifact URIs through XCom
for lineage, but the stages still derive their paths from config, so the
invariant holds while the pattern is demonstrated.

**Ingest is an anti-corruption layer.** This turned out to be the decision the
whole project was quietly betting on (see the real-data section). Ingest is the
one place that knows what the source warehouse actually looks like. Its SQL
aliases every column, casts types, and reshapes the source star schema into the
canonical schema the rest of the pipeline expects, so features, validate, train,
register, and score never learn where the data came from.

## The stages

- **ingest.** Exports the four source tables to raw Parquet, either from the live
  sneaker-intel Postgres (gated behind `SNEAKER_INTEL_DSN`) or from synthetic
  fixtures (the default for dev and CI). Both modes write byte-identical layouts.
- **features (PySpark).** Computes the premium economics: price premium, days
  since release, size premium, an ordinal release-type encoding, a time-range
  7-day rolling average premium with deliberate null semantics, a pre-drop demand
  signal from Google Trends, and a broadcast-joined per-brand average premium.
  Output is partitioned by brand so brand-filtered training reads one partition.
- **validate (Pandera).** A typed data contract. Schema and dtype checks plus
  statistical ones: premium bounds, non-negative age (within a bounded
  pre-release window), a custom cross-row check that the rolling average is null
  exactly when a shoe has too little history, and a ≥90% row-retention check that
  guards the stage boundary. Lazy validation reports every failing column, check,
  and row count at once, and the stage fails loudly rather than pass bad data on.
- **train (Ray Train + PyTorch + MLflow).** A temporal train/val split so the
  model is always judged predicting forward in time. The loop runs inside Ray's
  `TorchTrainer` so scaling to more workers or GPUs is a config change, not a
  rewrite. Preprocessing is fit on the train split only and saved inside
  `model.pt`. Every run logs hyperparameters, per-epoch losses, and the final
  metric to MLflow.
- **register (MLflow).** Registers every run for lineage, but moves the
  `staging` alias only if the candidate strictly beats the currently-deployed
  model's val RMSE. A worse model logs a warning instead of failing. Every
  decision writes a `promotion_report.json`.
- **score.** Batch inference: loads whatever model sits at `@staging` and scores
  a batch, writing predictions and a report.
- **serving (FastAPI).** The online counterpart. `/health`, `/predict`,
  `/predict/batch`, and a `/reload` that picks up an alias change without a
  restart. The model loads at startup, so a broken model fails a readiness check
  rather than surfacing as a 500.
- **monitor.** Computes per-feature PSI (Population Stability Index) against the
  distribution the deployed model was trained on.

**One inference path.** The scoring stage and the serving API both load the
`@staging` model through the same `stages/inference.py` and apply the exact
preprocessing saved in `model.pt`. There is no second implementation of the
transform to drift, so a batch row and an online request get byte-identical
treatment. That is the no-training/serving-skew guarantee made structural instead
of asserted.

## Orchestration

Two Airflow DAGs. The **training DAG** chains ingest → features → validate →
train → register → score. Tasks are thin: each one builds the run's config and
calls a stage's `run()`, and the heavy libraries import inside the callables, so
DAG parsing stays fast and the graph is testable without Spark or Torch
installed. A validation failure propagates and fails the whole run, and every
task carries a failure callback that writes a summary to a `failures/` prefix.

The **drift-monitor DAG** runs on a schedule: ingest → features → validate →
detect_drift → (short-circuit) → trigger the training DAG. Retraining is driven by
evidence, not a cron. If no feature has drifted past the threshold, it
short-circuits and nothing retrains. The reference distribution is resolved from
the registry, so "has the data drifted" always means "drifted from what the live
model actually learned."

## The real-data reconciliation (the part that mattered)

Everything up to this point ran on synthetic fixtures I designed myself, which is
circular: of course my code passes data I shaped to fit it. The project's real
claim, "production patterns, runs on real data," was unproven until I connected
the actual sneaker-intel Postgres. When I did, the real schema was materially
different from the idealized one I had built five stages against: different keys,
different column names, and the per-release retail price and release date living
in a different table than I assumed.

That was the moment the whole thing was at risk. If my abstractions had leaked the
schema everywhere, this is a five-stage rewrite and the "one env var from the
warehouse" story is a lie. Instead I absorbed the entire mismatch in one place,
ingest's SQL, as an anti-corruption layer, and nothing downstream changed. The bet
on the abstractions paid off, and it only got tested at the very end.

The quieter risk was the data itself. Real data had problems the clean fixtures
never would, and the validator caught each one loudly rather than letting it
poison the model silently:

| What real data surfaced | The fix |
|---|---|
| `release_type` vocabulary was `{general, limited, collab}`, not the synthetic set | Updated the config; validation's allowed set derives from it |
| Premium outliers reached ~20.3× retail, over the original `[-1, 20]` ceiling | Raised the ceiling to 50; over 50× is still a data error |
| Real resale legitimately repeats `(shoe, date, size)` | Dropped that uniqueness assumption; `sale_id` is the true key |
| ~5.6% of sales (5,601 rows) traded before the official drop | Made a bounded pre-release window a config value (−90 days) |

Each became a documented config change, not a silent patch. On the full run over
**99,956 StockX sales** (Off-White and Yeezy, 2017–2019), validation passed at
**100% row retention**, and the model trained to **val RMSE ≈ 0.21** on the 0–20
premium scale, using an 84/16 temporal split at 2019, then registered and promoted
to `@staging` as v1.

## Serving, deployment, and the demo

**Serving** is live locally through FastAPI, with the alias-as-source-of-truth
lifecycle handled properly: startup load with a real readiness signal, and
`/reload` for a promotion or rollback.

**AWS deployment** is written as Terraform under `infra/terraform/`: an S3
data-lake bucket, an ECR repo, and an App Runner service that serves the model
from S3 with a least-privilege instance role (`s3:GetObject` on `models/*` only,
no keys in the image). A dedicated serving image (CPU-only torch) and a
`deploy-serving` GitHub Actions workflow build and push it. I chose App Runner
over ECS/Fargate for a first deploy because it is a real, cost-modest managed
container host with far less to wire up. `DEPLOY.md` is the full runbook, cost and
teardown included.

**The live demo** is a Streamlit dashboard on Hugging Face Spaces (free, no card,
enough memory for CPU torch). It runs the real trained model behind an interactive
predictor, alongside a model card, the pipeline topology, and the real-run
monitoring numbers. It's styled to my own design system.

## Testing and CI

Over fifty tests, split so each CI job installs only what it needs: a lint job, a
core unit job (I/O including the S3-over-moto round-trip, ingest, validation,
inference, drift), a Spark job with a JRE, a Torch/Ray/MLflow job for training,
registry, scoring, and serving, and a DAG-integrity job with Airflow pinned to its
constraints file. Heavy-framework tests are guarded so nothing breaks at
collection when a framework is absent. CI never touches real S3, the real
Postgres, or the committed data. Twice, CI caught real portability bugs a
same-environment run would have missed (a read-only-array issue on newer pandas,
and a missing server dependency for the moto test), which is exactly what the
split is for.

## What this demonstrates

| Concern | Where it shows up |
|---|---|
| Training-data pipelines | S3 data lake + PySpark feature engineering |
| Distributed training | Ray `TorchTrainer` wrapping PyTorch |
| Workflow orchestration | Airflow DAG with XCom, retries, SLA, failure callbacks |
| Dataset validation & reproducibility | Pandera contract; MLflow tracking + registry with gated promotion |
| Data platforms / lakes | S3 as lake, Parquet throughout, partitioned by brand |
| Serving & MLOps loop | one inference path, FastAPI, PSI drift, retrain-on-drift |
| Infra / deployment | Terraform (S3 · ECR · App Runner), Docker, split CI |
| Integrating with a real source | ingest as an anti-corruption layer over the live warehouse |

## Closing

The honest summary is that the design earned its keep in about a day of
reconciliation at the end, not across the weeks of building it. Everywhere the
synthetic assumptions were wrong, the validator caught it and the fix was a
documented config change. The model stays simple; the reliability,
reproducibility, and scale of the pipeline around it are the deliverable, and they
now run end to end on 99,956 real sales, from the warehouse sneaker-intel built.

Built in the ruins of the present. © Sam Malik.
