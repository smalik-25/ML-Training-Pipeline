"""Stage 7 -- Drift monitoring.

Compares the current run's feature distributions against the distribution the
deployed model was trained on, and decides whether the data has drifted enough
to justify retraining. This is the monitoring half of the loop: train and
register produce a deployed model, score and serve consume it, and this stage
watches for the world moving out from under it.

Drift metric is PSI (Population Stability Index), the standard industry measure.
Per feature: PSI < 0.1 is stable, 0.1-0.2 is a moderate shift, > 0.2 is a
significant shift. It's numpy-only, so this whole stage is torch-free and
testable without the heavy frameworks.

Reference distribution: by default the validated features of the run that
produced the current staging model (resolved from the MLflow registry). If there
is no staging model yet, there's no baseline to compare against, so the stage
reports drift = True -- meaning "train the first model". You can also pass an
explicit ``--reference-run-date``.

CLI:
    python -m stages.monitor --run-date 2026-07-01 --reference-run-date 2026-06-28
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from stages.config import PipelineConfig, load_config
from stages.io import read_parquet, write_json
from stages.train import FEATURE_COLUMNS

logger = logging.getLogger("stages.monitor")

DEFAULT_PSI_THRESHOLD = 0.2


def compute_psi(
    reference: np.ndarray, current: np.ndarray, bins: int = 10
) -> float:
    """Population Stability Index between two 1-D samples.

    Bins are set from the reference quantiles (equal-frequency), with open tails
    so out-of-range current values still land in an edge bin. Returns 0.0 for a
    degenerate reference (a constant feature), where PSI is undefined.
    """
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:  # constant reference -> no measurable distribution
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    eps = 1e-6
    ref_frac = np.clip(np.histogram(reference, edges)[0] / len(reference), eps, None)
    cur_frac = np.clip(np.histogram(current, edges)[0] / len(current), eps, None)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_columns: list[str] = FEATURE_COLUMNS,
    psi_threshold: float = DEFAULT_PSI_THRESHOLD,
) -> dict:
    """Per-feature PSI drift report. ``drifted`` is True if any feature exceeds
    the threshold."""
    features = []
    for col in feature_columns:
        psi = compute_psi(
            reference[col].to_numpy(dtype=float),
            current[col].to_numpy(dtype=float),
        )
        features.append(
            {"feature": col, "psi": round(psi, 5), "drifted": psi > psi_threshold}
        )
    n_drifted = sum(f["drifted"] for f in features)
    return {
        "psi_threshold": psi_threshold,
        "n_reference": int(len(reference)),
        "n_current": int(len(current)),
        "features": features,
        "n_drifted_features": int(n_drifted),
        "drifted": n_drifted > 0,
    }


def _staging_reference_run_date() -> str | None:
    """Resolve the run_date the current staging model was trained on, if any."""
    import mlflow
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    from stages.inference import MODEL_NAME, STAGING_ALIAS
    from stages.train import resolve_tracking_uri

    uri = resolve_tracking_uri()
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(uri)
    try:
        version = client.get_model_version_by_alias(MODEL_NAME, STAGING_ALIAS)
        return client.get_run(version.run_id).data.params.get("run_date")
    except MlflowException:
        return None


def run(
    config: PipelineConfig,
    reference_run_date: str | None = None,
    psi_threshold: float = DEFAULT_PSI_THRESHOLD,
    feature_columns: list[str] | None = None,
    excluded_reasons: dict[str, str] | None = None,
) -> tuple[str, bool]:
    """Compute drift for the run_date's features. Returns (report_uri, drifted).

    If no reference is given and there is no staging model, reports drift = True
    (there's no baseline, so the first model should be trained).

    ``feature_columns`` restricts the PSI check to a subset. A current-only source
    (the KicksDB drift path) can't honestly compute every feature, so it passes
    the subset it can and the rest are recorded as excluded rather than PSI'd on
    degenerate data. Defaults to all of ``FEATURE_COLUMNS``.

    ``excluded_reasons`` maps an excluded feature to why it can't be measured from
    this source; the report carries those reasons so the coverage gap is explained,
    not just counted. Features without a listed reason get a generic one.
    """
    if reference_run_date is None:
        reference_run_date = _staging_reference_run_date()

    columns = list(feature_columns) if feature_columns else list(FEATURE_COLUMNS)
    excluded = [c for c in FEATURE_COLUMNS if c not in columns]

    current = read_parquet(config.validated_uri())

    if reference_run_date is None:
        report = {
            "run_date": config.run_date,
            "reference_run_date": None,
            "generated_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
            "drifted": True,
            "reason": "no staging model to compare against; train a baseline",
        }
        logger.warning("no reference model; reporting drift=True (train baseline)")
    else:
        reference_config = replace(config, run_date=reference_run_date)
        reference = read_parquet(reference_config.validated_uri())
        report = drift_report(
            reference, current, feature_columns=columns, psi_threshold=psi_threshold
        )
        if excluded:
            reasons = excluded_reasons or {}
            report["excluded_features"] = {
                c: reasons.get(c, "not computable from this source")
                for c in excluded
            }
            logger.info(
                "drift excludes %d feature(s) not computable from this source: %s",
                len(excluded),
                ", ".join(
                    f"{c} ({report['excluded_features'][c]})" for c in excluded
                ),
            )
        report.update(
            run_date=config.run_date,
            reference_run_date=reference_run_date,
            generated_at=datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        )
        logger.info(
            "drift: %d/%d features over PSI %.2f (drifted=%s) vs reference %s",
            report["n_drifted_features"], len(columns), psi_threshold,
            report["drifted"], reference_run_date,
        )

    report_uri = write_json(report, config.drift_report_uri())
    logger.info("drift report -> %s", report_uri)
    return report_uri, bool(report["drifted"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect feature drift (PSI).")
    parser.add_argument("--run-date", required=True, help="ISO run date (YYYY-MM-DD).")
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    parser.add_argument(
        "--reference-run-date",
        default=None,
        help="Baseline run_date; defaults to the staging model's training date.",
    )
    parser.add_argument(
        "--psi-threshold", type=float, default=DEFAULT_PSI_THRESHOLD,
        help="Per-feature PSI above which a feature is considered drifted.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    config = load_config(args.config, run_date=args.run_date)
    _, drifted = run(config, args.reference_run_date, args.psi_threshold)
    logger.info("monitoring stage complete, drifted=%s", drifted)


if __name__ == "__main__":
    main()
