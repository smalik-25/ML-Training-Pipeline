"""Stage 3 -- Dataset validation (Pandera).

A typed contract on the features Parquet that the training stage can trust. We
use Pandera's declarative ``Check`` API rather than ad-hoc ``if/else`` so the
checks are self-documenting, composable, and report *which* rows failed and how
many -- not just that something failed.

Checks enforced:
  * Schema: exact column set, dtypes (coerced), required fields non-null
    (price_premium, days_since_release, sale_date, ...).
  * price_premium in [-1.0, 20.0] -- flags outliers loudly; we never silently
    drop them.
  * days_since_release >= 0; retail_price/sale_price > 0; release_type and
    release_type_encoded within their allowed sets.
  * rolling_7d_avg_premium is null *iff* the shoe has < N days of history -- a
    custom cross-row check, not a plain null check, so a wrongly-populated or
    wrongly-nulled value is caught.
  * No duplicate (shoe_id, sale_date, size_us).
  * Row retention: the features output must keep >= 90% of the raw sales rows,
    guarding against silent data loss in the feature stage.

On failure the stage raises ``FeatureValidationError`` summarising every failed
check and the affected row counts -- the pipeline fails loudly rather than
passing bad data to training. On success it writes the validated Parquet and a
``validation_report.json`` alongside it.

CLI:
    python -m stages.validate --run-date 2025-01-01 \\
        --config config/pipeline_config.yaml
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from stages.config import (
    FeatureConfig,
    PipelineConfig,
    load_config,
    load_feature_config,
)
from stages.io import read_parquet, write_json, write_parquet

logger = logging.getLogger("stages.validate")

KEY_COLUMNS = ["shoe_id", "sale_date", "size_us"]
MIN_ROW_RETENTION_PCT = 90.0


class FeatureValidationError(Exception):
    """Raised when the features dataset fails validation. Fails the pipeline."""


def _rolling_null_consistency(window_days: int):
    """Wide check: rolling_7d_avg_premium null IFF history < window_days.

    Returns a row-aligned boolean Series (True = consistent), so Pandera reports
    exactly which rows violate the rule.
    """

    def check(df: pd.DataFrame) -> pd.Series:
        sale_date = pd.to_datetime(df["sale_date"])
        first_sale = sale_date.groupby(df["shoe_id"]).transform("min")
        days_of_history = (sale_date - first_sale).dt.days
        insufficient = days_of_history < window_days
        is_null = df["rolling_7d_avg_premium"].isna()
        return is_null == insufficient

    return check


def _unique_keys(df: pd.DataFrame) -> pd.Series:
    """Wide check: no duplicate (shoe_id, sale_date, size_us). Dup rows fail."""
    return ~df.duplicated(subset=KEY_COLUMNS, keep=False)


def build_schema(fcfg: FeatureConfig) -> pa.DataFrameSchema:
    """Build the Pandera schema for the features dataset."""
    release_types = list(fcfg.release_type_order.keys())
    encoded_ranks = list(fcfg.release_type_order.values())

    return pa.DataFrameSchema(
        columns={
            "sale_id": pa.Column(int, nullable=False, unique=True),
            "shoe_id": pa.Column(int, nullable=False),
            "size_us": pa.Column(float, pa.Check.gt(0), nullable=False),
            "sale_date": pa.Column("datetime64[ns]", nullable=False),
            "release_date": pa.Column("datetime64[ns]", nullable=False),
            "retail_price": pa.Column(float, pa.Check.gt(0), nullable=False),
            "sale_price": pa.Column(float, pa.Check.gt(0), nullable=False),
            "release_type": pa.Column(str, pa.Check.isin(release_types), nullable=False),
            "price_premium": pa.Column(
                float,
                pa.Check.in_range(
                    fcfg.price_premium_min, fcfg.price_premium_max
                ),
                nullable=False,
            ),
            "days_since_release": pa.Column(int, pa.Check.ge(0), nullable=False),
            "size_premium": pa.Column(float, nullable=False),
            "release_type_encoded": pa.Column(
                int, pa.Check.isin(encoded_ranks), nullable=False
            ),
            # Null is allowed but only with insufficient history (wide check below).
            "rolling_7d_avg_premium": pa.Column(float, nullable=True),
            "search_index_7d_pre_drop": pa.Column(float, nullable=True),
            "brand_avg_premium": pa.Column(float, nullable=False),
            "brand": pa.Column(str, nullable=False),
        },
        checks=[
            pa.Check(
                _rolling_null_consistency(fcfg.rolling_window_days),
                error=(
                    "rolling_7d_avg_premium must be null exactly when the shoe "
                    f"has < {fcfg.rolling_window_days} days of history"
                ),
            ),
            pa.Check(
                _unique_keys,
                error="duplicate (shoe_id, sale_date, size_us) combinations",
            ),
        ],
        strict=True,   # no unexpected columns
        coerce=True,   # normalise int32/datetime64[us]/category to declared types
    )


def validate_features(
    df: pd.DataFrame, fcfg: FeatureConfig
) -> tuple[pd.DataFrame, list[str]]:
    """Validate the features frame. Returns (validated_df, check_names).

    Raises:
        FeatureValidationError: with a per-check summary of failures.
    """
    schema = build_schema(fcfg)
    try:
        validated = schema.validate(df, lazy=True)
    except SchemaErrors as err:
        summary = _summarise_failures(err)
        logger.error("feature validation FAILED:\n%s", summary)
        raise FeatureValidationError(summary) from err

    check_names = [c.error or repr(c) for c in schema.checks]
    return validated, check_names


def _summarise_failures(err: SchemaErrors) -> str:
    """Build a human-readable per-check summary with affected row counts."""
    cases = err.failure_cases
    lines = ["feature validation failed:"]
    grouped = cases.groupby(["column", "check"], dropna=False).size()
    for (column, check), n in grouped.items():
        col = column if pd.notna(column) else "<dataframe>"
        lines.append(f"  - [{col}] {check}: {n} row(s) affected")
    lines.append(f"  total failure cases: {len(cases)}")
    return "\n".join(lines)


def run(config: PipelineConfig, fcfg: FeatureConfig) -> str:
    """Validate features, write validated Parquet + report. Returns validated URI."""
    features = read_parquet(config.features_uri())
    input_rows = len(read_parquet(config.raw_uri("sales")))
    output_rows = len(features)

    validated, check_names = validate_features(features, fcfg)

    retention_pct = 100.0 * output_rows / input_rows if input_rows else 0.0
    if retention_pct < MIN_ROW_RETENTION_PCT:
        msg = (
            f"row retention {retention_pct:.1f}% is below the "
            f"{MIN_ROW_RETENTION_PCT:.0f}% floor "
            f"(input={input_rows}, output={output_rows}) -- possible silent "
            "data loss in the feature stage"
        )
        logger.error(msg)
        raise FeatureValidationError(msg)

    out_uri = config.validated_uri()
    write_parquet(validated, out_uri)

    report = {
        "run_date": config.run_date,
        "validated_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "input_row_count": input_rows,
        "output_row_count": output_rows,
        "row_retention_pct": round(retention_pct, 2),
        "min_row_retention_pct": MIN_ROW_RETENTION_PCT,
        "rolling_null_rows": int(validated["rolling_7d_avg_premium"].isna().sum()),
        "checks": check_names,
        "status": "passed",
    }
    report_uri = config.validation_report_uri()
    write_json(report, report_uri)

    logger.info(
        "validation PASSED rows=%d retention=%.1f%% -> %s",
        output_rows, retention_pct, out_uri,
    )
    logger.info("validation report -> %s", report_uri)
    return out_uri


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate features with Pandera.")
    parser.add_argument("--run-date", required=True, help="ISO run date (YYYY-MM-DD).")
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    parser.add_argument("--feature-config", default="config/feature_config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    config = load_config(args.config, run_date=args.run_date)
    fcfg = load_feature_config(args.feature_config)
    uri = run(config, fcfg)
    logger.info("validation stage complete -> %s", uri)


if __name__ == "__main__":
    main()
