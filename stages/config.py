"""Pipeline configuration.

``pipeline_config.yaml`` is the single source of truth for every storage path
in the pipeline. No stage hardcodes a path; each loads this config and asks it
for the URI it needs. Because all URIs are built from ``storage_root``, pointing
the whole pipeline at S3 (prod) or the local filesystem (dev/CI) is a one-line
config change -- the stage code is identical either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

from stages.io import path_join


@dataclass(frozen=True)
class PipelineConfig:
    """Resolved configuration for a single pipeline run.

    ``storage_root`` is the only thing that changes between environments:
        prod -> "s3://my-bucket/sneaker-intel"
        dev  -> "./data"
    """

    storage_root: str
    raw_prefix: str
    features_prefix: str
    validated_prefix: str
    models_prefix: str
    failures_prefix: str
    run_date: str

    def raw_uri(self, table: str) -> str:
        """URI for a raw landed table, e.g. raw_uri('sales')."""
        return path_join(
            self.storage_root, self.raw_prefix, self.run_date, f"{table}.parquet"
        )

    def features_uri(self) -> str:
        return path_join(
            self.storage_root,
            self.features_prefix,
            self.run_date,
            "features.parquet",
        )

    def validated_uri(self) -> str:
        return path_join(
            self.storage_root,
            self.validated_prefix,
            self.run_date,
            "features_validated.parquet",
        )

    def validation_report_uri(self) -> str:
        return path_join(
            self.storage_root,
            self.validated_prefix,
            self.run_date,
            "validation_report.json",
        )

    def model_uri(self, run_id: str) -> str:
        return path_join(
            self.storage_root, self.models_prefix, self.run_date, run_id
        )

    def failure_uri(self, task: str) -> str:
        return path_join(
            self.storage_root,
            self.failures_prefix,
            self.run_date,
            f"{task}.json",
        )


def load_config(config_path: str, run_date: str | None = None) -> PipelineConfig:
    """Load and resolve pipeline config from a YAML file.

    Args:
        config_path: path to pipeline_config.yaml.
        run_date: ISO date string (YYYY-MM-DD). Overrides the file default;
            falls back to the file's run_date, then to today.

    Raises:
        FileNotFoundError: if the config file does not exist.
        KeyError: if a required key is missing (we fail loudly).
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Pipeline config not found: {config_path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    resolved_run_date = run_date or raw.get("run_date") or date.today().isoformat()

    return PipelineConfig(
        storage_root=raw["storage_root"],
        raw_prefix=raw["raw_prefix"],
        features_prefix=raw["features_prefix"],
        validated_prefix=raw["validated_prefix"],
        models_prefix=raw["models_prefix"],
        failures_prefix=raw["failures_prefix"],
        run_date=str(resolved_run_date),
    )


@dataclass(frozen=True)
class FeatureConfig:
    """Feature engineering parameters (from feature_config.yaml).

    Kept separate from PipelineConfig so feature logic can be tuned without
    touching storage/orchestration config.
    """

    rolling_window_days: int
    search_signal_window_days: int
    release_type_order: dict[str, int]
    partition_by: str
    price_premium_min: float
    price_premium_max: float
    # Lower bound for days_since_release. 0 for the synthetic fixtures; negative
    # for real data, where pre-release ("early pair") sales legitimately trade
    # before the official drop. Defaulted so existing callers stay valid.
    days_since_release_min: int = 0


def load_feature_config(config_path: str) -> FeatureConfig:
    """Load feature engineering config from a YAML file.

    Raises:
        FileNotFoundError: if the config file does not exist.
        KeyError: if a required key is missing (we fail loudly).
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Feature config not found: {config_path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    return FeatureConfig(
        rolling_window_days=int(raw["rolling_window_days"]),
        search_signal_window_days=int(raw["search_signal_window_days"]),
        release_type_order=dict(raw["release_type_order"]),
        partition_by=str(raw["partition_by"]),
        price_premium_min=float(raw["price_premium_min"]),
        price_premium_max=float(raw["price_premium_max"]),
        days_since_release_min=int(raw.get("days_since_release_min", 0)),
    )
