"""Live scoring of current sneakers pulled from KicksDB, through the one
inference path.

Phase 3 of the KicksDB integration. Where ``stages/score.py`` batch-scores
validated features, this scores TODAY's sneakers: it pulls a current KicksDB
snapshot via the Phase 1 adapter, builds the engineered features a single
snapshot can support, and scores them with the deployed ``@staging`` model
through ``stages.inference`` -- the exact impute + standardize + forward transform
the batch path uses. There is no second model transform; the only KicksDB-specific
step is turning a current product into a feature row.

The retail-price decision bites hardest here. The adapter only lands products with
a resolvable retail, so a requested SKU with none is reported as uncomputable
rather than scored on a guessed number. And the premium-derived features a lone
snapshot can't support (the rolling average, the pre-drop search signal) are left
null and imputed with the model's training means, exactly as the model handles a
missing signal anywhere else.

These sneakers are out-of-distribution: the model learned on 2017-2019 Off-White
and Yeezy StockX sales, and this scores current releases. The Phase 2 drift numbers
quantify that gap; this shows the model doing it anyway, honestly labeled.

CLI (needs a model; point MODEL_URI at a model.pt, or have a @staging model):
    MODEL_URI=demo/model.pt python -m stages.live_score --run-date 2026-07-04
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import pandas as pd

from stages import inference, kicksdb
from stages.config import (
    FeatureConfig,
    PipelineConfig,
    load_config,
    load_feature_config,
)

logger = logging.getLogger("stages.live_score")

# Columns carried alongside the prediction for a readable, traceable result.
IDENTITY_COLUMNS = ["shoe_id", "brand", "model", "colorway", "retail_price", "sale_price"]


def build_snapshot_features(
    frames: dict[str, pd.DataFrame], release_type_order: dict[str, int]
) -> pd.DataFrame:
    """Turn KicksDB canonical raw frames into the model's engineered features.

    This is the snapshot counterpart to the Spark feature stage, using the same
    definitions (premium, days_since_release, ordinal release_type) but for a
    current-only snapshot. The features a snapshot can't support are made explicit:
    ``size_premium`` is 0 (one representative size), and ``rolling_7d_avg_premium``
    / ``search_index_7d_pre_drop`` are null and imputed downstream by the shared
    transform. ``brand_avg_premium`` is the mean premium across the current set,
    exactly as the Spark stage aggregates it over a batch.
    """
    df = frames["sales"].merge(frames["shoes"], on="shoe_id", how="inner")
    premium = (df["sale_price"] - df["retail_price"]) / df["retail_price"]

    df["days_since_release"] = (
        pd.to_datetime(df["sale_date"]) - pd.to_datetime(df["release_date"])
    ).dt.days.astype("int64")
    df["release_type_encoded"] = (
        df["release_type"].map(release_type_order).astype("int64")
    )
    df["size_premium"] = 0.0
    df["rolling_7d_avg_premium"] = np.nan
    df["search_index_7d_pre_drop"] = np.nan
    df["brand_avg_premium"] = premium.groupby(df["brand"]).transform("mean")
    # size_us and retail_price already come from the sales/shoes frames.
    return df


def score_snapshot(
    frames: dict[str, pd.DataFrame],
    bundle: inference.ModelBundle,
    release_type_order: dict[str, int],
) -> pd.DataFrame:
    """Score a reshaped KicksDB snapshot through the shared inference transform."""
    feats = build_snapshot_features(frames, release_type_order)
    preds = inference.predict(bundle, feats)  # THE shared impute+standardize+forward

    out = feats[IDENTITY_COLUMNS].copy()
    out["predicted_premium"] = preds
    out["implied_resale"] = out["retail_price"] * (1.0 + preds)
    return out


def uncomputable_skus(
    requested: list[str], reference: kicksdb.RetailReference
) -> list[str]:
    """Requested SKUs with no retail in the reference -> premium uncomputable.

    Reported honestly rather than scored on a fabricated retail. This is the
    Phase 0 decision surfacing at inference time.
    """
    return [s for s in requested if reference.lookup(s) is None]


def score_current(
    config: PipelineConfig,
    fcfg: FeatureConfig,
    bundle: inference.ModelBundle | None = None,
    api_key: str | None = None,
    requested_skus: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch a current KicksDB snapshot and score it at ``@staging``.

    Live when ``api_key`` (or ``KICKSDB_API_KEY``) is set, else canned fixtures.
    ``bundle`` defaults to the model at ``MODEL_URI`` or the registry staging alias.

    ``requested_skus`` names the sneakers a caller wants scored; it defaults to the
    whole curated reference (the full current snapshot). A requested SKU with no
    retail in the reference has an uncomputable premium (the Phase 0 decision), so
    it is reported rather than scored on a guessed number. That list is logged and
    attached as ``results.attrs["uncomputable_skus"]``; the scored frame holds only
    the computable requested sneakers.
    """
    api_key = api_key or os.environ.get("KICKSDB_API_KEY") or None
    reference = kicksdb.load_retail_reference()
    requested = list(requested_skus) if requested_skus is not None else reference.skus()

    skipped = uncomputable_skus(requested, reference)
    if skipped:
        logger.warning(
            "%d of %d requested SKU(s) have no retail in the reference; premium is "
            "uncomputable, so they are reported and not scored: %s",
            len(skipped), len(requested), skipped,
        )

    frames = kicksdb.read_all(config.run_date, api_key=api_key)
    if bundle is None:
        bundle = _load_bundle()

    results = score_snapshot(frames, bundle, fcfg.release_type_order)

    # When the caller narrows the request, keep only the computable requested
    # shoes; the default request is the whole reference, so nothing is dropped.
    if requested_skus is not None:
        wanted_ids = {
            kicksdb.stable_shoe_id(kicksdb.normalize_sku(entry.sku))
            for sku in requested
            if (entry := reference.lookup(sku)) is not None
        }
        results = results[results["shoe_id"].isin(wanted_ids)].reset_index(drop=True)

    results.attrs["uncomputable_skus"] = skipped
    logger.info(
        "scored %d current KicksDB sneakers at model version=%s (out-of-distribution "
        "vs the 2017-2019 training slice); %d requested SKU(s) uncomputable",
        len(results), bundle.model_version, len(skipped),
    )
    return results


def _load_bundle() -> inference.ModelBundle:
    """Load from MODEL_URI if set, else the registry staging alias (like serving)."""
    model_uri = os.environ.get("MODEL_URI")
    if model_uri:
        return inference.load_bundle_from_uri(model_uri)
    return inference.load_staging_bundle()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score current KicksDB sneakers with the staging model."
    )
    parser.add_argument("--run-date", required=True, help="ISO run date (snapshot date).")
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    parser.add_argument("--feature-config", default="config/feature_config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    config = load_config(args.config, run_date=args.run_date)
    fcfg = load_feature_config(args.feature_config)

    results = score_current(config, fcfg)
    for row in results.itertuples(index=False):
        name = f"{row.brand} {row.model}"
        logger.info(
            "  %-24s %-20s retail=$%-6.0f -> premium %+.0f%% (resale ~$%s)",
            name, row.colorway, row.retail_price,
            row.predicted_premium * 100, f"{row.implied_resale:,.0f}",
        )


if __name__ == "__main__":
    main()
