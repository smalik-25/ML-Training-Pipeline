"""Stage 2 -- Feature engineering (PySpark).

Reads the raw Parquet landed by the ingest stage, computes premium economics and
temporal/demand features, and writes a flat features Parquet partitioned by
brand. This re-implements and extends the sneaker-intel dbt ``int_sales_enriched``
model on the Spark compute layer -- same economics, different framework, a
scale story rather than a SQL-warehouse story.

Features computed (target is ``price_premium``):
  * price_premium            (sale_price - retail_price) / retail_price
  * days_since_release       datediff(sale_date, release_date)
  * size_premium             size's avg premium minus the shoe's overall avg
                             premium (how much this size commands over baseline)
  * release_type_encoded     ordinal scarcity (limited > raffle > fcfs > general)
  * rolling_7d_avg_premium   7-day rolling avg of price_premium per shoe; NULL
                             when the shoe has < 7 days of history (Phase 3
                             validates exactly this null semantics)
  * search_index_7d_pre_drop avg Google Trends search_index in the 7 days before
                             the shoe's drop_date
  * brand_avg_premium        mean price_premium for the brand across all sales,
                             attached via a broadcast join (see DEVLOG)

Runs locally with a ``local[*]`` master for development. CLI:
    python -m stages.features --run-date 2025-01-01 \\
        --config config/pipeline_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import os

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from stages.config import (
    FeatureConfig,
    PipelineConfig,
    load_config,
    load_feature_config,
)
from stages.io import to_spark_path

logger = logging.getLogger("stages.features")

# Stable output column order -- this is the contract the validation stage and
# the training stage read against.
OUTPUT_COLUMNS = [
    "sale_id",
    "shoe_id",
    "brand",
    "size_us",
    "sale_date",
    "release_date",
    "retail_price",
    "sale_price",
    "release_type",
    "price_premium",
    "days_since_release",
    "size_premium",
    "release_type_encoded",
    "rolling_7d_avg_premium",
    "search_index_7d_pre_drop",
    "brand_avg_premium",
]


def build_spark(app_name: str = "ml-pipeline-features") -> SparkSession:
    """Create a local Spark session sized for small/medium tabular work."""
    # The JVM reads SPARK_LOCAL_IP at launch; pin it to loopback so an
    # unresolvable container hostname doesn't crash driver startup
    # (JAVA_GATEWAY_EXITED). Set before getOrCreate, don't override an explicit
    # operator value.
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    return (
        SparkSession.builder.master("local[*]")
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.session.timeZone", "UTC")
        # Bind to loopback so a missing/unresolvable hostname doesn't break
        # driver startup in containers/CI.
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )


def _encode_release_type(order: dict[str, int]):
    """Build an ordinal-encoding column expression from the config mapping."""
    col = F.lit(None).cast("int")
    for release_type, rank in order.items():
        col = F.when(F.col("release_type") == release_type, F.lit(rank)).otherwise(col)
    return col


def compute_features(
    sales: DataFrame,
    shoes: DataFrame,
    drops: DataFrame,
    search: DataFrame,
    fcfg: FeatureConfig,
) -> DataFrame:
    """Pure transformation: raw DataFrames -> flat feature DataFrame.

    Kept free of any I/O so it can be unit-tested directly against fixture
    DataFrames.
    """
    # dim_shoes is small -- broadcast it rather than shuffling the fact table.
    enriched = sales.join(
        F.broadcast(
            shoes.select(
                "shoe_id", "brand", "retail_price", "release_date", "release_type"
            )
        ),
        on="shoe_id",
        how="inner",
    )

    enriched = enriched.withColumn(
        "price_premium",
        (F.col("sale_price") - F.col("retail_price")) / F.col("retail_price"),
    ).withColumn(
        "days_since_release",
        F.datediff(F.col("sale_date"), F.col("release_date")),
    ).withColumn(
        "release_type_encoded",
        _encode_release_type(fcfg.release_type_order),
    )

    # --- rolling 7-day average premium, with explicit insufficient-history NULL.
    # Order a time-range window by seconds so rangeBetween spans real days.
    sale_secs = F.col("sale_date").cast("timestamp").cast("long")
    roll_window = (
        Window.partitionBy("shoe_id")
        .orderBy(sale_secs)
        .rangeBetween(-fcfg.rolling_window_days * 86400, 0)
    )
    shoe_window = Window.partitionBy("shoe_id")
    enriched = enriched.withColumn(
        "_rolling_avg", F.avg("price_premium").over(roll_window)
    ).withColumn(
        "_days_of_history",
        F.datediff(F.col("sale_date"), F.min("sale_date").over(shoe_window)),
    ).withColumn(
        "rolling_7d_avg_premium",
        F.when(
            F.col("_days_of_history") >= fcfg.rolling_window_days,
            F.col("_rolling_avg"),
        ).otherwise(F.lit(None).cast("double")),
    )

    # --- size_premium: this size's avg premium relative to the shoe's baseline.
    size_window = Window.partitionBy("shoe_id", "size_us")
    enriched = enriched.withColumn(
        "size_premium",
        F.avg("price_premium").over(size_window)
        - F.avg("price_premium").over(shoe_window),
    )

    # --- brand_avg_premium: tiny per-brand aggregate, attached via broadcast.
    brand_avg = enriched.groupBy("brand").agg(
        F.avg("price_premium").alias("brand_avg_premium")
    )
    enriched = enriched.join(F.broadcast(brand_avg), on="brand", how="left")

    # --- search_index_7d_pre_drop: avg demand signal in the N days before drop.
    drop_dates = drops.select("shoe_id", "drop_date")
    pre_drop = (
        search.join(F.broadcast(drop_dates), on="shoe_id", how="inner")
        .where(
            (F.col("signal_date") <= F.col("drop_date"))
            & (
                F.col("signal_date")
                >= F.date_sub(F.col("drop_date"), fcfg.search_signal_window_days)
            )
        )
        .groupBy("shoe_id")
        .agg(F.avg("search_index").alias("search_index_7d_pre_drop"))
    )
    enriched = enriched.join(F.broadcast(pre_drop), on="shoe_id", how="left")

    # brand is part of OUTPUT_COLUMNS and also serves as the write partition key.
    return enriched.select(*OUTPUT_COLUMNS)


def run(config: PipelineConfig, fcfg: FeatureConfig) -> str:
    """Read raw, compute features, write partitioned Parquet. Returns the URI."""
    spark = build_spark()
    try:
        sales = spark.read.parquet(to_spark_path(config.raw_uri("sales")))
        shoes = spark.read.parquet(to_spark_path(config.raw_uri("shoes")))
        drops = spark.read.parquet(to_spark_path(config.raw_uri("drops")))
        search = spark.read.parquet(to_spark_path(config.raw_uri("search_interest")))

        input_rows = sales.count()
        features = compute_features(sales, shoes, drops, search, fcfg).cache()
        output_rows = features.count()

        out_uri = config.features_uri()
        out_path = to_spark_path(out_uri)
        # coalesce(1) -> one file per brand partition (avoids small-file sprawl
        # on this dataset size). Partition by brand so brand-filtered training
        # reads a single partition instead of the whole dataset.
        (
            features.coalesce(1)
            .write.mode("overwrite")
            .partitionBy(fcfg.partition_by)
            .parquet(out_path)
        )

        logger.info("features: input_rows=%d output_rows=%d", input_rows, output_rows)
        logger.info("features written -> %s (partitioned by %s)", out_uri, fcfg.partition_by)
        return out_uri
    finally:
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute features with PySpark.")
    parser.add_argument("--run-date", required=True, help="ISO run date (YYYY-MM-DD).")
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    parser.add_argument("--feature-config", default="config/feature_config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    config = load_config(args.config, run_date=args.run_date)
    fcfg = load_feature_config(args.feature_config)
    uri = run(config, fcfg)
    logger.info("features stage complete -> %s", uri)


if __name__ == "__main__":
    main()
