"""Refresh the canned KicksDB responses used by fixture-mode ingest and CI.

The KicksDB adapter (``stages/kicksdb.py``) reads these when ``KICKSDB_API_KEY``
is unset, so the adapter, its tests, and a clean clone all run with no secret and
no network. Unlike ``generate_fixtures.py`` (which builds synthetic data), this
pulls REAL responses for the SKUs in the retail reference and trims them to the
handful of fields the adapter reads, dropping the affiliate ``link`` fields and
the image galleries. KicksDB's terms permit caching and redistributing the API
data; the trim keeps the committed sample small and free of tracking IDs.

Needs a key:  KICKSDB_API_KEY=... python generate_kicksdb_fixtures.py
"""

from __future__ import annotations

import argparse
import logging
import os

from stages.io import path_join, write_json
from stages.kicksdb import (
    DEFAULT_FIXTURES_DIR,
    DEFAULT_REFERENCE_PATH,
    KicksDBClient,
    load_retail_reference,
)

logger = logging.getLogger("generate_kicksdb_fixtures")

# Only the fields the reshape reads. Everything else (galleries, affiliate links,
# breadcrumbs, 360 spins) is dropped: not needed, and not ours to redistribute.
_STOCKX_KEEP = [
    "id", "sku", "brand", "model", "title", "primary_title", "secondary_title",
    "product_type", "category", "avg_price", "min_price", "max_price",
    "weekly_orders", "rank",
]
_GOAT_KEEP = [
    "id", "sku", "brand", "model", "name", "nickname", "colorway", "season",
    "release_date", "release_date_year", "product_type",
]


def _trim(obj: dict, keep: list[str]) -> dict:
    return {k: obj.get(k) for k in keep if k in obj}


def build_kicksdb_fixtures(
    out_dir: str = DEFAULT_FIXTURES_DIR,
    reference_path: str = DEFAULT_REFERENCE_PATH,
    api_key: str | None = None,
) -> dict[str, str]:
    """Pull + trim StockX and GOAT responses for every reference SKU."""
    api_key = api_key or os.environ.get("KICKSDB_API_KEY")
    if not api_key:
        raise SystemExit(
            "KICKSDB_API_KEY is required to refresh the KicksDB fixtures "
            "(this script hits the live API)."
        )
    reference = load_retail_reference(reference_path)
    client = KicksDBClient(api_key)

    stockx, goat = [], []
    for sku in reference.skus():
        sx = client.search_stockx(sku, limit=1)
        gt = client.search_goat(sku, limit=1)
        if sx:
            stockx.append(_trim(sx[0], _STOCKX_KEEP))
        if gt:
            goat.append(_trim(gt[0], _GOAT_KEEP))

    written = {
        "stockx": write_json({"data": stockx}, path_join(out_dir, "stockx_products.json")),
        "goat": write_json({"data": goat}, path_join(out_dir, "goat_products.json")),
    }
    logger.info(
        "wrote %d StockX + %d GOAT trimmed records to %s (%d requests)",
        len(stockx), len(goat), out_dir, client.request_count,
    )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh canned KicksDB fixtures.")
    parser.add_argument("--out", default=DEFAULT_FIXTURES_DIR)
    parser.add_argument("--reference", default=DEFAULT_REFERENCE_PATH)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    build_kicksdb_fixtures(args.out, args.reference)


if __name__ == "__main__":
    main()
