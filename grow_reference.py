"""Grow the KicksDB retail reference from the live current market.

The retail reference is the ceiling on what can have a premium computed: KicksDB's
Starter tier returns no retail, so only shoes in the reference get scored. This
scaffolds new entries from the live market. It searches KicksDB for a curated list
of current models, pulls each product's real SKU and market price, and proposes a
retail from a published-MSRP map.

The output is a DRAFT you review, never a silent overwrite. Model-level MSRP is a
fine default for a general release but not colorway-exact, so collabs and limiteds
are flagged for you to confirm the real retail before you merge. Nothing is
fabricated: the proposed numbers are published MSRPs, and the ones that vary by
colorway are marked for a human to check.

    KICKSDB_API_KEY=... python grow_reference.py --limit 3

Writes ``config/kicksdb_retail_reference.draft.json``. Review it, set the real
retail on the flagged rows, drop the ``_``-prefixed review fields, and append the
kept entries to ``config/kicksdb_retail_reference.json``. Then refresh the fixtures
(``make kicksdb-fixtures``) so the offline path matches.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from stages.kicksdb import (
    DEFAULT_REFERENCE_PATH,
    KicksDBClient,
    classify_release_type,
    load_retail_reference,
    normalize_sku,
)

logger = logging.getLogger("grow_reference")

DEFAULT_DRAFT_PATH = "config/kicksdb_retail_reference.draft.json"

# Curated current models with their published MSRP. Model-level, so it is a good
# default for a general release and a starting point (flagged) for collabs and
# limiteds, where the colorway retail often differs. Real published prices, not
# invented: the flag is there precisely so a human confirms the ones that vary.
MSRP_SEED = [
    {"query": "Nike Dunk Low", "brand": "Nike", "model": "Dunk Low",
     "retail": 115.0, "release_type": "general"},
    {"query": "Nike Dunk High", "brand": "Nike", "model": "Dunk High",
     "retail": 125.0, "release_type": "general"},
    {"query": "Nike Air Force 1 '07", "brand": "Nike", "model": "Air Force 1",
     "retail": 115.0, "release_type": "general"},
    {"query": "Nike Air Max 90", "brand": "Nike", "model": "Air Max 90",
     "retail": 130.0, "release_type": "general"},
    {"query": "Nike Air Max 1", "brand": "Nike", "model": "Air Max 1",
     "retail": 150.0, "release_type": "general"},
    {"query": "Nike Vomero 5", "brand": "Nike", "model": "Vomero 5",
     "retail": 160.0, "release_type": "general"},
    {"query": "Nike P-6000", "brand": "Nike", "model": "P-6000",
     "retail": 110.0, "release_type": "general"},
    {"query": "Air Jordan 1 High OG", "brand": "Jordan", "model": "Air Jordan 1 High",
     "retail": 180.0, "release_type": "limited"},
    {"query": "Air Jordan 1 Low OG", "brand": "Jordan", "model": "Air Jordan 1 Low",
     "retail": 150.0, "release_type": "general"},
    {"query": "Air Jordan 3 Retro", "brand": "Jordan", "model": "Air Jordan 3",
     "retail": 210.0, "release_type": "limited"},
    {"query": "Air Jordan 4 Retro", "brand": "Jordan", "model": "Air Jordan 4",
     "retail": 215.0, "release_type": "limited"},
    {"query": "Air Jordan 11 Retro", "brand": "Jordan", "model": "Air Jordan 11",
     "retail": 235.0, "release_type": "limited"},
    {"query": "Adidas Samba OG", "brand": "Adidas", "model": "Samba OG",
     "retail": 100.0, "release_type": "general"},
    {"query": "Adidas Gazelle", "brand": "Adidas", "model": "Gazelle",
     "retail": 100.0, "release_type": "general"},
    {"query": "Adidas Campus 00s", "brand": "Adidas", "model": "Campus 00s",
     "retail": 110.0, "release_type": "general"},
    {"query": "Adidas Yeezy Boost 350 V2", "brand": "Adidas", "model": "Yeezy Boost 350 V2",
     "retail": 230.0, "release_type": "limited"},
    {"query": "Adidas Yeezy Slide", "brand": "Adidas", "model": "Yeezy Slide",
     "retail": 70.0, "release_type": "limited"},
    {"query": "New Balance 550", "brand": "New Balance", "model": "550",
     "retail": 120.0, "release_type": "general"},
    {"query": "New Balance 990v6", "brand": "New Balance", "model": "990v6",
     "retail": 200.0, "release_type": "general"},
    {"query": "New Balance 2002R", "brand": "New Balance", "model": "2002R",
     "retail": 150.0, "release_type": "general"},
    {"query": "New Balance 9060", "brand": "New Balance", "model": "9060",
     "retail": 150.0, "release_type": "general"},
    {"query": "Asics Gel-Kayano 14", "brand": "Asics", "model": "Gel-Kayano 14",
     "retail": 160.0, "release_type": "general"},
    {"query": "Salomon XT-6", "brand": "Salomon", "model": "XT-6",
     "retail": 200.0, "release_type": "general"},
    {"query": "Puma Speedcat", "brand": "Puma", "model": "Speedcat",
     "retail": 100.0, "release_type": "general"},
]


def candidate_from_product(
    product: dict, seed: dict, existing: set[str]
) -> dict | None:
    """Build one proposed reference entry from a StockX product, or None to skip.

    Skips products with no SKU or a SKU already in the reference (dedup on the
    normalized form). Retail comes from the seed's model MSRP; the release type is
    refined from the title, and anything collab/limited is flagged for review
    because model-level retail is not colorway-exact for those.
    """
    sku = str(product.get("sku") or "").strip()
    nsku = normalize_sku(sku)
    if not nsku or nsku in existing:
        return None
    try:
        market = round(float(product.get("avg_price")), 2)
    except (TypeError, ValueError):
        market = 0.0
    title = product.get("title") or ""
    # Trust the seed's curated type; only upgrade to collab when the title names a
    # collaborator. So a plain retro keeps its seed "limited" and is flagged for
    # retail review, rather than being downgraded to "general" and skipped.
    release_type = (
        "collab" if classify_release_type(title) == "collab" else seed["release_type"]
    )
    return {
        "sku": sku,
        "retail_price": float(seed["retail"]),
        "release_type": release_type,
        "brand": seed.get("brand") or product.get("brand") or "",
        "model": seed.get("model") or product.get("model") or "",
        "colorway": product.get("secondary_title") or title,
        "silhouette": seed.get("model", ""),
        "_market": market,  # review context, not a reference field
        "_review": release_type in ("collab", "limited"),  # confirm retail
        "_query": seed["query"],
    }


def scaffold(client: KicksDBClient, seeds: list[dict], existing: set[str],
             limit: int) -> list[dict]:
    """Search each seed model and collect deduped candidate entries."""
    seen: set[str] = set()
    out: list[dict] = []
    for seed in seeds:
        for product in client.search_stockx(seed["query"], limit=limit):
            cand = candidate_from_product(product, seed, existing | seen)
            if cand is None:
                continue
            seen.add(normalize_sku(cand["sku"]))
            out.append(cand)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scaffold new KicksDB retail-reference entries from the live market."
    )
    parser.add_argument("--limit", type=int, default=3,
                        help="StockX results to consider per model query.")
    parser.add_argument("--reference", default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--out", default=DEFAULT_DRAFT_PATH)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    api_key = os.environ.get("KICKSDB_API_KEY")
    if not api_key:
        raise SystemExit(
            "grow_reference.py needs KICKSDB_API_KEY: it discovers live SKUs from "
            "the current market, which fixtures can't provide."
        )

    existing = set(load_retail_reference(args.reference).by_sku)
    client = KicksDBClient(api_key)
    entries = scaffold(client, MSRP_SEED, existing, args.limit)

    Path(args.out).write_text(json.dumps(entries, indent=2))
    n_review = sum(e["_review"] for e in entries)
    logger.info(
        "scaffolded %d new SKUs (%d already in the reference were skipped) in %d "
        "requests; %d flagged for retail review (collab/limited). Draft -> %s",
        len(entries), len(existing), client.request_count, n_review, args.out,
    )
    logger.info(
        "review the draft, set real retail on the flagged rows, drop the _-prefixed "
        "fields, append to %s, then refresh fixtures.", args.reference,
    )


if __name__ == "__main__":
    main()
