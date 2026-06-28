"""Generate small, schema-faithful synthetic Parquet fixtures.

Every downstream stage and all of CI run against these fixtures -- no live
Postgres required. The data is deterministic (fixed seed) so tests can assert on
exact values, and it is built to be statistically plausible rather than random
noise:

  * premiums track release scarcity (limited > raffle > fcfs > general), so the
    Phase 2 features and Phase 3 statistical checks have real signal to chew on;
  * each shoe gets enough clustered sales history that rolling_7d_avg_premium
    actually populates for later rows;
  * sale dates span 2021-2024 so the Phase 4 temporal train/val split
    (pre-2023 vs 2023+) has data on both sides;
  * (shoe_id, sale_date, size_us) is unique, matching the Pandera duplicate
    check in Phase 3.

Schemas mirror the sneaker-intel Postgres tables exactly so that fixture-mode
ingest and real Postgres ingest produce identical raw Parquet.

Run:  python generate_fixtures.py [--out data/fixtures]
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

from stages.io import path_join, write_parquet

logger = logging.getLogger("generate_fixtures")

SEED = 42

# Release scarcity -> base price premium over retail. Drives realistic spreads.
_BASE_PREMIUM = {"limited": 1.40, "raffle": 0.90, "fcfs": 0.40, "general": 0.05}
_SIZES = [7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 12.0]

# Eight shoes across brands and release types. (shoe_id, brand, model, colorway,
# retail_price, release_date, release_type, silhouette)
_SHOES: list[tuple] = [
    (1, "Jordan", "Air Jordan 1 High", "Chicago", 180.0, date(2021, 11, 13), "limited", "Air Jordan 1"),  # noqa: E501
    (2, "Jordan", "Air Jordan 4 Retro", "Bred", 210.0, date(2020, 12, 12), "raffle", "Air Jordan 4"),  # noqa: E501
    (3, "Nike", "Dunk Low", "Panda", 110.0, date(2021, 3, 10), "fcfs", "Dunk"),
    (4, "Nike", "Air Force 1 Low", "Triple White", 100.0, date(2020, 1, 15), "general", "Air Force 1"),  # noqa: E501
    (5, "Adidas", "Yeezy Boost 350 V2", "Zebra", 220.0, date(2022, 2, 9), "raffle", "Yeezy 350"),
    (6, "Adidas", "Samba OG", "Black White", 100.0, date(2022, 6, 1), "general", "Samba"),
    (7, "New Balance", "550", "White Green", 120.0, date(2021, 9, 20), "fcfs", "550"),
    (8, "New Balance", "990v5", "Grey", 185.0, date(2020, 5, 5), "general", "990"),
]


def _make_shoes() -> pd.DataFrame:
    df = pd.DataFrame(
        _SHOES,
        columns=[
            "shoe_id", "brand", "model", "colorway", "retail_price",
            "release_date", "release_type", "silhouette",
        ],
    )
    df["release_date"] = pd.to_datetime(df["release_date"])
    return df


def _make_sales(shoes: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """One row per resale. Clustered per shoe so rolling windows populate."""
    rows: list[dict] = []
    sale_id = 1
    for shoe in shoes.itertuples(index=False):
        n_sales = int(rng.integers(7, 12))
        # Day offsets after release, sorted; clustering happens naturally and we
        # nudge a few sales close together to guarantee 7-day-window neighbours.
        offsets = np.sort(rng.integers(1, 900, size=n_sales))
        for i in range(1, len(offsets)):
            if rng.random() < 0.4:
                offsets[i] = offsets[i - 1] + int(rng.integers(1, 6))
        base = _BASE_PREMIUM[shoe.release_type]
        for off in offsets:
            sale_dt = shoe.release_date.date() + timedelta(days=int(off))
            premium = base * (1.0 + rng.normal(0.0, 0.18))
            premium = max(premium, -0.45)  # never below ~half of retail
            sale_price = round(float(shoe.retail_price) * (1.0 + premium), 2)
            rows.append(
                {
                    "sale_id": sale_id,
                    "shoe_id": int(shoe.shoe_id),
                    "platform": "StockX",
                    "sale_price": sale_price,
                    "sale_date": pd.Timestamp(sale_dt),
                    "size_us": float(rng.choice(_SIZES)),
                    "condition": "new" if rng.random() < 0.85 else "used",
                    "source_item_id": f"SX-{int(shoe.shoe_id)}-{sale_id}",
                }
            )
            sale_id += 1

    df = pd.DataFrame(rows)
    # Enforce uniqueness of (shoe_id, sale_date, size_us) -- the Phase 3 check.
    df = df.drop_duplicates(subset=["shoe_id", "sale_date", "size_us"]).reset_index(drop=True)
    df["sale_id"] = range(1, len(df) + 1)
    return df


def _make_drops(shoes: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict] = []
    for drop_id, shoe in enumerate(shoes.itertuples(index=False), start=1):
        drop_dt = shoe.release_date.date()
        announced = drop_dt - timedelta(days=int(rng.integers(7, 22)))
        rows.append(
            {
                "drop_id": drop_id,
                "shoe_id": int(shoe.shoe_id),
                "drop_date": pd.Timestamp(drop_dt),
                "drop_type": shoe.release_type,
                "announced_at": pd.Timestamp(announced),
            }
        )
    return pd.DataFrame(rows)


def _make_search_interest(
    shoes: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Google Trends-style demand signal: weekly points around each drop.

    Higher for scarcer releases, and -- importantly -- includes points within
    the 7 days before drop_date so the Phase 2 search_index_7d_pre_drop join has
    rows to average.
    """
    rows: list[dict] = []
    for shoe in shoes.itertuples(index=False):
        drop_dt = shoe.release_date.date()
        peak = {"limited": 95, "raffle": 75, "fcfs": 55, "general": 30}[shoe.release_type]
        # Weekly points from 14 days before to 14 days after the drop.
        for week_off in range(-14, 15, 7):
            signal_dt = drop_dt + timedelta(days=week_off)
            # Demand peaks just before the drop, tapers after.
            proximity = max(0.0, 1.0 - abs(week_off + 3) / 18.0)
            idx = int(np.clip(peak * proximity + rng.normal(0, 5), 0, 100))
            rows.append(
                {
                    "shoe_id": int(shoe.shoe_id),
                    "signal_date": pd.Timestamp(signal_dt),
                    "platform": "google_trends",
                    "search_index": idx,
                }
            )
    return pd.DataFrame(rows)


def build_fixtures(out_dir: str = "data/fixtures") -> dict[str, str]:
    """Build all four fixtures and write them as Parquet. Returns name -> URI."""
    rng = np.random.default_rng(SEED)
    shoes = _make_shoes()
    tables = {
        "shoes": shoes,
        "sales": _make_sales(shoes, rng),
        "drops": _make_drops(shoes, rng),
        "search_interest": _make_search_interest(shoes, rng),
    }
    written: dict[str, str] = {}
    for name, df in tables.items():
        uri = path_join(out_dir, f"{name}.parquet")
        write_parquet(df, uri)
        written[name] = uri
        logger.info("fixture %-16s rows=%-4d -> %s", name, len(df), uri)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic fixtures.")
    parser.add_argument("--out", default="data/fixtures", help="Output directory.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    build_fixtures(args.out)


if __name__ == "__main__":
    main()
