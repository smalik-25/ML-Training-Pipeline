"""KicksDB ingest adapter -- reshape the live kicks.dev market API into the
canonical raw schema.

This is a second anti-corruption path, the sibling of the sneaker-intel Postgres
SQL in ``stages/ingest.py``. Where that SQL aliases a star schema into
``sales``/``shoes``/``drops``/``search_interest``, this module does the same job
for a REST API whose shape is nothing like the warehouse:

  * two endpoints, not one table set -- StockX for the current market price,
    GOAT for the release date, joined on SKU;
  * a product-grained current snapshot, not a transaction stream;
  * no retail price at all on the Starter tier.

Every one of those mismatches is absorbed here, so features, validate, train,
score, and serve never learn KicksDB exists.

Retail price is the load-bearing gap. The premium the model predicts is
``(sale_price - retail_price) / retail_price``, and the Starter tier returns no
retail: StockX has no such field, and GOAT's ``retail_prices`` was null for every
product sampled during recon. We do NOT fabricate it. Retail is carried from a
SKU-keyed reference table (``config/kicksdb_retail_reference.json``): real retail
for the shoes we have it for, nothing for the ones we don't. A product with no
resolvable retail is excluded from the landing and counted, so premium is never
computed on a guessed number. The tradeoff is written up in the DEVLOG recon
entry.

Grain. KicksDB Starter is a current snapshot, not a sales history -- the
``sales/daily`` route returns null and ``variants`` is 403 on this key type. So
each covered product lands as ONE canonical sale at the run's snapshot date,
priced at the current StockX market (``avg_price``). That is a current-market
observation, not a historical transaction; extending the training set from
KicksDB is deliberately out of scope (the per-sale history isn't there to support
it). What this source is for is drift monitoring against today's market and live
scoring of current sneakers.

No key present -> fixture mode: canned, trimmed KicksDB responses under
``data/fixtures/kicksdb/`` are read instead of the network, so the adapter, its
tests, and a clean clone all run with no secret. The live client caches per
product and backs off on 429 to stay inside the Starter budget (640 req/min,
50,000 req/month).
"""

from __future__ import annotations

import json
import logging
import time
import zlib
from dataclasses import dataclass, field

import pandas as pd

from stages.io import path_join, read_json

logger = logging.getLogger("stages.kicksdb")

BASE_URL = "https://api.kicks.dev/v3"
# StockX carries the current market price; GOAT carries the release date. Neither
# carries retail on Starter, which is why the reference table exists.
STOCKX_SEARCH = "/stockx/products"
GOAT_SEARCH = "/goat/products"

# The current snapshot price is a product-level aggregate across sizes, not a
# size-specific quote (per-size asks live behind the 403 variants route). We land
# it at one representative men's size so the row is well-formed; size_premium then
# collapses to 0 because there is no size dispersion to measure. Documented, not
# fabricated: the number is the real market price, the size is a stand-in for
# "not size-specific".
REPRESENTATIVE_SIZE_US = 9.0

DEFAULT_REFERENCE_PATH = "config/kicksdb_retail_reference.json"
DEFAULT_FIXTURES_DIR = "data/fixtures/kicksdb"

# Which of the model's 8 features a current-only KicksDB snapshot can drift-check
# honestly (Phase 2). The excluded four are degenerate on a snapshot -- not
# missing at random, structurally uncomputable from what the Starter tier gives
# -- so running PSI on them would measure the grain, not the market. The drift
# DAG passes SNAPSHOT_DRIFT_FEATURES to the monitor and logs the rest as excluded.
# Keep in sync with stages.train.FEATURE_COLUMNS.
SNAPSHOT_DRIFT_FEATURES = [
    "days_since_release",
    "retail_price",
    "release_type_encoded",
    "brand_avg_premium",
]
SNAPSHOT_EXCLUDED_FEATURES = {
    "size_us": "a snapshot lands one representative size, not real size dispersion",
    "size_premium": "needs several sizes per shoe; one size makes it identically 0",
    "rolling_7d_avg_premium": "needs per-transaction sales history the Starter tier omits",
    "search_index_7d_pre_drop": "needs a pre-drop demand signal a snapshot doesn't carry",
}

# Starter tier: 640 req/min. Keep a small floor between live calls so a batch pull
# stays polite well under the limit. Fixture mode never touches this.
_MIN_REQUEST_INTERVAL_S = 0.12
_MAX_RETRIES = 4

# Collaborator / tokens that mark a release as a collab when the reference table
# doesn't already say. The reference is authoritative; this is only a fallback so
# release_type is never left unset, and it stays inside the canonical vocabulary
# {general, limited, collab} the Pandera contract enforces.
_COLLAB_TOKENS = (
    "off-white", "travis scott", "fragment", "union", "supreme", "dior",
    " x ", "kaws", "sacai", "ambush", "cactus jack",
)
_LIMITED_TOKENS = ("yeezy", "retro high og", "limited")

# The canonical raw dtypes, table by table. These MUST match what
# generate_fixtures.py and the Postgres path produce, or the "byte-compatible
# layout" claim is a lie. reshape_to_canonical coerces to exactly these.
CANONICAL_DTYPES: dict[str, dict[str, str]] = {
    "sales": {
        "sale_id": "int64",
        "shoe_id": "int64",
        "platform": "object",
        "sale_price": "float64",
        "sale_date": "datetime64[ns]",
        "size_us": "float64",
        "condition": "object",
        "source_item_id": "object",
    },
    "shoes": {
        "shoe_id": "int64",
        "brand": "object",
        "model": "object",
        "colorway": "object",
        "retail_price": "float64",
        "release_date": "datetime64[ns]",
        "release_type": "object",
        "silhouette": "object",
    },
    "drops": {
        "drop_id": "int64",
        "shoe_id": "int64",
        "drop_date": "datetime64[ns]",
        "drop_type": "object",
        "announced_at": "datetime64[ns]",
    },
    "search_interest": {
        "shoe_id": "int64",
        "signal_date": "datetime64[ns]",
        "platform": "object",
        "search_index": "int64",
    },
}


class KicksDBError(RuntimeError):
    """Raised when the KicksDB adapter cannot produce a usable landing."""


# --------------------------------------------------------------------------- #
# Identity + classification helpers
# --------------------------------------------------------------------------- #
def normalize_sku(sku: str) -> str:
    """Canonical SKU form so StockX ('DD1391-100') and GOAT ('DD1391 100') join.

    Upper-cased, whitespace and underscores folded to a single dash, surrounding
    junk stripped. Empty/None -> "".
    """
    if not sku:
        return ""
    s = str(sku).strip().upper()
    out = []
    prev_dash = False
    for ch in s:
        if ch.isalnum():
            out.append(ch)
            prev_dash = False
        elif ch in " -_/":
            if not prev_dash:
                out.append("-")
                prev_dash = True
        # drop anything else
    return "".join(out).strip("-")


def sku_candidates(raw_sku: str) -> list[str]:
    """Normalized keys a raw SKU could match, whole first then components.

    StockX ships some shoes with a compound style code ("BZ0057/B75806"), where
    only one component is the SKU the reference and GOAT use. Returning both the
    whole normalized form and each ``/``-separated component lets the retail
    lookup and the StockX<->GOAT join match on the real code instead of a
    concatenation that matches nothing.
    """
    cands = [normalize_sku(raw_sku)]
    for part in str(raw_sku).split("/"):
        c = normalize_sku(part)
        if c and c not in cands:
            cands.append(c)
    return [c for c in cands if c]


def stable_shoe_id(normalized_sku: str) -> int:
    """Deterministic positive int shoe_id from a normalized SKU.

    Downstream the canonical schema keeps only ``shoe_id`` + ``brand`` for
    identity, so the adapter has to mint an integer key from the SKU. CRC32 is
    stable across runs and processes (unlike ``hash()``), and the 31-bit mask
    keeps it a positive int64. Collisions across a curated reference of a few
    dozen SKUs are vanishingly unlikely; the reshape asserts uniqueness anyway.
    """
    return zlib.crc32(normalized_sku.encode("utf-8")) & 0x7FFFFFFF


def classify_release_type(title: str) -> str:
    """Fallback release_type when the reference table doesn't specify one.

    Returns a value inside the canonical {general, limited, collab} vocabulary.
    The reference table is authoritative; this only fires if a reference entry
    omits release_type, so a landed row never carries a guessed scarcity when a
    curated one exists.
    """
    t = f" {title.lower()} "
    if any(tok in t for tok in _COLLAB_TOKENS):
        return "collab"
    if any(tok in t for tok in _LIMITED_TOKENS):
        return "limited"
    return "general"


# --------------------------------------------------------------------------- #
# Retail reference table (the confirmed retail-price decision)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReferenceEntry:
    """One curated sneaker: the retail KicksDB Starter does not give us."""

    sku: str
    retail_price: float
    release_type: str
    brand: str = ""
    model: str = ""
    colorway: str = ""
    silhouette: str = ""
    release_date: str = ""  # optional fallback if GOAT has none


@dataclass
class RetailReference:
    """SKU -> curated retail/release_type, keyed on the normalized SKU."""

    by_sku: dict[str, ReferenceEntry] = field(default_factory=dict)

    def lookup(self, sku: str) -> ReferenceEntry | None:
        """Resolve a SKU to its reference entry.

        Handles the compound style codes StockX sometimes returns (adidas ships
        one shoe as ``BZ0057/B75806``): try the whole normalized SKU, then each
        ``/``-separated component, so a product that has a real retail is matched
        rather than silently dropped and mis-counted as retail-less.
        """
        for cand in sku_candidates(sku):
            entry = self.by_sku.get(cand)
            if entry is not None:
                return entry
        return None

    def skus(self) -> list[str]:
        return [e.sku for e in self.by_sku.values()]

    def __len__(self) -> int:
        return len(self.by_sku)


def load_retail_reference(path: str = DEFAULT_REFERENCE_PATH) -> RetailReference:
    """Load the curated SKU->retail reference.

    Read through ``stages.io.read_json`` so it resolves the same way every other
    artifact does (local in dev/CI, s3:// in prod). Fails loudly if the file is
    missing rather than silently landing zero shoes.
    """
    records = read_json(path)
    if not isinstance(records, list):
        raise KicksDBError(
            f"retail reference at {path} must be a JSON list of records"
        )
    by_sku: dict[str, ReferenceEntry] = {}
    for rec in records:
        entry = ReferenceEntry(
            sku=str(rec["sku"]),
            retail_price=float(rec["retail_price"]),
            release_type=str(rec.get("release_type", "")),
            brand=str(rec.get("brand", "")),
            model=str(rec.get("model", "")),
            colorway=str(rec.get("colorway", "")),
            silhouette=str(rec.get("silhouette", "")),
            release_date=str(rec.get("release_date", "")),
        )
        by_sku[normalize_sku(entry.sku)] = entry
    logger.info("loaded retail reference: %d SKUs from %s", len(by_sku), path)
    return RetailReference(by_sku=by_sku)


# --------------------------------------------------------------------------- #
# Live client (only import path that needs `requests`, imported lazily)
# --------------------------------------------------------------------------- #
class KicksDBClient:
    """Thin, cached, backing-off client for the kicks.dev Starter API.

    Caches responses per (path, params) so re-pulling the same product inside a
    run is free, backs off on 429 to respect the rate limit, and counts requests
    so the ingest stage can log the budget spent.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = BASE_URL,
        min_interval_s: float = _MIN_REQUEST_INTERVAL_S,
        timeout_s: float = 25.0,
    ) -> None:
        try:
            import requests
        except ImportError as exc:  # fail loudly with an actionable message
            raise KicksDBError(
                "live KicksDB ingest requires the 'kicksdb' extra: "
                'pip install -e ".[kicksdb]"'
            ) from exc
        self._session = requests.Session()
        self._session.headers.update({"Authorization": api_key})
        self._base_url = base_url.rstrip("/")
        self._min_interval_s = min_interval_s
        self._timeout_s = timeout_s
        self._cache: dict[str, dict] = {}
        self._last_request_at = 0.0
        self.request_count = 0

    def _get(self, path: str, params: dict) -> dict:
        key = f"{path}?{json.dumps(params, sort_keys=True)}"
        if key in self._cache:
            return self._cache[key]

        import requests

        url = f"{self._base_url}{path}"
        for attempt in range(_MAX_RETRIES):
            elapsed = time.time() - self._last_request_at
            if elapsed < self._min_interval_s:
                time.sleep(self._min_interval_s - elapsed)
            self._last_request_at = time.time()
            self.request_count += 1
            resp = self._session.get(url, params=params, timeout=self._timeout_s)
            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", 2 ** attempt))
                logger.warning(
                    "KicksDB 429 on %s; backing off %.1fs (attempt %d/%d)",
                    path, retry_after, attempt + 1, _MAX_RETRIES,
                )
                time.sleep(retry_after)
                continue
            try:
                resp.raise_for_status()
            except requests.HTTPError as exc:
                raise KicksDBError(f"KicksDB GET {path} failed: {exc}") from exc
            body = resp.json()
            self._cache[key] = body
            return body
        raise KicksDBError(f"KicksDB GET {path} still rate-limited after retries")

    def search_stockx(self, query: str, limit: int = 1) -> list[dict]:
        return self._get(STOCKX_SEARCH, {"query": query, "limit": limit}).get(
            "data"
        ) or []

    def search_goat(self, query: str, limit: int = 1) -> list[dict]:
        return self._get(GOAT_SEARCH, {"query": query, "limit": limit}).get(
            "data"
        ) or []


# --------------------------------------------------------------------------- #
# Fetching: live vs fixtures. Both return the same (stockx, goat) raw lists.
# --------------------------------------------------------------------------- #
def fetch_live_products(
    client: KicksDBClient, skus: list[str]
) -> tuple[list[dict], list[dict]]:
    """Pull StockX (market) + GOAT (release date) for each SKU. One search each."""
    stockx: list[dict] = []
    goat: list[dict] = []
    for sku in skus:
        stockx.extend(client.search_stockx(sku, limit=1))
        goat.extend(client.search_goat(sku, limit=1))
    logger.info(
        "KicksDB live pull: %d SKUs -> %d StockX + %d GOAT records, %d requests",
        len(skus), len(stockx), len(goat), client.request_count,
    )
    return stockx, goat


def load_fixture_products(
    fixtures_dir: str = DEFAULT_FIXTURES_DIR,
) -> tuple[list[dict], list[dict]]:
    """Read canned, trimmed KicksDB responses (fixture / no-key mode)."""
    stockx = read_json(path_join(fixtures_dir, "stockx_products.json"))
    goat = read_json(path_join(fixtures_dir, "goat_products.json"))
    stockx = stockx.get("data", stockx) if isinstance(stockx, dict) else stockx
    goat = goat.get("data", goat) if isinstance(goat, dict) else goat
    logger.info(
        "KicksDB fixture mode: %d StockX + %d GOAT canned records from %s",
        len(stockx), len(goat), fixtures_dir,
    )
    return stockx, goat


# --------------------------------------------------------------------------- #
# The reshape: raw KicksDB records -> the four canonical frames.
# --------------------------------------------------------------------------- #
def _to_canonical_date(value) -> pd.Timestamp:
    """Normalize any release-date string to a tz-naive canonical date.

    GOAT returns ISO timestamps with a ``Z`` suffix ("2021-03-10T23:59:59.999Z"),
    which pandas parses as tz-aware. The canonical schema (and the Postgres path)
    carries release_date as a tz-naive DATE at midnight, so we drop the zone and
    normalize to midnight. Otherwise the raw layout wouldn't be byte-compatible
    with the other two sources, and datediff/Pandera would see a different dtype.
    """
    ts = pd.to_datetime(value, utc=True)
    return ts.tz_localize(None).normalize()


def _coerce(rows: list[dict], table: str) -> pd.DataFrame:
    """Build a frame with exactly the canonical columns and dtypes for a table."""
    dtypes = CANONICAL_DTYPES[table]
    df = pd.DataFrame(rows, columns=list(dtypes))
    for col, dtype in dtypes.items():
        if dtype.startswith("datetime64"):
            series = pd.to_datetime(df[col])
            # Defensive: never let a tz-aware column through -- the canonical
            # layout is tz-naive, and a mixed lake would read inconsistently.
            if getattr(series.dtype, "tz", None) is not None:
                series = series.dt.tz_localize(None)
            df[col] = series
        else:
            df[col] = df[col].astype(dtype)
    return df


def _stockx_market(product: dict) -> float | None:
    """Current market price for a StockX product, or None if it has no live ask.

    Uses ``avg_price`` (mean current ask across sizes) as the resale figure; 0 or
    missing means nothing is currently listed, so premium can't be computed.
    """
    avg = product.get("avg_price")
    try:
        avg = float(avg)
    except (TypeError, ValueError):
        return None
    return avg if avg > 0 else None


def _goat_release_date(product: dict) -> str | None:
    """Best release-date string GOAT exposes for a product."""
    for key in ("release_date", "release_date_year"):
        val = product.get(key)
        if val:
            return str(val)
    return None


def merge_by_sku(
    stockx: list[dict], goat: list[dict]
) -> dict[str, dict]:
    """Join StockX (market) and GOAT (release date) records on normalized SKU.

    StockX is the anchor because it carries the price the premium needs; GOAT
    contributes the release date. Returns normalized_sku -> merged raw fields.
    """
    goat_by_sku: dict[str, dict] = {}
    for g in goat:
        nsku = normalize_sku(g.get("sku", ""))
        if nsku and nsku not in goat_by_sku:
            goat_by_sku[nsku] = g

    merged: dict[str, dict] = {}
    for s in stockx:
        raw = s.get("sku", "")
        nsku = normalize_sku(raw)
        if not nsku or nsku in merged:
            continue
        # Attach GOAT by any candidate key so a compound StockX SKU still finds
        # the release date GOAT filed under the plain component.
        g = next(
            (goat_by_sku[c] for c in sku_candidates(raw) if c in goat_by_sku), {}
        )
        merged[nsku] = {
            "sku": raw,
            "normalized_sku": nsku,
            "brand": s.get("brand") or g.get("brand") or "",
            "model": s.get("model") or g.get("model") or "",
            "colorway": g.get("colorway", ""),
            "title": s.get("title") or g.get("name") or "",
            "silhouette": s.get("primary_title") or s.get("model") or "",
            "product_id": str(s.get("id", "")),
            "market_avg": _stockx_market(s),
            "release_date": _goat_release_date(g),
            "weekly_orders": s.get("weekly_orders") or 0,
        }
    return merged


def reshape_to_canonical(
    merged: dict[str, dict],
    reference: RetailReference,
    snapshot_date: str,
) -> dict[str, pd.DataFrame]:
    """Reshape merged KicksDB records into the four canonical raw frames.

    This is the whole anti-corruption job. A product lands only if we can build
    an honest row for it: a resolvable retail (from the reference), a live market
    price (StockX), and a release date (GOAT or reference fallback). Anything
    else is excluded and counted, never fabricated.

    Returns ``{table: DataFrame}`` with byte-compatible dtypes. Raises
    ``KicksDBError`` if nothing survives, with the exclusion breakdown, rather
    than letting the ingest stage fail on an empty frame with no explanation.
    """
    try:
        snapshot_ts = pd.Timestamp(snapshot_date)
    except (ValueError, TypeError) as exc:
        raise KicksDBError(
            f"snapshot_date {snapshot_date!r} is not a parseable date; the KicksDB "
            "adapter uses the run's run_date as the snapshot timestamp"
        ) from exc
    shoes_rows: list[dict] = []
    sales_rows: list[dict] = []
    drops_rows: list[dict] = []
    search_rows: list[dict] = []
    excluded = {
        "no_retail": 0, "identity_mismatch": 0, "no_market": 0, "no_release_date": 0
    }
    seen_shoe_ids: set[int] = set()

    sale_id = 1
    drop_id = 1
    for p in merged.values():
        # Look up on the raw SKU so compound codes ("BZ0057/B75806") resolve.
        ref = reference.lookup(p["sku"])
        if ref is None:
            excluded["no_retail"] += 1
            continue
        # Identity guard: the reference is authoritative for retail, so if the
        # curated brand disagrees with the brand the market source returned for
        # this SKU, the reference is mis-keyed and would pair the wrong retail
        # with this product's price. Exclude rather than fabricate a premium.
        ref_brand = (ref.brand or "").strip().lower()
        api_brand = (p["brand"] or "").strip().lower()
        if ref_brand and api_brand and ref_brand != api_brand:
            excluded["identity_mismatch"] += 1
            logger.warning(
                "SKU %s: reference brand %r != source brand %r; excluding to "
                "avoid pairing the wrong product's retail",
                p["sku"], ref.brand, p["brand"],
            )
            continue
        market = p.get("market_avg")
        if not market or market <= 0:
            excluded["no_market"] += 1
            continue
        release_date = p.get("release_date") or (ref.release_date or None)
        if not release_date:
            excluded["no_release_date"] += 1
            continue

        # Identity keys off the matched reference SKU (canonical), so a shoe gets
        # the same shoe_id whether StockX writes it plain or as a compound code.
        shoe_id = stable_shoe_id(normalize_sku(ref.sku))
        if shoe_id in seen_shoe_ids:  # crc32 collision guard; practically never
            raise KicksDBError(f"shoe_id collision for SKU {p['sku']!r}")
        seen_shoe_ids.add(shoe_id)

        release_type = ref.release_type or classify_release_type(p["title"])
        brand = ref.brand or p["brand"]
        release_ts = _to_canonical_date(release_date)

        shoes_rows.append(
            {
                "shoe_id": shoe_id,
                "brand": brand,
                "model": ref.model or p["model"],
                "colorway": ref.colorway or p["colorway"],
                "retail_price": float(ref.retail_price),
                "release_date": release_ts,
                "release_type": release_type,
                "silhouette": ref.silhouette or p["silhouette"],
            }
        )
        sales_rows.append(
            {
                "sale_id": sale_id,
                "shoe_id": shoe_id,
                "platform": "StockX",
                "sale_price": float(market),
                "sale_date": snapshot_ts,
                "size_us": REPRESENTATIVE_SIZE_US,
                "condition": "new",
                "source_item_id": p["product_id"],
            }
        )
        drops_rows.append(
            {
                "drop_id": drop_id,
                "shoe_id": shoe_id,
                "drop_date": release_ts,
                "drop_type": release_type,
                "announced_at": pd.NaT,
            }
        )
        search_rows.append(
            {
                "shoe_id": shoe_id,
                "signal_date": snapshot_ts,
                "platform": "kicksdb_weekly_orders",
                "search_index": int(p.get("weekly_orders") or 0),
            }
        )
        sale_id += 1
        drop_id += 1

    n_excluded = sum(excluded.values())
    logger.info(
        "KicksDB reshape: %d products landed, %d excluded (%s)",
        len(shoes_rows), n_excluded, excluded,
    )
    if not shoes_rows:
        raise KicksDBError(
            "no KicksDB products produced a canonical row "
            f"(candidates={len(merged)}, excluded={excluded}). "
            "Nothing has a resolvable retail + live market + release date."
        )

    return {
        "sales": _coerce(sales_rows, "sales"),
        "shoes": _coerce(shoes_rows, "shoes"),
        "drops": _coerce(drops_rows, "drops"),
        "search_interest": _coerce(search_rows, "search_interest"),
    }


def read_all(
    snapshot_date: str,
    api_key: str | None = None,
    reference_path: str = DEFAULT_REFERENCE_PATH,
    fixtures_dir: str = DEFAULT_FIXTURES_DIR,
) -> dict[str, pd.DataFrame]:
    """Produce the four canonical raw frames from KicksDB (live or fixtures).

    ``api_key`` present -> pull live for every reference SKU. Absent -> read the
    canned fixtures. Either way the reshape is identical, so the fixtures exercise
    the exact code path the live pull uses.
    """
    reference = load_retail_reference(reference_path)
    if api_key:
        client = KicksDBClient(api_key)
        stockx, goat = fetch_live_products(client, reference.skus())
        logger.info("KicksDB live mode: spent %d requests", client.request_count)
    else:
        logger.info("KicksDB fixture mode: no KICKSDB_API_KEY, reading canned responses")
        stockx, goat = load_fixture_products(fixtures_dir)

    merged = merge_by_sku(stockx, goat)
    return reshape_to_canonical(merged, reference, snapshot_date)
