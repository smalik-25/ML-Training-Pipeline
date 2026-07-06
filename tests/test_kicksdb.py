"""Tests for the KicksDB ingest adapter (fixture mode, no key, no network).

Covers the anti-corruption seam three ways, matching the ingest-stage contract:
the reshape turns KicksDB-shaped records into canonical rows with the right
columns and dtypes; the landed layout is byte-compatible with the synthetic and
Postgres modes; and with no ``KICKSDB_API_KEY`` the adapter falls back to canned
responses. Pure pandas/pyarrow, so it runs in the light CI job.
"""

from __future__ import annotations

import pandas as pd
import pytest

from generate_fixtures import build_fixtures
from stages import ingest
from stages.config import PipelineConfig
from stages.io import read_parquet
from stages.kicksdb import (
    CANONICAL_DTYPES,
    KicksDBError,
    ReferenceEntry,
    RetailReference,
    classify_release_type,
    load_fixture_products,
    load_retail_reference,
    merge_by_sku,
    normalize_sku,
    read_all,
    reshape_to_canonical,
    sku_candidates,
    stable_shoe_id,
)

# Same canonical column contract test_ingest.py asserts, re-used here so a drift
# in either the adapter or the fixtures fails a test.
EXPECTED_COLUMNS = {
    "sales": {
        "sale_id", "shoe_id", "platform", "sale_price", "sale_date",
        "size_us", "condition", "source_item_id",
    },
    "shoes": {
        "shoe_id", "brand", "model", "colorway", "retail_price",
        "release_date", "release_type", "silhouette",
    },
    "drops": {"drop_id", "shoe_id", "drop_date", "drop_type", "announced_at"},
    "search_interest": {"shoe_id", "signal_date", "platform", "search_index"},
}


def _config(root: str, run_date: str = "2026-07-04") -> PipelineConfig:
    return PipelineConfig(
        storage_root=root, raw_prefix="raw", features_prefix="features",
        validated_prefix="validated", models_prefix="models",
        failures_prefix="failures", run_date=run_date,
    )


def _merged(sku, market, release, brand="Nike", title="Nike Dunk Low"):
    """One entry as merge_by_sku would emit it, for controlled reshape tests."""
    return {
        "sku": sku, "normalized_sku": normalize_sku(sku), "brand": brand,
        "model": "Dunk Low", "colorway": "Panda", "title": title,
        "silhouette": "Dunk", "product_id": f"id-{sku}",
        "market_avg": market, "release_date": release, "weekly_orders": 3,
    }


def _reference(*entries: ReferenceEntry) -> RetailReference:
    return RetailReference(by_sku={normalize_sku(e.sku): e for e in entries})


# --------------------------------------------------------------------------- #
# Identity + classification
# --------------------------------------------------------------------------- #
def test_normalize_sku_folds_stockx_and_goat_forms() -> None:
    # StockX "DD1391-100" and GOAT "DD1391 100" must join.
    assert normalize_sku("DD1391-100") == normalize_sku("DD1391 100") == "DD1391-100"
    assert normalize_sku("cp9654") == "CP9654"
    assert normalize_sku("  555088_101  ") == "555088-101"
    assert normalize_sku("") == ""


def test_stable_shoe_id_is_deterministic_and_positive() -> None:
    a = stable_shoe_id(normalize_sku("DD1391-100"))
    b = stable_shoe_id(normalize_sku("DD1391 100"))
    assert a == b > 0
    assert stable_shoe_id("CP9654") != stable_shoe_id("HQ6316")


def test_classify_release_type_stays_in_vocabulary() -> None:
    assert classify_release_type("Off-White x Air Jordan 1") == "collab"
    assert classify_release_type("Yeezy Boost 350 V2 Zebra") == "limited"
    assert classify_release_type("Nike Dunk Low Panda") == "general"


def test_sku_candidates_split_compound_style_codes() -> None:
    # StockX ships some shoes with two "/"-joined codes; each is a candidate.
    assert sku_candidates("BZ0057/B75806") == ["BZ0057-B75806", "BZ0057", "B75806"]
    assert sku_candidates("DD1391-100") == ["DD1391-100"]


def test_compound_stockx_sku_resolves_reference_and_merges_goat() -> None:
    """A compound StockX SKU must still match its plain reference + GOAT record,
    or a coverable shoe is silently lost and mis-counted as retail-less."""
    ref = _reference(
        ReferenceEntry(sku="B75806", retail_price=100.0, release_type="general",
                       brand="Adidas"),
    )
    stockx = [{"sku": "BZ0057/B75806", "brand": "adidas", "avg_price": 120.0,
               "id": "sx", "title": "adidas Samba OG", "model": "Samba OG"}]
    goat = [{"sku": "B75806", "brand": "adidas",
             "release_date": "2018-05-09T23:59:59.999Z"}]
    frames = reshape_to_canonical(merge_by_sku(stockx, goat), ref, "2026-07-04")
    assert len(frames["shoes"]) == 1
    row = frames["shoes"].iloc[0]
    assert row["retail_price"] == 100.0
    # GOAT release date attached despite the compound StockX SKU.
    assert row["release_date"] == pd.Timestamp("2018-05-09")


def test_brand_mismatch_is_excluded_not_mis_paired() -> None:
    """If the reference brand disagrees with the market source's brand for a SKU,
    the reference is mis-keyed; exclude rather than pair the wrong retail."""
    ref = _reference(
        ReferenceEntry(sku="DN3707-100", retail_price=110.0,
                       release_type="general", brand="Nike", model="Dunk Low"),
    )
    # The market source returns a Jordan for this SKU, not the referenced Dunk.
    merged = {"DN3707-100": _merged("DN3707-100", 274.0, "2023-03-11Z",
                                    brand="Jordan", title="Jordan 3")}
    with pytest.raises(KicksDBError, match="no KicksDB products"):
        reshape_to_canonical(merged, ref, "2026-07-04")


# --------------------------------------------------------------------------- #
# Retail reference (the confirmed retail-price decision)
# --------------------------------------------------------------------------- #
def test_retail_reference_loads_committed_file() -> None:
    ref = load_retail_reference()
    assert len(ref) > 0
    dunk = ref.lookup("DD1391 100")  # GOAT-style SKU normalizes to the key
    assert dunk is not None
    assert dunk.retail_price == 110.0
    assert dunk.release_type == "general"


# --------------------------------------------------------------------------- #
# Merge + reshape
# --------------------------------------------------------------------------- #
def test_merge_joins_stockx_market_to_goat_release_on_sku() -> None:
    stockx = [{"sku": "DD1391-100", "brand": "Nike", "avg_price": 120.0,
               "id": "sx1", "title": "Nike Dunk Low", "model": "Dunk Low"}]
    goat = [{"sku": "DD1391 100", "release_date": "2021-03-10T23:59:59.999Z"}]
    merged = merge_by_sku(stockx, goat)
    assert set(merged) == {"DD1391-100"}
    row = merged["DD1391-100"]
    assert row["market_avg"] == 120.0
    assert row["release_date"] == "2021-03-10T23:59:59.999Z"


def test_reshape_produces_canonical_columns_and_dtypes() -> None:
    ref = _reference(
        ReferenceEntry(sku="DD1391-100", retail_price=110.0, release_type="general"),
    )
    merged = {"DD1391-100": _merged("DD1391-100", 220.0, "2021-03-10T23:59:59.999Z")}
    frames = reshape_to_canonical(merged, ref, "2026-07-04")

    for table, dtypes in CANONICAL_DTYPES.items():
        df = frames[table]
        assert set(df.columns) == EXPECTED_COLUMNS[table], f"{table} columns"
        assert list(df.dtypes.astype(str)) == list(dtypes.values()), f"{table} dtypes"

    # premium is real: (market 220 - retail 110) / 110 = 1.0
    row = frames["sales"].iloc[0]
    assert row["sale_price"] == 220.0
    assert frames["shoes"].iloc[0]["retail_price"] == 110.0
    # release_date landed tz-naive at midnight (canonical convention)
    rd = frames["shoes"].iloc[0]["release_date"]
    assert rd.tz is None and rd.hour == 0 and rd == pd.Timestamp("2021-03-10")


def test_reshape_excludes_products_without_retail_market_or_release() -> None:
    ref = _reference(
        ReferenceEntry(sku="AA3834-101", retail_price=190.0, release_type="collab"),
        ReferenceEntry(sku="CP9654", retail_price=220.0, release_type="limited"),
        ReferenceEntry(sku="DD1391-100", retail_price=110.0, release_type="general"),
    )
    merged = {
        # keeps: retail + market + release all present
        "AA3834-101": _merged("AA3834-101", 5000.0, "2017-09-06T23:59:59.999Z"),
        # dropped: SKU not in the reference -> no retail, never fabricated
        "UNKNOWN-999": _merged("UNKNOWN-999", 300.0, "2020-01-01T00:00:00Z"),
        # dropped: no live market (nothing currently listed)
        "CP9654": _merged("CP9654", 0.0, "2022-04-09T23:59:59.000Z"),
        # dropped: no release date anywhere
        "DD1391-100": _merged("DD1391-100", 150.0, None),
    }
    frames = reshape_to_canonical(merged, ref, "2026-07-04")
    assert len(frames["shoes"]) == 1
    assert frames["shoes"].iloc[0]["colorway"] == "Panda"


def test_reshape_falls_back_to_reference_release_date() -> None:
    ref = _reference(
        ReferenceEntry(sku="B75806", retail_price=100.0, release_type="general",
                       release_date="2018-05-09"),
    )
    # GOAT gave no release date; the reference fallback carries it.
    merged = {"B75806": _merged("B75806", 130.0, None)}
    frames = reshape_to_canonical(merged, ref, "2026-07-04")
    assert frames["shoes"].iloc[0]["release_date"] == pd.Timestamp("2018-05-09")


def test_reshape_raises_when_nothing_lands() -> None:
    ref = _reference(
        ReferenceEntry(sku="DD1391-100", retail_price=110.0, release_type="general"),
    )
    merged = {"UNKNOWN-1": _merged("UNKNOWN-1", 200.0, "2020-01-01Z")}
    with pytest.raises(KicksDBError, match="no KicksDB products"):
        reshape_to_canonical(merged, ref, "2026-07-04")


# --------------------------------------------------------------------------- #
# Fixture mode + byte-compatible layout (the whole point)
# --------------------------------------------------------------------------- #
def test_fixture_products_load_from_committed_responses() -> None:
    stockx, goat = load_fixture_products()
    assert len(stockx) > 0 and len(goat) > 0
    assert all("sku" in p for p in stockx)


def test_read_all_fixture_mode_lands_without_key() -> None:
    frames = read_all("2026-07-04", api_key=None)
    assert set(frames) == set(EXPECTED_COLUMNS)
    for table, cols in EXPECTED_COLUMNS.items():
        assert not frames[table].empty
        assert set(frames[table].columns) == cols


def test_ingest_kicksdb_mode_is_byte_compatible_with_fixtures(tmp_path, monkeypatch) -> None:
    """The KicksDB landing must match the synthetic/Postgres layout exactly."""
    monkeypatch.delenv("KICKSDB_API_KEY", raising=False)

    kick = _config(str(tmp_path / "kick"))
    written = ingest.run(kick, source="kicksdb")
    assert set(written) == set(EXPECTED_COLUMNS)

    # Oracle: the synthetic fixtures landed the normal way.
    syn_dir = str(tmp_path / "fix")
    build_fixtures(syn_dir)
    syn = _config(str(tmp_path / "syn"))
    syn_written = ingest.run(syn, dsn=None, fixtures_dir=syn_dir)

    for table in EXPECTED_COLUMNS:
        k = read_parquet(written[table])
        s = read_parquet(syn_written[table])
        assert set(k.columns) == set(s.columns) == EXPECTED_COLUMNS[table]
        assert list(k.dtypes.astype(str)) == list(s.dtypes.astype(str)), (
            f"{table} dtype layout drifted from the synthetic mode"
        )
        assert not k.empty


def test_ingest_kicksdb_premiums_are_computable_and_in_range(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("KICKSDB_API_KEY", raising=False)
    config = _config(str(tmp_path / "lake"))
    written = ingest.run(config, source="kicksdb")

    sales = read_parquet(written["sales"])
    shoes = read_parquet(written["shoes"])
    merged = sales.merge(shoes, on="shoe_id")
    premium = (merged["sale_price"] - merged["retail_price"]) / merged["retail_price"]
    # Every landed row carries a real retail from the reference, so premium is
    # always computable, and stays inside the Pandera contract's [-1, 50] band.
    assert premium.notna().all()
    assert premium.between(-1.0, 50.0).all()
    assert (shoes["retail_price"] > 0).all()


# --------------------------------------------------------------------------- #
# Live client control flow (mocked; no network)
# --------------------------------------------------------------------------- #
class _FakeResp:
    """Minimal stand-in for a requests.Response the client cares about."""

    def __init__(self, status_code=200, retry_after=None, payload=None):
        self.status_code = status_code
        self.headers = {"Retry-After": retry_after} if retry_after is not None else {}
        self._payload = payload or {"data": [{"sku": "DD1391-100"}]}

    def raise_for_status(self):
        import requests

        if self.status_code >= 400 and self.status_code != 429:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):
        return self._payload


def test_client_caches_repeated_requests(monkeypatch) -> None:
    """A repeated (path, params) request is served from the cache, so re-pulling
    the same product inside a run costs one network call, not two."""
    pytest.importorskip("requests")
    from stages.kicksdb import KicksDBClient

    calls = {"n": 0}

    def fake_get(url, params=None, timeout=None):
        calls["n"] += 1
        return _FakeResp()

    client = KicksDBClient("test-key", min_interval_s=0.0)
    monkeypatch.setattr(client._session, "get", fake_get)

    first = client.search_stockx("DD1391-100")
    second = client.search_stockx("DD1391-100")  # identical params -> cache hit
    assert first == second == [{"sku": "DD1391-100"}]
    assert calls["n"] == 1  # the second call never hits the network
    assert client.request_count == 1


def test_client_backs_off_on_429_then_succeeds(monkeypatch) -> None:
    """A 429 is retried after the Retry-After backoff, and the retry's body is
    returned. request_count counts both attempts."""
    pytest.importorskip("requests")
    from stages import kicksdb

    monkeypatch.setattr(kicksdb.time, "sleep", lambda *_: None)  # don't really wait
    statuses = [429, 200]

    def fake_get(url, params=None, timeout=None):
        return _FakeResp(status_code=statuses.pop(0), retry_after="0")

    client = kicksdb.KicksDBClient("test-key", min_interval_s=0.0)
    monkeypatch.setattr(client._session, "get", fake_get)

    out = client.search_stockx("DD1391-100")
    assert out == [{"sku": "DD1391-100"}]
    assert client.request_count == 2  # one 429 attempt, then the retry succeeds


def test_client_raises_kicksdberror_on_http_error(monkeypatch) -> None:
    """A non-429 HTTP error is surfaced as KicksDBError, not a bare requests one."""
    pytest.importorskip("requests")
    from stages import kicksdb

    def fake_get(url, params=None, timeout=None):
        return _FakeResp(status_code=500)

    client = kicksdb.KicksDBClient("test-key", min_interval_s=0.0)
    monkeypatch.setattr(client._session, "get", fake_get)
    with pytest.raises(kicksdb.KicksDBError, match="failed"):
        client.search_stockx("DD1391-100")


# --------------------------------------------------------------------------- #
# The snapshot feature split stays in sync with the model's feature list
# --------------------------------------------------------------------------- #
def test_snapshot_feature_split_covers_all_model_features() -> None:
    """The drift subset and the excluded set must together be exactly the model's
    features, with no overlap. Guards the 'keep in sync with FEATURE_COLUMNS'
    comment against a silent drift if the feature list ever changes."""
    from stages.kicksdb import SNAPSHOT_DRIFT_FEATURES, SNAPSHOT_EXCLUDED_FEATURES
    from stages.train import FEATURE_COLUMNS

    drift = set(SNAPSHOT_DRIFT_FEATURES)
    excluded = set(SNAPSHOT_EXCLUDED_FEATURES)
    assert drift.isdisjoint(excluded)
    assert drift | excluded == set(FEATURE_COLUMNS)
