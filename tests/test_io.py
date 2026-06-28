"""Tests for the shared I/O helper (stages/io.py).

This is the one piece of real logic in the Phase 0 scaffold, so it gets a real
round-trip test: write a DataFrame to a local path and read it back unchanged,
proving the fsspec-resolved I/O layer works end to end on the local filesystem
(the same code path that hits S3 in production).
"""

from __future__ import annotations

import pandas as pd

from stages.io import path_join, read_parquet, write_parquet


def test_parquet_roundtrip_local(tmp_path) -> None:
    df = pd.DataFrame({"shoe_id": [1, 2, 3], "price_premium": [0.5, 1.2, -0.1]})
    uri = str(tmp_path / "sales.parquet")
    write_parquet(df, uri)
    out = read_parquet(uri)
    pd.testing.assert_frame_equal(df, out)


def test_partitioned_roundtrip_local(tmp_path) -> None:
    df = pd.DataFrame(
        {"brand": ["Nike", "Adidas", "Nike"], "price_premium": [0.5, 0.2, 0.9]}
    )
    uri = str(tmp_path / "features")
    write_parquet(df, uri, partition_cols=["brand"])
    out = read_parquet(uri)
    assert set(out["brand"]) == {"Nike", "Adidas"}
    assert len(out) == 3


def test_path_join_preserves_s3_scheme() -> None:
    assert (
        path_join("s3://bucket/raw", "2025-01-01", "sales.parquet")
        == "s3://bucket/raw/2025-01-01/sales.parquet"
    )


def test_path_join_local() -> None:
    assert path_join("./data", "raw", "x.parquet") == "./data/raw/x.parquet"
