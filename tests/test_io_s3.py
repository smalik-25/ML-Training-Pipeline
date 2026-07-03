"""Prove the I/O layer works against the S3 API, using an in-process moto server.

This exercises the *same* ``stages.io`` code path that hits real AWS S3 in
production, pointed at a local moto endpoint via ``AWS_ENDPOINT_URL_S3``. It's
the zero-cost check that the fsspec/pyarrow storage abstraction actually round-
trips Parquet, bytes, and JSON over S3, not just the local filesystem.

Skipped where moto isn't installed (it's in the dev extra).
"""

from __future__ import annotations

import socket

import pandas as pd
import pytest

pytest.importorskip("moto")

import boto3  # noqa: E402
from moto.server import ThreadedMotoServer  # noqa: E402

from stages.io import (  # noqa: E402
    read_bytes,
    read_json,
    read_parquet,
    write_bytes,
    write_json,
    write_parquet,
)


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@pytest.fixture
def s3_bucket(monkeypatch):
    port = _free_port()
    server = ThreadedMotoServer(port=port)
    server.start()
    endpoint = f"http://127.0.0.1:{port}"
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", endpoint)
    boto3.client("s3", endpoint_url=endpoint).create_bucket(Bucket="lake")
    try:
        yield "s3://lake"
    finally:
        server.stop()


def test_parquet_roundtrip_over_s3(s3_bucket) -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
    uri = f"{s3_bucket}/features/x.parquet"
    write_parquet(df, uri)
    pd.testing.assert_frame_equal(read_parquet(uri), df)


def test_partitioned_parquet_over_s3(s3_bucket) -> None:
    df = pd.DataFrame({"brand": ["A", "B", "A"], "v": [1, 2, 3]})
    uri = f"{s3_bucket}/features/part"
    write_parquet(df, uri, partition_cols=["brand"])
    back = read_parquet(uri)
    assert set(back["brand"]) == {"A", "B"}
    assert len(back) == 3


def test_bytes_and_json_over_s3(s3_bucket) -> None:
    write_bytes(b"model-weights", f"{s3_bucket}/models/model.pt")
    assert read_bytes(f"{s3_bucket}/models/model.pt") == b"model-weights"

    write_json({"status": "passed", "rows": 42}, f"{s3_bucket}/reports/r.json")
    assert read_json(f"{s3_bucket}/reports/r.json") == {"status": "passed", "rows": 42}
