"""Shared filesystem I/O for all pipeline stages.

This is the single seam through which every stage reads and writes data. The
design goal: a stage's code does not know or care whether it is running against
S3 in production or the local filesystem in dev/CI. The filesystem is resolved
from the URI scheme:

    s3://bucket/raw/2025-01-01/sales.parquet   -> S3 (prod)
    ./data/fixtures/sales.parquet              -> local filesystem (dev/CI)
    file:///abs/path/sales.parquet             -> local filesystem (explicit)

Both paths flow through the exact same code, so fixtures exercise the real I/O
layer. We use ``pyarrow.fs`` (which sits on top of fsspec-style backends and
has native S3 support) rather than hand-rolling boto3 calls in each stage.

Tradeoff (documented in DEVLOG): the local path does not exercise true S3
semantics -- eventual consistency, multipart upload, IAM. We accept that in
exchange for zero-setup, fast CI. A MinIO compose file is provided for anyone
who wants a faithful local S3 API.
"""

from __future__ import annotations

import json
import posixpath
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq


def _resolve(uri: str) -> tuple[pafs.FileSystem, str]:
    """Resolve a URI to a (filesystem, path) pair.

    A bare or relative path (no ``scheme://``) is treated as a local path and
    normalised to an absolute one, so ``./data/x.parquet`` and
    ``file:///abs/data/x.parquet`` behave identically.
    """
    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        # Local path. urlparse puts everything in .path for "file://"; for a
        # bare path the whole string is the path. LocalFileSystem operates on
        # plain absolute paths.
        local_path = parsed.path if parsed.scheme == "file" else uri
        return pafs.LocalFileSystem(), _abspath(local_path)
    # Scheme-bearing URI (s3://, gs://, ...). Let pyarrow pick the backend.
    fs, path = pafs.FileSystem.from_uri(uri)
    return fs, path


def _abspath(path: str) -> str:
    import os

    return os.path.abspath(os.path.expanduser(path))


def read_parquet(uri: str) -> pd.DataFrame:
    """Read a Parquet file or partitioned dataset into a DataFrame.

    Handles both a single ``.parquet`` file and a directory of partitioned
    Parquet (e.g. output partitioned by brand).
    """
    fs, path = _resolve(uri)
    table = pq.read_table(path, filesystem=fs)
    return table.to_pandas()


def write_parquet(
    df: pd.DataFrame,
    uri: str,
    partition_cols: list[str] | None = None,
) -> str:
    """Write a DataFrame to Parquet at ``uri``.

    If ``partition_cols`` is given, writes a partitioned dataset (a directory)
    using Hive-style partitioning. Returns the URI written, so callers can push
    it to XCom.
    """
    fs, path = _resolve(uri)
    table = pa.Table.from_pandas(df, preserve_index=False)
    # Coerce timestamps to microseconds. pandas/pyarrow default to nanoseconds,
    # which Spark 3.5's Parquet reader rejects ("Illegal Parquet type: INT64
    # TIMESTAMP(NANOS)"). Microsecond precision keeps the raw lake readable by
    # both the pyarrow stages and the Spark feature stage.
    _ts_kwargs = {"coerce_timestamps": "us", "allow_truncated_timestamps": True}
    if partition_cols:
        pq.write_to_dataset(
            table,
            root_path=path,
            partition_cols=partition_cols,
            filesystem=fs,
            **_ts_kwargs,
        )
    else:
        _ensure_parent(fs, path)
        with fs.open_output_stream(path) as sink:
            pq.write_table(table, sink, **_ts_kwargs)
    return uri


def read_json(uri: str) -> dict[str, Any]:
    """Read a JSON document (e.g. a validation or promotion report)."""
    fs, path = _resolve(uri)
    with fs.open_input_stream(path) as src:
        return json.loads(src.read().decode("utf-8"))


def write_json(obj: dict[str, Any], uri: str) -> str:
    """Write a JSON document. Returns the URI written."""
    fs, path = _resolve(uri)
    _ensure_parent(fs, path)
    payload = json.dumps(obj, indent=2, default=str).encode("utf-8")
    with fs.open_output_stream(path) as sink:
        sink.write(payload)
    return uri


def write_bytes(data: bytes, uri: str) -> str:
    """Write raw bytes (e.g. a serialized model.pt) to ``uri``. Returns the URI."""
    fs, path = _resolve(uri)
    _ensure_parent(fs, path)
    with fs.open_output_stream(path) as sink:
        sink.write(data)
    return uri


def read_bytes(uri: str) -> bytes:
    """Read raw bytes from ``uri``."""
    fs, path = _resolve(uri)
    with fs.open_input_stream(path) as src:
        return src.read()


def path_join(base: str, *parts: str) -> str:
    """Join URI parts while preserving any ``scheme://`` prefix.

    ``path_join("s3://b/raw", "2025-01-01", "sales.parquet")`` ->
    ``"s3://b/raw/2025-01-01/sales.parquet"``.
    """
    parsed = urlparse(base)
    if parsed.scheme:
        prefix = f"{parsed.scheme}://{parsed.netloc}"
        joined = posixpath.join(parsed.path.lstrip("/"), *parts)
        return f"{prefix}/{joined}"
    return posixpath.join(base, *parts)


def _ensure_parent(fs: pafs.FileSystem, path: str) -> None:
    """Create the parent directory for ``path`` if the backend needs it.

    Object stores (S3) do not need directories, but the local filesystem does.
    ``create_dir`` is a no-op-safe call on S3.
    """
    parent = posixpath.dirname(path)
    if parent:
        fs.create_dir(parent, recursive=True)


def to_spark_path(uri: str) -> str:
    """Translate a pipeline URI into a path Spark's Hadoop layer understands.

    Spark does not use pyarrow's filesystem; it reads/writes through Hadoop, so
    S3 must be addressed as ``s3a://`` and local paths must be absolute. This is
    the Spark-side counterpart to ``_resolve`` for the pyarrow stages.

        s3://bucket/key   -> s3a://bucket/key
        ./data/raw/x      -> /abs/data/raw/x
        file:///abs/x     -> /abs/x
    """
    import os

    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        local_path = parsed.path if parsed.scheme == "file" else uri
        return os.path.abspath(os.path.expanduser(local_path))
    if parsed.scheme == "s3":
        return uri.replace("s3://", "s3a://", 1)
    return uri
