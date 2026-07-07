"""Refresh the demo's live market snapshot from KicksDB.

Run on a schedule (CI) or by hand. Pulls the current KicksDB market through the
real ingest adapter, scores the covered shoes through the one inference path
(``stages.live_score`` -> ``stages.inference``), and writes a compact JSON the
Streamlit Space reads. The Space renders that file, so it loads instantly and
never calls KicksDB itself on a normal visit.

Live when ``KICKSDB_API_KEY`` is set; otherwise it runs on the committed fixtures,
so CI and a clean clone produce a snapshot without a key. The board carries the
model's implied resale next to KicksDB's real current market price, so the
out-of-distribution gap is visible against live reality, not just asserted.

    KICKSDB_API_KEY=... python refresh_market_snapshot.py --run-date 2026-07-06

Writes ``demo/live_snapshot.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

from stages.config import PipelineConfig, load_feature_config
from stages.inference import load_bundle_from_uri
from stages.live_score import score_current

logger = logging.getLogger("refresh_market_snapshot")

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = str(REPO_ROOT / "demo" / "model.pt")
DEFAULT_OUT = str(REPO_ROOT / "demo" / "live_snapshot.json")
DEFAULT_HISTORY = str(REPO_ROOT / "demo" / "drift_history.json")
# Keep ~60 days of six-hourly points; the trend is a rolling window, not a log.
HISTORY_CAP = 240


def _note(brand: str, model: str, colorway: str) -> tuple[str, str]:
    """A display name and a colorway note from the canonical identity columns."""
    name = f"{brand} {model}".strip()
    return name, (colorway or "").strip()


def build_snapshot(run_date: str, model_uri: str) -> dict:
    """Pull + score the current KicksDB board. Returns the JSON-ready snapshot.

    Uses ``score_current`` so the board goes through the exact impute/standardize/
    forward the batch path uses. ``sale_price`` is KicksDB's real current market
    average; ``predicted_premium`` / ``implied_resale`` are the model's take, kept
    side by side so a viewer sees prediction against reality.
    """
    api_key = os.environ.get("KICKSDB_API_KEY") or None
    fcfg = load_feature_config(str(REPO_ROOT / "config" / "feature_config.yaml"))
    config = PipelineConfig(
        storage_root=str(REPO_ROOT / "data"), raw_prefix="raw",
        features_prefix="features", validated_prefix="validated",
        models_prefix="models", failures_prefix="failures", run_date=run_date,
    )
    bundle = load_bundle_from_uri(model_uri)

    scored = score_current(config, fcfg, bundle=bundle, api_key=api_key)

    board = []
    for row in scored.itertuples(index=False):
        name, note = _note(row.brand, row.model, row.colorway)
        board.append({
            "name": name,
            "note": note,
            "retail": round(float(row.retail_price), 2),
            "market": round(float(row.sale_price), 2),  # real current KicksDB ask
            "premium": round(float(row.predicted_premium), 4),  # model's take
            "implied_resale": round(float(row.implied_resale), 2),
        })
    board.sort(key=lambda b: b["premium"], reverse=True)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "source": "live" if api_key else "fixtures",
        "run_date": run_date,
        "model_version": bundle.model_version,
        "n_scored": len(board),
        "uncomputable_skus": scored.attrs.get("uncomputable_skus", []),
        "board": board,
    }


def append_history(snapshot: dict, path: str, cap: int = HISTORY_CAP) -> dict | None:
    """Append one market-vs-model point to the rolling drift history.

    Each refresh records the board's mean model premium next to the mean real
    market premium (``(market - retail) / retail``). Over time that pair is the
    out-of-distribution gap as a trend: the model's view stays put while the live
    market moves and shoes age, so the distance between the two lines is the drift
    a single snapshot can only assert. Rolling window, not an append-only log.
    """
    board = snapshot.get("board") or []
    if not board:
        return None
    # Median, not mean: a couple of hyped shoes (an Off-White AJ1 trading at 25x
    # retail) blow up the average and hide the typical-shoe story, which is the
    # honest out-of-distribution point -- the median current shoe trades near
    # retail while the model still runs hot.
    point = {
        "ts": snapshot["generated_at"],
        "n": len(board),
        "model_premium": round(statistics.median(b["premium"] for b in board), 4),
        "market_premium": round(
            statistics.median((b["market"] - b["retail"]) / b["retail"] for b in board),
            4,
        ),
    }
    try:
        history = json.loads(Path(path).read_text())
        if not isinstance(history, list):
            history = []
    except (FileNotFoundError, json.JSONDecodeError):
        history = []
    history.append(point)
    Path(path).write_text(json.dumps(history[-cap:], indent=2))
    return point


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the demo live snapshot.")
    parser.add_argument("--run-date", required=True, help="ISO snapshot date.")
    parser.add_argument("--model-uri", default=DEFAULT_MODEL)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--history", default=DEFAULT_HISTORY)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    snapshot = build_snapshot(args.run_date, args.model_uri)
    Path(args.out).write_text(json.dumps(snapshot, indent=2))
    point = append_history(snapshot, args.history)
    logger.info(
        "wrote %s: %d shoes scored (source=%s); trend point market=%s model=%s",
        args.out, snapshot["n_scored"], snapshot["source"],
        point and point["market_premium"], point and point["model_premium"],
    )


if __name__ == "__main__":
    main()
