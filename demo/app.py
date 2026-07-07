"""SAM·MALIK — sneaker ML platform, live demo.

A Streamlit dashboard over the offline ML training pipeline, told as one story
across three tabs: The Model (the honest 2017-2019 predictor and its evaluation),
Live Market (current KicksDB sneakers scored through the same path, running hot and
out of distribution, with the drift monitor that measures the gap), and How It
Works (the pipeline and the seams, now landing two real sources through one
anti-corruption layer). Styled to the Sam Malik design system (terminal-meets-
gothic: phosphor on void, Cormorant / Space Grotesk / IBM Plex Mono, hairline
structure, decimal indices, no emoji).
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from net import FEATURE_COLUMNS, load_bundle, predict

MODEL_PATH = str(Path(__file__).parent / "model.pt")
SNAPSHOT_PATH = Path(__file__).parent / "live_snapshot.json"
HISTORY_PATH = Path(__file__).parent / "drift_history.json"
REPO_URL = "https://github.com/smalik-25/ML-Training-Pipeline"
SNEAKER_INTEL_URL = "https://sneaker-intel-2.streamlit.app/"
WEBSITE_URL = "https://sam-malik.com"

# Real numbers from the 99,956-row StockX run (see the repo DEVLOG).
RUN = {
    "model_name": "sneaker-price-model",
    "model_version": "1",
    "val_rmse": 0.209,
    "n_train": "83.8K",
    "n_val": "16.2K",
    "split_year": 2019,
    "n_features": len(FEATURE_COLUMNS),
    "brands": "Off-White · Yeezy",
    "rows_scored": "99,956",
    "scoring_rmse": 0.295,
    "retention": "100%",
    "drift_features": "0 / 8",
    "pre_release": "5,601",
}

RELEASE_TYPES = {"General": 0, "Collab": 1, "Limited": 2}

# Preset sales: full 8-feature vectors, realistic. None = imputed at inference.
# All three are in-distribution: shapes the 2017-2019 model actually saw.
PRESETS = {
    "Off-White · limited": dict(
        days_since_release=120, size_us=9.0, retail_price=190.0, size_premium=0.05,
        release_type_encoded=2, rolling_7d_avg_premium=1.4,
        search_index_7d_pre_drop=None, brand_avg_premium=1.10,
    ),
    "Nike Dunk · general": dict(
        days_since_release=210, size_us=10.0, retail_price=110.0, size_premium=0.0,
        release_type_encoded=0, rolling_7d_avg_premium=0.20,
        search_index_7d_pre_drop=None, brand_avg_premium=0.22,
    ),
    "Yeezy · resale": dict(
        days_since_release=320, size_us=9.5, retail_price=220.0, size_premium=0.02,
        release_type_encoded=2, rolling_7d_avg_premium=0.55,
        search_index_7d_pre_drop=None, brand_avg_premium=0.50,
    ),
}

# Current sneakers pulled from KicksDB (kicks.dev), scored through the same model.
# Retail comes from a curated SKU reference (the Starter tier doesn't expose it);
# the snapshot-uncomputable features (rolling, pre-drop search, brand average) are
# left null and imputed with the training means. These are out of distribution vs
# the 2017-2019 training slice, so the model runs hot -- shown as-is, not corrected.
CURRENT_KICKSDB = [
    {"name": "Nike Dunk Low", "note": "Panda", "retail": 110.0, "days": 1942, "rt": 0},
    {"name": "Off-White × AJ1", "note": "Chicago", "retail": 190.0, "days": 3223, "rt": 1},
    {"name": "Travis Scott × AJ1 Low", "note": "Mocha", "retail": 150.0, "days": 2542, "rt": 1},
    {"name": "Yeezy 350 V2", "note": "Zebra", "retail": 220.0, "days": 1547, "rt": 2},
]

# A current release with no retail in the 13-SKU reference. Premium can't be
# computed against a retail we don't have, so it is reported, never scored on a
# guess. This is the retail-reference coverage ceiling, made concrete.
UNCOMPUTABLE_KICKSDB = {"name": "KAWS × Air Force 1", "note": "no retail reference"}

# The real drift run: the current KicksDB snapshot against the staging model's
# 2017-2019 training distribution. PSI per feature; the retrain gate trips at 0.2.
# A current-only snapshot honestly supports four of the eight features; the two
# that moved most blow past the threshold. The other four are structurally
# uncomputable from a snapshot and excluded with a reason rather than measured on
# degenerate data.
PSI_THRESHOLD = 0.2
DRIFT_COMPUTED = [
    ("days_since_release", 12.0, True),
    ("retail_price", 1.8, True),
    ("release_type_encoded", None, False),
    ("brand_avg_premium", None, False),
]
DRIFT_EXCLUDED = [
    ("size_us", "a snapshot lands one representative size, not real dispersion"),
    ("size_premium", "needs several sizes per shoe; collapses to 0 on one"),
    ("rolling_7d_avg_premium", "needs the per-sale history the Starter tier omits"),
    ("search_index_7d_pre_drop", "needs a pre-drop demand signal a snapshot lacks"),
]


def _kicksdb_record(item: dict) -> dict:
    """One current sneaker's feature row: real retail/release, the rest imputed."""
    return {
        "days_since_release": item["days"], "size_us": 9.0,
        "retail_price": item["retail"], "size_premium": 0.0,
        "release_type_encoded": item["rt"], "rolling_7d_avg_premium": None,
        "search_index_7d_pre_drop": None, "brand_avg_premium": None,
    }


st.set_page_config(page_title="SAM·MALIK — sneaker ML", layout="wide")

# --------------------------------------------------------------------------- #
# Design system
# --------------------------------------------------------------------------- #
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant:ital,wght@0,300..700;1,400..600&family=Space+Grotesk:wght@300..700&family=IBM+Plex+Mono:ital,wght@0,400..600;1,400..500&display=swap');

:root {
  --pitch:#060608; --void:#0b0b0f; --slab:#131318; --slab-2:#1b1b22;
  --hairline:#26262e; --hairline-2:#34343e;
  --bone:#ece7d8; --bone-dim:#b6b1a4; --ash:#847f74; --ash-dim:#555049;
  --phosphor:#c8f24a; --phosphor-dim:#9bbd38; --phosphor-deep:#2c360f;
  --phosphor-glow:rgba(200,242,74,.32);
  --oxblood:#8e1c24; --oxblood-lift:#b3303a; --oxblood-deep:#2a0c0f;
  --warn:#e0a838; --info:#6f8fb8;
  --font-display:'Cormorant',serif; --font-ui:'Space Grotesk',sans-serif;
  --font-mono:'IBM Plex Mono',monospace;
}

.stApp, [data-testid="stAppViewContainer"] { background: var(--void); }
/* hauntological grain: restrained fractal noise, screen-blended */
.stApp::before { content:""; position:fixed; inset:0; z-index:9999;
  pointer-events:none; opacity:.035; mix-blend-mode:screen;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E"); }
#MainMenu, header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display:none !important; }
.block-container { max-width: 1080px; padding-top: 2.5rem; padding-bottom: 5rem; }

html, body, [class*="css"], .stMarkdown, p, span, div, label {
  font-family: var(--font-ui); color: var(--bone-dim);
}
h1,h2,h3,h4 { font-family: var(--font-display); color: var(--bone);
  letter-spacing:-.02em; line-height:1.02; font-weight:500; }

/* mono label */
.sm-label { font-family:var(--font-mono); font-size:11px; letter-spacing:.14em;
  text-transform:uppercase; color:var(--ash); }
.sm-mid { color: var(--phosphor); }

/* hero */
.sm-eyebrow { font-family:var(--font-mono); font-size:12px; letter-spacing:.16em;
  text-transform:uppercase; color:var(--ash); margin-bottom:1.4rem; }
.sm-wordmark { font-family:var(--font-display); font-size:1.5rem; color:var(--bone);
  letter-spacing:.02em; }
.sm-display { font-family:var(--font-display); font-style:italic; font-weight:500;
  font-size:4rem; line-height:1.0; color:var(--bone); letter-spacing:-.03em;
  margin:.6rem 0 1rem; }
.sm-status { font-family:var(--font-mono); font-size:12px; letter-spacing:.06em;
  color:var(--phosphor); }
.sm-status .dot { display:inline-block; width:7px; height:7px; border-radius:999px;
  background:var(--phosphor); box-shadow:0 0 8px var(--phosphor-glow); margin-right:8px;
  vertical-align:middle; }
.sm-lede { font-size:1.05rem; color:var(--bone-dim); line-height:1.6; max-width:60ch;
  margin-top:1.2rem; }

/* rule / section divider */
.sm-rule { display:flex; align-items:center; gap:14px; margin:3rem 0 1.4rem; }
.sm-rule .line { flex:1; height:1px; background:var(--hairline); }
.sm-rule .lab { font-family:var(--font-mono); font-size:11px; letter-spacing:.14em;
  text-transform:uppercase; color:var(--ash); }

/* datafield grid */
.sm-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
  gap:1px; background:var(--hairline); border:1px solid var(--hairline); }
.sm-field { background:var(--slab); padding:16px 18px; }
.sm-field .k { font-family:var(--font-mono); font-size:10.5px; letter-spacing:.14em;
  text-transform:uppercase; color:var(--ash); }
.sm-field .idx { color:var(--phosphor-dim); }
.sm-field .v { font-family:var(--font-mono); font-size:1.35rem; color:var(--bone);
  margin-top:6px; }
.sm-field .u { font-size:.8rem; color:var(--ash); }
/* out-of-distribution / hot field */
.sm-field.hot { border-left:2px solid var(--oxblood); }
.sm-field.hot .v { color:var(--oxblood-lift); }
.sm-field.dim { opacity:.62; }
.sm-field.dim .v { color:var(--ash); }

/* stage flow */
.sm-flow { display:flex; flex-wrap:wrap; gap:0; border:1px solid var(--hairline); }
.sm-stage { flex:1; min-width:120px; background:var(--slab); padding:14px 16px;
  border-right:1px solid var(--hairline); }
.sm-stage:last-child { border-right:none; }
.sm-stage .i { font-family:var(--font-mono); font-size:10px; letter-spacing:.14em;
  color:var(--phosphor-dim); }
.sm-stage .n { font-family:var(--font-display); font-size:1.15rem; color:var(--bone);
  margin:4px 0 2px; }
.sm-stage .t { font-family:var(--font-mono); font-size:10px; letter-spacing:.08em;
  text-transform:uppercase; color:var(--ash); }
.sm-arrow { color:var(--ash); font-family:var(--font-mono); }

/* prediction readout */
.sm-readout { border:1px solid var(--phosphor); background:var(--phosphor-deep);
  padding:22px 24px; }
.sm-readout .k { font-family:var(--font-mono); font-size:11px; letter-spacing:.14em;
  text-transform:uppercase; color:var(--phosphor-dim); }
.sm-readout .v { font-family:var(--font-mono); font-size:2.6rem; color:var(--phosphor);
  text-shadow:0 0 14px var(--phosphor-glow); line-height:1.1; margin-top:4px; }
.sm-readout .sub { font-size:.95rem; color:var(--bone-dim); margin-top:6px; }
/* hot readout: the out-of-distribution prediction, marked not hidden */
.sm-readout.hot { border-color:var(--oxblood); background:var(--oxblood-deep); }
.sm-readout.hot .k { color:var(--oxblood-lift); }
.sm-readout.hot .v { color:var(--oxblood-lift); text-shadow:0 0 14px rgba(179,48,58,.35); }

/* then vs now comparison */
.sm-compare { display:grid; grid-template-columns:1fr auto 1fr; gap:0;
  border:1px solid var(--hairline); }
.sm-side { background:var(--slab); padding:20px 22px; }
.sm-side.now { background:var(--oxblood-deep); }
.sm-side .h { font-family:var(--font-mono); font-size:10.5px; letter-spacing:.14em;
  text-transform:uppercase; color:var(--ash); }
.sm-side.now .h { color:var(--oxblood-lift); }
.sm-side .b { font-family:var(--font-display); font-size:1.4rem; color:var(--bone);
  margin:8px 0 4px; line-height:1.1; }
.sm-side .d { font-size:.92rem; color:var(--bone-dim); line-height:1.5; }
.sm-vs { display:flex; align-items:center; justify-content:center; padding:0 16px;
  background:var(--void); font-family:var(--font-mono); font-size:11px; color:var(--ash);
  letter-spacing:.12em; }

/* psi drift bars */
.sm-psi { border:1px solid var(--hairline); }
.sm-psi-row { display:grid; grid-template-columns:200px 1fr 96px; align-items:center;
  gap:14px; padding:11px 16px; border-bottom:1px solid var(--hairline);
  background:var(--slab); }
.sm-psi-row:last-child { border-bottom:none; }
.sm-psi-row .f { font-family:var(--font-mono); font-size:11px; color:var(--bone-dim);
  letter-spacing:.03em; }
.sm-psi-track { position:relative; height:9px; background:var(--pitch);
  border:1px solid var(--hairline); }
.sm-psi-fill { position:absolute; top:0; bottom:0; left:0; }
.sm-psi-thresh { position:absolute; top:-3px; bottom:-3px; width:1px;
  background:var(--bone-dim); }
.sm-psi-val { font-family:var(--font-mono); font-size:11px; text-align:right; }
.sm-psi-note { font-family:var(--font-mono); font-size:10.5px; letter-spacing:.03em;
  color:var(--ash); padding:11px 16px; background:var(--slab);
  border-bottom:1px solid var(--hairline); }
.sm-psi-note:last-child { border-bottom:none; }
.sm-psi-note b { color:var(--bone-dim); font-weight:400; }

/* tab bar */
[data-baseweb="tab-list"] { gap:2px; background:transparent;
  border-bottom:1px solid var(--hairline); margin-bottom:1.8rem; }
[data-baseweb="tab"] { background:transparent !important; border-radius:0;
  padding:11px 20px 12px; height:auto; }
[data-baseweb="tab"] [data-testid="stMarkdownContainer"] p,
[data-baseweb="tab"] p { font-family:var(--font-mono) !important; font-size:11px;
  letter-spacing:.14em; text-transform:uppercase; color:var(--ash); margin:0; }
[data-baseweb="tab"][aria-selected="true"] [data-testid="stMarkdownContainer"] p,
[data-baseweb="tab"][aria-selected="true"] p { color:var(--phosphor); }
[data-baseweb="tab-highlight"] { background:var(--phosphor); height:2px; }
[data-baseweb="tab-border"] { background:transparent; }

.sm-foot { font-family:var(--font-mono); font-size:11px; letter-spacing:.1em;
  color:var(--ash); }
.sm-foot a, .sm-lede a { color:var(--phosphor); text-decoration:none;
  border-bottom:1px solid var(--phosphor-dim); }

/* widgets */
.stButton>button { font-family:var(--font-mono); text-transform:uppercase;
  letter-spacing:.14em; font-size:11px; border-radius:2px; background:transparent;
  color:var(--bone); border:1px solid var(--hairline-2); transition:all .18s; }
.stButton>button:hover { border-color:var(--phosphor); color:var(--phosphor);
  background:var(--phosphor-deep); }
.stButton>button[kind="primary"] { background:var(--phosphor); color:var(--void);
  border-color:var(--phosphor); box-shadow:0 0 14px var(--phosphor-glow); }
.stButton>button[kind="primary"]:hover { background:var(--phosphor-dim); }
[data-testid="stNumberInput"] input, [data-baseweb="select"] > div {
  background:var(--pitch) !important; border:1px solid var(--hairline) !important;
  border-radius:2px !important; color:var(--bone) !important; }
label, .stSelectbox label, [data-testid="stWidgetLabel"] p {
  font-family:var(--font-mono) !important; font-size:10.5px !important;
  letter-spacing:.14em !important; text-transform:uppercase !important;
  color:var(--ash) !important; }
[data-testid="stExpander"] { border:1px solid var(--hairline) !important;
  border-radius:2px !important; background:var(--slab); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def _bundle():
    return load_bundle(MODEL_PATH)


@st.cache_data(ttl=300)
def _load_snapshot() -> dict | None:
    """The scheduled KicksDB market snapshot, if the refresh job has published one.

    Cached for five minutes so the Space picks up a freshly pushed snapshot without
    a restart and never blocks a page load on the file read. Missing file means a
    clean clone with no snapshot yet, so the tab falls back to canned records.
    """
    try:
        return json.loads(SNAPSHOT_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


@st.cache_data(ttl=300)
def _load_history() -> list | None:
    """The rolling market-vs-model drift history the refresh job accumulates."""
    try:
        data = json.loads(HISTORY_PATH.read_text())
        return data if isinstance(data, list) else None
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _trend_svg(history: list) -> str:
    """A two-line sparkline: market premium vs the model's, over time.

    Phosphor is the real market, oxblood is the model. The hairline is retail
    (premium 0). As shoes age and the market moves, the gap between the lines is
    the drift a single snapshot can only assert.
    """
    market = [h["market_premium"] for h in history]
    model = [h["model_premium"] for h in history]
    lo = min(min(market), min(model), 0.0)
    hi = max(max(market), max(model))
    span = (hi - lo) or 1.0
    w, h = 100.0, 30.0
    n = len(history)

    def _line(series: list) -> str:
        return " ".join(
            f"{i / (n - 1) * w:.2f},{h - (v - lo) / span * h:.2f}"
            for i, v in enumerate(series)
        )

    zero_y = h - (0.0 - lo) / span * h
    return (
        f'<svg viewBox="0 0 {w:.0f} {h:.0f}" preserveAspectRatio="none" '
        'style="width:100%;height:130px;display:block;border:1px solid var(--hairline);'
        'background:var(--slab)">'
        f'<line x1="0" y1="{zero_y:.2f}" x2="{w:.0f}" y2="{zero_y:.2f}" '
        'stroke="var(--hairline-2)" stroke-width="0.4" stroke-dasharray="2 2"/>'
        f'<polyline points="{_line(model)}" fill="none" '
        'stroke="var(--oxblood-lift)" stroke-width="0.9" '
        'vector-effect="non-scaling-stroke"/>'
        f'<polyline points="{_line(market)}" fill="none" '
        'stroke="var(--phosphor)" stroke-width="0.9" '
        'vector-effect="non-scaling-stroke"/>'
        "</svg>"
    )


def rule(label: str) -> None:
    st.markdown(
        f'<div class="sm-rule"><span class="lab">{label}</span>'
        f'<span class="line"></span></div>', unsafe_allow_html=True)


def field(idx: str, key: str, value: str, unit: str = "", cls: str = "") -> str:
    unit = f' <span class="u">{unit}</span>' if unit else ""
    cls = f" {cls}" if cls else ""
    return (f'<div class="sm-field{cls}"><div class="k"><span class="idx">{idx}</span> '
            f'{key}</div><div class="v">{value}{unit}</div></div>')


def lede(text: str, top: str = "1rem") -> None:
    st.markdown(
        f'<p class="sm-lede" style="margin-top:{top}">{text}</p>',
        unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Hero (shared, above the tabs)
# --------------------------------------------------------------------------- #
st.markdown(
    f"""
    <div class="sm-eyebrow">§ 0.0 — ML INFRASTRUCTURE · SNEAKER-INTEL PHASE 2</div>
    <div class="sm-wordmark">SAM<span class="sm-mid">·</span>MALIK</div>
    <div class="sm-display">Price premiums, predicted<br>to the edge of the data.</div>
    <div class="sm-status"><span class="dot"></span>pipeline sneaker.0.8 // green
      &nbsp;·&nbsp; {RUN['rows_scored']} real StockX sales · two live sources</div>
    <p class="sm-lede">I build the plumbing that moves data quietly and correctly.
    This is the ML layer I deferred in <a href="{SNEAKER_INTEL_URL}">sneaker-intel</a>
    and came back to build: ingest, feature engineering, a loud data contract,
    distributed training, a model registry, batch and online serving, and drift
    monitoring. The model is simple on purpose. What follows is one predictor read
    three ways: the model on its own turf, the same model hitting the edge of what
    it learned, and the machinery that makes both legible.</p>
    """,
    unsafe_allow_html=True,
)

tab_model, tab_live, tab_arch = st.tabs(
    ["The Model", "Live Market", "How It Works"]
)

# =========================================================================== #
# TAB 1 — THE MODEL (the in-distribution story)
# =========================================================================== #
with tab_model:
    lede(
        "The original story, and the honest one. A feedforward model trained on "
        f"{RUN['rows_scored']} StockX sales from 2017 to 2019, {RUN['brands']}, "
        "predicting resale premium over retail. It knows this era. Everything on "
        "this tab is in distribution: the predictor defaults to an input the model "
        "actually saw, so the first number you get back is a sane one.",
        top=".2rem",
    )

    # ----------------------------------------------------------------------- #
    # Predictor
    # ----------------------------------------------------------------------- #
    rule("§ 1.1 — PREDICT")
    lede(
        "Enter a sale's engineered features; the model returns the predicted resale "
        "premium over retail. The same imputation and standardization fit during "
        "training are applied here from the stats saved inside the model, so this "
        "matches what the model learned.",
        top=".2rem",
    )

    # seed defaults from an in-distribution preset
    for _k, _v in PRESETS["Off-White · limited"].items():
        st.session_state.setdefault(_k, _v)

    pcols = st.columns(len(PRESETS))
    for col, (name, vals) in zip(pcols, PRESETS.items()):
        if col.button(name, use_container_width=True):
            for _k, _v in vals.items():
                st.session_state[_k] = _v
            st.rerun()

    c1, c2, c3, c4 = st.columns(4)
    retail = c1.number_input("retail price · $", min_value=40.0, max_value=1000.0,
                             step=10.0, key="retail_price")
    days = c2.number_input("days since release", min_value=-90, max_value=3000,
                           step=10, key="days_since_release")
    _rt_names = list(RELEASE_TYPES)
    rt_name = c3.selectbox("release type", _rt_names,
                           index=[RELEASE_TYPES[n] for n in _rt_names].index(
                               int(st.session_state["release_type_encoded"])))
    size = c4.selectbox("us size", [7.0, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 12.0],
                        key="size_us")

    with st.expander("§ advanced — engineered signals (default to training means)"):
        a1, a2, a3 = st.columns(3)
        brand_avg = a1.number_input("brand avg premium", min_value=-1.0, max_value=20.0,
                                    step=0.05, key="brand_avg_premium")
        size_prem = a2.number_input("size premium", min_value=-2.0, max_value=2.0,
                                    step=0.01, key="size_premium")
        rolling = a3.number_input("rolling 7d premium", min_value=-1.0, max_value=20.0,
                                  step=0.05, key="rolling_7d_avg_premium")

    if st.button("PREDICT →", type="primary"):
        record = {
            "days_since_release": days,
            "size_us": size,
            "retail_price": retail,
            "size_premium": size_prem,
            "release_type_encoded": RELEASE_TYPES[rt_name],
            "rolling_7d_avg_premium": rolling,
            "search_index_7d_pre_drop": None,  # imputed with the training mean
            "brand_avg_premium": brand_avg,
        }
        premium = float(predict(_bundle(), [record])[0])
        resale = retail * (1 + premium)
        st.markdown(
            f'<div class="sm-readout"><div class="k">Predicted resale premium</div>'
            f'<div class="v">{premium * 100:+.0f}%</div>'
            f'<div class="sub">≈ ${resale:,.0f} resale on a ${retail:,.0f} retail '
            f'· {premium + 1:.2f}× · model @staging v{RUN["model_version"]}</div></div>',
            unsafe_allow_html=True,
        )

    # ----------------------------------------------------------------------- #
    # Model card
    # ----------------------------------------------------------------------- #
    rule("§ 1.2 — MODEL")
    st.markdown(
        '<div class="sm-grid" style="grid-template-columns:repeat(3,1fr)">'
        + field(":01", "registry", f"{RUN['model_name']} @staging v{RUN['model_version']}")
        + field(":02", "val rmse", f"{RUN['val_rmse']:.3f}", "0–20 scale")
        + field(":03", "temporal split", f"{RUN['n_train']} / {RUN['n_val']}",
                f"@ {RUN['split_year']}")
        + field(":04", "architecture", "8 → 64 → 64 → 1", "feedforward")
        + field(":05", "features", str(RUN["n_features"]), "engineered")
        + field(":06", "brands", RUN["brands"])
        + "</div>",
        unsafe_allow_html=True,
    )

    # ----------------------------------------------------------------------- #
    # Evaluation on the real run
    # ----------------------------------------------------------------------- #
    rule("§ 1.3 — EVALUATION")
    st.markdown(
        '<div class="sm-grid" style="grid-template-columns:repeat(4,1fr)">'
        + field(":01", "rows scored", RUN["rows_scored"], "real sales")
        + field(":02", "scoring rmse", f"{RUN['scoring_rmse']:.3f}", "full set")
        + field(":03", "row retention", RUN["retention"], "validated")
        + field(":04", "in-era drift", RUN["drift_features"], "PSI > 0.2")
        + "</div>",
        unsafe_allow_html=True,
    )
    lede(
        f"From a real run over {RUN['rows_scored']} StockX sales "
        f"({RUN['brands']}, 2017–2019). The split holds out the last slice of time "
        f"({RUN['n_train']} / {RUN['n_val']} at {RUN['split_year']}) rather than a "
        f"random sample, so the validation number is a forward-in-time test, not a "
        f"leak. The validator caught {RUN['pre_release']} pre-release sales and "
        "premium outliers past the ceiling; each became a documented config decision "
        "rather than a silent patch. That is the validator doing its job on real data."
    )
    lede(
        "What the model knows: one era, well. Within that era nothing drifts, which "
        "is what the 0 / 8 above says. Point it at today's market and it runs hot. "
        "That is the next tab, and it is on purpose, not a bug.",
        top="1rem",
    )

# =========================================================================== #
# TAB 2 — LIVE MARKET (the KicksDB / out-of-distribution story)
# =========================================================================== #
with tab_live:
    lede(
        "Same model, same inference path, current sneakers. These are pulled from "
        '<a href="https://kicks.dev">KicksDB</a> and scored through the exact '
        "transform saved inside the model, so nothing about the preprocessing "
        "changed between a 2019 training row and a 2026 one. The point of this tab is "
        "the distance between what the model learned and what it is now being asked.",
        top=".2rem",
    )

    # ----------------------------------------------------------------------- #
    # Then vs now: the interpretive spine
    # ----------------------------------------------------------------------- #
    rule("§ 2.1 — THEN vs NOW")
    st.markdown(
        '<div class="sm-compare">'
        '<div class="sm-side"><div class="h">The model\'s world · 2017–2019</div>'
        '<div class="b">The hyped StockX era</div>'
        '<div class="d">Off-White and Yeezy, limited drops trading at multiples of '
        f'retail. {RUN["rows_scored"]} sales. This is the shape of premium the model '
        'learned to predict.</div></div>'
        '<div class="sm-vs">VS</div>'
        '<div class="sm-side now"><div class="h">Today\'s market · 2026 snapshot</div>'
        '<div class="b">Current releases from KicksDB</div>'
        '<div class="d">Mostly years past their drop, many trading near or below '
        'retail. The model has never seen a single one of them.</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ----------------------------------------------------------------------- #
    # The OOD framing: the load-bearing copy
    # ----------------------------------------------------------------------- #
    rule("§ 2.2 — OFF THE END OF THE DATA")
    lede(
        "The predictions below run hot. The model puts a premium of several times "
        "retail on a current Dunk that KicksDB's live market has trading near or "
        "below it, and it under-shoots the genuinely hyped pairs the other way. The "
        "gap between the model column and the KicksDB market column is the whole "
        "story. This is not the model breaking. It is the model doing exactly what a "
        "model does off the end of its training data: it learned that hyped 2017–2019 "
        "pairs carried large premiums, it has never seen a 2026 general-release Dunk, "
        "so it extrapolates the only thing it knows.",
        top=".2rem",
    )
    lede(
        "The drift monitor measures that same distance and agrees. The two features "
        "that moved most, <b>days_since_release</b> and <b>retail_price</b>, blow "
        "past the PSI threshold, which is why the retrain gate trips. The hot "
        "prediction and the drift number are one fact told twice: the market the "
        "model learned is not the market it is now being shown.",
        top="1rem",
    )
    lede(
        "That is the point, not the caveat. A drift monitor that never fired would be "
        "the useless one. This one fires exactly when the data moves out from under "
        "the model, which is the signal to retrain rather than trust a stale number.",
        top="1rem",
    )

    # The board prefers the scheduled snapshot (the model's premium next to
    # KicksDB's real current market price); a clean clone with no snapshot falls
    # back to canned records scored live through the same transform.
    _snap = _load_snapshot()
    if _snap and _snap.get("board"):
        _stamp = _snap["generated_at"][:16].replace("T", " ")
        _src = "live KicksDB" if _snap.get("source") == "live" else "canned snapshot"
        st.markdown(
            f'<p class="sm-label" style="margin:.2rem 0 .9rem">updated {_stamp} UTC '
            f"· {_snap['n_scored']} shoes scored · {_src}</p>",
            unsafe_allow_html=True,
        )
        _kick_cards = "".join(
            f'<div class="sm-field hot"><div class="k">{_b["name"]} '
            f'<span class="idx">· {_b["note"]}</span></div>'
            f'<div class="v">{_b["premium"] * 100:+.0f}%<span class="u"> model</span></div>'
            f'<div class="k" style="margin-top:6px">${_b["retail"]:.0f} retail · '
            f'implies ${_b["implied_resale"]:,.0f}</div>'
            f'<div class="k" style="color:var(--bone-dim)">KicksDB market '
            f'${_b["market"]:,.0f}</div></div>'
            for _b in _snap["board"]
        )
    else:
        _kick_cards = ""
        for _item in CURRENT_KICKSDB:
            _prem = float(predict(_bundle(), [_kicksdb_record(_item)])[0])
            _kick_cards += (
                f'<div class="sm-field hot"><div class="k">{_item["name"]} '
                f'<span class="idx">· {_item["note"]}</span></div>'
                f'<div class="v">{_prem * 100:+.0f}%<span class="u"> model</span></div>'
                f'<div class="k" style="margin-top:6px">${_item["retail"]:.0f} retail '
                f'· implies ${_item["retail"] * (1 + _prem):,.0f}</div></div>'
            )
        _kick_cards += (
            f'<div class="sm-field dim"><div class="k">{UNCOMPUTABLE_KICKSDB["name"]} '
            f'<span class="idx">· {UNCOMPUTABLE_KICKSDB["note"]}</span></div>'
            '<div class="v">—<span class="u"> uncomputable</span></div>'
            '<div class="k" style="margin-top:6px">no retail · not scored</div></div>'
        )
    st.markdown(
        '<div class="sm-grid" style="grid-template-columns:repeat(auto-fit,minmax(210px,1fr))">'
        + _kick_cards + "</div>",
        unsafe_allow_html=True,
    )
    lede(
        "Retail is the boundary on what can be scored at all. The Starter API returns "
        "no retail, so it is carried from a curated 13-SKU reference. A current "
        "release outside that reference, like the KAWS pair above, has no retail to "
        "compute a premium against, so it is reported uncomputable and left unscored "
        "rather than handed a fabricated number.",
        top="1rem",
    )

    # ----------------------------------------------------------------------- #
    # The monitor: PSI drift (expanded visually in a later pass)
    # ----------------------------------------------------------------------- #
    rule("§ 2.3 — THE MONITOR")
    lede(
        "The same snapshot, run through the drift stage against the model's training "
        "distribution. Two of the four computable features drift far past the 0.2 "
        "threshold and the retrain gate trips. The other four can't be computed from "
        "a current-only snapshot, so they are excluded with a reason rather than "
        "measured on degenerate data.",
        top=".2rem",
    )

    # PSI bars for the four snapshot-computable features
    _vis_max = 2.5
    _thresh_pct = PSI_THRESHOLD / _vis_max * 100
    _rows = ""
    for _feat, _psi, _drifted in DRIFT_COMPUTED:
        if _psi is None:  # computable, under threshold; exact value not recorded
            _rows += (
                f'<div class="sm-psi-row"><div class="f">{_feat}</div>'
                f'<div class="sm-psi-track"><div class="sm-psi-thresh" '
                f'style="left:{_thresh_pct:.1f}%"></div></div>'
                '<div class="sm-psi-val" style="color:var(--ash)">under 0.2</div></div>'
            )
            continue
        _pct = min(_psi / _vis_max, 1.0) * 100
        _color = "var(--oxblood-lift)" if _drifted else "var(--phosphor-dim)"
        _label = f"~{_psi:.1f}" + (" →" if _psi > _vis_max else "")
        _rows += (
            f'<div class="sm-psi-row"><div class="f">{_feat}</div>'
            f'<div class="sm-psi-track">'
            f'<div class="sm-psi-fill" style="width:{_pct:.1f}%;background:{_color}"></div>'
            f'<div class="sm-psi-thresh" style="left:{_thresh_pct:.1f}%"></div></div>'
            f'<div class="sm-psi-val" style="color:{_color}">{_label}</div></div>'
        )
    st.markdown(f'<div class="sm-psi">{_rows}</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sm-label" style="margin:.6rem 0 0">the vertical rule is the 0.2 '
        "retrain threshold · days_since_release and retail_price clear it by a wide "
        "margin, which is a 2026 snapshot of shoes years past their drop</p>",
        unsafe_allow_html=True,
    )

    # The four a snapshot can't honestly compute, excluded with the reason
    _excl = "".join(
        f'<div class="sm-psi-note"><b>{_f}</b> — excluded · {_r}</div>'
        for _f, _r in DRIFT_EXCLUDED
    )
    st.markdown(
        '<div class="sm-psi" style="margin-top:1.3rem">'
        '<div class="sm-psi-note" style="color:var(--bone-dim)">Four features a '
        "current-only snapshot can't support, excluded rather than measured on "
        "degenerate data:</div>" + _excl + "</div>",
        unsafe_allow_html=True,
    )

    # The retrain gate
    st.markdown(
        '<div class="sm-grid" style="grid-template-columns:repeat(3,1fr);margin-top:1.3rem">'
        + field(":01", "features drifted", "2 / 4", "computable")
        + field(":02", "retrain gate", "TRIPPED", "PSI > 0.2", cls="hot")
        + field(":03", "would trigger", "retrain", "in the live pipeline")
        + "</div>",
        unsafe_allow_html=True,
    )
    lede(
        "In the deployed pipeline this is a scheduled Airflow DAG. When the gate "
        "trips it fires the training pipeline; when nothing drifts, it short-circuits "
        "and nothing retrains. The retrain is evidence-driven, not a blind cron.",
        top="1rem",
    )

    # Market-vs-model drift over time, accumulated by the scheduled KicksDB refresh.
    rule("§ 2.4 — MARKET vs MODEL, OVER TIME")
    _hist = _load_history()
    if _hist and len(_hist) >= 2:
        st.markdown(_trend_svg(_hist), unsafe_allow_html=True)
        _last = _hist[-1]
        st.markdown(
            '<p class="sm-label" style="margin:.7rem 0 0">'
            '<span style="color:var(--phosphor)">market premium</span> &nbsp; '
            '<span style="color:var(--oxblood-lift)">model premium</span> &nbsp;·&nbsp; '
            f'{len(_hist)} points &nbsp;·&nbsp; latest: market '
            f'{_last["market_premium"] * 100:+.0f}% vs model '
            f'{_last["model_premium"] * 100:+.0f}%</p>',
            unsafe_allow_html=True,
        )
    else:
        lede(
            "The refresh job records a market-vs-model point every six hours, so this "
            "trend fills in as the market moves and the shoes age. It begins with the "
            "first scheduled pull.",
            top=".2rem",
        )

# =========================================================================== #
# TAB 3 — HOW IT WORKS (architecture and seams; reworked in a later pass)
# =========================================================================== #
with tab_arch:
    lede(
        "The machinery under both tabs. The model is deliberately simple; the "
        "pipeline around it is the part that took the work.",
        top=".2rem",
    )

    rule("§ 3.1 — THE PIPELINE")
    stages = [
        (":01", "ingest", "postgres · kicksdb"),
        (":02", "features", "pyspark"),
        (":03", "validate", "pandera"),
        (":04", "train", "ray + pytorch"),
        (":05", "register", "mlflow"),
        (":06", "score", "batch + api"),
    ]
    flow = '<div class="sm-flow">' + "".join(
        f'<div class="sm-stage"><div class="i">{i}</div><div class="n">{n}</div>'
        f'<div class="t">{t}</div></div>' for i, n, t in stages) + "</div>"
    st.markdown(flow, unsafe_allow_html=True)
    lede(
        "Every stage runs on its own from the CLI and speaks to the next only through "
        "Parquet at an S3 (or local) path. Two Airflow DAGs sit on top: a training "
        "pipeline, and a drift-monitor that watches for movement and retrains only "
        "when the data has moved. Config is the single source of truth, the data "
        "contract is explicit and loud, and every non-obvious decision is written down."
    )

    rule("§ 3.2 — TWO SOURCES, ONE SCHEMA")
    st.markdown(
        '<div class="sm-grid" style="grid-template-columns:repeat(4,1fr)">'
        + field(":01", "real sources", "2", "postgres · kicksdb")
        + field(":02", "downstream changed", "0", "files, adding kicksdb")
        + field(":03", "inference paths", "1", "batch + live")
        + field(":04", "retail reference", "13", "curated SKUs")
        + "</div>",
        unsafe_allow_html=True,
    )
    lede(
        "Ingest is an anti-corruption layer. It lands two real sources, the "
        "sneaker-intel Postgres warehouse and the KicksDB market API, into one "
        "canonical schema, byte-compatible, so nothing downstream knows there is a "
        "second source. Features, validate, train, and serve were unchanged when "
        "KicksDB was added, a zero-line diff. That is the claim the whole "
        "architecture was built to make, and it held.",
        top="1rem",
    )

    rule("§ 3.3 — ONE INFERENCE PATH")
    lede(
        "Batch scoring and live KicksDB scoring both go through the same transform, "
        "with the exact imputation and standardization saved inside model.pt. There "
        "is no second copy of the preprocessing to drift, so a batch row and a live "
        "row get identical treatment. The +460% Dunk on the last tab went through the "
        "same path a 2019 training row did.",
        top=".2rem",
    )
    lede(
        "Retail is the honest ceiling. The Starter API exposes no retail, so it is "
        "carried from a 13-SKU reference and never fabricated. That reference is the "
        "limit on what can have a premium computed at all, which is why some current "
        "sneakers are reported instead of scored.",
        top="1rem",
    )

# --------------------------------------------------------------------------- #
# Footer (shared, below the tabs)
# --------------------------------------------------------------------------- #
st.markdown('<div class="sm-rule"><span class="line"></span></div>',
            unsafe_allow_html=True)
st.markdown(
    f'<div class="sm-foot">↳ <a href="{REPO_URL}">github.com/smalik-25/'
    'ML-Training-Pipeline</a><br>'
    f'↳ <a href="{WEBSITE_URL}">sam-malik.com</a><br><br>'
    'SAM·MALIK · Data Engineer · Theory-Fiction · Seattle · 47.6°N<br>'
    'Built in the ruins of the present.</div>',
    unsafe_allow_html=True,
)
