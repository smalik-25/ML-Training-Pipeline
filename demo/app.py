"""SAM·MALIK — sneaker ML platform, live demo.

A Streamlit dashboard over the offline ML training pipeline: a live price-premium
predictor (the real trained model), a model card, the pipeline topology, and the
real-run monitoring artifacts. Styled to the Sam Malik design system
(terminal-meets-gothic: phosphor on void, Cormorant / Space Grotesk / IBM Plex
Mono, hairline structure, decimal indices, no emoji).
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from net import FEATURE_COLUMNS, load_bundle, predict

MODEL_PATH = str(Path(__file__).parent / "model.pt")
REPO_URL = "https://github.com/smalik-25/ML-Training-Pipeline"

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


def rule(label: str) -> None:
    st.markdown(
        f'<div class="sm-rule"><span class="lab">{label}</span>'
        f'<span class="line"></span></div>', unsafe_allow_html=True)


def field(idx: str, key: str, value: str, unit: str = "") -> str:
    unit = f' <span class="u">{unit}</span>' if unit else ""
    return (f'<div class="sm-field"><div class="k"><span class="idx">{idx}</span> '
            f'{key}</div><div class="v">{value}{unit}</div></div>')


# --------------------------------------------------------------------------- #
# Hero
# --------------------------------------------------------------------------- #
st.markdown(
    f"""
    <div class="sm-eyebrow">§ 0.0 — ML INFRASTRUCTURE · SNEAKER-INTEL PHASE 2</div>
    <div class="sm-wordmark">SAM<span class="sm-mid">·</span>MALIK</div>
    <div class="sm-display">Price premiums,<br>predicted forward in time.</div>
    <div class="sm-status"><span class="dot"></span>pipeline sneaker.0.8 // green
      &nbsp;·&nbsp; {RUN['rows_scored']} real StockX sales</div>
    <p class="sm-lede">I build the plumbing that moves data quietly and correctly.
    This is the ML layer I deferred in <a href="{REPO_URL}">sneaker-intel</a> and came
    back to build: ingest, feature engineering, a loud data contract, distributed
    training, a model registry, batch and online serving, and drift monitoring. The
    model is simple on purpose. The pipeline is the point.</p>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Predictor
# --------------------------------------------------------------------------- #
rule("§ 0.1 — PREDICT")
st.markdown(
    '<p class="sm-lede" style="margin-top:.2rem">Enter a sale\'s engineered '
    "features; the model returns the predicted resale premium over retail. The "
    "same imputation and standardization fit during training are applied here from "
    "the stats saved inside the model, so this matches what the model learned.</p>",
    unsafe_allow_html=True,
)

# seed defaults from a preset
for k, v in PRESETS["Off-White · limited"].items():
    st.session_state.setdefault(k, v)

pcols = st.columns(len(PRESETS))
for col, (name, vals) in zip(pcols, PRESETS.items()):
    if col.button(name, use_container_width=True):
        for k, v in vals.items():
            st.session_state[k] = v
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

# --------------------------------------------------------------------------- #
# Model card
# --------------------------------------------------------------------------- #
rule("§ 0.2 — MODEL")
st.markdown(
    '<div class="sm-grid">'
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

# --------------------------------------------------------------------------- #
# Pipeline topology
# --------------------------------------------------------------------------- #
rule("§ 0.3 — THE PIPELINE")
stages = [
    (":01", "ingest", "postgres → parquet"),
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
st.markdown(
    '<p class="sm-lede" style="margin-top:1rem">Every stage runs on its own from '
    "the CLI and speaks to the next only through Parquet at an S3 (or local) path. "
    "Airflow orchestrates the topology; a scheduled monitor watches for drift and "
    "retrains only when the data has moved. Config is the single source of truth, "
    "the data contract is explicit and loud, and every non-obvious decision is "
    "written down.</p>",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Observed
# --------------------------------------------------------------------------- #
rule("§ 0.4 — OBSERVED")
st.markdown(
    '<div class="sm-grid">'
    + field(":01", "rows scored", RUN["rows_scored"], "real sales")
    + field(":02", "scoring rmse", f"{RUN['scoring_rmse']:.3f}", "full set")
    + field(":03", "row retention", RUN["retention"], "validated")
    + field(":04", "feature drift", RUN["drift_features"], "PSI > 0.2")
    + "</div>",
    unsafe_allow_html=True,
)
st.markdown(
    f'<p class="sm-lede" style="margin-top:1rem">From a real run over '
    f"{RUN['rows_scored']} StockX sales ({RUN['brands']}, 2017–2019). The validator "
    f"caught {RUN['pre_release']} pre-release sales and premium outliers past the "
    "ceiling; each became a documented config decision rather than a silent patch. "
    "That is the validator doing its job on real data.</p>",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Footer
# --------------------------------------------------------------------------- #
st.markdown('<div class="sm-rule"><span class="line"></span></div>',
            unsafe_allow_html=True)
st.markdown(
    f'<div class="sm-foot">↳ <a href="{REPO_URL}">github.com/smalik-25/'
    'ML-Training-Pipeline</a><br><br>SAM·MALIK · Data Engineer · Theory-Fiction · '
    'Seattle · 47.6°N<br>Built in the ruins of the present.</div>',
    unsafe_allow_html=True,
)
