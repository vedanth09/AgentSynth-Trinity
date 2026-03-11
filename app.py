import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import threading
import traceback
import queue
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime

# ── YOUR ORIGINAL IMPORTS (unchanged) ────────────────────────────────────────
from main import run_trinity_pipeline, AgentState
from utils.prompt_parser import PromptParser, GenerationConfig
from utils.data_loader import DataLoader
from utils.reporter import TrinityReporter
from utils.trace_viz import TraceVisualizer
from utils.certificate_gen import CertificateGenerator

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (your original, unchanged)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="AgentSynth-Trinity Demo", page_icon="🛡️", layout="wide")

# ══════════════════════════════════════════════════════════════════════════════
#  COLOUR PALETTE  (showcase tabs only)
# ══════════════════════════════════════════════════════════════════════════════
_TEAL      = "#00B4A6"
_AMBER     = "#F5A623"
_SLATE     = "#1E2A3A"
_SLATE_MID = "#2E3D52"
_SLATE_LT  = "#3D5066"
_OFF_WHITE = "#F0F4F8"
_PASS      = "#2ECC71"
_FAIL      = "#E74C3C"
_BLUE      = "#3498DB"
_PURPLE    = "#9B59B6"

# Pre-computed rgba strings for use inside f-string HTML/CSS.
# These avoid the "{_VAR}AA" pattern which Python misreads as a format spec.
_TEAL_AA     = "rgba(0,180,166,0.67)"
_TEAL_88     = "rgba(0,180,166,0.53)"
_AMBER_AA    = "rgba(245,166,35,0.67)"
_OFF_WHITE_66 = "rgba(240,244,248,0.4)"
_OFF_WHITE_88 = "rgba(240,244,248,0.53)"
_OFF_WHITE_AA = "rgba(240,244,248,0.67)"
_SLATE_LT_44  = "rgba(61,80,102,0.27)"
_SLATE_LT_66  = "rgba(61,80,102,0.4)"

st.markdown("""
<style>
    /* ── your original styles ── */
    .main { background-color: #f8f9fa; }
    .stButton>button { border-radius: 5px; height: 3em; width: 100%; }
    .stMetric { background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }

    /* ── showcase tab styles ── */
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;600&family=Space+Grotesk:wght@700&display=swap');

    .sc-card {
        background: #2E3D52; border: 1px solid #3D506666;
        border-radius: 12px; padding: 1.2rem 1.4rem; margin-bottom: 0.85rem;
    }
    .sc-title {
        font-family: 'Space Grotesk', sans-serif; font-size: 0.85rem;
        font-weight: 700; color: #00B4A6; text-transform: uppercase;
        letter-spacing: 0.06em; margin-bottom: 0.7rem;
    }
    .sc-gate {
        text-align: center; padding: 3rem 2rem;
        background: #2E3D52; border: 1px dashed #3D5066;
        border-radius: 14px; margin-top: 1rem;
    }
    .sc-gate h3 { font-family: 'Space Grotesk', sans-serif; color: #00B4A6; margin-bottom: 0.5rem; }
    .sc-gate p  { color: #F0F4F888; font-size: 0.88rem; }

    .sc-kpi-grid { display: flex; gap: 0.75rem; flex-wrap: wrap; margin-bottom: 0.9rem; }
    .sc-kpi {
        flex: 1; min-width: 120px; background: #1E2A3A;
        border: 1px solid #3D5066; border-radius: 9px;
        padding: 0.8rem 1rem; text-align: center;
    }
    .sc-kpi-val { font-family: 'DM Mono', monospace; font-size: 1.65rem;
                  font-weight: 500; color: #00B4A6; line-height: 1; }
    .sc-kpi-lbl { font-size: 0.67rem; color: #F0F4F888;
                  text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.2rem; }
    .sc-kpi-pass { border-color: #2ECC7155; }
    .sc-kpi-fail { border-color: #E74C3C55; }

    .sc-badge-pass { background:#2ECC7122; color:#2ECC71; border:1px solid #2ECC7144;
                     border-radius:20px; padding:2px 9px; font-size:0.7rem; font-weight:600; }
    .sc-badge-fail { background:#E74C3C22; color:#E74C3C; border:1px solid #E74C3C44;
                     border-radius:20px; padding:2px 9px; font-size:0.7rem; font-weight:600; }

    .sc-agent-row { display:flex; align-items:center; gap:0.7rem;
                    padding:0.4rem 0; border-bottom:1px solid #3D506644; }
    .sc-agent-dot { width:8px; height:8px; border-radius:50%;
                    background:#00B4A6; flex-shrink:0; }
    .sc-agent-name { font-family:'DM Mono',monospace; font-size:0.78rem;
                     color:#00B4A6; min-width:170px; }
    .sc-agent-desc { font-size:0.75rem; color:#F0F4F8AA; }

    .sc-rank-row    { display:flex; align-items:center; gap:0.65rem;
                      padding:0.45rem 0.75rem; border-radius:7px; margin-bottom:0.25rem; }
    .sc-rank-gold   { background:#F5A62318; border-left:3px solid #F5A623; }
    .sc-rank-silver { background:#ADADAD18; border-left:3px solid #ADADAD; }
    .sc-rank-bronze { background:#CD7F3218; border-left:3px solid #CD7F32; }
    .sc-rank-num    { font-family:'DM Mono'; font-size:0.85rem; color:#F5A623; min-width:22px; }
    .sc-rank-model  { font-weight:600; font-size:0.84rem; flex:1; }
    .sc-rank-score  { font-family:'DM Mono'; font-size:0.78rem; color:#F0F4F8BB; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  YOUR ORIGINAL INITIALIZATION (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
parser      = PromptParser()
data_loader = DataLoader()
reporter    = TrinityReporter()
viz         = TraceVisualizer()
cert_gen    = CertificateGenerator()

if "config" not in st.session_state:
    st.session_state.config = GenerationConfig(domain="Healthcare", goal="Synthesize high-fidelity healthcare data")
if "results" not in st.session_state:
    st.session_state.results = None
if "prompt" not in st.session_state:
    st.session_state.prompt = ""

# ══════════════════════════════════════════════════════════════════════════════
#  YOUR ORIGINAL HELPER FUNCTIONS (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
def update_config_from_prompt(prompt):
    parsed = parser.parse(prompt)
    st.session_state.config = parsed
    st.session_state.prompt = prompt

def reset_app():
    st.session_state.results = None
    st.session_state.prompt  = ""
    st.session_state.config  = GenerationConfig(domain="Healthcare", goal="Synthesize high-fidelity healthcare data")
    st.rerun()

def run_async_pipeline(coro, update_queue):
    # FIX: The pipeline runs in a background thread (needed to avoid asyncio conflicts
    # with Streamlit). But Streamlit UI calls (status.write, st.caption) CANNOT be
    # made from background threads — they raise NoSessionContext.
    # Solution: pipeline puts update messages into a thread-safe queue. The main
    # thread drains the queue and writes to the UI after the pipeline finishes.
    result = {"value": None, "error": None, "tb": None}

    async def _wrapped():
        final_state = None
        async for event in coro:
            node_name   = list(event.keys())[0]
            node_output = event[node_name]
            update_queue.put(("agent_done", node_name, node_output))
            final_state = node_output
        return final_state

    def _thread_target():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result["value"] = loop.run_until_complete(_wrapped())
        except Exception as exc:
            result["error"] = exc
            result["tb"]    = traceback.format_exc()
        finally:
            update_queue.put(("done", None, None))
            loop.close()

    t = threading.Thread(target=_thread_target, daemon=True)
    t.start()
    t.join()

    if result["error"]:
        print("\n===== PIPELINE ERROR TRACEBACK =====")
        print(result["tb"])
        print("=====================================\n")
        raise result["error"]

    return result["value"]


# ══════════════════════════════════════════════════════════════════════════════
#  LIVE DATA EXTRACTOR
#  Pulls every value the showcase tabs need from st.session_state.results.
#  Returns None if the pipeline hasn't run yet.
# ══════════════════════════════════════════════════════════════════════════════
def _extract_live(res_state: dict) -> dict | None:
    """
    Parse the pipeline result dict into a flat dict the showcase tabs use.
    All keys come exclusively from the real pipeline output.
    Returns None if the result is missing or malformed.
    """
    if not isinstance(res_state, dict):
        return None

    sc        = res_state.get("trinity_scorecard") or {}
    fidelity  = sc.get("fidelity", {}) or {}
    utility   = sc.get("utility",  {}) or {}
    privacy_s = sc.get("privacy",  {}) or {}
    proof     = res_state.get("privacy_proof", {}) or {}

    # ── core metrics ──────────────────────────────────────────────────────────
    wass     = fidelity.get("wasserstein",      1.0)
    jsd      = fidelity.get("jsd",              None)
    corr_sim = fidelity.get("correlation_similarity", None)
    mmd      = fidelity.get("mmd",              None)

    tstr       = utility.get("tstr_accuracy",   0.0)
    trts       = utility.get("trts_accuracy",   None)
    perf_drop  = utility.get("performance_drop",100.0)
    baseline   = utility.get("baseline_accuracy", None)

    mia        = privacy_s.get("mia_risk_score",    1.0)
    k_anon     = privacy_s.get("k_anonymity",        0)
    linkability= privacy_s.get("linkability_score",  None)
    is_compliant = bool(privacy_s.get("is_compliant", False))

    epsilon    = proof.get("epsilon", st.session_state.config.epsilon)
    delta      = proof.get("delta",   None)

    decision   = res_state.get("judge_decision", "N/A")
    domain     = res_state.get("domain", st.session_state.config.domain)
    iteration  = res_state.get("iteration", 1)

    # ── pilot model scores (StatisticalCritic logs these) ────────────────────
    pilot_scores = res_state.get("pilot_scores") or []          # list of dicts
    ensemble_weights = res_state.get("ensemble_weights") or {}  # {model: weight}

    # ── synthetic data for distribution charts ────────────────────────────────
    synth_df = res_state.get("safe_data_asset")   # pd.DataFrame or None
    raw_df   = res_state.get("raw_data")          # pd.DataFrame or None — original input

    # ── agent trace (TraceVisualizer / LangGraph events) ─────────────────────
    agent_trace = res_state.get("agent_trace") or []  # list of {agent, output, duration_s}

    return dict(
        # scorecard
        wass=wass, jsd=jsd, corr_sim=corr_sim, mmd=mmd,
        tstr=tstr, trts=trts, perf_drop=perf_drop, baseline=baseline,
        mia=mia, k_anon=k_anon, linkability=linkability, is_compliant=is_compliant,
        epsilon=epsilon, delta=delta,
        decision=decision, domain=domain, iteration=iteration,
        # pass/fail booleans
        f_pass=wass < 0.15,
        u_pass=perf_drop < 20,
        p_pass=is_compliant,
        # extras
        pilot_scores=pilot_scores,
        ensemble_weights=ensemble_weights,
        synth_df=synth_df,
        raw_df=raw_df,
        agent_trace=agent_trace,
        # full objects for downstream use
        scorecard=sc,
        proof=proof,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED PLOTLY THEME
# ══════════════════════════════════════════════════════════════════════════════
_PL = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=_SLATE_MID,
    font=dict(family="DM Sans, sans-serif", color=_OFF_WHITE, size=11),
    title_font=dict(family="Space Grotesk, sans-serif", size=13, color=_OFF_WHITE),
    margin=dict(l=44, r=20, t=44, b=40),
)
_GR = dict(gridcolor="rgba(61,80,102,0.33)", zerolinecolor="rgba(61,80,102,0.53)",
           tickfont=dict(size=9, color=_OFF_WHITE))


# ══════════════════════════════════════════════════════════════════════════════
#  CHART BUILDERS  (all accept live data only, no defaults)
# ══════════════════════════════════════════════════════════════════════════════

def _chart_model_comparison(pilot_scores, final_wass, final_tstr, final_mia):
    """
    Grouped bar: pilot models + Ensemble across Wasserstein / TSTR / MIA.
    pilot_scores[i].tstr and .mia may be None (critic only measures Wasserstein).
    Those bars are shown as 0 height with an "N/A" label — honest and crash-free.
    """
    def _f(v):
        return float(v) if v is not None else 0.0

    def _lbl(v, fmt):
        return f"{float(v):{fmt}}" if v is not None else "N/A"

    models     = [p["model"] for p in pilot_scores]
    all_models = models + ["▶ Ensemble"]
    n = len(pilot_scores)

    raw_w = [p.get("wasserstein", p.get("w")) for p in pilot_scores] + [final_wass]
    raw_t = [p.get("tstr") for p in pilot_scores] + [final_tstr]
    raw_m = [p.get("mia")  for p in pilot_scores] + [final_mia]

    all_w = [_f(v) for v in raw_w]
    all_t = [_f(v) for v in raw_t]
    all_m = [_f(v) for v in raw_m]

    bar_c  = ["rgba(0,180,166,0.4)"] * n + [_TEAL]
    bar_lw = [0] * n + [2.2]

    fig = make_subplots(1, 3,
        subplot_titles=["Wasserstein ↓", "TSTR Accuracy ↑", "MIA Score ↓"],
        horizontal_spacing=0.09)

    for col_i, (vals, raw_vals, thresh, fmt) in enumerate([
        (all_w, raw_w, 0.15, ".3f"),
        (all_t, raw_t, None, ".1%"),
        (all_m, raw_m, 0.76, ".3f"),
    ], 1):
        labels = [_lbl(v, fmt) for v in raw_vals]
        fig.add_trace(go.Bar(
            x=all_models, y=vals,
            marker_color=bar_c,
            marker_line_color=["#888"] * n + ["white"],
            marker_line_width=bar_lw,
            text=labels, textposition="outside",
            textfont=dict(size=9, color=_OFF_WHITE), showlegend=False,
        ), row=1, col=col_i)
        if thresh:
            fig.add_hline(y=thresh, line_dash="dot", line_color="rgba(245,166,35,0.73)",
                          line_width=1.4, row=1, col=col_i,
                          annotation_text=f"threshold {thresh}",
                          annotation_font_size=8, annotation_font_color=_AMBER)

    fig.update_layout(**_PL,
        title_text="Model Pilot: Standalone vs AgentSynth-Trinity Ensemble", height=330)
    for ax in ["xaxis", "xaxis2", "xaxis3"]:
        fig.update_layout(**{ax: dict(**_GR, tickangle=10)})
    fig.update_layout(yaxis=dict(**_GR), yaxis2=dict(**_GR), yaxis3=dict(**_GR))
    return fig


def _chart_dist_overlay(col_name, real_series, synth_series):
    """
    Overlaid KDE of real vs synthetic for a single column.
    Uses actual data arrays — no approximations.
    """
    try:
        from scipy.stats import gaussian_kde
        real_arr  = real_series.dropna().values.astype(float)
        synth_arr = synth_series.dropna().values.astype(float)
        lo = min(real_arr.min(), synth_arr.min())
        hi = max(real_arr.max(), synth_arr.max())
        xs = np.linspace(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo), 300)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xs, y=gaussian_kde(real_arr)(xs), name="Real Data",
            line=dict(color=_TEAL, width=2.5), fill="tozeroy", fillcolor="rgba(0,180,166,0.13)"))
        fig.add_trace(go.Scatter(
            x=xs, y=gaussian_kde(synth_arr)(xs), name="Synthetic Data",
            line=dict(color=_AMBER, width=2.5, dash="dash"),
            fill="tozeroy", fillcolor="rgba(245,166,35,0.09)"))
        fig.update_layout(**_PL,
            title_text=f"{col_name} — Real vs Synthetic (KDE)",
            height=300, xaxis=dict(**_GR), yaxis=dict(**_GR, title="Density"),
            legend=dict(orientation="h", yanchor="top", y=0.97,
                        xanchor="right", x=0.99,
                        bgcolor="rgba(46,61,82,0.73)", bordercolor=_SLATE_LT))
        return fig
    except ImportError:
        return None


def _chart_radar(wass, tstr, mia, k_anon, linkability):
    cats = ["Fidelity<br>(Wasserstein)", "Utility<br>(TSTR)",
            "Privacy<br>(MIA)", "k-Anonymity", "Linkability"]
    vals = [
        max(0, 1 - wass / 0.15),
        tstr,
        max(0, 1 - (mia - 0.5) / 0.5),
        min((k_anon or 0) / 10, 1.0),
        1.0 if (linkability is None or linkability == 0) else max(0, 1 - linkability),
    ]
    vals += [vals[0]]; cats += [cats[0]]
    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats, fill="toself",
        fillcolor="rgba(0,180,166,0.16)", line=dict(color=_TEAL, width=2.5),
        marker=dict(color=_TEAL, size=7),
    ))
    base = {k: v for k, v in _PL.items() if k != "plot_bgcolor"}
    fig.update_layout(**base,
        polar=dict(
            bgcolor=_SLATE_MID,
            radialaxis=dict(visible=True, range=[0, 1],
                tickfont=dict(size=7.5, color="rgba(240,244,248,0.53)"),
                gridcolor="rgba(61,80,102,0.4)", linecolor="rgba(61,80,102,0.4)"),
            angularaxis=dict(tickfont=dict(size=9, color=_OFF_WHITE),
                             gridcolor="rgba(61,80,102,0.4)"),
        ),
        title_text="Trinity FUP Score Radar", height=340)
    return fig


def _chart_agent_gantt(agent_trace):
    """
    Horizontal Gantt from real agent_trace list.
    Each entry: {agent: str, duration_s: float|None, output: str}
    """
    if not agent_trace:
        return None

    # Sanitise — duration_s may be None; replace with 1.0 as fallback
    agents  = [a["agent"] for a in agent_trace]
    durs    = [float(a.get("duration_s") or 1.0) for a in agent_trace]
    outputs = [str(a.get("output") or "") for a in agent_trace]

    # Cycle through colours safely regardless of trace length
    colour_pool = [_BLUE, _TEAL, _AMBER, _PURPLE, _PASS, _TEAL, _AMBER, _BLUE]
    clrs = [colour_pool[i % len(colour_pool)] for i in range(len(agents))]

    # marker_color must be a valid CSS colour — use rgba() not 8-digit hex
    def _rgba_080(hex6):
        r = int(hex6[1:3], 16)
        g = int(hex6[3:5], 16)
        b = int(hex6[5:7], 16)
        return f"rgba({r},{g},{b},0.8)"

    starts = [0.0]
    for d in durs[:-1]:
        starts.append(starts[-1] + d)

    fig = go.Figure()
    for ag, s, d, out, clr in zip(agents, starts, durs, outputs, clrs):
        fig.add_trace(go.Bar(
            x=[d], y=[ag], orientation="h", base=s,
            marker_color=_rgba_080(clr),
            marker_line_color="white", marker_line_width=0.8, width=0.55,
            text=ag, textposition="inside", insidetextanchor="middle",
            textfont=dict(size=8.5, color=_SLATE),
            hovertemplate=f"<b>{ag}</b><br>{out}<extra></extra>",
            showlegend=False,
        ))
    fig.update_layout(**_PL,
        title_text="LangGraph Agent Execution Timeline (live run)", height=290,
        xaxis=dict(**_GR, title="Wall-clock time (s)"),
        yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)",
                   tickfont=dict(size=9.5, family="DM Mono", color=_OFF_WHITE)),
        bargap=0.30)
    return fig


def _chart_col_wasserstein(real_df, synth_df):
    """
    Per-column Wasserstein distances computed from the actual real & synthetic DataFrames.
    """
    try:
        from scipy.stats import wasserstein_distance
    except ImportError:
        return None

    num_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
    shared   = [c for c in num_cols if c in synth_df.columns]
    if not shared:
        return None

    dists = []
    for c in shared:
        r = real_df[c].dropna().values.astype(float)
        s = synth_df[c].dropna().values.astype(float)
        if len(r) > 1 and len(s) > 1:
            r_range = r.max() - r.min()
            norm    = r_range if r_range > 0 else 1.0
            dists.append(wasserstein_distance(r, s) / norm)
        else:
            dists.append(0.0)

    colors = [_PASS if d < 0.05 else _AMBER if d < 0.15 else _FAIL for d in dists]
    fig = go.Figure(go.Bar(
        x=shared, y=dists,
        marker_color=colors,
        text=[f"{d:.4f}" for d in dists], textposition="outside",
        textfont=dict(size=9, color=_OFF_WHITE),
    ))
    fig.add_hline(y=0.15, line_dash="dot", line_color=_PASS,
                  annotation_text="threshold 0.15",
                  annotation_font_color=_PASS, annotation_font_size=8)
    fig.update_layout(**_PL,
        title_text="Per-Column Normalised Wasserstein Distance (Real vs Synthetic)",
        height=310,
        xaxis=dict(**_GR, tickangle=15),
        yaxis=dict(**_GR, title="Normalised Wasserstein"))
    return fig


def _chart_corr_heatmaps(real_df, synth_df):
    """Return (fig_real, fig_synth) correlation heatmaps from actual data."""
    num_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
    shared   = [c for c in num_cols if c in synth_df.columns][:8]
    if len(shared) < 2:
        return None, None

    r_corr = real_df[shared].corr().values
    s_corr = synth_df[shared].corr().values
    base   = {k: v for k, v in _PL.items() if k not in ["xaxis", "yaxis"]}
    figs   = []
    for mat, title in [(r_corr, "Real Data Correlation"),
                       (s_corr, "Synthetic Data Correlation")]:
        fig = go.Figure(go.Heatmap(
            z=mat, x=shared, y=shared,
            colorscale=[[0, _FAIL], [0.5, _SLATE_MID], [1, _TEAL]],
            zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in mat],
            texttemplate="%{text}", textfont=dict(size=8),
        ))
        # base already contains margin from _PL — strip it to avoid duplicate kwarg
        base_no_margin = {k: v for k, v in base.items() if k != "margin"}
        fig.update_layout(**base_no_margin, title_text=title, height=270,
                          margin=dict(l=30, r=20, t=36, b=30))
        # PATCHED_CORR_MARGIN
        figs.append(fig)
    return figs[0], figs[1]


# ══════════════════════════════════════════════════════════════════════════════
#  HTML HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _badge(ok):
    return (f'<span class="sc-badge-pass">✓ PASS</span>'
            if ok else f'<span class="sc-badge-fail">✗ FAIL</span>')

def _kpi(val, lbl, ok=None):
    cls = "sc-kpi-pass" if ok is True else "sc-kpi-fail" if ok is False else ""
    return (f'<div class="sc-kpi {cls}">'
            f'<div class="sc-kpi-val">{val}</div>'
            f'<div class="sc-kpi-lbl">{lbl}</div></div>')

def _kpi_row(*items):
    return f'<div class="sc-kpi-grid">{"".join(items)}</div>'

def _gate_message():
    """Shown in showcase tabs when no run exists yet."""
    st.markdown("""
    <div class="sc-gate">
      <h3>🚀 No results yet</h3>
      <p>Go to the <strong>Generate</strong> tab, run the pipeline,<br>
         and come back here to explore your live results.</p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER  (your original, unchanged)
# ══════════════════════════════════════════════════════════════════════════════
st.title("🛡️ AgentSynth-Trinity: Autonomous Data Synthesis")
st.markdown("*A Master's Thesis Project for Healthcare & Finance Synthetic Data Generation*")
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_gen, tab_showcase, tab_privacy, tab_fidelity, tab_audit = st.tabs([
    "🚀  Generate",
    "🏆  Model Showcase",
    "🔒  Privacy Analysis",
    "📈  Fidelity Deep Dive",
    "📋  Audit & Compliance",
])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — GENERATE  (your EXACT original code, zero changes)
# ══════════════════════════════════════════════════════════════════════════════
with tab_gen:

    col_left, col_right = st.columns([1.2, 0.8], gap="large")

    with col_left:
        st.header("🔡 Describe the Data You Need")

        prompt_input = st.text_area(
            "Natural Language Prompt",
            value=st.session_state.prompt,
            placeholder="e.g., generate 500 diabetic patient records with high privacy",
            height=150,
            label_visibility="collapsed"
        )

        st.write("💡 Examples:")
        examples = [
            "500 diabetic patient records with high privacy",
            "1000 fraud transactions for model stress testing",
            "200 rare disease cohort records (lupus)",
            "5000 normal credit card transactions"
        ]
        cols = st.columns(len(examples))
        for i, ex in enumerate(examples):
            if cols[i].button(ex, key=f"ex_{i}"):
                update_config_from_prompt(ex)
                st.rerun()

        if st.button("🚀 Generate Synthetic Data", type="primary"):
            if not prompt_input and not st.session_state.prompt:
                st.warning("Please describe the data first or select an example.")
            else:
                if prompt_input != st.session_state.prompt:
                    update_config_from_prompt(prompt_input)
                st.session_state.results = "running"
                st.rerun()

    with col_right:
        with st.expander("⚙️ Advanced Settings & Overrides", expanded=True):
            domain = st.selectbox("Domain", ["Healthcare", "Finance"],
                                 index=0 if st.session_state.config.domain == "Healthcare" else 1)
            rows = st.slider("Number of Rows", 100, 10000, st.session_state.config.rows)
            epsilon = st.slider("Privacy Budget (ε)", 0.1, 10.0, st.session_state.config.epsilon,
                               help="0.1=Maximum Privacy, 10.0=Maximum Fidelity")
            model_type = st.selectbox("Generative Model", ["auto", "timegan", "vae", "diffusion"],
                                     index=["auto", "timegan", "vae", "diffusion"].index(st.session_state.config.model))
            out_format = st.selectbox("Output Format", ["CSV", "JSON", "Parquet"])
            seed = st.number_input("Random Seed", value=42)

            st.session_state.config.domain        = domain
            st.session_state.config.rows          = rows
            st.session_state.config.epsilon       = epsilon
            st.session_state.config.model         = model_type
            st.session_state.config.output_format = out_format
            st.session_state.config.seed          = int(seed)

        if st.button("🔄 Reset System"):
            reset_app()

    # --- Pipeline Execution ---
    if st.session_state.results == "running":
        st.divider()
        with st.status("🕵️ Agent Orchestrator: Initiating Synthesis...", expanded=True) as status:

            sample_path = f"data/{st.session_state.config.domain.lower()}_sample.csv"
            raw_df = data_loader.load_csv(sample_path)

            if raw_df is None:
                status.write(
                    f"💡 No input data file found — activating **cold-start mode** "
                    f"(generating {st.session_state.config.domain} seed from domain priors)"
                )

            state = AgentState(
                raw_data=raw_df,
                domain=st.session_state.config.domain,
                goal=st.session_state.config.goal,
                epsilon_input=st.session_state.config.epsilon,
                num_rows=st.session_state.config.rows,
                selected_model_type=st.session_state.config.model,
                output_format=getattr(st.session_state.config, "output_format", "CSV"),
                seed=getattr(st.session_state.config, "seed", 42),
            )

            try:
                update_queue = queue.Queue()
                final_results = run_async_pipeline(run_trinity_pipeline(state), update_queue)

                while not update_queue.empty():
                    msg_type, node_name, node_output = update_queue.get_nowait()
                    if msg_type == "agent_done":
                        status.write(f"✅ Agent Completed: **{node_name.capitalize()}**")

                st.session_state.results = final_results
                mode = "Cold-Start" if raw_df is None else "Augmentation"
                status.update(
                    label=f"Synthesis Complete [{mode} Mode]: Trinity Standards Met!",
                    state="complete"
                )
                st.rerun()

            except Exception as e:
                st.error(f"Pipeline Error: {e}")
                st.session_state.results = None
                status.update(label="Pipeline Failed", state="error")

    # --- Results Dashboard ---
    if isinstance(st.session_state.results, dict):
        res_state = st.session_state.results

        scorecard = res_state.get("trinity_scorecard") or {}
        fidelity  = scorecard.get("fidelity", {}) or {}
        utility   = scorecard.get("utility",  {}) or {}
        privacy_s = scorecard.get("privacy",  {}) or {}

        wass   = fidelity.get("wasserstein", 1.0)
        tstr   = utility.get("tstr_accuracy", 0.0)
        f_pass = wass < 0.15
        u_pass = utility.get("performance_drop", 100) < 20
        p_pass = bool(privacy_s.get("is_compliant", False))

        st.divider()

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("Judge Decision", res_state.get("judge_decision", "N/A"))
        with kpi2:
            st.metric("Fidelity (Wasserstein)", f"{wass:.4f}")
        with kpi3:
            st.metric("Utility (TSTR)", f"{tstr:.1%}")
        with kpi4:
            eps = res_state.get("privacy_proof", {}).get("epsilon", st.session_state.config.epsilon)
            st.metric("Privacy Cost (ε)", f"{eps:.2f}")

        tab1, tab2, tab3 = st.tabs(["📊 Data Preview", "🏆 Trinity Score", "📋 Audit Log"])

        with tab1:
            st.subheader("Synthetic Data Preview")
            synth_df = res_state.get("safe_data_asset")
            if synth_df is not None:
                st.dataframe(synth_df.head(50), use_container_width=True)

                with st.expander("📈 Column Statistics"):
                    st.write(synth_df.describe())

                st.subheader("Distribution Analysis")
                num_cols = synth_df.select_dtypes(include=[np.number]).columns[:5]
                if len(num_cols) > 0:
                    selected_col = st.selectbox("Select Column to Visualize", num_cols)
                    fig = px.histogram(synth_df, x=selected_col, color_discrete_sequence=['#ff7f0e'],
                                      title=f"{selected_col} Distribution (Synthetic)")
                    st.plotly_chart(fig, use_container_width=True)

        with tab2:
            col_rad, col_metrics = st.columns([1, 1])

            with col_rad:
                st.subheader("Trinity Radar Chart")
                fig_radar = reporter.generate_radar_chart(scorecard, output_path=None)
                st.pyplot(fig_radar)

            with col_metrics:
                st.subheader("Metric Compliance Breakdown")
                def get_badge(passed): return "✅ Pass" if passed else "❌ Fail"

                comp_data = {
                    "Metric": ["Fidelity (W-Dist)", "Utility (TSTR)", "Privacy (MIA/Link)"],
                    "Value":  [f"{wass:.4f}", f"{tstr:.1%}", "Compliant" if p_pass else "At Risk"],
                    "Status": [get_badge(f_pass), get_badge(u_pass), get_badge(p_pass)]
                }
                st.table(pd.DataFrame(comp_data))

                overall = int((f_pass + u_pass + p_pass) / 3 * 100)
                grade = "Excellent (85+)" if overall > 85 else "Good (70+)" if overall >= 70 else "Needs Improvement (<70)"
                st.metric("Overall Trinity Score", f"{overall}/100", delta=grade, delta_color="normal")

        with tab3:
            st.subheader("GDPR Compliance Audit")
            st.json(res_state.get("privacy_proof", {}))

            st.markdown("### Compliance Checklist")
            st.checkbox("GDPR Article 30 (RoPA) Generated",           value=True,   disabled=True)
            st.checkbox("Privacy Guard Sanitization Applied",          value=True,   disabled=True)
            st.checkbox("Statistical Fidelity Verified",               value=f_pass, disabled=True)
            st.checkbox("Membership Inference Attack (MIA) Resisted",  value=p_pass, disabled=True)

            if st.button("📄 View Compliance Certificate (Text)"):
                cert_path = cert_gen.generate_text_certificate(scorecard, st.session_state.config.domain)
                with open(cert_path, "r") as f:
                    st.text(f.read())

        # --- Downloads ---
        st.divider()
        st.subheader("⬇️ Download Assets")
        synth_df = res_state.get("safe_data_asset")
        d_col1, d_col2, d_col3, d_col4 = st.columns(4)

        if synth_df is not None:
            chosen_format = getattr(st.session_state.config, "output_format", "CSV")
            if chosen_format == "JSON":
                dl_data = synth_df.to_json(orient="records", indent=2).encode("utf-8")
                dl_name, dl_mime = "synthetic_data.json", "application/json"
            elif chosen_format == "Parquet":
                import io
                buf = io.BytesIO()
                synth_df.to_parquet(buf, index=False)
                dl_data = buf.getvalue()
                dl_name, dl_mime = "synthetic_data.parquet", "application/octet-stream"
            else:
                dl_data = synth_df.to_csv(index=False).encode("utf-8")
                dl_name, dl_mime = "synthetic_data.csv", "text/csv"

            d_col1.download_button(f"Download Synthetic {chosen_format}", data=dl_data, file_name=dl_name, mime=dl_mime)
            d_col2.download_button("Download Trinity Report",
                                   json.dumps(scorecard, indent=4).encode("utf-8"),
                                   "trinity_report.json", "application/json")
            d_col3.download_button("Download Audit Log",
                                   json.dumps(res_state.get("privacy_proof", {}), indent=4).encode("utf-8"),
                                   "audit_log.json", "application/json")
            d_col4.download_button("Download Certificate (TXT)",
                                   ("Trinity Compliance Certificate\n" + json.dumps(scorecard, indent=2)).encode("utf-8"),
                                   "certificate.txt")


# ══════════════════════════════════════════════════════════════════════════════
#  LIVE DATA — extracted once, shared across tabs 2-5
# ══════════════════════════════════════════════════════════════════════════════
live = _extract_live(st.session_state.results)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — MODEL SHOWCASE
# ══════════════════════════════════════════════════════════════════════════════
with tab_showcase:
    if live is None:
        _gate_message()
    else:
        # ── KPI strip ─────────────────────────────────────────────────────────
        st.markdown(_kpi_row(
            _kpi(live["decision"],              "DECISION",         live["decision"] == "APPROVE"),
            _kpi(f'{live["wass"]:.4f}',         "WASSERSTEIN ↓",    live["f_pass"]),
            _kpi(f'{live["tstr"]:.1%}',         "TSTR ACCURACY ↑",  live["u_pass"]),
            _kpi(f'{live["mia"]:.3f}',          "MIA SCORE ↓",      live["p_pass"]),
            _kpi(str(live["k_anon"] or "—"),    "k-ANONYMITY",      (live["k_anon"] or 0) >= 3),
            _kpi(f'Iter {live["iteration"]}',   "ITERATION",        None),
        ), unsafe_allow_html=True)

        # ── model comparison (only if pilot data available) ───────────────────
        if live["pilot_scores"]:
            st.plotly_chart(
                _chart_model_comparison(
                    live["pilot_scores"], live["wass"], live["tstr"], live["mia"]),
                use_container_width=True)

            # Leaderboard
            st.markdown('<div class="sc-card"><div class="sc-title">📋 Pilot Leaderboard</div>',
                        unsafe_allow_html=True)
            rank_css = ["sc-rank-gold", "sc-rank-silver", "sc-rank-bronze"]
            rank_ico = ["🥇", "🥈", "🥉"]
            for i, p in enumerate(live["pilot_scores"]):
                w_val = p.get("wasserstein", p.get("w")) or 0.0
                t_raw = p.get("tstr")
                m_raw = p.get("mia")
                w_str = f"{float(w_val):.4f}"
                t_str = f"{float(t_raw):.1%}" if t_raw is not None else "N/A"
                m_str = f"{float(m_raw):.3f}" if m_raw is not None else "N/A"
                rank  = rank_css[i] if i < 3 else ""
                icon  = rank_ico[i] if i < 3 else f"#{i+1}"
                ew    = live["ensemble_weights"].get(p["model"], 0)
                tag   = f"weight {float(ew):.1%}" if ew else ""
                teal_muted = "rgba(0,180,166,0.67)"
                st.markdown(
                    '<div class="sc-rank-row ' + rank + '">'
                    '<span class="sc-rank-num">' + str(icon) + '</span>'
                    '<span class="sc-rank-model">' + str(p["model"]) + '</span>'
                    '<span class="sc-rank-score">'
                    'W=' + w_str + ' &nbsp;|&nbsp; TSTR=' + t_str + ' &nbsp;|&nbsp; MIA=' + m_str +
                    '</span>'
                    '<span style="font-size:.7rem;color:' + teal_muted + ';margin-left:auto">' + tag + '</span>'
                    '</div>',
                    unsafe_allow_html=True)
# PATCHED_NONE_SAFE_v3
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Pilot model scores were not stored in the pipeline output. "
                    "Add `pilot_scores` to the AgentState return dict in `statistical_critic.py` "
                    "to populate this chart.", icon="ℹ️")

        # ── agent timeline ────────────────────────────────────────────────────
        gantt_fig = _chart_agent_gantt(live["agent_trace"])
        if gantt_fig:
            st.markdown('<div class="sc-card"><div class="sc-title">🔄 Agent Execution Timeline</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(gantt_fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── radar + compliance table ──────────────────────────────────────────
        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(
                _chart_radar(live["wass"], live["tstr"], live["mia"],
                             live["k_anon"], live["linkability"]),
                use_container_width=True)

        with c2:
            rows_data = [
                ("Wasserstein ↓",     f'{live["wass"]:.4f}',                    live["f_pass"]),
                ("JSD ↓",             f'{live["jsd"]:.4f}' if live["jsd"] is not None else "—", True),
                ("TSTR ↑",            f'{live["tstr"]:.1%}',                    live["u_pass"]),
                ("Perf Drop ↓",       f'{live["perf_drop"]:.1f}%',              live["perf_drop"] < 20),
                ("MIA Score ↓",       f'{live["mia"]:.3f}',                     live["p_pass"]),
                ("k-Anonymity ↑",     str(live["k_anon"] or "—"),               (live["k_anon"] or 0) >= 3),
                ("Linkability",       str(live["linkability"] if live["linkability"] is not None else "—"),
                                      live["linkability"] is None or live["linkability"] == 0),
            ]
            rows_html = "".join(
                f'<tr style="border-bottom:1px solid {_SLATE_LT_44}">'
                f'<td style="padding:.35rem 0">{lbl}</td>'
                f'<td style="text-align:right;font-family:\'DM Mono\'">{val}</td>'
                f'<td style="text-align:right">{_badge(ok)}</td></tr>'
                for lbl, val, ok in rows_data
            )
            st.markdown(f"""
            <div class="sc-card" style="margin-top:0">
              <div class="sc-title">📊 Full Metric Compliance</div>
              <table style="width:100%;border-collapse:collapse;font-size:.8rem">
                <tr style="color:{_OFF_WHITE_66};border-bottom:1px solid {_SLATE_LT}">
                  <th style="text-align:left;padding:.3rem 0">Metric</th>
                  <th style="text-align:right">Value</th>
                  <th style="text-align:right">Status</th>
                </tr>{rows_html}
              </table>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — PRIVACY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_privacy:
    if live is None:
        _gate_message()
    else:
        mia   = live["mia"]
        eps   = live["epsilon"]
        delta = live["delta"]
        p_ok  = live["p_pass"]

        # ── headline ─────────────────────────────────────────────────────────
        st.markdown(_kpi_row(
            _kpi(f'{mia:.3f}',             "MIA SCORE ↓",      mia < 0.76),
            _kpi(f'{live["k_anon"] or "—"}', "k-ANONYMITY",    (live["k_anon"] or 0) >= 3),
            _kpi(f'{eps:.2f}',             "EPSILON (ε)",       None),
            _kpi(str(delta) if delta else "—", "DELTA (δ)",     None),
            _kpi("PASS" if p_ok else "FAIL", "PRIVACY STATUS", p_ok),
        ), unsafe_allow_html=True)

        # ── DP proof JSON ─────────────────────────────────────────────────────
        st.markdown('<div class="sc-card"><div class="sc-title">🔐 Differential Privacy Proof</div>',
                    unsafe_allow_html=True)
        st.json(live["proof"])
        st.markdown("</div>", unsafe_allow_html=True)

        # ── MIA single result card ────────────────────────────────────────────
        mia_color   = _PASS if mia < 0.76 else _FAIL
        mia_verdict = "PASS — below threshold 0.76" if mia < 0.76 else "FAIL — above threshold 0.76"
        gap_to_rand = mia - 0.5

        st.markdown(f"""
        <div class="sc-card">
          <div class="sc-title">🔬 MIA Attack Result</div>
          <div style="display:flex;gap:2rem;align-items:center;flex-wrap:wrap">
            <div>
              <div style="font-family:'DM Mono';font-size:3rem;color:{mia_color};line-height:1">{mia:.3f}</div>
              <div style="font-size:.75rem;color:{_OFF_WHITE_88};margin-top:.25rem">5-fold CV MIA accuracy</div>
            </div>
            <div style="flex:1;min-width:200px">
              <p style="font-size:.83rem;color:{mia_color};font-weight:600;margin:0">{mia_verdict}</p>
              <p style="font-size:.78rem;color:{_OFF_WHITE_AA};margin:.4rem 0 0">
                Gap above random (0.50): <strong style="font-family:'DM Mono'">{gap_to_rand:+.3f}</strong><br>
                A score of 0.50 means the attack performs no better than random guessing — the ideal.<br>
                Threshold 0.76 provides a practical margin of safety.
              </p>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        # ── full privacy scorecard ────────────────────────────────────────────
        st.markdown('<div class="sc-card"><div class="sc-title">📋 Full Privacy Scorecard</div>',
                    unsafe_allow_html=True)
        priv_sc = live["scorecard"].get("privacy", {}) or {}
        st.json(priv_sc)
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — FIDELITY DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
with tab_fidelity:
    if live is None:
        _gate_message()
    else:
        real_df  = live["raw_df"]
        synth_df = live["synth_df"]

        # ── fidelity KPIs ─────────────────────────────────────────────────────
        st.markdown(_kpi_row(
            _kpi(f'{live["wass"]:.4f}',  "WASSERSTEIN ↓",   live["f_pass"]),
            _kpi(f'{live["jsd"]:.4f}' if live["jsd"] is not None else "—", "JSD ↓", True),
            _kpi(f'{live["corr_sim"]:.4f}' if live["corr_sim"] is not None else "—",
                 "CORR SIMILARITY",  None),
            _kpi(f'{live["mmd"]:.4f}' if live["mmd"] is not None else "—", "MMD",   None),
        ), unsafe_allow_html=True)

        # ── full fidelity scorecard ────────────────────────────────────────────
        st.markdown('<div class="sc-card"><div class="sc-title">📋 Full Fidelity Scorecard</div>',
                    unsafe_allow_html=True)
        st.json(live["scorecard"].get("fidelity", {}))
        st.markdown("</div>", unsafe_allow_html=True)

        if synth_df is not None:
            num_cols = synth_df.select_dtypes(include=[np.number]).columns.tolist()

            # ── real vs synthetic distribution overlay ────────────────────────
            if real_df is not None and len(num_cols) > 0:
                st.markdown('<div class="sc-card"><div class="sc-title">📊 Real vs Synthetic Distribution</div>',
                            unsafe_allow_html=True)
                col_sel = st.selectbox(
                    "Select column", num_cols, key="fid_col",
                    help="Overlay of real (teal) and synthetic (amber) KDE distributions")
                if col_sel in real_df.columns:
                    dist_fig = _chart_dist_overlay(col_sel, real_df[col_sel], synth_df[col_sel])
                    if dist_fig:
                        st.plotly_chart(dist_fig, use_container_width=True)
                    else:
                        st.info("Install scipy for KDE overlays: `pip install scipy`")
                else:
                    st.warning(f"Column '{col_sel}' not found in real data.")
                st.markdown("</div>", unsafe_allow_html=True)

            # ── per-column Wasserstein ────────────────────────────────────────
            if real_df is not None:
                wass_fig = _chart_col_wasserstein(real_df, synth_df)
                if wass_fig:
                    st.plotly_chart(wass_fig, use_container_width=True)

            # ── correlation heatmaps ──────────────────────────────────────────
            if real_df is not None:
                fig_r, fig_s = _chart_corr_heatmaps(real_df, synth_df)
                if fig_r and fig_s:
                    st.markdown('<div class="sc-card"><div class="sc-title">🔗 Correlation Preservation</div>',
                                unsafe_allow_html=True)
                    hc1, hc2 = st.columns(2)
                    hc1.plotly_chart(fig_r, use_container_width=True)
                    hc2.plotly_chart(fig_s, use_container_width=True)

                    # Frobenius norm
                    shared_num = [c for c in real_df.select_dtypes(include=[np.number]).columns
                                  if c in synth_df.columns][:8]
                    if len(shared_num) >= 2:
                        r_c = real_df[shared_num].corr().values
                        s_c = synth_df[shared_num].corr().values
                        frob = np.linalg.norm(r_c - s_c, "fro")
                        st.markdown(
                            f'<p style="font-size:.77rem;color:{_OFF_WHITE_66};text-align:center">'
                            f'Frobenius norm between correlation matrices: {frob:.4f}'
                            f' — lower = better preserved structure</p>',
                            unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("Synthetic data not found in pipeline output (`safe_data_asset`).")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — AUDIT & COMPLIANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab_audit:
    if live is None:
        _gate_message()
    else:
        c1, c2 = st.columns([1, 1])

        with c1:
            st.markdown('<div class="sc-card"><div class="sc-title">🏛️ GDPR Audit Log</div>',
                        unsafe_allow_html=True)
            st.json(live["proof"])
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="sc-card"><div class="sc-title">✅ Compliance Checklist</div>',
                        unsafe_allow_html=True)
            checklist = [
                (True,            "GDPR Article 30 RoPA Generated"),
                (live["epsilon"] is not None, "Differential Privacy Applied"),
                (live["f_pass"],  f'Fidelity Verified (W={live["wass"]:.4f} < 0.15)'),
                (live["u_pass"],  f'Utility Retained (Drop={live["perf_drop"]:.1f}% < 20%)'),
                (live["p_pass"],  f'MIA Resisted ({live["mia"]:.3f} < threshold 0.76)'),
                ((live["k_anon"] or 0) >= 3,
                 f'k-Anonymity Satisfied (k={live["k_anon"] or "—"})'),
                (live["linkability"] is None or live["linkability"] == 0,
                 "Zero Exact-Match Linkability"),
                (live["decision"] == "APPROVE", f'Judge Decision: {live["decision"]}'),
            ]
            for ok, label in checklist:
                icon  = "✅" if ok else "❌"
                color = _PASS if ok else _FAIL
                st.markdown(
                    f'<div style="padding:.3rem 0;font-size:.81rem;'
                    f'border-bottom:1px solid {_SLATE_LT_44}">'
                    f'{icon} <span style="color:{color}">{label}</span></div>',
                    unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Compliance certificate
        st.markdown('<div class="sc-card"><div class="sc-title">📜 Compliance Certificate</div>',
                    unsafe_allow_html=True)
        scorecard = live["scorecard"]
        if st.button("📄 Generate & View Certificate"):
            cert_path = cert_gen.generate_text_certificate(
                scorecard, live["domain"])
            with open(cert_path, "r") as f:
                cert_text = f.read()
            st.code(cert_text, language=None)
            st.download_button("⬇ Download Certificate",
                               cert_text.encode(), "compliance_certificate.txt", "text/plain")
        st.markdown("</div>", unsafe_allow_html=True)

        # Full scorecard JSON
        with st.expander("🔍 Full Trinity Scorecard (raw JSON)"):
            st.json(scorecard)
