import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="Raman Mineral Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #1e1a16; }
[data-testid="stSidebar"] * { color: #d4c5a9 !important; }
.main { background: #13110e; color: #d4c5a9; }
.stApp { background: #13110e; }
.metric-card {
    background: #2a2318; border: 1px solid #4a3c2a;
    border-radius: 8px; padding: 14px 18px; margin: 6px 0;
}
.mineral-rank { font-size: 0.75rem; color: #8a7a62; font-weight: 600; }
.mineral-name { font-size: 1.1rem; font-weight: 700; color: #e8d5b0; }
.mineral-formula { font-size: 0.85rem; color: #a08060; font-style: italic; }
.conf-bar-wrap { height: 6px; background:#3a3020; border-radius:3px; margin:6px 0; }
.conf-bar { height:6px; border-radius:3px; }
.badge {
    display:inline-block; padding:2px 8px; border-radius:10px;
    font-size:0.7rem; font-weight:600; margin-left:6px;
}
.badge-ok  { background:#1a3a1a; color:#6abf6a; border:1px solid #2a5a2a; }
.badge-off { background:#3a1a1a; color:#bf6a6a; border:1px solid #5a2a2a; }
.badge-src { background:#2a2838; color:#8a8abf; border:1px solid #3a3a5a; }
h1, h2, h3 { color: #e8d5b0 !important; }
.stButton button {
    background: #6b4c2a; color: #f0e0c0; border: none;
    width: 100%; font-weight: 600; border-radius: 6px; padding: 10px;
}
.stButton button:hover { background: #8a6040; }
div[data-testid="stSelectbox"] label { color: #c0a878 !important; }
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading reference database…")
def load_database():
    from database.reference_store import get_database, get_match_database
    matrix, names, metadata = get_database()
    d2_matrix, _, _ = get_match_database()
    return matrix, names, metadata, d2_matrix


@st.cache_resource(show_spinner="Loading CNN model…")
def load_cnn():
    import core.cnn_model as cnn_mod
    return cnn_mod


@st.cache_resource(show_spinner="Loading Random Forest model…")
def load_rf():
    import core.rf_model as rf_mod
    return rf_mod


# ── Helpers ───────────────────────────────────────────────────────────────────

def _conf_color(pct: float) -> str:
    if pct >= 80:
        return "#4caf50"
    if pct >= 60:
        return "#ffa726"
    return "#ef5350"


def build_spectrum_fig(grid, raw, processed, reference_overlays=None, title="Uploaded Spectrum"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=grid, y=raw,
        name="Raw", line=dict(color="#6a5a4a", width=1, dash="dot"),
        hovertemplate="<b>%{x:.0f} cm⁻¹</b><br>Intensity: %{y:.4f}<extra>Raw</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=grid, y=processed,
        name="Processed", line=dict(color="#d4a97a", width=2),
        hovertemplate="<b>%{x:.0f} cm⁻¹</b><br>Intensity: %{y:.4f}<extra>Processed</extra>",
    ))
    if reference_overlays:
        colors = ["#7abfbf", "#bf7abf", "#7abf7a", "#bfbf7a", "#bf7a7a"]
        for i, ov in enumerate(reference_overlays[:3]):
            fig.add_trace(go.Scatter(
                x=grid, y=ov["spectrum"],
                name=ov["mineral"], line=dict(color=colors[i % len(colors)], width=1.5, dash="dash"),
                opacity=0.7,
                hovertemplate=f"<b>%{{x:.0f}} cm⁻¹</b><br>Intensity: %{{y:.4f}}<extra>{ov['mineral']}</extra>",
            ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#e8d5b0", size=14)),
        paper_bgcolor="#1e1a16", plot_bgcolor="#13110e",
        font=dict(color="#c0a878"),
        xaxis=dict(title="Wavenumber (cm⁻¹)", gridcolor="#2a2318", color="#a08060"),
        yaxis=dict(title="Normalised Intensity", gridcolor="#2a2318", color="#a08060"),
        legend=dict(bgcolor="#1e1a16", bordercolor="#4a3c2a", borderwidth=1),
        margin=dict(l=60, r=20, t=50, b=50),
        height=380,
    )
    return fig


def build_mixture_fig(components):
    labels = [c["mineral"] for c in components]
    values = [c["fraction_pct"] for c in components]
    colors = [c.get("color", "#7c5c3e") for c in components]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=colors, line=dict(color="#1e1a16", width=2)),
        textinfo="label+percent",
        textfont=dict(color="#e8d5b0", size=13),
        hole=0.4,
        hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#1e1a16",
        font=dict(color="#c0a878"),
        margin=dict(l=20, r=20, t=30, b=20),
        height=300,
        showlegend=False,
    )
    return fig


def render_match_cards(matches):
    for i, m in enumerate(matches):
        pct = m.get("confidence_pct", 0)
        color = _conf_color(pct)
        src_badge = f'<span class="badge badge-src">{m.get("source","")}</span>'
        formula = m.get("formula", "")
        desc = m.get("description", "")
        st.markdown(f"""
<div class="metric-card">
  <div class="mineral-rank">#{i+1} match</div>
  <div class="mineral-name">{m['mineral']} {src_badge}</div>
  <div class="mineral-formula">{formula}</div>
  <div class="conf-bar-wrap">
    <div class="conf-bar" style="width:{pct}%;background:{color}"></div>
  </div>
  <div style="font-size:0.8rem;color:#8a7a62">{pct:.1f}% confidence</div>
  {'<div style="font-size:0.78rem;color:#6a5a42;margin-top:4px">'+desc[:120]+'</div>' if desc else ''}
</div>""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔬 Raman Mineral Analyzer")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload Spectrum (CSV)",
        type=["csv", "txt"],
        help="Two-column CSV: wavenumber (cm⁻¹), intensity. Any delimiter.",
    )

    mode = st.selectbox(
        "Identification mode",
        ["Auto", "Spectral Matching", "CNN", "Random Forest"],
        help="Auto picks the highest-confidence result across all available models.",
    )
    mode_map = {
        "Auto": "auto",
        "Spectral Matching": "matching",
        "CNN": "cnn",
        "Random Forest": "rf",
    }

    show_steps = st.checkbox("Show preprocessing steps", value=False)

    run_btn = st.button("⚡ Identify Mineral", disabled=(uploaded is None))

    st.markdown("---")
    st.markdown("**System status**")

    cnn_mod = load_cnn()
    rf_mod  = load_rf()
    _, _, _, _ = load_database()  # warm the DB cache

    cnn_ok = cnn_mod.CNN_AVAILABLE
    rf_ok  = rf_mod.RF_AVAILABLE

    st.markdown(
        f'CNN &nbsp;<span class="badge {"badge-ok" if cnn_ok else "badge-off"}">{"✓ ready" if cnn_ok else "✗ offline"}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'Random Forest &nbsp;<span class="badge {"badge-ok" if rf_ok else "badge-off"}">{"✓ ready" if rf_ok else "✗ offline"}</span>',
        unsafe_allow_html=True,
    )

    matrix, names, metadata, _ = load_database()
    st.markdown(
        f'Reference spectra &nbsp;<span class="badge badge-src">{len(names):,}</span>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption("Upload a Raman spectrum CSV to get started. Sample files are in the `samples/` folder.")


# ── Main area ─────────────────────────────────────────────────────────────────

st.markdown("# Raman Mineral Analyzer")
st.markdown("Identify minerals from Raman spectroscopy data using spectral matching, CNN, and Random Forest classifiers.")

if uploaded is None:
    st.info("👈 Upload a CSV spectrum in the sidebar to begin. Try `samples/quartz.csv` for a demo.", icon="ℹ️")
    with st.expander("About this tool"):
        st.markdown("""
**How it works**

1. **Upload** a two-column CSV: wavenumber (cm⁻¹) and intensity.
2. The spectrum is preprocessed: Savitzky–Golay smoothing + ALS baseline removal, then L2-normalised onto a standard 100–4000 cm⁻¹ grid.
3. **Three classifiers** can identify the mineral:
   - *Spectral Matching* — dual-metric cosine + 2nd-derivative Pearson against ~4,100 RRUFF reference spectra
   - *CNN* — 1D ResNet trained with 80 augmentation variants per spectrum
   - *Random Forest* — PCA (120 components) + ExtraTrees (300 trees)
4. **Mixture decomposition** uses Non-Negative Least Squares (NNLS) to detect and quantify multi-mineral samples.
        """)
    st.stop()


# ── Run analysis ──────────────────────────────────────────────────────────────

if "result" not in st.session_state:
    st.session_state.result = None
if "filename" not in st.session_state:
    st.session_state.filename = None

if run_btn or (st.session_state.result is None and uploaded is not None):
    file_bytes = uploaded.read()
    selected_mode = mode_map[mode]

    with st.spinner("Analysing spectrum…"):
        try:
            from core.pipeline import run_full_analysis
            result = run_full_analysis(file_bytes, mode=selected_mode)
            st.session_state.result = result
            st.session_state.filename = uploaded.name
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

result = st.session_state.result
if result is None:
    st.stop()

grid       = result["spectrum"]["wavenumber"]
raw        = result["spectrum"]["raw"]
processed  = result["spectrum"]["processed"]
steps      = result["spectrum"]["steps"]
matches    = result["matches"]
mixture    = result["mixture"]
overlays   = result["reference_overlays"]
mode_used  = result["mode_used"]
is_mix     = result["is_likely_mixture"]

# ── Spectrum plot ─────────────────────────────────────────────────────────────

fname = st.session_state.filename or "spectrum"
fig = build_spectrum_fig(grid, raw, processed, overlays, title=f"Spectrum: {fname}")
st.plotly_chart(fig, use_container_width=True)

if show_steps:
    with st.expander("Preprocessing steps"):
        fig2 = go.Figure()
        for label, color, y in [
            ("Raw",       "#6a5a4a", steps["raw"]),
            ("Smoothed",  "#a08060", steps["smoothed"]),
            ("Baseline",  "#bf6a6a", steps["baseline"]),
            ("Corrected", "#d4a97a", steps["corrected"]),
        ]:
            fig2.add_trace(go.Scatter(x=grid, y=y, name=label, line=dict(color=color, width=1.5)))
        fig2.update_layout(
            paper_bgcolor="#1e1a16", plot_bgcolor="#13110e",
            font=dict(color="#c0a878"),
            xaxis=dict(title="Wavenumber (cm⁻¹)", gridcolor="#2a2318"),
            yaxis=dict(title="Intensity", gridcolor="#2a2318"),
            legend=dict(bgcolor="#1e1a16"),
            height=300, margin=dict(l=60, r=20, t=30, b=50),
        )
        st.plotly_chart(fig2, use_container_width=True)

# ── Results ───────────────────────────────────────────────────────────────────

col_id, col_mix = st.columns([1, 1], gap="large")

with col_id:
    method_label = mode_used.upper().replace("_", " ")
    st.markdown(f"### Identification Results <span style='font-size:0.75rem;color:#8a7a62'>via {method_label}</span>", unsafe_allow_html=True)

    if is_mix:
        st.warning("⚠️ Low top-match confidence — this spectrum may be a **mixture**. See the decomposition panel →", icon="⚠️")

    if not matches:
        st.error("No matches found.")
    else:
        render_match_cards(matches)

    if mode_used == "matching" and not cnn_ok and not rf_ok:
        st.caption("CNN and RF models offline — only spectral matching is available.")


with col_mix:
    st.markdown("### Mixture Decomposition")
    comps = mixture.get("components", [])
    is_mixture_flag = mixture.get("is_mixture", False)
    residual = mixture.get("residual_norm", 0)

    if not comps:
        st.info("No significant mixture components detected.")
    elif not is_mixture_flag:
        st.success(f"✅ **Pure mineral** — {comps[0]['mineral']} ({comps[0]['fraction_pct']:.1f}%)")
        st.caption(f"NNLS residual: {residual:.4f}")
    else:
        st.info(f"🪨 **Mixture detected** — {len(comps)} components (NNLS residual: {residual:.4f})")
        st.plotly_chart(build_mixture_fig(comps), use_container_width=True)
        for c in comps:
            pct = c["fraction_pct"]
            st.markdown(
                f"**{c['mineral']}** (*{c['formula']}*) — "
                f"<span style='color:#d4a97a;font-weight:600'>{pct:.1f}%</span>",
                unsafe_allow_html=True,
            )

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption("Raman Mineral Analyzer · RRUFF database spectra · [GitHub](https://github.com/Coder-bot-vBeta/raman-spectroscopy)")
