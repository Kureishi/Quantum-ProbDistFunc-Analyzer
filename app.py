"""
app.py — Schrödinger Parameter Analyser
========================================
Streamlit UI: upload a |ψ(x)|² probability density function image,
send it to a fine-tuned VLM running in LM Studio, and display the
extracted Schrödinger equation parameters.

Run:
  streamlit run app.py

Requirements:
  pip install streamlit openai Pillow

LM Studio must be running with the local server enabled (port 1234).
"""

import base64
import json
import re
import time
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Schrödinger Analyser",
    page_icon="⚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Syne:wght@400;600;700;800&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #080C14;
    color: #C8D8E8;
}

/* ── Scanline overlay ── */
body::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,255,200,0.015) 2px,
        rgba(0,255,200,0.015) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* ── Main content area ── */
.main .block-container {
    padding: 2rem 3rem;
    max-width: 1200px;
}

/* ── Header ── */
.site-header {
    border-bottom: 1px solid rgba(0,255,180,0.2);
    padding-bottom: 1.4rem;
    margin-bottom: 2rem;
}
.site-header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    letter-spacing: -0.02em;
    color: #E8F4F0;
    margin: 0 0 0.2rem 0;
    line-height: 1.1;
}
.site-header h1 span {
    color: #00FFB2;
}
.site-header p {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    color: #4A7A6A;
    letter-spacing: 0.08em;
    margin: 0;
}

/* ── Upload zone ── */
.upload-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    color: #00FFB2;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* ── Streamlit file uploader override ── */
[data-testid="stFileUploader"] {
    background: rgba(0,255,178,0.03);
    border: 1px solid rgba(0,255,178,0.18);
    border-radius: 6px;
    padding: 0.5rem;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,255,178,0.5);
}

/* ── Image preview panel ── */
.image-panel {
    background: #0B1220;
    border: 1px solid rgba(0,255,178,0.12);
    border-radius: 8px;
    padding: 1rem;
    position: relative;
}
.panel-tag {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    color: #00FFB2;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    opacity: 0.7;
}

/* ── Analyse button ── */
.stButton > button {
    background: transparent !important;
    border: 1.5px solid #00FFB2 !important;
    color: #00FFB2 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.15em !important;
    padding: 0.65rem 2rem !important;
    border-radius: 3px !important;
    width: 100% !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    background: rgba(0,255,178,0.08) !important;
    box-shadow: 0 0 20px rgba(0,255,178,0.2) !important;
}
.stButton > button:active {
    background: rgba(0,255,178,0.18) !important;
}

/* ── Result card ── */
.result-card {
    background: #0B1220;
    border: 1px solid rgba(0,255,178,0.15);
    border-radius: 8px;
    padding: 1.5rem;
    margin-top: 0.5rem;
    animation: fadeSlideIn 0.4s ease-out;
}
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

.result-section-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    color: #4A7A6A;
    text-transform: uppercase;
    margin-bottom: 1rem;
    border-bottom: 1px solid rgba(0,255,178,0.08);
    padding-bottom: 0.5rem;
}

/* ── Potential badge ── */
.potential-badge {
    display: inline-block;
    background: rgba(0,255,178,0.08);
    border: 1px solid rgba(0,255,178,0.3);
    border-radius: 3px;
    padding: 0.3rem 0.8rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.82rem;
    color: #00FFB2;
    letter-spacing: 0.06em;
    margin-bottom: 1.2rem;
}

/* ── Parameter rows ── */
.param-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 0.55rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.param-row:last-child { border-bottom: none; }
.param-key {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    color: #5A8A7A;
    letter-spacing: 0.04em;
}
.param-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.92rem;
    color: #E0F0EA;
    font-weight: 600;
}
.param-unit {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem;
    color: #3A6A5A;
    margin-left: 0.4rem;
}

/* ── Energy highlight ── */
.energy-highlight .param-value {
    color: #00FFB2;
    font-size: 1.05rem;
}

/* ── Sub-params block ── */
.subparams-block {
    background: rgba(0,0,0,0.2);
    border-left: 2px solid rgba(0,255,178,0.2);
    border-radius: 0 4px 4px 0;
    padding: 0.6rem 0.8rem;
    margin-top: 0.4rem;
}

/* ── Latency tag ── */
.latency-tag {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    color: #3A6A5A;
    letter-spacing: 0.1em;
    text-align: right;
    margin-top: 1rem;
}

/* ── Error box ── */
.error-box {
    background: rgba(255, 60, 60, 0.07);
    border: 1px solid rgba(255,60,60,0.25);
    border-radius: 6px;
    padding: 1rem 1.2rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    color: #FF8080;
    letter-spacing: 0.04em;
    animation: fadeSlideIn 0.3s ease-out;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0A0F1A !important;
    border-right: 1px solid rgba(0,255,178,0.08) !important;
}
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stNumberInput input,
[data-testid="stSidebar"] .stSelectbox select {
    background: #0D1525 !important;
    border: 1px solid rgba(0,255,178,0.2) !important;
    color: #C8D8E8 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.82rem !important;
    border-radius: 3px !important;
}
.sidebar-section {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    color: #00FFB2;
    text-transform: uppercase;
    opacity: 0.6;
    margin: 1.2rem 0 0.5rem 0;
}

/* ── JSON expander ── */
[data-testid="stExpander"] {
    background: rgba(0,0,0,0.2) !important;
    border: 1px solid rgba(0,255,178,0.08) !important;
    border-radius: 4px !important;
}

/* ── History items ── */
.history-item {
    background: rgba(0,255,178,0.03);
    border: 1px solid rgba(0,255,178,0.08);
    border-radius: 4px;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.5rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    color: #5A8A7A;
    cursor: pointer;
    transition: border-color 0.15s;
}
.history-item:hover {
    border-color: rgba(0,255,178,0.25);
}
.history-item .hi-potential { color: #C8D8E8; font-size: 0.78rem; }
.history-item .hi-meta { color: #3A6A5A; font-size: 0.65rem; margin-top: 0.15rem; }

/* ── Idle state ── */
.idle-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 280px;
    opacity: 0.35;
}
.idle-atom {
    font-size: 3.5rem;
    margin-bottom: 0.8rem;
    animation: pulse 3s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 0.35; transform: scale(1); }
    50%       { opacity: 0.6;  transform: scale(1.04); }
}
.idle-text {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    color: #4A7A6A;
    text-align: center;
}

/* ── Potential colour map ── */
.pot-infinite_square_well  { color: #60C8FF !important; border-color: rgba(96,200,255,0.3) !important; background: rgba(96,200,255,0.06) !important; }
.pot-harmonic_oscillator   { color: #C080FF !important; border-color: rgba(192,128,255,0.3) !important; background: rgba(192,128,255,0.06) !important; }
.pot-finite_square_well    { color: #60FFA0 !important; border-color: rgba(96,255,160,0.3) !important; background: rgba(96,255,160,0.06) !important; }
.pot-hydrogen_radial       { color: #FFD060 !important; border-color: rgba(255,208,96,0.3) !important; background: rgba(255,208,96,0.06) !important; }
.pot-double_well           { color: #FF7070 !important; border-color: rgba(255,112,112,0.3) !important; background: rgba(255,112,112,0.06) !important; }

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a quantum mechanics analysis assistant. "
    "Given an image of a probability density function |ψ(x)|² plot for a solution "
    "to the time-independent Schrödinger equation, extract and return the physical "
    "parameters as a JSON object. "
    "Always respond with valid JSON only — no explanation, no markdown fences."
    "Always include all potential_params keys — never return an empty object."
)

USER_PROMPT = (
    "Analyse this |ψ(x)|² probability density function plot and return the "
    "Schrödinger equation parameters as JSON with these exact keys:\n"
    "  potential_type   (string: infinite_square_well | harmonic_oscillator | "
    "finite_square_well | hydrogen_radial | double_well)\n"
    "  quantum_number_n (integer)\n"
    "  mass_au          (float, in atomic units)\n"
    "  energy_au        (float, in atomic units)\n"
    "  potential_params (object with potential-specific keys)\n\n"
    "Respond with JSON only."
)

POTENTIAL_LABELS = {
    "infinite_square_well": "Infinite Square Well",
    "harmonic_oscillator":  "Quantum Harmonic Oscillator",
    "finite_square_well":   "Finite Square Well",
    "hydrogen_radial":      "Hydrogen Atom (Radial)",
    "double_well":          "Symmetric Double Well",
}

PARAM_DESCRIPTIONS = {
    "L":     ("Well width",         "a.u."),
    "V0":    ("Barrier height",     "a.u."),
    "omega": ("Angular frequency",  "a.u."),
    "a":     ("Well separation",    "a.u."),
    "b":     ("Shape factor",       ""),
    "l":     ("Angular momentum ℓ", ""),
}

# ── Session state defaults ────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []   # list of result dicts
if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def image_to_data_uri(pil_img: Image.Image) -> str:
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def parse_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(l for l in lines[1:] if not l.startswith("```")).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = re.sub(r",\s*([}\]])", r"\1", text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"_error": "Could not parse model response", "_raw": text}


def get_lmstudio_client(base_url: str, api_key: str):
    try:
        from openai import OpenAI
        return OpenAI(base_url=base_url, api_key=api_key)
    except ImportError:
        return None


def fetch_models(client) -> list[str]:
    try:
        return [m.id for m in client.models.list().data]
    except Exception:
        return []


def run_inference(client, model_id: str, pil_img: Image.Image,
                  temperature: float, max_tokens: int) -> tuple[dict, float]:
    data_uri = image_to_data_uri(pil_img)
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text",      "text": USER_PROMPT},
                ],
            },
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    elapsed = time.perf_counter() - t0
    raw = response.choices[0].message.content
    return parse_response(raw), elapsed


# ── Render helpers ────────────────────────────────────────────────────────────

def render_result(result: dict, elapsed: float):
    """
    Render extracted parameters using only native Streamlit components.
    No st.markdown HTML — avoids Streamlit's HTML sanitiser stripping
    display:flex and other CSS, which caused raw HTML to leak as plain text.
    """

    # Potential → emoji + colour label for st.success / st.info etc.
    POTENTIAL_ICONS = {
        "infinite_square_well": "🟦",
        "harmonic_oscillator":  "🟣",
        "finite_square_well":   "🟩",
        "hydrogen_radial":      "🟡",
        "double_well":          "🔴",
    }

    if "_error" in result:
        st.error(f"Model error: {result['_error']}")
        if result.get("_raw"):
            with st.expander("Raw response"):
                st.text(result["_raw"][:500])
        return

    pt       = result.get("potential_type", "unknown")
    pt_label = POTENTIAL_LABELS.get(pt, pt.replace("_", " ").title())
    icon     = POTENTIAL_ICONS.get(pt, "⚛")
    n        = result.get("quantum_number_n", "—")
    mass     = result.get("mass_au", "—")
    energy   = result.get("energy_au", "—")
    pp       = result.get("potential_params") or {}

    energy_str = f"{energy:.6f} a.u." if isinstance(energy, float) else str(energy)
    mass_str   = f"{mass:.4f} a.u."   if isinstance(mass,   float) else str(mass)

    # ── Potential type header ─────────────────────────────────────────────────
    st.markdown(f"### {icon} {pt_label}")
    st.divider()

    # ── Core parameters in a 3-column metric row ──────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Quantum number n", str(n))
    c2.metric("Energy eigenvalue", energy_str)
    c3.metric("Particle mass", mass_str)

    # ── Potential-specific parameters ─────────────────────────────────────────
    st.markdown("**Potential parameters**")
    if pp:
        cols = st.columns(max(len(pp), 1))
        for col, (k, v) in zip(cols, pp.items()):
            desc, unit = PARAM_DESCRIPTIONS.get(k, (k, "a.u."))
            val_str    = f"{v:.4g}" if isinstance(v, (int, float)) else str(v)
            label      = f"{desc} ({k})"
            col.metric(label, f"{val_str} {unit}".strip())
    else:
        st.caption("_No potential params returned by model_")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.caption(f"⏱ {elapsed:.2f}s · {model_id_display()} · atomic units")
    st.divider()

    # ── Raw JSON ──────────────────────────────────────────────────────────────
    with st.expander("Raw JSON", expanded=False):
        st.code(json.dumps(result, indent=2), language="json")


def model_id_display() -> str:
    return st.session_state.get("active_model", "—")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:0.5rem 0 1rem 0">
        <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.1rem;
                    color:#E8F4F0;letter-spacing:-0.01em;">⚛ Config</div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;
                    color:#3A6A5A;letter-spacing:0.1em;margin-top:0.2rem;">
            LM STUDIO CONNECTION
        </div>
    </div>
    """, unsafe_allow_html=True)

    base_url = st.text_input(
        "Server URL",
        value="http://localhost:1234/v1",
        help="LM Studio local server endpoint",
    )
    api_key = st.text_input(
        "API Key",
        value="lm-studio",
        type="password",
        help="Ignored by LM Studio but required by the OpenAI client",
    )

    client = get_lmstudio_client(base_url, api_key)

    st.markdown('<div class="sidebar-section">Model</div>', unsafe_allow_html=True)
    if client:
        with st.spinner("Fetching models…"):
            available_models = fetch_models(client)
    else:
        available_models = []
        st.error("openai package not installed. Run: pip install openai")

    if available_models:
        selected_model = st.selectbox(
            "Loaded model",
            available_models,
            help="Models currently loaded in LM Studio",
        )
        st.session_state["active_model"] = selected_model
        st.markdown(
            f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.65rem;'
            f'color:#00FFB2;opacity:0.7;margin-top:0.3rem;">● CONNECTED</div>',
            unsafe_allow_html=True,
        )
    else:
        selected_model = None
        st.session_state["active_model"] = "—"
        st.markdown(
            '<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.65rem;'
            'color:#FF6060;opacity:0.8;margin-top:0.3rem;">○ NO MODEL DETECTED</div>',
            unsafe_allow_html=True,
        )
        st.caption("Start the server in LM Studio → Developer tab")

    st.markdown('<div class="sidebar-section">Inference</div>', unsafe_allow_html=True)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05,
                            help="Lower = more deterministic output")
    max_tokens  = st.slider("Max tokens",  64, 1024, 512, 64)

    # ── History ───────────────────────────────────────────────────────────────
    if st.session_state.history:
        st.markdown('<div class="sidebar-section">History</div>', unsafe_allow_html=True)
        for i, h in enumerate(reversed(st.session_state.history[-8:])):
            pt = h["result"].get("potential_type", "unknown")
            label = POTENTIAL_LABELS.get(pt, pt)
            n_val = h["result"].get("quantum_number_n", "?")
            e_val = h["result"].get("energy_au", "?")
            e_str = f"{e_val:.4f}" if isinstance(e_val, float) else str(e_val)
            st.markdown(f"""
            <div class="history-item">
                <div class="hi-potential">{label}</div>
                <div class="hi-meta">n={n_val} &nbsp;·&nbsp; E={e_str} a.u. &nbsp;·&nbsp; {h['elapsed']:.2f}s</div>
            </div>""", unsafe_allow_html=True)


# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="site-header">
    <h1>Schrödinger<span> Analyser</span></h1>
    <p>QUANTUM STATE PARAMETER EXTRACTION  ·  TIME-INDEPENDENT SCHRÖDINGER EQUATION  ·  ATOMIC UNITS</p>
</div>
""", unsafe_allow_html=True)

left_col, right_col = st.columns([1, 1], gap="large")

# ── Left: upload + image preview ─────────────────────────────────────────────
with left_col:
    st.markdown('<div class="upload-label">▸ Upload |ψ(x)|² image</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        label="Upload PDF plot",
        type=["png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed",
    )

    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        st.markdown('<div class="image-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-tag">↳ Input image</div>', unsafe_allow_html=True)
        st.image(pil_img, use_container_width=True)
        w, h = pil_img.size
        st.markdown(
            f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.65rem;'
            f'color:#3A6A5A;margin-top:0.4rem;">{w} × {h} px &nbsp;·&nbsp; {uploaded.name}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        analyse_clicked = st.button("⚛  Analyse Parameters", use_container_width=True)
    else:
        # Idle state
        st.markdown("""
        <div class="idle-state">
            <div class="idle-atom">⚛</div>
            <div class="idle-text">Drop a |ψ(x)|² plot above<br>to extract quantum parameters</div>
        </div>""", unsafe_allow_html=True)
        analyse_clicked = False

# ── Right: results ────────────────────────────────────────────────────────────
with right_col:
    st.markdown('<div class="upload-label">▸ Extracted parameters</div>', unsafe_allow_html=True)

    if analyse_clicked:
        if not client:
            st.markdown(
                '<div class="error-box">openai not installed — pip install openai</div>',
                unsafe_allow_html=True,
            )
        elif not selected_model:
            st.markdown(
                '<div class="error-box">No model loaded in LM Studio.<br>'
                'Load your fine-tuned GGUF and start the server.</div>',
                unsafe_allow_html=True,
            )
        else:
            with st.spinner("Analysing…"):
                try:
                    result, elapsed = run_inference(
                        client, selected_model, pil_img, temperature, max_tokens
                    )
                    st.session_state.last_result = {"result": result, "elapsed": elapsed}
                    st.session_state.history.append({"result": result, "elapsed": elapsed})
                except Exception as e:
                    st.session_state.last_result = {
                        "result": {"_error": str(e)}, "elapsed": 0.0
                    }

    if st.session_state.last_result:
        render_result(
            st.session_state.last_result["result"],
            st.session_state.last_result["elapsed"],
        )
    else:
        st.markdown("""
        <div class="idle-state">
            <div style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;
                        letter-spacing:0.1em;color:#3A6A5A;text-align:center;">
                Results will appear here<br>after analysis
            </div>
        </div>""", unsafe_allow_html=True)
