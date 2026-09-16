import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image
import streamlit as st
from src.pipeline.pipeline import run_pipeline


st.set_page_config(page_title="Paint Estimator", page_icon="PE", layout="wide")


st.markdown(
    """
<style>
:root {
    --navy: #071A2B;
    --navy-2: #0B2D47;
    --blue: #1877F2;
    --aqua: #16C7B7;
    --paint: #FFB84D;
    --bg: #F5F8FC;
    --card: #FFFFFF;
    --text: #102A43;
    --muted: #627D98;
    --border: #D9E2EC;
    --soft-blue: #EAF3FF;
    --soft-aqua: #EAF9F7;
}

/* Keep every pipeline image centered inside its parent content boundary. */
div[data-testid="stImage"] {
    display: flex;
    justify-content: center;
    width: 100%;
}

div[data-testid="stImage"] img {
    display: block;
    margin-left: auto;
    margin-right: auto;
}

.upload-showcase {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 18px 20px 22px;
    box-shadow: 0 10px 30px rgba(7,26,43,.06);
}


.stApp {
    background: var(--bg);
    color: var(--text);
}

.block-container {
    max-width: 1320px;
    padding-top: 2.2rem;
    padding-bottom: 4rem;
}

h1, h2, h3, h4, p, label, .stMarkdown {
    color: var(--text);
}

.hero {
    background:
        radial-gradient(circle at 90% 20%, rgba(22,199,183,.22), transparent 28%),
        linear-gradient(135deg, var(--navy), var(--navy-2));
    border-radius: 28px;
    padding: 48px 52px;
    margin-bottom: 30px;
    box-shadow: 0 22px 50px rgba(7,26,43,.18);
}

.hero-kicker {
    color: #55E4D6;
    font-size: .78rem;
    font-weight: 800;
    letter-spacing: .16em;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.hero h1 {
    color: white !important;
    font-size: clamp(2.6rem, 5vw, 4.4rem);
    line-height: 1;
    letter-spacing: -.045em;
    margin: 0 0 16px;
    font-weight: 850;
}

.hero-subtitle {
    color: #DDECF8 !important;
    font-size: 1.08rem;
    margin: 0;
}

.hero-pill {
    display: inline-block;
    margin-top: 22px;
    padding: 9px 14px;
    border-radius: 999px;
    background: rgba(255,255,255,.08);
    border: 1px solid rgba(255,255,255,.15);
    color: #F3FAFF;
    font-size: .82rem;
    font-weight: 650;
}

.section-title {
    font-size: 1.55rem;
    font-weight: 850;
    letter-spacing: -.02em;
    margin: 32px 0 5px;
}

.section-subtitle {
    color: var(--muted) !important;
    margin: 0 0 18px;
    font-size: .96rem;
}

.upload-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 22px;
    box-shadow: 0 10px 30px rgba(7,26,43,.06);
}

.pipeline-shell {
    background: linear-gradient(180deg, #FFFFFF, #F8FBFF);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 24px;
    box-shadow: 0 12px 34px rgba(7,26,43,.07);
}

.pipeline-intro.standalone {
    background: white;
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 18px 20px;
    box-shadow: 0 7px 22px rgba(7,26,43,.045);
    margin-bottom: 22px;
}

.input-label {
    color: var(--text);
    font-size: .92rem;
    font-weight: 800;
}

.input-help {
    color: var(--muted);
    font-size: .75rem;
    line-height: 1.35;
    margin-top: 3px;
}

.pipeline-intro {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 22px;
}

.pipeline-intro-number {
    width: 38px;
    height: 38px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--navy);
    color: white;
    font-weight: 850;
}

.pipeline-intro-title {
    font-size: 1.12rem;
    font-weight: 850;
    color: var(--text);
}

.pipeline-intro-copy {
    color: var(--muted);
    font-size: .84rem;
}

.pipeline-stage {
    position: relative;
    display: grid;
    grid-template-columns: 52px 1fr;
    gap: 16px;
    padding-bottom: 24px;
}

.pipeline-stage:last-child {
    padding-bottom: 2px;
}

.pipeline-stage:not(:last-child)::before {
    content: "";
    position: absolute;
    left: 25px;
    top: 52px;
    bottom: 4px;
    width: 2px;
    background: linear-gradient(180deg, #BFD4E8, #E6EEF6);
}

.stage-number {
    position: relative;
    z-index: 1;
    width: 52px;
    height: 52px;
    border-radius: 16px;
    background: var(--soft-blue);
    border: 1px solid #CFE1F5;
    color: var(--blue);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: .9rem;
    font-weight: 850;
}

.stage-number.complete {
    background: var(--soft-aqua);
    border-color: #BFEAE4;
    color: #087A70;
}

.stage-content {
    background: white;
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 18px 20px;
    box-shadow: 0 7px 22px rgba(7,26,43,.045);
}

.stage-title {
    font-size: 1.02rem;
    font-weight: 850;
    color: var(--text);
    margin-bottom: 4px;
}

.stage-model {
    color: var(--blue);
    font-size: .77rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .07em;
    margin-bottom: 7px;
}

.stage-description {
    color: var(--muted);
    font-size: .88rem;
    line-height: 1.5;
    margin-bottom: 10px;
}

.stage-output {
    background: #F6F9FC;
    border-radius: 11px;
    padding: 9px 11px;
    color: #486581;
    font-size: .82rem;
}

.settings-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 22px 22px 18px;
    margin-bottom: 18px;
    box-shadow: 0 10px 28px rgba(7,26,43,.055);
}

.settings-title {
    font-size: 1.12rem;
    font-weight: 850;
    margin-bottom: 4px;
}

.settings-caption {
    color: var(--muted);
    font-size: .83rem;
    line-height: 1.45;
    margin-bottom: 15px;
}

.value-guide {
    background: #F7FAFD;
    border: 1px solid #E4ECF3;
    border-radius: 12px;
    padding: 11px 13px;
    margin: 12px 0 6px;
    color: #526D82;
    font-size: .78rem;
    line-height: 1.5;
}

.value-guide strong {
    color: var(--text);
}

.stTextInput label,
.stNumberInput label,
.stCheckbox label {
    color: var(--text) !important;
    font-weight: 650 !important;
}

.stTextInput input,
.stNumberInput input {
    color: var(--text) !important;
    background: white !important;
}

div[data-baseweb="input"] {
    background: white !important;
}

div[data-baseweb="select"] > div {
    background: white !important;
}

section[data-testid="stFileUploaderDropzone"] {
    background: #FFFFFF !important;
    border: 1px dashed #B7C9DA !important;
    border-radius: 16px !important;
    padding: 14px !important;
}

section[data-testid="stFileUploaderDropzone"] button {
    background: #071A2B !important;
    border: 1px solid #071A2B !important;
    color: #FFFFFF !important;
    border-radius: 10px !important;
}

section[data-testid="stFileUploaderDropzone"] button *,
section[data-testid="stFileUploaderDropzone"] small,
section[data-testid="stFileUploaderDropzone"] span {
    color: #16324A !important;
}

section[data-testid="stFileUploaderDropzone"] button *,
section[data-testid="stFileUploaderDropzone"] button {
    color: #FFFFFF !important;
}

.result-table {
    width: 100%;
    border-collapse: collapse;
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
}

.result-table th {
    background: #F1F6FB;
    color: #486581;
    text-align: left;
    font-size: .76rem;
    text-transform: uppercase;
    letter-spacing: .06em;
    padding: 13px 14px;
    border-bottom: 1px solid var(--border);
}

.result-table td {
    padding: 14px;
    border-bottom: 1px solid #E5ECF3;
    color: var(--text);
    font-size: .86rem;
}

.result-table tr:last-child td {
    border-bottom: 0;
    font-weight: 850;
    background: #F7FAFD;
}

.result-table .money {
    text-align: right;
    font-weight: 800;
}

.tech-box {
    background: #F1F6FB;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 8px 16px;
    margin: 8px 0 20px;
}

.estimate-note {
    background: var(--soft-aqua);
    border: 1px solid #BFEAE4;
    color: #176B64;
    border-radius: 14px;
    padding: 13px 15px;
    margin: 12px 0 18px;
    font-size: .86rem;
}

div.stButton > button[kind="primary"] {
    width: 100%;
    min-height: 58px;
    border-radius: 16px;
    border: 0;
    background: linear-gradient(135deg, var(--blue), #16B8CF);
    color: white;
    font-size: 1.05rem;
    font-weight: 850;
    box-shadow: 0 12px 26px rgba(24,119,242,.22);
}

.result-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 22px;
    min-height: 128px;
    box-shadow: 0 10px 28px rgba(7,26,43,.06);
}

.result-card.total {
    background: linear-gradient(135deg, var(--navy), #0B4261);
    border: none;
}

.result-label {
    color: var(--muted);
    font-size: .76rem;
    font-weight: 800;
    letter-spacing: .09em;
    text-transform: uppercase;
}

.total .result-label {
    color: #BFE7F7;
}

.result-value {
    color: var(--text);
    font-size: 2.05rem;
    font-weight: 900;
    margin-top: 8px;
}

.total .result-value {
    color: white;
}

.cost-box {
    background: white;
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 20px 22px;
    box-shadow: 0 10px 28px rgba(7,26,43,.055);
}

.cost-row {
    display: flex;
    justify-content: space-between;
    gap: 18px;
    padding: 14px 0;
    border-bottom: 1px solid #E5ECF3;
}

.cost-row:last-child {
    border-bottom: 0;
}

.cost-name {
    color: var(--text);
    font-weight: 800;
}

.cost-detail {
    color: var(--muted);
    font-size: .82rem;
    margin-top: 3px;
}

.cost-value {
    color: var(--text);
    font-weight: 850;
    white-space: nowrap;
}

.footer {
    text-align: center;
    color: var(--muted);
    font-size: .78rem;
    border-top: 1px solid var(--border);
    padding-top: 22px;
    margin-top: 42px;
}

[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 15px !important;
    background: white !important;
}
</style>
""",
    unsafe_allow_html=True,
)


st.markdown(
    """
<div class="hero">
    <div class="hero-kicker">AI • COMPUTER VISION • 3D MEASUREMENT</div>
    <h1>Paint Estimator</h1>
    <p class="hero-subtitle">AI-Powered Wall Measurement &amp; Painting Cost Estimation</p>
    <div class="hero-pill">Upload a room image → understand the wall → measure the surface → estimate the job</div>
</div>
""",
    unsafe_allow_html=True,
)


st.markdown(
    '<div class="section-title">1. Upload your room image</div>'
    '<div class="section-subtitle">Start with a clear indoor RGB image where the target wall is visible.</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    uploaded_file = st.file_uploader(
        "Upload an indoor RGB image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.markdown(
            """
            <div class="upload-showcase">
                <strong>Ready for your room image</strong><br>
                <span style="color:#627D98;">Supported formats: JPG, JPEG, PNG. The image is processed locally by the project pipeline.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.image(uploaded_file, caption="Uploaded room image", width=560)

    # This is the user-facing scientific showcase. It is deliberately always visible
    # after upload rather than hidden inside expanders.
    st.markdown(
        '<div class="section-title">2. How your image is processed</div>'
        '<div class="section-subtitle">The complete production pipeline is shown below in the order used by the system.</div>',
        unsafe_allow_html=True,
    )

    pipeline_stages = [
        (
            "01",
            "Wall Segmentation",
            "U-Net + ResNet34",
            "The wall model identifies the wall pixels in the RGB image and creates the wall mask.",
        ),
        (
            "02",
            "Door & Window Segmentation",
            "U-Net + ResNet34",
            "A second segmentation model identifies doors and windows so non-paintable openings can be excluded.",
        ),
        (
            "03",
            "Paintable Surface Extraction",
            "Mask combination",
            "The system removes detected doors and windows from the wall mask to obtain the final paintable-wall mask.",
        ),
        (
            "04",
            "Relative Depth Estimation",
            "MiDaS",
            "MiDaS estimates relative scene depth from the single RGB image.",
        ),
        (
            "05",
            "Metric Depth Calibration",
            "Single-anchor inverse-scale calibration",
            "The relative depth is converted into metric depth using the configured calibration factor.",
        ),
        (
            "06",
            "3D Surface Reconstruction & Area Measurement",
            "Pinhole backprojection + triangular mesh",
            "Valid paintable pixels are back-projected into 3D. Neighboring 3D points form triangles whose areas are summed to estimate the paintable surface.",
        ),
        (
            "07",
            "Material & Cost Estimation",
            "Paint + Primer + Putty + Labour",
            "The measured paintable area is passed to the material and labour models using the values you enter below.",
        ),
    ]

    completed = st.session_state.get("last_results") is not None

    st.markdown(
        '<div class="pipeline-intro standalone">'
        '<div class="pipeline-intro-number">AI</div>'
        '<div><div class="pipeline-intro-title">End-to-end measurement pipeline</div>'
        '<div class="pipeline-intro-copy">The system moves through these seven stages in order. Each stage below explains the model, input, processing and output.</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    for num, title, model, description in pipeline_stages:
        complete_class = " complete" if completed else ""
        output = (
            "Completed • output is available in the results section below."
            if completed
            else "Ready • this stage runs when you select “Estimate My Room”."
        )
        st.markdown(
            f"""
            <div class="pipeline-stage">
                <div class="stage-number{complete_class}">{num}</div>
                <div class="stage-content">
                    <div class="stage-model">{model}</div>
                    <div class="stage-title">{title}</div>
                    <div class="stage-description">{description}</div>
                    <div class="stage-output">{output}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">3. Customize your estimate</div>'
        '<div class="section-subtitle">These values control the material quantities and costs. Each field explains exactly what you are changing.</div>',
        unsafe_allow_html=True,
    )

    def inline_number(label, help_text, *, min_value, value, step, key, max_value=None, disabled=False):
        left, right = st.columns([1.35, 1], gap="medium", vertical_alignment="center")
        with left:
            st.markdown(
                f'<div class="input-label">{label}</div><div class="input-help">{help_text}</div>',
                unsafe_allow_html=True,
            )
        with right:
            kwargs = dict(
                min_value=min_value,
                value=value,
                step=step,
                key=key,
                disabled=disabled,
                label_visibility="collapsed",
            )
            if max_value is not None:
                kwargs["max_value"] = max_value
            return st.number_input(label, **kwargs)

    def inline_select(label, help_text, options, *, index, key):
        left, right = st.columns([1.35, 1], gap="medium", vertical_alignment="center")
        with left:
            st.markdown(
                f'<div class="input-label">{label}</div><div class="input-help">{help_text}</div>',
                unsafe_allow_html=True,
            )
        with right:
            return st.selectbox(
                label,
                options,
                index=index,
                key=key,
                label_visibility="collapsed",
            )

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown(
            '<div class="settings-card">'
            '<div class="settings-title">Paint</div>'
            '<div class="settings-caption">Choose the finish type and the assumptions used to calculate paint quantity and cost.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        paint_type = inline_select(
            "Paint type",
            "Choose the type of finish you plan to use.",
            ["Interior Emulsion", "Exterior Emulsion", "Distemper", "Enamel", "Texture Paint"],
            index=0,
            key="paint_type",
        )
        paint_coats = inline_number(
            "Number of coats",
            "Number of paint layers applied to the wall.",
            min_value=1, max_value=5, value=2, step=1, key="paint_coats",
        )
        paint_coverage = inline_number(
            "Coverage (m² per litre)",
            "Wall area that 1 litre of paint can cover.",
            min_value=0.01, value=12.0, step=0.5, key="paint_coverage",
        )
        paint_price = inline_number(
            "Paint price (₹ per litre)",
            "Amount paid for 1 litre of the selected paint.",
            min_value=0.0, value=30.0, step=1.0, key="paint_price",
        )
        paint_wastage = inline_number(
            "Wastage allowance (%)",
            "Extra percentage added to the calculated paint quantity.",
            min_value=0.0, max_value=100.0, value=5.0, step=1.0, key="paint_wastage",
        )

    with c2:
        st.markdown(
            '<div class="settings-card">'
            '<div class="settings-title">Primer</div>'
            '<div class="settings-caption">Optional preparation layer applied before the finishing paint.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        primer_enabled = st.checkbox("Use primer", value=True, key="primer_enabled")
        primer_coats = inline_number(
            "Number of primer coats",
            "Number of primer layers applied to the wall.",
            min_value=1, max_value=5, value=1, step=1, key="primer_coats", disabled=not primer_enabled,
        )
        primer_coverage = inline_number(
            "Coverage (m² per litre)",
            "Wall area that 1 litre of primer can cover.",
            min_value=0.01, value=15.0, step=0.5, key="primer_coverage", disabled=not primer_enabled,
        )
        primer_price = inline_number(
            "Primer price (₹ per litre)",
            "Amount paid for 1 litre of primer.",
            min_value=0.0, value=18.0, step=1.0, key="primer_price", disabled=not primer_enabled,
        )
        primer_wastage = inline_number(
            "Wastage allowance (%)",
            "Extra percentage added to the calculated primer quantity.",
            min_value=0.0, max_value=100.0, value=5.0, step=1.0, key="primer_wastage", disabled=not primer_enabled,
        )

    c3, c4 = st.columns(2, gap="large")

    with c3:
        st.markdown(
            '<div class="settings-card">'
            '<div class="settings-title">Putty</div>'
            '<div class="settings-caption">Optional surface-preparation layer applied before painting.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        putty_enabled = st.checkbox("Use putty", value=True, key="putty_enabled")
        putty_coats = inline_number(
            "Number of putty coats",
            "Number of putty layers applied to the wall.",
            min_value=1, max_value=5, value=1, step=1, key="putty_coats", disabled=not putty_enabled,
        )
        putty_coverage = inline_number(
            "Coverage (m² per kg)",
            "Wall area that 1 kg of putty can cover.",
            min_value=0.01, value=4.0, step=0.5, key="putty_coverage", disabled=not putty_enabled,
        )
        putty_price = inline_number(
            "Putty price (₹ per kg)",
            "Amount paid for 1 kg of putty.",
            min_value=0.0, value=25.0, step=1.0, key="putty_price", disabled=not putty_enabled,
        )
        putty_wastage = inline_number(
            "Wastage allowance (%)",
            "Extra percentage added to the calculated putty quantity.",
            min_value=0.0, max_value=100.0, value=5.0, step=1.0, key="putty_wastage", disabled=not putty_enabled,
        )

    with c4:
        st.markdown(
            '<div class="settings-card">'
            '<div class="settings-title">Labour</div>'
            '<div class="settings-caption">Application labour calculated from the measured paintable surface.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        labour_rate = inline_number(
            "Labour rate (₹ per m²)",
            "Amount charged for applying the work to 1 m² of paintable surface.",
            min_value=0.0, value=20.0, step=1.0, key="labour_rate",
        )

        st.markdown(
            '<div class="value-guide"><strong>What these inputs control:</strong> coats control the number of layers; '
            'coverage controls material required per area; price controls material cost; wastage adds extra quantity; '
            'labour rate controls application cost per m².</div>',
            unsafe_allow_html=True,
        )

    paint_overrides = {
        "coats": paint_coats,
        "coverage_m2_per_litre": paint_coverage,
        "price_per_litre": paint_price,
        "wastage_percent": paint_wastage,
    }
    labour_overrides = {"rate_per_m2": labour_rate}
    primer_overrides = {
        "coats": primer_coats,
        "coverage_m2_per_litre": primer_coverage,
        "price_per_litre": primer_price if primer_enabled else 0.0,
        "wastage_percent": primer_wastage if primer_enabled else 0.0,
    }
    putty_overrides = {
        "coats": putty_coats,
        "coverage_m2_per_kg": putty_coverage,
        "price_per_kg": putty_price if putty_enabled else 0.0,
        "wastage_percent": putty_wastage if putty_enabled else 0.0,
    }

    with st.expander("Advanced / Technical Settings", expanded=False):
        st.caption("Production defaults used by the measurement pipeline. Change these only when you have a technical reason.")
        t1, t2, t3 = st.columns(3)
        with t1:
            calibration_factor = st.number_input("Calibration factor", min_value=0.000001, value=1059.0, format="%.6f")
            camera_fx = st.number_input("Camera fx", min_value=0.000001, value=529.5, format="%.6f")
        with t2:
            camera_fy = st.number_input("Camera fy", min_value=0.000001, value=529.5, format="%.6f")
            camera_cx = st.number_input("Camera cx", min_value=0.0, value=365.0, format="%.6f")
        with t3:
            camera_cy = st.number_input("Camera cy", min_value=0.0, value=265.0, format="%.6f")
            max_depth_jump = st.number_input("Maximum depth jump (m)", min_value=0.0, value=0.025, format="%.6f")

    st.markdown(
        '<div class="estimate-note"><strong>Calculation note:</strong> Your material values are runtime inputs. '
        'The AI measurement methodology and 3D wall-area calculation remain unchanged.</div>',
        unsafe_allow_html=True,
    )

    run_clicked = st.button("Estimate My Room", type="primary")

    if run_clicked:
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_image_path = temp_file.name

        try:
            with st.spinner("Running the complete AI measurement pipeline…"):
                results = run_pipeline(
                    temp_image_path,
                    calibration_factor,
                    camera_fx,
                    camera_fy,
                    camera_cx,
                    camera_cy,
                    max_depth_jump=max_depth_jump,
                    paint_overrides=paint_overrides,
                    primer_overrides=primer_overrides,
                    putty_overrides=putty_overrides,
                    labour_overrides=labour_overrides,
                )

            st.session_state.last_results = results

            # Display quantities respect the optional enable/disable controls.
            # The backend cost calculation remains unchanged; disabled materials are
            # presented as not included in the user-facing result.
            primer_quantity_display = results["primer_quantity_l"] if primer_enabled else 0.0
            putty_quantity_display = results["putty_quantity_kg"] if putty_enabled else 0.0

            st.success("AI measurement completed — the seven production stages are shown below in sequence.")

            st.markdown(
                '<div class="section-title">5. Step-by-step AI processing</div>'
                '<div class="section-subtitle">Now that the estimate is complete, the actual output of every production stage is shown below in sequence.</div>',
                unsafe_allow_html=True,
            )

            original_image = Image.open(uploaded_file).convert("RGB")
            original_array = np.array(original_image)

            # Stage 01
            st.markdown(
                '<div class="pipeline-stage">'
                '<div class="stage-number complete">01</div>'
                '<div class="stage-content"><div class="stage-model">U-Net + ResNet34</div>'
                '<div class="stage-title">Wall Segmentation</div>'
                '<div class="stage-description">Detected wall region used as the starting surface for measurement.</div></div></div>',
                unsafe_allow_html=True,
            )
            wall_overlay = original_array.copy()
            wall_mask = results["wall_mask"]
            wall_overlay[wall_mask] = (
                0.55 * wall_overlay[wall_mask] + 0.45 * np.array([0, 255, 0])
            ).astype(np.uint8)
            img_left, img_center, img_right = st.columns([1, 2, 1])
            with img_center:
                st.image(wall_overlay, caption="Wall segmentation overlay", width=620)
            st.caption(f"Detected wall pixels: {int(wall_mask.sum()):,}")

            # Stage 02
            st.markdown(
                '<div class="pipeline-stage">'
                '<div class="stage-number complete">02</div>'
                '<div class="stage-content"><div class="stage-model">U-Net + ResNet34</div>'
                '<div class="stage-title">Door & Window Segmentation</div>'
                '<div class="stage-description">Detected openings that should not be counted as paintable wall.</div></div></div>',
                unsafe_allow_html=True,
            )
            opening_overlay = original_array.copy()
            door_mask = results["door_mask"]
            window_mask = results["window_mask"]
            opening_overlay[door_mask] = (
                0.45 * opening_overlay[door_mask] + 0.55 * np.array([255, 0, 0])
            ).astype(np.uint8)
            opening_overlay[window_mask] = (
                0.45 * opening_overlay[window_mask] + 0.55 * np.array([0, 80, 255])
            ).astype(np.uint8)
            img_left, img_center, img_right = st.columns([1, 2, 1])
            with img_center:
                st.image(opening_overlay, caption="Door and window segmentation overlay", width=620)
            st.caption(
                f"Detected door pixels: {int(door_mask.sum()):,} • Detected window pixels: {int(window_mask.sum()):,}"
            )

            # Stage 03
            st.markdown(
                '<div class="pipeline-stage">'
                '<div class="stage-number complete">03</div>'
                '<div class="stage-content"><div class="stage-model">Mask combination</div>'
                '<div class="stage-title">Paintable Surface Extraction</div>'
                '<div class="stage-description">Wall pixels are combined with the opening masks to isolate the surface that can actually be painted.</div></div></div>',
                unsafe_allow_html=True,
            )
            paintable_overlay = original_array.copy()
            paintable_mask = results["paintable_mask"]
            paintable_overlay[paintable_mask] = (
                0.45 * paintable_overlay[paintable_mask] + 0.55 * np.array([0, 255, 0])
            ).astype(np.uint8)
            img_left, img_center, img_right = st.columns([1, 2, 1])
            with img_center:
                st.image(paintable_overlay, caption="Final paintable wall mask", width=620)
            st.caption(f"Paintable wall pixels: {int(paintable_mask.sum()):,}")

            # Stage 04
            st.markdown(
                '<div class="pipeline-stage">'
                '<div class="stage-number complete">04</div>'
                '<div class="stage-content"><div class="stage-model">MiDaS</div>'
                '<div class="stage-title">Relative Depth Estimation</div>'
                '<div class="stage-description">The single RGB image is converted into a relative depth representation.</div></div></div>',
                unsafe_allow_html=True,
            )
            relative_depth = results["relative_depth"]
            depth_visual = np.zeros_like(relative_depth, dtype=np.uint8)
            valid = np.isfinite(relative_depth)
            if np.any(valid):
                vals = relative_depth[valid]
                low = np.percentile(vals, 1)
                high = np.percentile(vals, 99)
                if high > low:
                    depth_visual[valid] = (
                        np.clip((relative_depth[valid] - low) / (high - low), 0, 1) * 255
                    ).astype(np.uint8)
            img_left, img_center, img_right = st.columns([1, 2, 1])
            with img_center:
                st.image(depth_visual, caption="MiDaS relative depth map", width=620)
            st.caption("The map represents relative depth before metric calibration.")

            # Stage 05
            st.markdown(
                '<div class="pipeline-stage">'
                '<div class="stage-number complete">05</div>'
                '<div class="stage-content"><div class="stage-model">Single-anchor inverse-scale calibration</div>'
                '<div class="stage-title">Metric Depth Calibration</div>'
                '<div class="stage-description">Relative depth is converted into metric depth using the production calibration factor.</div></div></div>',
                unsafe_allow_html=True,
            )
            metric_depth = results["metric_depth"]
            metric_visual = np.zeros_like(metric_depth, dtype=np.uint8)
            metric_valid = np.isfinite(metric_depth) & (metric_depth > 0)
            if np.any(metric_valid):
                metric_values = metric_depth[metric_valid]
                low = np.percentile(metric_values, 1)
                high = np.percentile(metric_values, 99)
                if high > low:
                    metric_visual[metric_valid] = (
                        np.clip((metric_depth[metric_valid] - low) / (high - low), 0, 1) * 255
                    ).astype(np.uint8)
                img_left, img_center, img_right = st.columns([1, 2, 1])
                with img_center:
                    st.image(metric_visual, caption="Calibrated metric depth map", width=620)
                st.caption(
                    f"Valid depth range: {metric_values.min():.2f}–{metric_values.max():.2f} m • "
                    f"Median depth: {np.median(metric_values):.2f} m • "
                    f"Calibration factor: {results['calibration_factor']:.6f}"
                )
            else:
                st.warning("No valid positive metric-depth values are available.")

            # Stage 06
            st.markdown(
                '<div class="pipeline-stage">'
                '<div class="stage-number complete">06</div>'
                '<div class="stage-content"><div class="stage-model">Pinhole backprojection + triangular mesh</div>'
                '<div class="stage-title">3D Surface Reconstruction & Area Measurement</div>'
                '<div class="stage-description">Metric depth and camera intrinsics are used to reconstruct the paintable surface in 3D. Valid triangular surface elements are summed to obtain area.</div></div></div>',
                unsafe_allow_html=True,
            )
            paintable_metric_depth = np.where(results["paintable_mask"], results["metric_depth"], np.nan)
            visual = np.zeros_like(paintable_metric_depth, dtype=np.uint8)
            valid = np.isfinite(paintable_metric_depth) & (paintable_metric_depth > 0)
            if np.any(valid):
                vals = paintable_metric_depth[valid]
                low = np.percentile(vals, 1)
                high = np.percentile(vals, 99)
                if high > low:
                    visual[valid] = (
                        np.clip((paintable_metric_depth[valid] - low) / (high - low), 0, 1) * 255
                    ).astype(np.uint8)
                depth_left, depth_center, depth_right = st.columns([1, 2, 1])
                with depth_center:
                    st.image(visual, caption="Metric depth restricted to the paintable surface", width=620)
            st.info(
                f"Paintable surface area: {results['paintable_area_m2']:.2f} m² • "
                f"Depth-jump filter: {max_depth_jump:.3f} m • "
                f"Camera: fx={camera_fx:.2f}, fy={camera_fy:.2f}, cx={camera_cx:.2f}, cy={camera_cy:.2f}"
            )

            # Stage 07
            st.markdown(
                '<div class="pipeline-stage">'
                '<div class="stage-number complete">07</div>'
                '<div class="stage-content"><div class="stage-model">Runtime estimation inputs</div>'
                '<div class="stage-title">Material & Cost Estimation</div>'
                '<div class="stage-description">The measured paintable area is converted into material quantities and labour cost using the values you entered above.</div></div></div>',
                unsafe_allow_html=True,
            )
            st.info(
                f"Stage 07 complete • Paint: {results['paint_quantity_l']:.2f} L • "
                f"Primer: {primer_quantity_display:.2f} L • "
                f"Putty: {putty_quantity_display:.2f} kg • "
                f"Labour: ₹{results['labour_cost']:.2f}"
            )

            st.markdown(
                '<div class="section-title">8. Final result</div>'
                '<div class="section-subtitle">Your final estimate appears only after all seven pipeline stages have completed successfully.</div>',
                unsafe_allow_html=True,
            )

            r1, r2, r3 = st.columns(3, gap="large")
            with r1:
                st.markdown(
                    f'<div class="result-card"><div class="result-label">Paintable Area</div>'
                    f'<div class="result-value">{results["paintable_area_m2"]:.2f} m²</div></div>',
                    unsafe_allow_html=True,
                )
            with r2:
                st.markdown(
                    f'<div class="result-card"><div class="result-label">Paint Quantity</div>'
                    f'<div class="result-value">{results["paint_quantity_l"]:.2f} L</div></div>',
                    unsafe_allow_html=True,
                )
            with r3:
                st.markdown(
                    f'<div class="result-card total"><div class="result-label">Total Cost</div>'
                    f'<div class="result-value">₹{results["total_cost"]:.2f}</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                '<div class="section-title">Result breakdown</div>'
                '<div class="section-subtitle">The table shows exactly which user-entered assumptions produced each cost.</div>',
                unsafe_allow_html=True,
            )

            primer_rate_display = primer_price if primer_enabled else 0.0
            putty_rate_display = putty_price if putty_enabled else 0.0
            primer_wastage_display = primer_wastage if primer_enabled else 0.0
            putty_wastage_display = putty_wastage if putty_enabled else 0.0

            result_rows = [
                ("Paint", f"{results['paint_quantity_l']:.2f} L", paint_type, f"{paint_price:.2f} / L", f"{paint_coats}", f"{paint_wastage:.1f}%", results["paint_cost"]),
                ("Primer", f"{primer_quantity_display:.2f} L" if primer_enabled else "Not included", "Primer", f"{primer_rate_display:.2f} / L" if primer_enabled else "—", f"{primer_coats}" if primer_enabled else "—", f"{primer_wastage_display:.1f}%" if primer_enabled else "—", results["primer_cost"]),
                ("Putty", f"{putty_quantity_display:.2f} kg" if putty_enabled else "Not included", "Putty", f"{putty_rate_display:.2f} / kg" if putty_enabled else "—", f"{putty_coats}" if putty_enabled else "—", f"{putty_wastage_display:.1f}%" if putty_enabled else "—", results["putty_cost"]),
                ("Labour", "—", "Application labour", f"{labour_rate:.2f} / m²", "—", "—", results["labour_cost"]),
            ]

            table_html = (
                '<table class="result-table"><thead><tr>'
                '<th>Item</th><th>Quantity</th><th>Type / basis</th><th>Rate (₹)</th><th>Coats</th><th>Wastage</th><th style="text-align:right;">Cost (₹)</th>'
                '</tr></thead><tbody>'
            )
            for item, quantity, basis, rate, coats, wastage, cost in result_rows:
                table_html += (
                    f'<tr><td>{item}</td><td>{quantity}</td><td>{basis}</td><td>{rate}</td>'
                    f'<td>{coats}</td><td>{wastage}</td><td class="money">{cost:.2f}</td></tr>'
                )
            table_html += (
                f'<tr><td colspan="6">Total Estimate</td><td class="money">{results["total_cost"]:.2f}</td></tr>'
                '</tbody></table>'
            )
            st.markdown(table_html, unsafe_allow_html=True)

            with st.expander("Technical Details"):
                st.write(f"Calibration factor: {results['calibration_factor']:.6f}")
                st.write(
                    f"Camera intrinsics: fx={camera_fx:.2f}, fy={camera_fy:.2f}, "
                    f"cx={camera_cx:.2f}, cy={camera_cy:.2f}"
                )
                st.write(f"Maximum depth jump: {max_depth_jump:.3f} m")

        except Exception as exc:
            st.error(
                "The estimation could not be completed. Please check the uploaded image and technical settings."
            )
            with st.expander("Technical error details"):
                st.exception(exc)
        finally:
            Path(temp_image_path).unlink(missing_ok=True)


st.markdown(
    '<div class="footer">Paint Estimator • AI-based wall measurement and painting cost estimation</div>',
    unsafe_allow_html=True,
)
