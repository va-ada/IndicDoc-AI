"""
IndicDoc AI - Streamlit Application
Deep Learning-Based Multilingual Indian Document Layout Understanding

Premium, Indic-inspired UI with full detection, evaluation, and model comparison.
"""

import streamlit as st
import numpy as np
import cv2
import json
import time
import os
import sys
from pathlib import Path
from PIL import Image
import io
import base64
import hashlib

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="IndicDoc AI",
    page_icon="assets/logo.jpg",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════
#  INDIC-INSPIRED DESIGN SYSTEM (CSS)
# ══════════════════════════════════════════════════════════════

INDIC_CSS = """
<style>
/* ─────── Imports & Base ─────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@500;600;700;800&display=swap');

:root {
    /* Indic Heritage Premium Theme */
    --bg-main: #FBF8F1; /* Warm Ivory */
    --bg-card: #FFFFFF;
    --text-main: #1E2A5A; /* Deep Indigo */
    --text-muted: #5A6B8A;
    --primary: #1E2A5A;
    --accent: #C89B3C; /* Antique Gold */
    --accent-hover: #DFB768;
    --border: rgba(30, 42, 90, 0.12);
    --terracotta: #8B4A3A;
    --radius-sm: 6px;
    --radius-md: 12px;
    --radius-lg: 18px;
    --shadow-sm: 0 4px 15px rgba(30, 42, 90, 0.04);
    --shadow-md: 0 8px 25px rgba(30, 42, 90, 0.08);
    --transition: 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Global overides */
.stApp {
    background-color: var(--bg-main) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-main);
}
.block-container {
    max-width: 1440px !important;
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
}
/* Keep Streamlit's header (it holds the button that reopens a collapsed sidebar), but make it
   transparent and hide only the Deploy button and the main menu */
header[data-testid="stHeader"] {
    background: transparent !important;
}
[data-testid="stAppDeployButton"],
[data-testid="stMainMenu"],
#MainMenu {
    display: none !important;
}
[data-testid="stExpandSidebarButton"] {
    color: var(--primary) !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}
footer {
    display: none !important;
}

/* ─────── Header & Hero ─────── */
.hero-container {
    text-align: center;
    padding: 3rem 1rem 2rem 1rem;
    margin-bottom: 2rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.4rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 0.3rem;
    color: var(--primary);
}
.hero-title span {
    color: var(--accent);
}
.hero-subtitle {
    font-size: 1.1rem;
    color: var(--text-muted);
    font-weight: 400;
    letter-spacing: 0.02em;
    max-width: 600px;
    margin: 0 auto;
}
.hero-divider {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1rem;
    margin: 1.5rem 0 2rem 0;
}
.hero-divider::before, .hero-divider::after {
    content: "";
    height: 1px;
    width: 60px;
    background-color: var(--accent);
    opacity: 0.5;
}
.hero-divider-icon {
    color: var(--accent);
    font-size: 0.8rem;
    opacity: 0.9;
}

/* ─────── Metric Cards ─────── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.5rem;
    text-align: center;
    box-shadow: var(--shadow-sm);
    transition: transform var(--transition), box-shadow var(--transition);
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    position: relative;
    overflow: hidden;
    border-top: 3px solid var(--accent);
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-md);
}
.metric-value {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--primary);
    line-height: 1.1;
    margin-bottom: 0.25rem;
}
.metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ─────── Section Titles ─────── */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--primary);
    margin: 2rem 0 1.25rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
    display: inline-block;
}

/* ─────── Custom HTML Tables ─────── */
.custom-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    margin: 1rem 0;
    font-family: 'Inter', sans-serif;
    background: var(--bg-card);
    border-radius: var(--radius-sm);
    overflow: hidden;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border);
}
.custom-table th {
    background: var(--primary);
    color: #FFFFFF;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    padding: 14px 16px;
    text-align: left;
}
.custom-table td {
    padding: 14px 16px;
    border-bottom: 1px solid var(--border);
    color: var(--text-main);
    font-size: 0.9rem;
}
.custom-table tbody tr:last-of-type td {
    border-bottom: none;
}
.custom-table tbody tr:hover {
    background-color: rgba(200, 155, 60, 0.04);
}

/* ─────── Empty States ─────── */
.empty-state {
    text-align: center;
    padding: 5rem 2rem;
    background: var(--bg-card);
    border: 1.5px dashed var(--accent);
    border-radius: var(--radius-md);
    color: var(--text-muted);
    transition: all var(--transition);
}
.empty-state:hover {
    background: rgba(200, 155, 60, 0.03);
}
.empty-icon {
    font-size: 3.5rem;
    margin-bottom: 1rem;
    opacity: 0.8;
}
.empty-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--primary);
    margin-bottom: 0.5rem;
}
.empty-subtitle {
    font-size: 0.95rem;
    line-height: 1.5;
    max-width: 450px;
    margin: 0 auto;
}

/* ─────── Panels & Cards ─────── */
.result-panel, .about-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.75rem;
    box-shadow: var(--shadow-sm);
    margin-bottom: 1rem;
    transition: box-shadow var(--transition);
}
.result-panel:hover, .about-card:hover {
    box-shadow: var(--shadow-md);
}
.about-card h4 {
    font-family: 'Playfair Display', serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--primary);
    margin-top: 0;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.75rem;
    margin-bottom: 1.25rem;
}
.about-card p, .about-card li {
    font-size: 0.95rem;
    line-height: 1.6;
}

/* ─────── Demo Banner ─────── */
.demo-banner {
    background: rgba(139, 74, 58, 0.08);
    color: var(--terracotta);
    border: 1px solid rgba(139, 74, 58, 0.2);
    border-left: 4px solid var(--terracotta);
    padding: 14px 18px;
    border-radius: var(--radius-sm);
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    font-weight: 500;
}

/* ─────── Sidebar Customization ─────── */
[data-testid="stSidebar"] {
    background-color: var(--primary) !important;
    border-right: 1px solid rgba(200,155,60,0.2) !important;
}
[data-testid="stSidebar"] * {
    color: #F8FAFC !important;
}
/* Light text needs a dark field: the light theme would otherwise give the model dropdown a white box */
[data-testid="stSidebar"] [data-testid="stSelectbox"] [role="group"] {
    background-color: rgba(255, 255, 255, 0.08) !important;
    border-color: rgba(255, 255, 255, 0.25) !important;
}
[data-testid="stSidebar"] .stSelectbox label, 
[data-testid="stSidebar"] .stSlider label {
    font-family: 'Inter', sans-serif;
    font-weight: 600 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--accent) !important;
}
.sidebar-footer {
    text-align: center;
    font-size: 0.75rem;
    color: rgba(255,255,255,0.6);
    padding: 1.5rem 0;
    border-top: 1px solid rgba(200,155,60,0.2);
    margin-top: 2rem;
}
.lang-badge {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 4px;
    padding: 3px 8px;
    font-size: 0.75rem;
    color: #FBF8F1;
    display: inline-block;
    margin: 3px;
}

/* ─────── Footer ─────── */
.app-footer {
    text-align: center;
    padding: 3rem 1rem;
    margin-top: 4rem;
    color: var(--text-muted);
    font-size: 0.85rem;
    border-top: 1px solid var(--border);
}

/* ─────── Tabs Customization ─────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
    background-color: transparent;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 1.05rem;
    padding-top: 1rem;
    padding-bottom: 1rem;
    color: var(--text-muted) !important;
}
.stTabs [aria-selected="true"] {
    color: var(--primary) !important;
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem;
}
.stTabs [data-baseweb="tab-highlight"] {
    background-color: var(--accent);
}

/* ─────── Responsive ─────── */
@media (max-width: 768px) {
    .hero-title { font-size: 2.5rem; }
    .hero-subtitle { font-size: 0.95rem; }
    .metric-value { font-size: 1.8rem; }
}
</style>
"""

st.markdown(INDIC_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  CONSTANTS & UTILITIES
# ══════════════════════════════════════════════════════════════

from ml.config import (CLASS_NAMES as DEFAULT_CLASSES, AUTHORS_CHECKPOINT, BASELINE_CHECKPOINT, PROPOSED_CHECKPOINT,
                       get_device, METRICS_DIR, PLOTS_DIR, PREDICTION_DIR)
from ml.ocr import OCR_LANGUAGES, extract_document, tesseract_available
from ml.inference.predict import demo_inference, disable_runtime_installs

disable_runtime_installs()  # serving must never pip-install packages

# Color palette for detection boxes
def get_class_colors(n):
    np.random.seed(42)
    colors = []
    for i in range(n):
        hue = int(180 * i / n)
        c = cv2.cvtColor(np.array([[[hue, 180, 220]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(x) for x in c))
    return colors

COLORS = get_class_colors(len(DEFAULT_CLASSES))


def draw_detections(image, boxes, labels, scores, class_names, conf_threshold=0.5):
    img = image.copy()
    h, w = img.shape[:2]
    for box, label, score in zip(boxes, labels, scores):
        if score < conf_threshold:
            continue
        idx = int(label) if int(label) < len(class_names) else 0
        color = COLORS[idx % len(COLORS)]
        name = class_names[idx] if idx < len(class_names) else f"cls_{idx}"
        x1, y1, x2, y2 = [int(c) for c in box[:4]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        thick = max(2, int(min(h, w) / 350))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        text = f"{name}: {score:.2f}"
        fs = max(0.6, min(h, w) / 1200.0)
        t_thick = max(1, int(fs * 1.5))
        (tw, th_t), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, t_thick)
        
        # If the label goes above the image, move it inside the box
        if y1 - th_t - bl - 6 < 0:
            y_rect_top = y1
            y_rect_bot = y1 + th_t + bl + 6
            y_text = y1 + th_t + 3
        else:
            y_rect_top = y1 - th_t - bl - 6
            y_rect_bot = y1
            y_text = y1 - bl - 3
            
        cv2.rectangle(img, (x1, y_rect_top), (x1 + tw + 4, y_rect_bot), color, -1)
        cv2.putText(img, text, (x1 + 2, int(y_text)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), t_thick, cv2.LINE_AA)
    return img


def load_metrics(metrics_dir):
    metrics = {}
    if metrics_dir.exists():
        for f in metrics_dir.glob("*.json"):
            try:
                with open(f) as fp:
                    metrics[f.stem] = json.load(fp)
            except Exception:
                pass
    return metrics


MODEL_TYPES = {"DocLayout-YOLO (IndicDLP authors)": "doclayout", "YOLOv8 (Proposed)": "yolov8",
               "Faster R-CNN (Baseline)": "fasterrcnn"}
METRICS_NAMES = {"doclayout": "authors_metrics", "yolov8": "proposed_metrics", "fasterrcnn": "baseline_metrics"}
# Inference keeps every box down to this score; the sidebar slider then filters live, without re-running the model
INFERENCE_FLOOR = 0.05


@st.cache_resource(show_spinner="Loading model...")
def get_model(model_type):
    """Load a trained checkpoint once per server process, not on every click."""
    from ml.inference.predict import load_model
    device = get_device()
    path = {"doclayout": AUTHORS_CHECKPOINT, "yolov8": PROPOSED_CHECKPOINT, "fasterrcnn": BASELINE_CHECKPOINT}[model_type]
    model = load_model(str(path), model_type, device)
    disable_runtime_installs()  # again, now that ultralytics may have been imported
    return model, device


# Evaluated models, best first: (metrics file prefix, display name)
EVALUATED_MODELS = [("authors", "DocLayout-YOLO (IndicDLP authors)"), ("proposed", "YOLOv8s (ours)"),
                    ("baseline", "Faster R-CNN (ours)")]


def fmt(value, digits=3):
    """Metric for display: fixed decimals for numbers, N/A when missing."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.{digits}f}"
    return "N/A"


def dataset_label(m):
    if not m.get("num_images"):
        return "Not evaluated yet"
    return f"{m.get('dataset', 'Unknown dataset')} ({m['num_images']} pages)"


def show_training_history():
    """Per-epoch loss curves written by the two training scripts."""
    import pandas as pd
    baseline_file = METRICS_DIR / "baseline_training_history.json"
    proposed_file = METRICS_DIR / "proposed_training_history.csv"
    if not baseline_file.exists() and not proposed_file.exists():
        st.info("Loss curves will appear after training.")
        return
    hc1, hc2 = st.columns(2)
    with hc1:
        if baseline_file.exists():
            h = json.loads(baseline_file.read_text())
            df = pd.DataFrame({"Train loss": h["train_losses"], "Val loss": h["val_losses"]},
                              index=range(1, len(h["train_losses"]) + 1))
            st.markdown("**Faster R-CNN** &mdash; loss per epoch")
            st.line_chart(df)
    with hc2:
        if proposed_file.exists():
            r = pd.read_csv(proposed_file)
            r.columns = [c.strip() for c in r.columns]
            parts = ["box_loss", "cls_loss", "dfl_loss"]
            df = pd.DataFrame({"Train loss": r[[f"train/{p}" for p in parts]].sum(axis=1).values,
                               "Val loss": r[[f"val/{p}" for p in parts]].sum(axis=1).values},
                              index=r["epoch"].values)
            st.markdown("**YOLOv8** &mdash; loss per epoch (box + class + DFL)")
            st.line_chart(df)


def run_inference_frcnn(image, model, device, conf_threshold):
    import torch
    from torchvision import transforms as T
    model.eval()
    
    # Prevent OOM on massive images by capping max dimension
    max_dim = 1024
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image_resized = cv2.resize(image, (int(w * scale), int(h * scale)))
    else:
        image_resized = image

    from ml.inference.predict import predict_single

    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
    start = time.time()
    # predict_single shifts Faster R-CNN labels back to 0-indexed class ids
    pred = predict_single(model, img_tensor, 'fasterrcnn', device)
    inference_time = time.time() - start
    mask = pred['scores'] >= conf_threshold

    boxes = pred['boxes'][mask]

    # Scale boxes back if we resized
    if max(h, w) > max_dim:
        boxes = boxes / scale

    return {
        'boxes': boxes,
        'labels': pred['labels'][mask],
        'scores': pred['scores'][mask],
        'inference_time': inference_time
    }


def run_inference_yolo(image, model, conf_threshold, model_type="yolov8", device="cpu"):
    """YOLOv8 or DocLayout-YOLO on a BGR image; labels are 0-indexed class ids."""
    from ml.inference.predict import predict_single
    start = time.time()
    out = predict_single(model, Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), model_type, device,
                         conf=conf_threshold)
    out['inference_time'] = time.time() - start
    return out


def filter_results(results, conf_threshold):
    """The boxes at or above the slider's threshold, with duplicates (same box, several labels) removed."""
    from ml.inference.predict import dedupe_boxes
    keep = np.asarray(results["scores"]) >= conf_threshold
    return dedupe_boxes({**results, "boxes": np.asarray(results["boxes"])[keep],
                         "labels": np.asarray(results["labels"])[keep], "scores": np.asarray(results["scores"])[keep]})


def metric_card_html(value, label, icon=""):
    return f"""
    <div class="metric-card">
        <div class="metric-value">{icon} {value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


def generate_html_table(headers, rows):
    header_html = "".join([f"<th>{h}</th>" for h in headers])
    rows_html = ""
    for row in rows:
        row_html = "".join([f"<td>{str(cell)}</td>" for cell in row])
        rows_html += f"<tr>{row_html}</tr>"
    return f"""
    <table class="custom-table">
        <thead><tr>{header_html}</tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    """


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo
    logo_path = Path(__file__).parent / "assets" / "logo.jpg"
    if logo_path.exists():
        st.image(str(logo_path), width="stretch")
    else:
        st.markdown('<div style="text-align:center; font-size:1.5rem; font-weight:700; color:#C89B3C; margin:1rem 0;">IndicDoc AI</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Offer only the models that have a trained checkpoint; demo mode only when there is none
    available = {"DocLayout-YOLO (IndicDLP authors)": AUTHORS_CHECKPOINT, "YOLOv8 (Proposed)": PROPOSED_CHECKPOINT,
                 "Faster R-CNN (Baseline)": BASELINE_CHECKPOINT}
    model_options = [name for name, path in available.items() if path.exists()] or ["Demo (No Model)"]
    model_choice = st.selectbox(
        "MODEL",
        model_options,
        index=0,
        help="Select the model for inference"
    )

    detection_mode = st.radio(
        "DETECTION MODE", ["Find every part", "Balanced"],
        help="Find every part: the lowest threshold that kept at least 70% of boxes correct on validation pages "
             "(finds more, some boxes wrong). Balanced: the threshold with the best F1.")
    # Thresholds chosen on the validation pages by scripts/evaluate.py
    metrics_name = METRICS_NAMES.get(MODEL_TYPES.get(model_choice), "baseline_metrics")
    model_metrics = load_metrics(METRICS_DIR).get(metrics_name, {})
    recommended = model_metrics.get("recommended_conf", 0.5)
    if detection_mode == "Find every part":
        recommended = model_metrics.get("high_recall_conf", recommended)
    conf_threshold = st.slider(
        "CONFIDENCE THRESHOLD",
        min_value=0.05, max_value=0.95, value=float(min(0.95, max(0.05, recommended))), step=0.05,
        key=f"conf_{model_choice}_{detection_mode}",  # each model and mode starts from its own value
        help="Boxes below this confidence are hidden; move it to show more or fewer parts instantly.",
    )

    st.markdown("---")
    st.markdown('<p style="font-size:0.72rem; color:rgba(200,155,60,0.7); font-weight:600; letter-spacing:0.08em;">DATASET</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.82rem; line-height:1.6;'>
    <b>IndicDLP</b> &mdash; AIKosh<br>
    119,806 images<br>
    12 languages &bull; 42 classes<br>
    COCO-format annotations
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="font-size:0.72rem; color:rgba(200,155,60,0.7); font-weight:600; letter-spacing:0.08em;">LANGUAGES</p>', unsafe_allow_html=True)
    languages = ["Assamese", "Bengali", "English", "Gujarati", "Hindi", "Kannada",
                 "Malayalam", "Marathi", "Odia", "Punjabi", "Tamil", "Telugu"]
    badge_html = "".join(f'<span class="lang-badge">{l}</span>' for l in languages)
    st.markdown(f'<div style="display:flex; flex-wrap:wrap;">{badge_html}</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-footer">
        IndicDoc AI v1.0<br>
        B.E. AI&ML Deep Learning Project<br>
        SDG 4 &bull; SDG 9 &bull; SDG 16
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  HERO HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center; padding:0.8rem 0 0.3rem;">
    <div class="hero-title">Indic<span>Doc</span> AI</div>
    <div class="hero-subtitle">Check question papers and official documents before they go out. 12 Indian languages, offline.</div>
    <div class="hero-divider"><span class="hero-divider-icon">&loz; &loz; &loz;</span></div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  NAVIGATION TABS
# ══════════════════════════════════════════════════════════════
# Analyze first: the app opens on the upload screen
tab_verify, tab_analyze, tab_dash, tab_insights, tab_eval, tab_about = st.tabs([
    "Verify Paper",
    "Analyze",
    "Dashboard",
    "Model Insights",
    "Evaluation",
    "About",
])


# ══════════════════════════════════════════════════════════════
#  TAB 0: VERIFY PAPER (IndicDoc Verify)
# ══════════════════════════════════════════════════════════════
def _pages_from_bytes(data: bytes, name: str):
    """Every page of a PDF (200 dpi: every digit must be read exactly) or the one image, as RGB arrays."""
    from ml.preprocessing.preprocess import decode_image, pdf_page_count, render_pdf_page
    if name.lower().endswith(".pdf"):
        return [cv2.cvtColor(render_pdf_page(data, i, dpi=200), cv2.COLOR_BGR2RGB) for i in range(pdf_page_count(data))]
    return [cv2.cvtColor(decode_image(data), cv2.COLOR_BGR2RGB)]


def _read_paper(images, languages, progress, offset, total):
    """Layout model at its high-recall threshold, then OCR with the number pass, page by page."""
    model, device = get_model("doclayout")
    conf = load_metrics(METRICS_DIR).get("authors_metrics", {}).get("high_recall_conf", 0.15)
    pages = []
    for i, image_rgb in enumerate(images):
        results = filter_results(run_inference_yolo(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), model, conf,
                                                    "doclayout", device), conf)
        pages.append(extract_document(image_rgb, results, languages, number_pass=True))
        progress.progress((offset + i + 1) / total, text=f"Reading page {offset + i + 1} of {total}")
    return pages


class _ExampleFile:
    """A bundled example paper with the interface of an uploaded file (name, getvalue)."""
    def __init__(self, path):
        self.path, self.name = path, path.name

    def getvalue(self):
        return self.path.read_bytes()


TEST_PAPERS = PROJECT_ROOT / "docs" / "test_papers"
# label -> (paper, source version or None); answer key: docs/test_papers/ANSWER_KEY.md
VERIFY_EXAMPLES = {
    "CBSE bilingual paper (one PDF): no error": ("13_cbse_bilingual/no_error.pdf", None),
    "CBSE bilingual paper (one PDF): a number changed in Hindi Q9": ("13_cbse_bilingual/number_changed.pdf", None),
    "CBSE bilingual paper (one PDF): internal choice missing in Hindi Q33":
        ("13_cbse_bilingual/internal_choice_missing.pdf", None),
    "Hindi translation vs English: a number changed in Q15":
        ("02_hindi_number_changed/hindi_version.pdf", "02_hindi_number_changed/english_source.pdf"),
    "Hindi translation vs English: no error":
        ("01_hindi_no_error/hindi_version.pdf", "01_hindi_no_error/english_source.pdf"),
    "Tamil translation vs English: an option missing in Q9":
        ("08_tamil_option_missing/tamil_version.pdf", "08_tamil_option_missing/english_source.pdf"),
    "Tamil paper: question 13 printed twice": ("09_tamil_question_printed_twice/tamil_version.pdf", None),
}

with tab_verify:
    from ml.verify import list_profiles, verify
    from ml.verify.report import annotate, to_html as verify_html
    st.markdown('<div class="section-title">Pre-print check of a question paper</div>', unsafe_allow_html=True)
    st.markdown("Upload the paper as it will be printed. IndicDoc Verify reads every page offline and flags "
                "numbering gaps, repeated questions, missing options, marks that do not add up, questions repeated "
                "from earlier papers and, against the source version, every number, option and mark that differs in "
                "the translation. Each finding points to the question and page for the vetter to confirm.")
    with st.expander("How it works: two readers, one verdict", expanded=False):
        st.markdown("""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:4px 0 10px">
 <div style="border-left:5px solid #14284B;background:#fff;border-radius:10px;padding:10px 14px">
  <b>Reader A — deep learning</b><br>
  <span style="font-size:0.92em">DocLayout-YOLO finds every question, option block, sub-part, figure and table on the
  page image; OCR reads the text inside each box.</span></div>
 <div style="border-left:5px solid #D9A441;background:#fff;border-radius:10px;padding:10px 14px">
  <b>Reader B — independent</b><br>
  <span style="font-size:0.92em">No layout model: it reads the PDF's own text layer (exact characters) or, for a
  scanned page, a second OCR pass with different settings, and rebuilds questions from line positions.</span></div>
</div>
<div style="text-align:center;color:#D9A441;font-size:1.3em;line-height:1">▼ &nbsp; both readings are matched question by question &nbsp; ▼</div>
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:8px;font-size:0.9em">
 <div style="background:#E8F5E9;border-radius:8px;padding:8px"><b>Agree</b><br>number, options, numbers, marks and parts match → the reading is trusted</div>
 <div style="background:#FFF8E1;border-radius:8px;padding:8px"><b>Disagree</b><br>the PDF's own text or the more confident reader decides</div>
 <div style="background:#E3F2FD;border-radius:8px;padding:8px"><b>Still unsure</b><br>that question is read a third time (two crops, English re-reading for numbers)</div>
 <div style="background:#FFEBEE;border-radius:8px;padding:8px"><b>Unsettled</b><br>shown as "needs a human look" — never reported as an error</div>
</div>
<p style="font-size:0.9em;margin-top:10px">A finding is <b>confirmed</b> only when the agreed reading still shows it.
A finding that only one reader produced, because it misread something, is <b>dismissed</b>. After a check, the table
"Reader A vs reader B" shows this for every question.</p>
""", unsafe_allow_html=True)
    v_mode = st.radio("What to check", ["One paper", "Translation against its source", "One bilingual paper",
                                        "Several papers"],
                      horizontal=True,
                      help="One paper: a paper printed in two languages page by page (CBSE: a Hindi page, then the "
                           "English page) is recognised by itself and its versions are checked against each other. "
                           "One bilingual paper: both languages on the same pages, scripts chosen below. Several "
                           "papers: a verdict for each and all reports in one ZIP.")
    lang_names = ["Auto-detect"] + list(OCR_LANGUAGES)

    def _codes(choice):
        code = OCR_LANGUAGES.get(choice)
        return "auto" if code is None else (code if code == "eng" else f"{code}+eng")

    c1, c2 = st.columns(2)
    with c1:
        v_files = []
        if v_mode == "Several papers":
            v_file = None
            v_files = st.file_uploader("Papers to check", type=["pdf", "png", "jpg", "jpeg", "tiff"],
                                       accept_multiple_files=True, key="verify_files")
        else:
            v_file = st.file_uploader("Paper to check", type=["pdf", "png", "jpg", "jpeg", "tiff"], key="verify_file")
        v_lang = st.selectbox("Language of this paper", lang_names, key="verify_lang")
    v_src, v_src_lang, v_split = None, "auto", None
    with c2:
        if v_mode == "Translation against its source":
            v_src = st.file_uploader("Source-language version", type=["pdf", "png", "jpg", "jpeg", "tiff"],
                                     key="verify_src")
            v_src_lang = _codes(st.selectbox("Language of the source", lang_names, key="verify_src_lang"))
        elif v_mode == "One bilingual paper":
            scripts = ["Latin", "Devanagari", "Bengali", "Gurmukhi", "Gujarati", "Oriya", "Tamil", "Telugu",
                       "Kannada", "Malayalam"]
            s1 = st.selectbox("Source script", scripts, index=0)
            s2 = st.selectbox("Translation script", scripts, index=1)
            v_split = {s1: s1.lower(), s2: s2.lower()} if s1 != s2 else None
    examples = {k: v for k, v in VERIFY_EXAMPLES.items() if (TEST_PAPERS / v[0]).exists()}
    v_example = st.selectbox("Or try an example paper (answer key: docs/test_papers/ANSWER_KEY.md)",
                             ["None"] + list(examples)) if examples and v_mode != "Several papers" else "None"
    if v_example != "None":
        paper_path, source_path = examples[v_example]
        v_file = _ExampleFile(TEST_PAPERS / paper_path)
        v_src = _ExampleFile(TEST_PAPERS / source_path) if source_path else None
        v_mode = "Translation against its source" if source_path else "One paper"
        v_lang, v_src_lang, v_split = "Auto-detect", "auto", None
        st.caption(f"Example: {v_file.name}" + (f" checked against {v_src.name}" if v_src else ""))
    if v_file is not None and v_mode != "Several papers":
        with st.expander("Preview of the paper", expanded=True):
            try:
                _prev = _pages_from_bytes(v_file.getvalue(), v_file.name)
                cols = st.columns(4)
                for k, pg in enumerate(_prev):
                    cols[k % 4].image(pg, caption=f"Page {k + 1}", width="stretch")
            except Exception as e:  # a preview must never block the check
                st.caption(f"No preview: {e}")
    v_dual = st.checkbox("Dual-reader consensus (recommended)", value=True,
                         help="The file is read twice, independently: by the deep-learning layout model with OCR, and "
                              "without the layout model from the PDF's own text (or a second OCR pass). A finding is "
                              "confirmed only when both readings support it; where they disagree the question is read "
                              "a third time, and anything still unsettled is marked for a human look.")
    with st.expander("More checks"):
        v_bank = st.file_uploader("Question bank of earlier papers (JSON list of {text, source})", type=["json"])
        profiles = list_profiles()
        v_profile = st.selectbox("Format profile", ["None"] + list(profiles),
                                 format_func=lambda k: "None" if k == "None" else profiles[k])

    ready = AUTHORS_CHECKPOINT.exists() and tesseract_available()
    if not ready:
        st.info("Verification needs the DocLayout-YOLO model and Tesseract (python scripts/download_models.py).")
    def _check(paper, source, lang, src_lang, split, auto_bilingual, bar):
        """Read and check one paper (and its source version): reader A, reader B when asked, then the checks."""
        images = _pages_from_bytes(paper.getvalue(), paper.name)
        src_images = _pages_from_bytes(source.getvalue(), source.name) if source is not None else None
        total = len(images) + len(src_images or [])
        pages = _read_paper(images, lang, bar, 0, total)
        src_pages = _read_paper(src_images, src_lang, bar, len(images), total) if src_images else None
        kwargs = {}
        if v_dual:
            from ml.verify.consensus import make_rereader
            from ml.verify.pipeline import second_reading, version_langs

            def _second(imgs, pgs, upload):
                data = upload.getvalue()
                return second_reading(imgs, pgs, data if upload.name.lower().endswith(".pdf") else None, 200,
                                      progress=lambda k, n: bar.progress(k / n, text=f"Second reading: page {k} of {n}"))
            kwargs = {"pages_b": _second(images, pages, paper), "rereader": make_rereader(images),
                      "langs": {"target": version_langs(pages)}}
            if src_images:
                kwargs["source_pages_b"] = _second(src_images, src_pages, source)
                kwargs["rereader"] = {"target": make_rereader(images), "source": make_rereader(src_images)}
                kwargs["langs"]["source"] = version_langs(src_pages)
        bar.progress(1.0, text="Checking")
        report = verify(pages, source_pages=src_pages, bank=v_bank_items, split_scripts=split,
                        profile=None if v_profile == "None" else v_profile,
                        meta={"File": paper.name, "Pages": len(images)},
                        bilingual="auto" if auto_bilingual else "off", **kwargs)
        return report, images, src_images

    def _verdict(summary):
        if summary["errors"]:
            return "Do not print"
        return "Check before printing" if summary["warnings"] or summary.get("review") else "Ready for printing"

    try:
        v_bank_items = json.loads(v_bank.getvalue()) if v_bank is not None else None
    except Exception as e:
        st.error(f"Could not read the question bank: {e}")
        v_bank_items = None
    if v_mode == "Several papers" and st.button("Verify all", type="primary", disabled=not (ready and v_files)):
        import io
        import zipfile
        rows, archive = [], io.BytesIO()
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            for n, f in enumerate(v_files, 1):
                bar = st.progress(0.0, text=f"Paper {n} of {len(v_files)}: {f.name}")
                try:
                    report, images, _ = _check(f, None, _codes(v_lang), "auto", None, True, bar)
                except Exception as e:
                    rows.append({"paper": f.name, "verdict": f"Not checked: {e}"})
                    bar.empty()
                    continue
                s = report["summary"]
                rows.append({"paper": f.name, "verdict": _verdict(s), "questions": s["questions"],
                             "errors": s["errors"], "warnings": s["warnings"],
                             "versions": report["meta"].get("Versions", "")})
                stem = f.name.rsplit(".", 1)[0]
                zf.writestr(f"{stem}_report.html", verify_html(report, {"paper": images}))
                zf.writestr(f"{stem}_report.json", json.dumps(report, ensure_ascii=False, indent=1))
                bar.empty()
        st.session_state["verify_batch"] = (rows, archive.getvalue())
    if v_mode == "Several papers" and st.session_state.get("verify_batch"):
        import pandas as pd
        rows, archive = st.session_state["verify_batch"]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.download_button("Download all reports (ZIP)", data=archive, file_name="indicdoc_verify_reports.zip",
                           mime="application/zip", width="stretch")

    if v_mode != "Several papers" and st.button("Verify", type="primary", disabled=not (ready and v_file is not None)):
        bar = st.progress(0.0, text="Reading pages")
        try:
            st.session_state["verify"] = _check(v_file, v_src, _codes(v_lang), v_src_lang, v_split,
                                                v_mode == "One paper", bar)
        except Exception as e:
            st.error(f"Could not read the files: {e}")
            st.stop()
        bar.empty()

    if v_mode != "Several papers" and st.session_state.get("verify"):
        report, images, src_images = st.session_state["verify"]
        s = report["summary"]
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(metric_card_html(str(s["questions"]), "Questions read"), unsafe_allow_html=True)
        k2.markdown(metric_card_html(str(s["errors"]), "Errors"), unsafe_allow_html=True)
        k3.markdown(metric_card_html(str(s["warnings"]), "Warnings"), unsafe_allow_html=True)
        k4.markdown(metric_card_html(str(s["info"]), "Notes"), unsafe_allow_html=True)
        if s["errors"]:
            st.error("Do not print: errors found. Each one is listed below with its question and page.")
        elif s["warnings"]:
            st.warning("Check before printing: see the warnings below.")
        elif s.get("review"):
            st.warning(f"Check before printing: {s['review']} place(s) the two readers read differently are listed "
                       "below as uncertain readings, for a look by eye.")
        else:
            st.success("No problems found by the automatic checks.")
        if report.get("versions", {}).get("mode") in ("pages", "lines"):
            st.info(f"Bilingual paper: {report['meta'].get('Versions', '')}. The two versions were checked against "
                    "each other.")
        cons = report.get("consensus")
        if cons:
            stats = cons["stats"]
            agree = [v["field_agreement"] for v in stats.values() if v.get("field_agreement") is not None]
            ca = [v["mean_confidence_a"] for v in stats.values() if v.get("mean_confidence_a") is not None]
            cb = [v["mean_confidence_b"] for v in stats.values() if v.get("mean_confidence_b") is not None]
            st.markdown(
                f"**Dual-reader consensus.** Reader A ({cons['readers']['a']}) and reader B ({cons['readers']['b']}) "
                f"agreed on {100 * sum(agree) / len(agree):.0f}% of what they read" if agree else "**Dual-reader consensus.**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Confirmed by both", s.get("confirmed", 0))
            c2.metric("Need a human look", s.get("review", 0))
            c3.metric("Dismissed as misreadings", s.get("dismissed", 0))
            c4.metric("Mean confidence A / B", f"{sum(ca) / len(ca):.2f} / {sum(cb) / len(cb):.2f}" if ca and cb else "-")
        if report["issues"]:
            import pandas as pd

            def _row(i):
                c = i.get("consensus") or {}
                row = {"severity": i["severity"], "check": i["code"].replace("_", " ").lower(),
                       "question": i["question"], "page": None if i["page"] is None else i["page"] + 1,
                       "version": i["version"], "finding": i["message"]}
                if c:
                    row.update({"consensus": c.get("status"), "conf. A": c.get("confidence_a"),
                                "conf. B": c.get("confidence_b")})
                return row
            st.dataframe(pd.DataFrame([_row(i) for i in report["issues"]]), hide_index=True, width="stretch")
        if cons and cons.get("comparison"):
            import pandas as pd
            comp = pd.DataFrame([{"version": r["version"], "reader A (layout + OCR)": r["reader_a"],
                                  "reader B (independent)": r["reader_b"],
                                  "agreed on": ", ".join(r["agree"]) or "-",
                                  "differed on": ", ".join(r["disagree"]) or "-", "outcome": r["outcome"]}
                                 for r in cons["comparison"]])
            n_agree = int((comp["outcome"] == "both agree").sum())
            with st.expander(f"Reader A vs reader B, question by question ({n_agree} of {len(comp)} agree fully)"):
                st.dataframe(comp, hide_index=True, width="stretch")
        if cons and cons["dismissed"]:
            with st.expander(f"{len(cons['dismissed'])} single-reader finding(s) dismissed by the consensus"):
                st.dataframe(pd.DataFrame([{"reader": i["raised_by"].upper(), "check": i["code"].replace("_", " ").lower(),
                                            "question": i["question"], "finding": i["message"]}
                                           for i in cons["dismissed"]]), hide_index=True, width="stretch")
        numbered = list(enumerate(report["issues"], 1))
        same_file = src_images is None
        from ml.verify.report import mark_questions
        detected = (report.get("paper") or {}).get("questions", []) + ((report.get("source") or {}).get("questions", [])
                                                                      if same_file else [])
        st.markdown("**The paper as read:** blue boxes are the detected questions; numbered red/orange boxes are findings.")
        for p, image_rgb in enumerate(images):
            on_page = [(n, i) for n, i in numbered if i.get("page") == p
                       and (same_file or i["version"] in ("paper", "target"))]
            marked = mark_questions(image_rgb, detected, p)
            st.image(annotate(marked, on_page), caption=f"Page {p + 1}: "
                     f"{sum(q.get('page') == p for q in detected)} question(s) detected, {len(on_page)} finding(s)",
                     width="stretch")
        d1, d2 = st.columns(2)
        imgs = {"paper": images, **({"source": src_images} if src_images else {})}
        d1.download_button("Download report (HTML)", data=verify_html(report, imgs).encode("utf-8"),
                           file_name="indicdoc_verify_report.html", mime="text/html", width="stretch")
        d2.download_button("Download findings (JSON)", data=json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"),
                           file_name="indicdoc_verify_report.json", mime="application/json", width="stretch")


# ══════════════════════════════════════════════════════════════
#  TAB 1: DASHBOARD
# ══════════════════════════════════════════════════════════════
with tab_dash:
    # Metric row: the default (best) model
    saved_metrics = load_metrics(METRICS_DIR)
    baseline_m = saved_metrics.get("baseline_metrics", {})
    proposed_m = saved_metrics.get("proposed_metrics", {})
    authors_m = saved_metrics.get("authors_metrics", {})
    primary_name, primary_m = next(((label, saved_metrics[f"{n}_metrics"]) for n, label in EVALUATED_MODELS
                                    if saved_metrics.get(f"{n}_metrics")), ("", {}))
    if primary_m:
        st.caption(f"{primary_name} on {dataset_label(primary_m)}")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(metric_card_html(fmt(primary_m.get('map50')), "mAP@50"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card_html(
            fmt(primary_m.get('at_recommended_conf', {}).get('precision')), "Precision"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card_html(
            fmt(primary_m.get('at_recommended_conf', {}).get('recall')), "Recall"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card_html(
            fmt(primary_m.get('at_recommended_conf', {}).get('f1')), "F1-Score"), unsafe_allow_html=True)
    with c5:
        st.markdown(metric_card_html(
            f"{fmt(primary_m.get('inference_time'))}s", "Time per Page"), unsafe_allow_html=True)
    if primary_m.get("recommended_conf") is not None:
        st.caption(f"Precision, recall and F1 at the model's recommended confidence ({primary_m['recommended_conf']}), "
                   "chosen on validation pages.")

    st.markdown("")
    st.markdown('<div class="section-title">Models</div>', unsafe_allow_html=True)
    headers = ["Property", "DocLayout-YOLO (IndicDLP authors)", "YOLOv8s (ours)", "Faster R-CNN (ours)"]
    rows = [
        ["Trained by", "IndicDLP authors, full training set (~96K pages)", "This project, 2,293 pages", "This project, 2,293 pages"],
        ["Architecture", "DocLayout-YOLO (YOLOv10-based)", "YOLOv8s (CSPDarknet)", "ResNet-50 + FPN, two-stage"],
        ["Evaluated on", dataset_label(authors_m), dataset_label(proposed_m), dataset_label(baseline_m)],
        ["mAP@50", fmt(authors_m.get("map50")), fmt(proposed_m.get("map50")), fmt(baseline_m.get("map50"))],
        ["Input size", "1024 px", "640 px", "640 px (longest side)"],
        ["Available here", "Yes" if AUTHORS_CHECKPOINT.exists() else "No",
         "Yes" if PROPOSED_CHECKPOINT.exists() else "No", "Yes" if BASELINE_CHECKPOINT.exists() else "No"],
    ]
    st.markdown(generate_html_table(headers, rows), unsafe_allow_html=True)

    st.markdown('<div class="section-title">Training Overview</div>', unsafe_allow_html=True)
    show_training_history()
    comp_plot = PLOTS_DIR / "model_comparison.png"
    if comp_plot.exists():
        st.image(str(comp_plot), caption="Model Performance Comparison (same test pages)", width="stretch")


# ══════════════════════════════════════════════════════════════
#  TAB 2: ANALYZE
# ══════════════════════════════════════════════════════════════
with tab_analyze:
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown('<div class="section-title">Upload & Analyze</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a document image or PDF",
            type=["png", "jpg", "jpeg", "tiff", "bmp", "pdf"],
            help="Upload a document image or PDF for layout analysis"
        )

        # Real IndicDLP test pages (MIT licence) ship with the app; the synthetic pages are only a fallback
        sample_dir = PROJECT_ROOT / "app" / "assets" / "samples"
        if not any(sample_dir.glob("*.png")):
            sample_dir = PROJECT_ROOT / "data" / "processed" / "sample" / "images"
        sample_images = sorted(list(sample_dir.glob("*.png")) + list(sample_dir.glob("*.jpg"))) if sample_dir.exists() else []

        if sample_images:
            use_sample = st.checkbox("Use a sample page (real IndicDLP test pages)")
            if use_sample:
                sample_choice = st.selectbox("Sample", [s.name for s in sample_images[:10]])
                uploaded_file = sample_dir / sample_choice

        if uploaded_file is not None:
            from ml.preprocessing.preprocess import decode_image, pdf_page_count, render_pdf_page
            if isinstance(uploaded_file, Path):
                file_ext = uploaded_file.suffix.lower()
                file_data = uploaded_file.read_bytes()
            else:
                file_ext = Path(uploaded_file.name).suffix.lower()
                file_data = uploaded_file.getvalue()
            page_no = 1
            if file_ext == '.pdf':
                try:
                    n_pages = pdf_page_count(file_data)
                    page_no = st.number_input(f"Page (1-{n_pages})", min_value=1, max_value=n_pages, value=1) if n_pages > 1 else 1
                    image = render_pdf_page(file_data, page_index=page_no - 1)
                except Exception:
                    st.error("Could not read this PDF.")
                    st.stop()
            else:
                try:
                    image = decode_image(file_data)
                except ValueError as e:
                    st.error(f"Could not use this file: {e}.")
                    st.stop()
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Results belong to one page: identify it by content (and PDF page) so they never carry over
            page_key = f"{hashlib.sha1(file_data).hexdigest()}:{page_no}"

            h, w = image.shape[:2]
            st.markdown(f"**Dimensions:** {w} x {h} px")
            run_btn = st.button("Analyze Document", type="primary", width="stretch")
            reset_btn = st.button("Reset", width="stretch")
        else:
            run_btn = False
            reset_btn = False

        st.markdown('<div class="section-title" style="margin-top: 2rem;">Analysis Results</div>', unsafe_allow_html=True)
        results = None
        if uploaded_file is not None and st.session_state.get("last_results_key") == page_key:
            results = st.session_state.get("last_results")

        if results:
            results = filter_results(results, conf_threshold)
            n_det = len(results.get("boxes", []))
            avg_conf = float(np.mean(results["scores"])) if n_det > 0 else 0.0
            inf_time = results.get("inference_time", 0)

            st.markdown(metric_card_html(str(n_det), "Elements Detected"), unsafe_allow_html=True)
            st.markdown(metric_card_html(f"{avg_conf:.2f}", "Avg Confidence"), unsafe_allow_html=True)
            st.markdown(metric_card_html(f"{inf_time:.3f}s", "Inference Time"), unsafe_allow_html=True)

            if n_det > 0:
                st.markdown("<br>**Detected Elements:**", unsafe_allow_html=True)
                for i, (box, label, score) in enumerate(zip(results["boxes"], results["labels"], results["scores"])):
                    if score < conf_threshold:
                        continue
                    idx = int(label) if int(label) < len(DEFAULT_CLASSES) else 0
                    cls = DEFAULT_CLASSES[idx]
                    st.markdown(f"`{i+1}.` **{cls}** &mdash; {score:.2f}")

            if n_det > 0:
                annotated_bgr = cv2.cvtColor(
                    draw_detections(image_rgb.copy(), results["boxes"], results["labels"], results["scores"], DEFAULT_CLASSES, conf_threshold),
                    cv2.COLOR_RGB2BGR
                )
                _, buf = cv2.imencode('.png', annotated_bgr)
                st.markdown("<br>", unsafe_allow_html=True)
                st.download_button("Download Image", data=buf.tobytes(), file_name="indicdoc_result.png", mime="image/png", width="stretch")
        else:
            st.markdown("""
            <div style="color:var(--text-muted); font-size:0.9rem;">
                Results will appear here after running analysis.
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-title">Document Viewer</div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            view_mode = st.radio("View", ["Original", "Prediction", "Comparison"], index=1, horizontal=True, label_visibility="collapsed")
            has_results = st.session_state.get("last_results_key") == page_key

            if run_btn or has_results:
                # Run inference
                if run_btn:
                    with st.spinner("Analyzing document..."):
                        model_type = MODEL_TYPES.get(model_choice)
                        if model_type is None:
                            results = demo_inference(image)
                        else:
                            try:
                                model, device = get_model(model_type)
                                if model_type == "fasterrcnn":
                                    results = run_inference_frcnn(image, model, device, INFERENCE_FLOOR)
                                else:
                                    results = run_inference_yolo(image, model, INFERENCE_FLOOR, model_type, device)
                            except Exception as e:
                                st.warning(f"Model loading failed: {e}")
                                results = demo_inference(image)

                    st.session_state["last_results"] = results
                    st.session_state["last_results_key"] = page_key
                    # The left results panel was drawn before this inference ran; rerun so it shows the new results
                    st.rerun()

                results = st.session_state.get("last_results")
                if results:
                    if results.get("is_demo"):
                        st.markdown('<div class="demo-banner"><b>Demo Mode</b> &mdash; Showing synthetic detections. Train a model for real predictions.</div>', unsafe_allow_html=True)

                    shown = filter_results(results, conf_threshold)
                    annotated = draw_detections(image_rgb.copy(), shown["boxes"], shown["labels"], shown["scores"], DEFAULT_CLASSES, conf_threshold)

                    if view_mode == "Original":
                        st.image(image_rgb, caption="Original Document", width="stretch")
                    elif view_mode == "Prediction":
                        st.image(annotated, caption="Detected Layout Elements", width="stretch")
                    else:
                        c_l, c_r = st.columns(2)
                        with c_l:
                            st.image(image_rgb, caption="Original", width="stretch")
                        with c_r:
                            st.image(annotated, caption="Prediction", width="stretch")

                    # ── Text extraction (OCR) of the detected regions ──
                    st.markdown('<div class="section-title">Extracted Text</div>', unsafe_allow_html=True)
                    if not tesseract_available():
                        st.info("Text extraction needs Tesseract and its language models (python scripts/download_models.py).")
                    else:
                        lang_choice = st.selectbox("Text language", ["Auto-detect"] + list(OCR_LANGUAGES),
                                                   help="Auto-detect reads one text block with each script's model and keeps the most confident.")
                        if st.button("Read text (OCR)", width="stretch"):
                            code = OCR_LANGUAGES.get(lang_choice)
                            languages = "auto" if code is None else (code if code == "eng" else f"{code}+eng")
                            with st.spinner("Reading text..."):
                                st.session_state["ocr_result"] = extract_document(
                                    image_rgb, filter_results(results, conf_threshold), languages)
                                st.session_state["ocr_key"] = (page_key, conf_threshold, lang_choice)
                        doc = st.session_state.get("ocr_result")
                        if doc and st.session_state.get("ocr_key", (None,))[0] == page_key:
                            detected = f"detected script: {doc['script']}, " if doc.get("script") else ""
                            st.caption(f"Machine-read text ({detected}Tesseract languages: {doc['languages']}). "
                                       "It may contain errors; please review before use.")
                            t_doc, t_regions = st.tabs(["Document", "By region"])
                            with t_doc:
                                st.markdown(doc["markdown"])
                            with t_regions:
                                import pandas as pd
                                st.dataframe(pd.DataFrame([{"#": i + 1, "label": r["label"], "confidence": r["score"],
                                                            "text": r["text"]} for i, r in enumerate(doc["regions"])]),
                                             hide_index=True, width="stretch")
                            d1, d2 = st.columns(2)
                            with d1:
                                st.download_button("Download Markdown", data=doc["markdown"].encode("utf-8"),
                                                   file_name="indicdoc_text.md", mime="text/markdown", width="stretch")
                            with d2:
                                st.download_button("Download JSON", data=json.dumps(doc, ensure_ascii=False, indent=2).encode("utf-8"),
                                                   file_name="indicdoc_text.json", mime="application/json", width="stretch")
                            # Accessible formats for screen-reader users (headings, lists, tables, language tag)
                            from ml.accessible import to_epub, to_html
                            a1, a2 = st.columns(2)
                            with a1:
                                st.download_button("Accessible HTML", data=to_html(doc, "IndicDoc page").encode("utf-8"),
                                                   file_name="indicdoc_accessible.html", mime="text/html", width="stretch")
                            with a2:
                                st.download_button("Accessible EPUB3", data=to_epub([doc], "IndicDoc page"),
                                                   file_name="indicdoc_accessible.epub", mime="application/epub+zip", width="stretch")
            else:
                st.image(image_rgb, caption="Original Document", width="stretch")
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">&#x1F4C4;</div>
                <div class="empty-title">No Document Selected</div>
                <div class="empty-subtitle">Upload a document image or PDF from the left panel to begin layout analysis. Supports PNG, JPG, TIFF, BMP, and PDF formats.</div>
            </div>
            """, unsafe_allow_html=True)

    if reset_btn:
        st.session_state.pop("last_results", None)
        st.session_state.pop("last_results_key", None)
        st.rerun()


# ══════════════════════════════════════════════════════════════
#  TAB 3: MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════
with tab_insights:
    st.markdown('<div class="section-title">Model Comparison (same test pages)</div>', unsafe_allow_html=True)

    saved = load_metrics(METRICS_DIR)
    present = [(n, label, saved[f"{n}_metrics"]) for n, label in EVALUATED_MODELS if saved.get(f"{n}_metrics")]
    if present:
        headers = ["Metric"] + [label for _, label, _ in present]
        spec = [("mAP@50", "map50", 3), ("mAP@50:95", "map50_95", 3), ("Precision (conf >= 0.5)", "precision", 3),
                ("Recall (conf >= 0.5)", "recall", 3), ("F1 (conf >= 0.5)", "f1", 3), ("Avg IoU", "avg_iou", 3),
                ("Recommended confidence", "recommended_conf", 2), ("Inference Time (s/page)", "inference_time", 3),
                ("Model Size (MB)", "model_size_mb", 1)]
        rows = [[name] + [fmt(m.get(key), d) for _, _, m in present] for name, key, d in spec]
        rows.append(["F1 at recommended confidence"] + [fmt(m.get("at_recommended_conf", {}).get("f1")) for _, _, m in present])
        st.markdown(generate_html_table(headers, rows), unsafe_allow_html=True)
        for _, label, m in present:
            if m.get("note"):
                st.caption(f"{label}: {m['note']}")
    else:
        st.info("No evaluation metrics available yet.")

    # Architecture cards
    st.markdown('<div class="section-title">Architecture Details</div>', unsafe_allow_html=True)
    ac1, ac2 = st.columns(2)
    with ac1:
        st.markdown("""
<div class="about-card">
    <h4>Faster R-CNN (Baseline)</h4>
    <table style="width:100%; font-size:0.85rem;">
        <tr><td><b>Backbone</b></td><td>ResNet-50 + FPN</td></tr>
        <tr><td><b>Type</b></td><td>Two-stage detector</td></tr>
        <tr><td><b>Pretrained</b></td><td>COCO (torchvision)</td></tr>
        <tr><td><b>Parameters</b></td><td>~41.8M</td></tr>
        <tr><td><b>Optimizer</b></td><td>SGD (lr=0.005, momentum=0.9, weight decay 5e-4)</td></tr>
        <tr><td><b>Scheduler</b></td><td>Cosine annealing</td></tr>
        <tr><td><b>Augmentation</b></td><td>Colour jitter, Gaussian blur (no flips)</td></tr>
    </table>
</div>
""", unsafe_allow_html=True)
    with ac2:
        st.markdown("""
<div class="about-card">
    <h4>YOLOv8s (Proposed)</h4>
    <table style="width:100%; font-size:0.85rem;">
        <tr><td><b>Backbone</b></td><td>CSPDarknet</td></tr>
        <tr><td><b>Type</b></td><td>Single-stage detector</td></tr>
        <tr><td><b>Pretrained</b></td><td>COCO (ultralytics)</td></tr>
        <tr><td><b>Parameters</b></td><td>~11.2M</td></tr>
        <tr><td><b>Optimizer</b></td><td>Chosen automatically by ultralytics</td></tr>
        <tr><td><b>Scheduler</b></td><td>Linear decay with warm-up (ultralytics default)</td></tr>
        <tr><td><b>Augmentation</b></td><td>Mosaic, HSV, translate/scale (no flips)</td></tr>
    </table>
</div>
""", unsafe_allow_html=True)

    # Training curves & visualizations
    st.markdown('<div class="section-title">Training Curves</div>', unsafe_allow_html=True)
    show_training_history()

    st.markdown('<div class="section-title">Precision-Recall Curves</div>', unsafe_allow_html=True)
    tc1, tc2 = st.columns(2)
    for col, name, label in ((tc1, "baseline", "Faster R-CNN"), (tc2, "proposed", "YOLOv8")):
        with col:
            pr = PLOTS_DIR / f"{name}_pr_curve.png"
            if pr.exists():
                st.image(str(pr), caption=f"{label}: precision-recall at IoU 0.50", width="stretch")
            else:
                st.info(f"{label}: PR curve will appear after evaluation.")

    # Sample predictions
    pred_images = sorted(PREDICTION_DIR.glob("*_prediction_*.png")) if PREDICTION_DIR.exists() else []
    if pred_images:
        st.markdown('<div class="section-title">Sample Predictions</div>', unsafe_allow_html=True)
        pc = st.columns(min(3, len(pred_images)))
        for i, pi in enumerate(pred_images[:6]):
            with pc[i % len(pc)]:
                st.image(str(pi), caption=pi.stem.replace("_", " ").title(), width="stretch")


# ══════════════════════════════════════════════════════════════
#  TAB 4: EVALUATION
# ══════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown('<div class="section-title">Evaluation Dashboard</div>', unsafe_allow_html=True)

    saved = load_metrics(METRICS_DIR)
    present = [(n, label, saved[f"{n}_metrics"]) for n, label in EVALUATED_MODELS if saved.get(f"{n}_metrics")]

    for n, label, m in present:
        st.markdown(f"**{label}**")
        ec1, ec2, ec3, ec4, ec5, ec6 = st.columns(6)
        with ec1: st.markdown(metric_card_html(fmt(m.get('map50')), "mAP@50"), unsafe_allow_html=True)
        with ec2: st.markdown(metric_card_html(fmt(m.get('map50_95')), "mAP@50:95"), unsafe_allow_html=True)
        with ec3: st.markdown(metric_card_html(fmt(m.get('at_recommended_conf', {}).get('precision')), "Precision"), unsafe_allow_html=True)
        with ec4: st.markdown(metric_card_html(fmt(m.get('at_recommended_conf', {}).get('recall')), "Recall"), unsafe_allow_html=True)
        with ec5: st.markdown(metric_card_html(fmt(m.get('at_recommended_conf', {}).get('f1')), "F1-Score"), unsafe_allow_html=True)
        with ec6: st.markdown(metric_card_html(f"{fmt(m.get('inference_time'))}s", "Latency"), unsafe_allow_html=True)
    if present:
        st.caption("Precision, recall and F1 at each model's recommended confidence (best F1 on validation pages).")
    else:
        st.info("No evaluation metrics available yet.")

    # Per-class AP and confusion matrices, one row per model
    for name, label in EVALUATED_MODELS:
        ap_plot, cm_plot = PLOTS_DIR / f"{name}_per_class_ap.png", PLOTS_DIR / f"{name}_confusion_matrix.png"
        if ap_plot.exists() or cm_plot.exists():
            st.markdown(f'<div class="section-title">{label}: Per-Class Results</div>', unsafe_allow_html=True)
            ev1, ev2 = st.columns(2)
            with ev1:
                if ap_plot.exists():
                    st.image(str(ap_plot), caption=f"{label}: AP@50 per class", width="stretch")
            with ev2:
                if cm_plot.exists():
                    st.image(str(cm_plot), caption=f"{label}: confusion matrix (12 most frequent classes)", width="stretch")

    # Class distribution
    cd_plot = PLOTS_DIR / "class_distribution.png"
    if cd_plot.exists():
        st.markdown('<div class="section-title">Class Distribution</div>', unsafe_allow_html=True)
        st.image(str(cd_plot), caption="Class distribution of the evaluation pages", width="stretch")

    # Error analysis
    err_dir = PREDICTION_DIR / "errors"
    err_images = sorted(err_dir.glob("*.png")) if err_dir.exists() else []
    if err_images:
        st.markdown('<div class="section-title">Error Analysis</div>', unsafe_allow_html=True)
        for ei in err_images:
            st.image(str(ei), caption=ei.stem.replace("_", " ").title(), width="stretch")

    # Per-language table
    st.markdown('<div class="section-title">Per-Language Evaluation</div>', unsafe_allow_html=True)
    groups = [(label, m.get("per_group") or {}) for _, label, m in present]
    languages = sorted({lang for _, g in groups for lang in g})
    if languages:
        headers = ["Language", "Test pages"] + [f"{label} mAP@50" for label, _ in groups]
        rows = []
        for lang in languages:
            pages = next(g[lang]["images"] for _, g in groups if lang in g)
            rows.append([lang.title(), pages] + [fmt((g.get(lang) or {}).get("map50")) for _, g in groups])
        st.markdown(generate_html_table(headers, rows), unsafe_allow_html=True)
        st.caption("mAP@50 computed separately on each language's test pages. Languages with few pages give less reliable numbers.")
    else:
        st.info("Per-language results will appear after evaluation.")


# ══════════════════════════════════════════════════════════════
#  TAB 5: ABOUT
# ══════════════════════════════════════════════════════════════
with tab_about:
    ab1, ab2 = st.columns([3, 2])

    with ab1:
        st.markdown("""
<div class="about-card">
    <h4>About IndicDoc AI</h4>
    <p><b>IndicDoc AI</b> is a deep learning system for multilingual Indian document
    layout understanding. It detects and classifies structural elements &mdash;
    paragraphs, tables, headers, footers, figures, and more &mdash; from scanned
    document images across 12 Indian languages.</p>

    <h4>Problem Statement</h4>
    <p>Indian documents exhibit significant diversity in scripts (Devanagari, Tamil, Bengali, etc.),
    layouts (newspapers, textbooks, forms), and quality (scanned, photographed, degraded).
    Automated understanding requires robust deep learning models that can handle this diversity.</p>

    <h4>Methodology</h4>
    <ol>
        <li><b>Dataset:</b> IndicDLP (AIKosh / Hugging Face) &mdash; 119,806 pages, 12 languages, 42 classes. The models here were trained on a language-balanced subset of it (see the Evaluation tab for the test-set size)</li>
        <li><b>Baseline:</b> Faster R-CNN with ResNet-50 FPN backbone</li>
        <li><b>Proposed:</b> YOLOv8s fine-tuned for document layout detection</li>
        <li><b>Evaluation:</b> mAP, Precision, Recall, F1-Score, IoU</li>
    </ol>

    <h4>Pipeline</h4>
    <p><code>Document Image &rarr; Preprocessing &rarr; Deep Learning Model &rarr;
    NMS &rarr; Bounding Boxes + Classes + Confidence &rarr; Visualization</code></p>

    <h4>SDG Alignment</h4>
    <p><b>SDG 9: Industry, Innovation and Infrastructure</b> &mdash; Enables AI-based
    document digitization, supports multilingual information processing, and contributes
    to intelligent document infrastructure for India.</p>
</div>
""", unsafe_allow_html=True)

    with ab2:
        st.markdown("""
<div class="about-card">
    <h4>Technology Stack</h4>
    <ul>
        <li>Python 3.11</li>
        <li>PyTorch + torchvision</li>
        <li>Ultralytics YOLOv8</li>
        <li>OpenCV</li>
        <li>Streamlit</li>
        <li>Albumentations</li>
        <li>scikit-learn</li>
        <li>matplotlib + seaborn</li>
    </ul>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="about-card">
    <h4>Links</h4>
    <ul>
        <li><a href="https://aikosh.indiaai.gov.in/home" target="_blank">AIKosh Dataset Platform</a></li>
        <li><a href="https://github.com/Prashantj44/IndicDoc-AI" target="_blank">GitHub Repository</a></li>
    </ul>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="about-card">
    <h4>Project Info</h4>
    <table style="width:100%; font-size:0.85rem;">
        <tr><td><b>Type</b></td><td>B.E. AI&ML Mini-Project</td></tr>
        <tr><td><b>Domain</b></td><td>Computer Vision / DL</td></tr>
        <tr><td><b>SDG</b></td><td>9 &mdash; Industry & Innovation</td></tr>
        <tr><td><b>Dataset</b></td><td>IndicDLP (AIKosh)</td></tr>
    </table>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-footer">
    <span class="gold">&loz;</span> &nbsp;
    <b>IndicDoc AI</b> &mdash; Deep Learning-Based Multilingual Indian Document Layout Understanding
    &nbsp; <span class="gold">&loz;</span><br>
    B.E. Artificial Intelligence & Machine Learning &bull; Deep Learning Mini-Project &bull; SDG 9
</div>
""", unsafe_allow_html=True)
