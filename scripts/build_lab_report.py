"""
IndicDoc AI - Build the Expt.-11 (SDG-based Deep Learning Mini Project) lab report.

The project as delivered is IndicDoc Verify: offline, layout-aware pre-print checking of question papers and their
translations in 12 Indian languages. Structure follows the Expt.-11 document (Aim, Lab Outcome, Methodology,
Expected Outcomes, Deliverables, Evaluation Criteria); the body follows the Phase-2 content list (16 sections);
tables, figures, citations and references follow IEEE style, with references mostly from 2021-2026. Every number is
read from files the pipeline wrote (dataset/*.csv, outputs/metrics); nothing is typed in by hand. The PDF is printed
with headless Chrome/Chromium.

Usage:
    python scripts/build_lab_report.py        # -> docs/report/IndicDoc_AI_Expt11_Lab_Report.pdf
    CHROME=/path/to/chrome python scripts/build_lab_report.py
"""
import html
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / 'docs' / 'report'
ASSETS = REPORT_DIR / 'assets'
METRICS = PROJECT_ROOT / 'outputs' / 'metrics'
FIGURES = PROJECT_ROOT / 'docs' / 'figures'
PDF_PATH = REPORT_DIR / 'IndicDoc_AI_Expt11_Lab_Report.pdf'

TITLE = ('IndicDoc Verify: Deep-Learning Layout Analysis for Pre-Print Checking of Indian Question Papers '
         'and Their Translations')
TEAM = [('Riddhi Patil', 39), ('Mariam Badure', 40), ('Prashant Jha', 41), ('Aditya Soni', 42)]
GUIDE = 'Dr. Joanne Gomes'
REPO = 'https://github.com/va-ada/IndicDoc-AI'
MODEL_NAMES = {'authors': 'DocLayout-YOLO (IndicDLP authors)', 'proposed': 'YOLOv8s (ours)',
               'baseline': 'Faster R-CNN (ours)'}
Q_LABELS = ['first-level-question', 'second-level-question', 'third-level-question', 'options', 'answer']


# ----------------------------------------------------------------------------------------------- helpers
def uri(path: Path) -> str:
    return Path(path).resolve().as_uri()


def roman(n: int) -> str:
    vals = [(10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]
    out = ''
    for v, s in vals:
        while n >= v:
            out, n = out + s, n - v
    return out


class Numbering:
    """IEEE numbering: tables in Roman numerals (caption above), figures in Arabic numerals (caption below)."""

    def __init__(self):
        self.tables = 0
        self.figures = 0

    def table(self, caption: str, head: list, rows: list, align: str = None, width: str = '100%',
              cls: str = '') -> str:
        self.tables += 1
        align = align or 'l' + 'r' * (len(head) - 1)
        css = {'l': 'left', 'r': 'right', 'c': 'center'}
        th = ''.join(f'<th style="text-align:{css[a]}">{h}</th>' for h, a in zip(head, align))
        body = ''
        for r in rows:
            if r == '---':
                body += '<tr class="mid"><td colspan="%d"></td></tr>' % len(head)
                continue
            body += '<tr>' + ''.join(f'<td style="text-align:{css[a]}">{c}</td>' for c, a in zip(r, align)) + '</tr>'
        return (f'<div class="tbl {cls}"><table style="width:{width}"><caption class="tcap"><span class="tno">Table '
                f'{roman(self.tables)}</span><br><span class="ttl">{caption}</span></caption><thead><tr>{th}</tr>'
                f'</thead><tbody>{body}</tbody></table></div>')

    def figure(self, content: str, caption: str, cls: str = '') -> str:
        self.figures += 1
        return (f'<figure class="{cls}">{content}<figcaption><span class="fno">Fig. {self.figures}.</span> '
                f'{caption}</figcaption></figure>')


def img(path: Path, width: str = '100%', extra: str = '') -> str:
    return f'<img src="{uri(path)}" style="width:{width};{extra}" alt="">'


def f3(x: float) -> str:
    return f'{x:.3f}'


# ------------------------------------------------------------------------------------------------- data
def load():
    m = {k: json.loads((METRICS / f'{k}_metrics.json').read_text()) for k in MODEL_NAMES}
    study = json.loads((METRICS / 'study_summary.json').read_text())
    ocr = json.loads((METRICS / 'ocr_evaluation.json').read_text())
    tests = json.loads((METRICS / 'test_suite.json').read_text())
    langs = pd.read_csv(PROJECT_ROOT / 'dataset' / 'language_split.csv')
    classes = pd.read_csv(PROJECT_ROOT / 'dataset' / 'class_counts.csv')
    base_hist = json.loads((METRICS / 'baseline_training_history.json').read_text())
    yolo_hist = pd.read_csv(METRICS / 'proposed_training_history.csv')
    yolo_hist.columns = [c.strip() for c in yolo_hist.columns]
    return m, study, ocr, tests, langs, classes, base_hist, yolo_hist


# ------------------------------------------------------------------------------------------------- data
def jload(name: str) -> dict:
    p = METRICS / name
    return json.loads(p.read_text()) if p.exists() else {}


def load():
    m = {k: json.loads((METRICS / f'{k}_metrics.json').read_text()) for k in MODEL_NAMES}
    study = json.loads((METRICS / 'study_summary.json').read_text())
    ocr = json.loads((METRICS / 'ocr_evaluation.json').read_text())
    tests = json.loads((METRICS / 'test_suite.json').read_text())
    langs = pd.read_csv(PROJECT_ROOT / 'dataset' / 'language_split.csv')
    classes = pd.read_csv(PROJECT_ROOT / 'dataset' / 'class_counts.csv')
    base_hist = json.loads((METRICS / 'baseline_training_history.json').read_text())
    yolo_hist = pd.read_csv(METRICS / 'proposed_training_history.csv')
    yolo_hist.columns = [c.strip() for c in yolo_hist.columns]
    return m, study, ocr, tests, langs, classes, base_hist, yolo_hist


def training_figure(base_hist: dict, yolo_hist: pd.DataFrame, out: Path) -> Path:
    """Loss curves of both trained models and the YOLOv8s validation mAP per epoch (measured)."""
    plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 9})
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.25))
    ep = range(1, len(base_hist['train_losses']) + 1)
    ax[0].plot(ep, base_hist['train_losses'], 'o-', color='black', label='train')
    ax[0].plot(ep, base_hist['val_losses'], 's--', color='0.45', label='validation')
    ax[0].set_title('(a) Faster R-CNN loss')
    parts = ['box_loss', 'cls_loss', 'dfl_loss']
    ax[1].plot(yolo_hist['epoch'], yolo_hist[[f'train/{p}' for p in parts]].sum(axis=1), 'o-', color='black',
               ms=3, label='train')
    ax[1].plot(yolo_hist['epoch'], yolo_hist[[f'val/{p}' for p in parts]].sum(axis=1), 's--', color='0.45', ms=3,
               label='validation')
    ax[1].set_title('(b) YOLOv8s loss (box+cls+dfl)')
    ax[2].plot(yolo_hist['epoch'], yolo_hist['metrics/mAP50(B)'], 'o-', color='black', ms=3, label='mAP@50')
    ax[2].plot(yolo_hist['epoch'], yolo_hist['metrics/mAP50-95(B)'], 's--', color='0.45', ms=3, label='mAP@50:95')
    ax[2].set_title('(c) YOLOv8s validation mAP')
    from matplotlib.ticker import MaxNLocator
    for a, yl in zip(ax, ['loss', 'loss', 'mAP']):
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
        a.set_xlabel('epoch')
        a.set_ylabel(yl)
        a.grid(alpha=0.3, lw=0.5)
        a.legend(fontsize=7, frameon=False)
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------------------------- diagrams
def svg_box(x, y, w, h, title, sub='', fill='#F4F4F4', bold=True):
    t = html.escape(title)
    s = html.escape(sub)
    cy = y + h / 2 - (6 if sub else -4)
    out = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="{fill}" stroke="#222" stroke-width="0.9"/>'
    out += (f'<text x="{x + w / 2}" y="{cy}" text-anchor="middle" font-size="10" '
            f'font-weight="{700 if bold else 400}">{t}</text>')
    for i, line in enumerate(s.split('|') if s else []):
        out += f'<text x="{x + w / 2}" y="{cy + 13 + 11 * i}" text-anchor="middle" font-size="8.6">{line}</text>'
    return out


def arrow(x1, y1, x2, y2, dashed=False):
    d = ' stroke-dasharray="4 3"' if dashed else ''
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#222" stroke-width="1" marker-end="url(#ah)"{d}/>'


SVG_DEFS = ('<defs><marker id="ah" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">'
            '<path d="M0,0 L7,3 L0,6 z" fill="#222"/></marker></defs>')


def block_diagram() -> str:
    """IndicDoc Verify: two lanes of processing (paper, source version) meeting in the checks."""
    W, bw, bh, gap = 684, 118, 66, 20
    xs = [6 + i * (bw + gap) for i in range(5)]
    y0, y1, y2 = 4, 74, 190
    g = [f'<svg viewBox="0 0 {W} 272" xmlns="http://www.w3.org/2000/svg" font-family="CMU Serif, serif">', SVG_DEFS]
    g.append(svg_box(xs[0], y0, bw, 46, 'Streamlit / REST API', 'Verify Paper · /verify', '#FFFFFF'))
    g.append(svg_box(xs[1], y0, bw, 46, 'Source version', 'optional, any language', '#FFFFFF'))
    g.append(svg_box(xs[2], y0, bw, 46, 'Question bank', 'earlier papers', '#FFFFFF'))
    g.append(svg_box(xs[3], y0, bw, 46, 'Format profile', 'JSON rules', '#FFFFFF'))
    top = [('Paper PDF / scans', 'rendered at 200 dpi|12 languages'), ('Layout detector', 'DocLayout-YOLO|questions, options'),
           ('Indic OCR', 'Tesseract, the page|script\'s own model'), ('Number pass', 'English model; digits|must agree'),
           ('Build questions', 'number, stem, options|(by layout gaps), marks')]
    for i, (t, s) in enumerate(top):
        g.append(svg_box(xs[i], y1, bw, bh, t, s, '#E3E7F2' if i == 1 else '#F4F4F4'))
        if i:
            g.append(arrow(xs[i - 1] + bw, y1 + 33, xs[i], y1 + 33))
    g.append(arrow(xs[0] + bw / 2, y0 + 46, xs[0] + bw / 2, y1, dashed=True))
    g.append(arrow(xs[1] + bw / 2, y0 + 46, xs[1] + bw / 2, y1, dashed=True))
    bottom = [('Report', 'verdict, findings|pages outlined, HTML/JSON'), ('Format check', 'required elements,|order, bilingual'),
              ('Repeat check', 'similar to an|earlier question'), ('Parity check', 'align versions; numbers,|options, marks'),
              ('Structure check', 'numbering, duplicates,|options, marks total')]
    for i, (t, s) in enumerate(bottom):
        g.append(svg_box(xs[i], y2, bw, bh, t, s, '#F4F4F4'))
    g.append(arrow(xs[4] + bw / 2, y1 + bh, xs[4] + bw / 2, y2))
    for i in range(4, 0, -1):
        g.append(arrow(xs[i], y2 + 33, xs[i - 1] + bw, y2 + 33))
    g.append(f'<rect x="{xs[0]}" y="{y2 + bh + 8}" width="12" height="10" fill="#E3E7F2" stroke="#222" stroke-width="0.7"/>')
    g.append(f'<text x="{xs[0] + 17}" y="{y2 + bh + 17}" font-size="9" font-style="italic">deep-learning stage; '
             'the source version takes the same path; dashed: optional inputs</text>')
    g.append('</svg>')
    return ''.join(g)


def architecture_diagram(params: dict) -> str:
    rows = [
        (f"DocLayout-YOLO [11]  ({params['authors']:.1f} M params, used by IndicDoc Verify)",
         [('Input', '1024×1024×3'), ('Backbone', 'CSP + GL-CRM|global-to-local'), ('Neck', 'PAN pyramid|3 scales'),
          ('Head', 'YOLOv10-style|NMS-free'), ('Output', 'N × (box, 42 scores)')]),
        (f"YOLOv8s [13]  ({params['proposed']:.1f} M params, fine-tuned here)",
         [('Input', '640×640×3'), ('Backbone', 'CSPDarknet|C2f + SPPF'), ('Neck', 'PAN-FPN|3 scales'),
          ('Head', 'decoupled|anchor-free'), ('Output', 'boxes + 42 classes')]),
        (f"Faster R-CNN [12]  ({params['baseline']:.1f} M params, fine-tuned here)",
         [('Input', 'short side 640–1024'), ('Backbone', 'ResNet-50|frozen BN'), ('Neck', 'FPN|5 levels'),
          ('RPN + RoIAlign', 'region proposals'), ('Box head', '43 outputs|(42 + background)')]),
    ]
    W, bw, bh, gap = 684, 118, 40, 20
    g = [f'<svg viewBox="0 0 {W} {len(rows) * 78}" xmlns="http://www.w3.org/2000/svg" '
         f'font-family="CMU Serif, serif">', SVG_DEFS]
    for r, (label, blocks) in enumerate(rows):
        y = r * 78 + 18
        g.append(f'<text x="6" y="{y - 5}" font-size="10" font-weight="700">{html.escape(label)}</text>')
        for i, (t, s) in enumerate(blocks):
            x = 6 + i * (bw + gap)
            g.append(svg_box(x, y, bw, bh + 8, t, s, '#E3E7F2' if 0 < i < 4 else '#F4F4F4'))
            if i:
                g.append(arrow(x - gap, y + 24, x, y + 24))
    g.append('</svg>')
    return ''.join(g)


# ------------------------------------------------------------------------------------------------- report
REFS = [
    # literature survey (Table II)
    'O. Nath, S. Kukkala, M. Khapra, and R. K. Sarvadevabhatla, &ldquo;IndicDLP: A foundational dataset for '
    'multi-lingual and multi-domain document layout parsing,&rdquo; in <i>Proc. Int. Conf. Document Analysis and '
    'Recognition (ICDAR)</i>, 2025, pp. 23&ndash;39.',
    'B. V. Kasuba <i>et al.</i>, &ldquo;PLATTER: A page-level handwritten text recognition system for Indic '
    'scripts,&rdquo; arXiv:2502.06172, 2025.',
    'A. Mondal <i>et al.</i>, &ldquo;UniTabBank: A large scale multi-lingual, multi-layout, multi-type, multi-format '
    'dataset for table detection,&rdquo; in <i>Proc. IEEE/CVF Winter Conf. Applications of Computer Vision (WACV)</i>, 2026.',
    'V. Pal, E. Kanoulas, A. Yates, and M. de Rijke, &ldquo;Table question answering for low-resourced Indic '
    'languages,&rdquo; arXiv:2410.03576, 2024.',
    'Y. Xu <i>et al.</i>, &ldquo;LayoutXLM: Multimodal pre-training for multilingual visually-rich document '
    'understanding,&rdquo; arXiv:2104.08836, 2021.',
    'B. V. Kasuba, P. Chaudhuri, and G. Ramakrishnan, &ldquo;DRISHTIKON: Visual grounding at multiple granularities '
    'in documents,&rdquo; arXiv:2506.21316, 2025.',
    'S. Bhattacharyya <i>et al.</i>, &ldquo;Adapting vision-language models for Hindi OCR,&rdquo; IIIT Hyderabad, 2025.',
    'A. R. Dumada and S. G. Shah, &ldquo;Incorporate structure and content for Devanagari table extraction,&rdquo; '
    '<i>Int. J. Sci. Adv. Technol. (IJSAT)</i>, vol. 17, no. 1, 2026.',
    'M. Babayeva, J. Dunlap, M. Sn&#283;tinov&aacute;, and R. Widenhorn, &ldquo;Translating the Force Concept '
    'Inventory in the age of AI,&rdquo; arXiv:2508.13908, 2025.',
    'R. Rei <i>et al.</i>, &ldquo;CometKiwi: IST-Unbabel 2022 submission for the quality estimation shared '
    'task,&rdquo; in <i>Proc. 7th Conf. Machine Translation (WMT)</i>, 2022, pp. 634&ndash;645.',
    # models and tools
    'Z. Zhao, H. Kang, B. Wang, and C. He, &ldquo;DocLayout-YOLO: Enhancing document layout analysis through '
    'diverse synthetic data and global-to-local adaptive perception,&rdquo; arXiv:2410.12628, 2024.',
    'S. Ren, K. He, R. Girshick, and J. Sun, &ldquo;Faster R-CNN: Towards real-time object detection with region '
    'proposal networks,&rdquo; in <i>Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)</i>, 2015, pp. 91&ndash;99.',
    'G. Jocher, A. Chaurasia, and J. Qiu, &ldquo;Ultralytics YOLOv8,&rdquo; 2023. [Online]. Available: '
    '<span class="tt">https://github.com/ultralytics/ultralytics</span>',
    'Y. Chen <i>et al.</i>, &ldquo;RoDLA: Benchmarking the robustness of document layout analysis models,&rdquo; in '
    '<i>Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)</i>, 2024, pp. 15556&ndash;15566.',
    'J. Ansel <i>et al.</i>, &ldquo;PyTorch 2: Faster machine learning through dynamic Python bytecode '
    'transformation and graph compilation,&rdquo; in <i>Proc. ACM ASPLOS</i>, 2024.',
    'S. Verma <i>et al.</i>, &ldquo;MILU: A multi-task Indic language understanding benchmark,&rdquo; in <i>Proc. '
    'NAACL</i>, 2025.',
    'AI4Bharat, &ldquo;IndicDLP dataset,&rdquo; Hugging Face, 2025. [Online]. Available: '
    '<span class="tt">https://huggingface.co/datasets/ai4bharat/indicdlp</span>',
    'O. Nath <i>et al.</i>, &ldquo;IndicDLP model checkpoints,&rdquo; Zenodo, record 15881917, 2025, CC-BY 4.0.',
    'Tesseract OCR, version 5, tessdata_best models, 2021&ndash;2025. [Online]. Available: '
    '<span class="tt">https://github.com/tesseract-ocr/tesseract</span>; tesseract.js 7, 2025.',
    'United Nations General Assembly, &ldquo;Transforming our world: The 2030 Agenda for Sustainable '
    'Development,&rdquo; Resolution A/RES/70/1, 2015.',
    # evidence of the problem (news and official records)
    '&ldquo;Madras HC orders CBSE to give 196 marks to Tamil medium students, for errors in 49 questions,&rdquo; '
    '<i>The News Minute</i>, Jul. 2018; Supreme Court of India, judgment in CBSE v. T. K. Rangarajan, Nov. 2018.',
    '&ldquo;NEET UG 2021: NTA to examine alleged error in Hindi translation of physics question,&rdquo; '
    '<i>Careers360</i>, 2021.',
    '&ldquo;When Kannada was lost in translation: KPSC forces a do-over,&rdquo; <i>The South First</i>, 2024; '
    '&ldquo;Conduct KAS prelims examination again,&rdquo; <i>ETV Bharat</i>, Mar. 2025.',
    '&ldquo;Supreme Court issues notice to Andhra Pradesh Public Service Commission over primacy to English '
    'translation despite holding test in Telugu,&rdquo; <i>India Legal</i>.',
    '&ldquo;NTA admits defects in 3 UGC-NET papers, orders re-tests in September,&rdquo; <i>The Week</i>, Aug. 2026; '
    '&ldquo;UGC-NET re-exam for English, Commerce, Sociology on September 9-10,&rdquo; <i>SCC Online</i>, Aug. 2026.',
    '&ldquo;HC grants relief to MBBS student, directs NTRUHS to give 1 compensatory mark for question paper '
    'error,&rdquo; <i>Medical Dialogues</i>, Jul. 2026.',
    'Rajya Sabha, Unstarred Question 2356 (AI tools in the Supreme Court e-filing system), Mar. 2025; DAKSH, '
    '&ldquo;The silent gatekeeper,&rdquo; 2025.',
    f'P. Jha, A. Soni, R. Patil, and M. Badure, &ldquo;IndicDoc AI,&rdquo; source code, 2026. [Online]. Available: '
    f'<span class="tt">{REPO}</span>',
]


def build_html(tmp: Path) -> str:
    m, study, ocr, tests, langs, classes, base_hist, yolo_hist = load()
    A, P, B = m['authors'], m['proposed'], m['baseline']
    V, VN = jload('verify_benchmark_oracle.json'), jload('verify_benchmark_noisy.json')
    V_NP, V_NG = jload('verify_benchmark_oracle_no_number_pass.json'), jload('verify_benchmark_oracle_no_geometry.json')
    VP = jload('verify_benchmark_oracle_posthoc.json')
    N = Numbering()

    pages = {s: int(langs[s].sum()) for s in ('train', 'val', 'test')}
    boxes = {s: int(classes[s].sum()) for s in ('train', 'val', 'test')}
    total_pages, total_boxes = sum(pages.values()), sum(boxes.values())
    para_share = classes.loc[classes.class_name == 'paragraph', 'total'].iloc[0] / total_boxes
    cnt = dict(zip(classes.class_name, classes.total))
    pages_csv = pd.read_csv(PROJECT_ROOT / 'dataset' / 'pages.csv')
    qp_pages = int((pages_csv.domain == 'question papers').sum())
    params = {k: m[k]['parameters'] / 1e6 for k in m}
    b_minutes = base_hist['epoch_end_seconds'][-1] / 60
    y_minutes = float(yolo_hist['time'].iloc[-1]) / 60
    y_epochs, b_epochs = len(yolo_hist), len(base_hist['train_losses'])
    qap = {k: A['per_class_ap50'].get(k, 0) for k in Q_LABELS}
    vo, vq, vl = V['overall'], V['by_quality'], V['by_language']
    cl, dg = vq['clean_render'], vq['scan_degraded']
    n_pairs, n_err = vo['pairs'], vo['errors_injected']
    labels = V['labels']

    loss_png = training_figure(base_hist, yolo_hist, tmp / 'fig_training.png')
    F = FIGURES
    PL = PROJECT_ROOT / 'outputs' / 'plots'
    SC = PROJECT_ROOT / 'docs' / 'screenshots'

    def pct(x):
        return f'{100 * x:.0f}%'

    s = []
    a = s.append

    # ------------------------------------------------------------------ front matter (Expt.-11 structure)
    team_rows = ''.join(f'<tr><td>{n}</td><td>{r}</td></tr>' for n, r in TEAM)
    name, subtitle = TITLE.split(': ', 1)
    a(f'''
<div class="title">{name}</div>
<div class="subtitle">{subtitle}</div>
<div class="expt">Expt.-11: SDG-Based Deep Learning Mini Project</div>
<div class="rule"></div>

<table class="meta">
<tr><th>Team</th><td><table class="team"><tr><th>Name</th><th>Roll No.</th></tr>{team_rows}</table></td></tr>
<tr><th>Project guide</th><td>{GUIDE}</td></tr>
<tr><th>SDGs addressed</th><td>SDG 4 (Quality Education), SDG 9 (Industry, Innovation and Infrastructure),
SDG 10 (Reduced Inequalities), SDG 16 (Peace, Justice and Strong Institutions)</td></tr>
<tr><th>Code repository</th><td><span class="tt">{REPO.replace('https://', '')}</span></td></tr>
<tr><th>Date of report</th><td>{date.today().strftime('%d %B %Y')}</td></tr>
</table>

<h2>Aim</h2>
<p>To design, implement, and analyze a Deep Learning solution addressing a real-world problem aligned with the
Sustainable Development Goals (SDGs).</p>

<h2>Lab Outcome</h2>
<p>Students will be able to apply Deep Learning techniques for real world problem-solving and evaluate model
performance.</p>

<h2>Methodology</h2>
<p>The project is carried out in multiple phases to ensure systematic development and implementation:</p>
<ol class="phases">
<li><b>Topic Selection (SDG Mapping):</b> Identify a real-world problem aligned with Sustainable Development Goals
(SDGs). <i>Done:</i> errors in Indian exam papers and their translations, which reach lakhs of candidates
(Section 1).</li>
<li><b>Phase-1: Literature Survey:</b> Review 4&ndash;8 research papers and identify the research gap.
<i>Done:</i> ten papers from 2021&ndash;2026 reviewed, four gaps identified (Sections 2&ndash;3).</li>
<li><b>Phase-2: Final Project Implementation with Performance Analysis:</b> Develop system architecture, collect
dataset, and implement DL model. Analyze results, improve performance, design front end, integrate the system.
Demonstrate the finished project and submit the code along with final working project. <i>Done:</i>
Sections 4&ndash;16.</li>
<li><b>Documentation:</b> Prepare a research paper in standard format. <i>Done:</i> IEEE conference-format paper
in <span class="tt">paper/</span>.</li>
</ol>

<div class="abstract"><b><i>Abstract</i></b>&mdash;Indian exam papers are set in one language and translated into
Hindi and regional languages; every version is checked by hand, and errors keep reaching candidates: 49 wrong
questions in the Tamil NEET 2018 paper, 59 errors and a re-exam in the 2024 Karnataka civil-services paper, re-tests
of three UGC-NET papers in 2026. We present IndicDoc&nbsp;Verify, an offline system that reads a printed paper in
any of 12 Indian languages and flags structural and numerical errors, and every difference from its source version,
before printing. A deep-learning layout detector finds the questions, sub-questions, options and answers on each
page (DocLayout-YOLO, AP@50 {min(qap.values()):.2f}&ndash;{max(qap.values()):.2f} on these labels, chosen after
comparing it with a YOLOv8s and a Faster R-CNN that we fine-tuned); Tesseract reads the text with an English
<i>number pass</i> that must agree on every digit; the questions are then checked for numbering, duplicates,
options, marks and repeats of earlier papers, and two language versions are compared question by question without
any translation. On a benchmark of {n_pairs} English&ndash;Hindi and English&ndash;Tamil paper pairs with injected
errors, read with real OCR, the system finds {pct(cl['found_at_place'])} of the errors at the right question on
born-digital papers ({pct(dg['found_at_place'])} on scan-degraded pages) with
{cl['false_alarms_per_clean_pair']:.1f} false alarms per clean pair. It ships as a web app, REST API and offline
Docker image.</div>
<p class="kw"><b><i>Index Terms</i></b>&mdash;document layout analysis, Indian languages, object detection, OCR,
question paper verification, translation quality, examinations.</p>
''')

    # ------------------------------------------------------------------ 1. SDG mapping
    a('<h2>1. Topic Selection and SDG Mapping</h2>')
    a('<p><b>Problem.</b> National and state exams in India are bilingual or multilingual: NEET-UG is offered in 13 '
      'languages, and UPSC, the state Public Service Commissions, school boards and universities print papers in '
      'English and a regional language. Each version is checked by hand before printing. When the check fails, a '
      'number, an option, a mark or a whole question differs between versions, and candidates who chose a regional '
      'language are judged on a different paper. Table I lists recent cases; each ended in grace marks, a re-exam or '
      'a court case. No tool checks a paper and its translations automatically, and a confidential pre-print paper '
      'cannot be sent to a cloud service. The mapping to the SDG targets of the 2030 Agenda [20] is given in Table II.</p>')
    a(N.table('Recent errors in Indian exam papers', ['Exam', 'What went wrong', 'Consequence', 'Ref.'], [
        ['NEET-UG 2018', 'Errors in 49 questions of the Tamil version', 'HC ordered 196 grace marks; Supreme Court set '
         'it aside and asked for bilingual subject experts', '[21]'],
        ['NEET-UG 2021', 'A term dropped in the Hindi version of a physics question', 'NTA agreed to examine it before '
         'the Supreme Court', '[22]'],
        ['KPSC KAS 2024', '59 errors in the Kannada paper; machine translation suspected', 'Re-exam for about 2 lakh '
         'candidates, errors again', '[23]'],
        ['APPSC Group-I', '51 of 120 questions alleged mistranslated into Telugu', 'Supreme Court notice', '[24]'],
        ['UGC-NET Jun 2026', 'Factual, typographical and translation errors; questions repeated from earlier papers',
         'Re-test of three papers on 9&ndash;10 Sep 2026', '[25]'],
        ['MBBS 2026 (NTRUHS)', 'The same MCQ printed twice', 'High Court ordered a compensatory mark', '[26]'],
    ], align='llll', cls='split small'))
    a(N.table('Mapping of the project to the Sustainable Development Goals', ['SDG', 'Target', 'How IndicDoc Verify contributes'], [
        ['SDG 4 Quality Education', '4.3 equal access to tertiary education; 4.5 equal access for the vulnerable',
         'Entrance and board exams in which a regional-language candidate gets the same paper as an English one'],
        ['SDG 9 Industry, Innovation &amp; Infrastructure', '9.5 research and technological capability',
         'Open, offline document-AI for 12 Indian languages, with a measured benchmark'],
        ['SDG 10 Reduced Inequalities', '10.2 inclusion irrespective of ethnicity or origin',
         'Removes a disadvantage that falls only on candidates writing in Indian languages'],
        ['SDG 16 Strong Institutions', '16.6 effective, accountable institutions; 16.10 access to information',
         'Exam bodies catch errors before they become litigation; bilingual circulars checked for compliance'],
    ], align='lll', cls='split small'))

    # ------------------------------------------------------------------ 2. literature
    a('<h2>2. Literature Survey Summary Table</h2>')
    a('<p>Ten papers from 2021&ndash;2026 were reviewed (Table III): Indian-language document layout and recognition '
      '[1]&ndash;[8], from Phase-1, and two works on checking translations [9], [10], added when the project turned to '
      'exam papers.</p>')
    lit = [
        ['[1]', 'Nath <i>et al.</i>, 2025', 'Layout parsing of Indian-language pages', 'IndicDLP: 119,806 pages, 12 languages, 42 labels incl. questions and options',
         'YOLOv10x, DocLayout-YOLO, RoDLA, DINO, DiT, Florence-2', 'Regions only; question labels not used to check papers'],
        ['[2]', 'Kasuba <i>et al.</i>, 2025', 'Page-level handwritten Indic text recognition', 'CHIPS, ~25K pages, 10 languages',
         'DBNet word detection + 6 recognisers', 'User must name the language; text only'],
        ['[3]', 'Mondal <i>et al.</i>, 2026', 'Multilingual table detection', 'UniTabBank: 55K pages, 28 languages',
         'Semi-automatic labelling; YOLOv11', 'Borderless tables and degraded scans fail'],
        ['[4]', 'Pal <i>et al.</i>, 2024', 'Table QA for low-resource Indic languages', 'BanglaTabQA 2M, HindiTabQA 643K',
         'Fine-tuned mBART / M2M / Llama', 'Wikipedia tables only'],
        ['[5]', 'Xu <i>et al.</i>, 2021', 'Multilingual document understanding', '30M documents; XFUND, 7 languages',
         'LayoutXLM: text + layout + image', 'No Indian language'],
        ['[6]', 'Kasuba <i>et al.</i>, 2025', 'Grounding answers on the page', 'MGVG: 70 circulars, 509 QA pairs',
         'OCR + Llama-3.1-8B + region matching', 'OCR-dependent; single page'],
        ['[7]', 'Bhattacharyya <i>et al.</i>, 2025', 'Hindi OCR with vision-language models', '1.1M printed Hindi words',
         'LoRA-tuned GOT-OCR 2.0', 'Hindi only; mixed-language lines hard'],
        ['[8]', 'Dumada &amp; Shah, 2026', 'Devanagari table extraction', '500+ documents',
         'Structure + Tesseract, weighted fusion', 'No ablation'],
        ['[9]', 'Babayeva <i>et al.</i>, 2025', 'Translating a test into other languages with an LLM', 'Force Concept Inventory',
         'GPT-4o + back-translation + human review', 'Every item still checked by hand; no page structure'],
        ['[10]', 'Rei <i>et al.</i>, 2022', 'Reference-free translation quality estimation', 'WMT 2022 QE task',
         'CometKiwi: multilingual encoder scores sentences', 'Plain sentences: no numbers, options or marks'],
    ]
    a(N.table('Literature survey summary', ['Ref', 'Author', 'Problem', 'Dataset', 'Method', 'Limitation'], lit,
              align='llllll', cls='split small lit'))

    # ------------------------------------------------------------------ 3. gaps
    a('<h2>3. Research Gaps</h2>')
    a('<ol class="gaps">'
      '<li><b>G1: No automatic pre-print check.</b> Papers are translated into up to 13 languages and vetted by hand; '
      'the cases in Table I show that this step fails.</li>'
      '<li><b>G2: Translation tools ignore the page.</b> Quality estimation [10] and LLM back-translation [9] work on '
      'plain sentences; they cannot see question numbers, options, marks or figures, they need clean text, and they '
      'run in the cloud.</li>'
      '<li><b>G3: Layout models are not used for checking.</b> IndicDLP [1] labels questions, options and answers in '
      '12 languages, but detected structure has not been used to verify a document.</li>'
      '<li><b>G4: Numbers are unreliable in Indic OCR.</b> Indic models misread Latin digits in mixed text [7]; a '
      'checker of numbers needs every digit right.</li></ol>')
    a('<p><b>Choice of problem.</b> We also examined court-filing scrutiny, where AI defect detection is already '
      'deployed (the Supreme Court with IIT Madras, and a commercial tool in 16 High Courts [27]), and chose instead '
      'the problem that no tool addresses yet.</p>')

    # ------------------------------------------------------------------ 4. problem statement
    a('<h2>4. Problem Statement and Objectives</h2>')
    a('<p><b>Problem statement.</b> Given the pages of a question paper in any of 12 Indian languages, and optionally '
      'its source-language version, find every structural or numerical error and every difference between the '
      'versions before the paper is printed, and show where each one is.</p>')
    a('<p><b>Main objective.</b> Build an offline deep-learning system that verifies a question paper and its '
      'translations, and measure how many real kinds of error it finds and how many false alarms it raises.</p>')
    a('<p><b>Specific objectives.</b></p><ol class="obj">'
      '<li>Detect question-paper structure with a layout detector; fine-tune two detectors and compare them with the '
      'released IndicDLP model.</li>'
      '<li>Read text in 12 languages with reliable numbers (number pass).</li>'
      '<li>Implement structure, translation-parity, repeat and format checks with a report that points to the page.</li>'
      '<li>Evaluate on a benchmark with injected errors; deliver a web app, REST API and Docker image.</li></ol>')
    a(N.table('Research questions', ['RQ', 'Question'], [
        ['RQ1', 'How well do deep-learning layout detectors recover the structure of Indian question papers?'],
        ['RQ2', 'What share of real kinds of error does layout-aware checking find, at what false-alarm rate?'],
        ['RQ3', 'What do the number pass and layout-based option splitting contribute?'],
        ['RQ4', 'How robust is it to scanned pages and to layout-model mistakes?'],
    ], align='ll', cls='small'))
    a('<p><b>Expected output.</b> A verdict (ready / check / do not print), each finding with its question and page, '
      'the pages with every problem outlined, as HTML and JSON.</p>')

    # ------------------------------------------------------------------ 5. proposed system
    a('<h2>5. Proposed System, Methodology and Novelty</h2>')
    a('<p><b>Workflow (input to output).</b></p><ol class="steps">'
      '<li><b>Input:</b> the paper as PDF (rendered at 200 dpi) or scans; optionally its source version, a question '
      'bank of earlier papers and a format profile.</li>'
      '<li><b>Find the structure:</b> DocLayout-YOLO [11] detects questions, sub-questions, options, answers, figures, '
      'tables and page furniture on every page (high-recall threshold chosen on validation pages).</li>'
      '<li><b>Read the text:</b> Tesseract [19] reads the page with its own script\'s model; an English <i>number '
      'pass</i> re-reads it and fixes the Latin digits the Indic model misread. Words go to the region that contains '
      'them.</li>'
      '<li><b>Build the questions:</b> a question region starts a question; options are separated by the gaps between '
      'words (layout), so garbled option markers do not matter; marks, figures, tables and formulas are attached.</li>'
      '<li><b>Check:</b> numbering, duplicates, options, marks total; repeats of earlier questions; parity with the '
      'source version; format rules.</li>'
      '<li><b>Report:</b> verdict and findings, each outlined on its page.</li></ol>')
    a('<p><b>How deep learning is used.</b> The structure of a page is found by a convolutional object detector '
      'fine-tuned on Indian-language pages (transfer learning from COCO and synthetic documents). Its regions, and '
      'the word positions inside them, turn a picture of a page in any script into structured questions; the checks '
      'then work on that structure.</p>')
    a('<p><b>Novelty.</b> (i) <i>Translation parity without translation:</i> numbers, option counts, marks, '
      'sub-questions and figures are the same in every language, so two versions are aligned question by question '
      '(Needleman&ndash;Wunsch on these language-independent features) and compared without a translation model. '
      '(ii) <i>Layout-aware checking:</i> detected question and option regions plus word geometry, instead of plain '
      'text. (iii) <i>Number pass:</i> two OCR models must agree on every digit, with page-level detection of '
      'whether the paper prints Latin or Indian digits. (iv) <i>Offline and general:</i> one Docker image, 12 '
      'languages, and JSON format profiles for other documents (e.g. bilingual government circulars).</p>')

    # ------------------------------------------------------------------ 6. block diagram
    a('<h2>6. System Block Diagram</h2>')
    a('<p>Fig. 1 shows the components and the flow of data. The app, the API and the benchmark run the same code.</p>')
    a(N.figure(block_diagram(), 'System block diagram of IndicDoc Verify. The paper flows left to right through '
               'the deep-learning layout stage and OCR, then right to left through the checks to the report. The '
               'question bank feeds the repeat check and the profile the format check.'))

    # ------------------------------------------------------------------ 7. dataset
    a('<h2>7. Dataset Description and Pre-processing</h2>')
    a(f'<p><b>Layout data.</b> IndicDLP [1], [17]: 119,806 human-annotated pages, 12 languages, 12 domains, 42 labels, '
      f'including first-, second- and third-level questions, options and answers. Our language-balanced subset has '
      f'{total_pages:,} pages and {total_boxes:,} boxes from the <i>official</i> splits (Table V); {qp_pages} pages are '
      f'question papers, with {cnt.get("first-level-question", 0):,} question and {cnt.get("options", 0):,} option '
      f'boxes. It is released as CSV files in <span class="tt">dataset/</span>.</p>')
    rows = [[r.language.capitalize(), r.train, r.val, r.test, r.total] for r in langs.itertuples()]
    rows += ['---', ['<b>Total pages</b>', f"<b>{pages['train']:,}</b>", f"<b>{pages['val']}</b>",
                     f"<b>{pages['test']}</b>", f'<b>{total_pages:,}</b>'],
             ['Labelled boxes', f"{boxes['train']:,}", f"{boxes['val']:,}", f"{boxes['test']:,}", f'{total_boxes:,}']]
    a(N.table('IndicDLP subset used for the layout models (pages per split)', ['Language', 'Train', 'Validation', 'Test', 'Total'],
              rows, width='72%', cls='small'))
    by_type_n = {t: V['by_type'][t]['errors_injected'] for t in labels}
    a(f'<p><b>Verification benchmark.</b> No exam body publishes erroneous pre-print papers, so we built one. Each of '
      f'the {n_pairs} pairs is an English source and its Hindi or Tamil version: 18 questions (15 MCQs from numeric '
      f'templates and fixed questions, 3 short-answer questions with marks), set like a state-board model paper and '
      f'rendered at 200 dpi with Noto fonts. {n_err} pairs carry one injected error of the kinds in Table I (Table VI); '
      f'{n_pairs - n_err} are clean, to count false alarms. Half of the pairs are degraded like scans (Gaussian blur '
      f'0.7 px, noise &sigma; = 8, JPEG quality 55). A separate <i>development</i> set (48 pairs, other random seed) '
      f'was used to build the parser; the test set was run once. Fig. 2 shows sample pages.</p>')
    a(N.table('Injected errors in the verification benchmark (test set)', ['Error', 'How it is made', 'Pairs'], [
        [labels['number_changed'], 'one value in a translated stem changed', by_type_n['number_changed']],
        [labels['option_number_changed'], 'one option value changed in the translation', by_type_n['option_number_changed']],
        [labels['option_dropped'], 'one option removed from the translation', by_type_n['option_dropped']],
        [labels['question_dropped'], 'a question removed (numbers renumbered or left with a gap)', by_type_n['question_dropped']],
        [labels['question_duplicated'], 'an MCQ printed again in place of another (both versions)', by_type_n['question_duplicated']],
        [labels['marks_changed'], 'marks of a short-answer question changed in the translation', by_type_n['marks_changed']],
        [labels['total_wrong'], 'maximum marks changed in the translation', by_type_n['total_wrong']],
        [labels['numbering_error'], 'a question printed with the previous number', by_type_n['numbering_error']],
        [labels['bank_repeat'], 'a question copied from a "previous year" paper (both versions)', by_type_n['bank_repeat']],
    ], align='llr', cls='small'))
    grid = ''.join(f'<div>{img(F / f"bench_{k}.jpg")}<span>{c}</span></div>' for k, c in
                   (('en', 'English source'), ('hi', 'Hindi version'), ('ta', 'Tamil version, scan-degraded')))
    grid += f'<div>{img(PROJECT_ROOT / "dataset" / "samples" / "hindi_newspapers.jpg")}<span>IndicDLP page (Hindi)</span></div>'
    a(N.figure(f'<div class="grid4">{grid}</div>', 'Sample pages: the verification benchmark (left three) and IndicDLP.'))
    a(N.table('Pre-processing and augmentation', ['Step', 'What was done'], [
        ['Layout data', 'Parquet row groups streamed from Hugging Face; PNG + COCO JSON; boxes clipped, &lt;1 px boxes dropped; COCO &rarr; YOLO labels'],
        ['Resizing', 'Letterbox to 640 px (our models) or 1024 px (DocLayout-YOLO)'],
        ['Augmentation', 'YOLOv8s: mosaic, HSV, scale, translate; Faster R-CNN: colour jitter, blur; no mirror flips'],
        ['Verification input', 'PDF pages rendered at 200 dpi so every digit is legible; images used as they are'],
        ['Scan degradation', 'Gaussian blur 0.7 px, Gaussian noise &sigma; = 8, JPEG quality 55, on half of the benchmark'],
    ], align='ll', cls='split small'))
    a(f'<p><b>Challenges.</b> Class imbalance (<i>paragraph</i> is {para_share:.0%} of boxes); no OCR ground truth in '
      f'IndicDLP; no public data of real erroneous papers, hence the benchmark; Indic OCR of Latin digits (Section 15).</p>')

    # ------------------------------------------------------------------ 8. models
    a('<h2>8. Deep Learning Model Architecture</h2>')
    a('<p>Three detectors were compared (Fig. 3, Table VIII). All follow the backbone&ndash;neck&ndash;head design. '
      'The released DocLayout-YOLO is used by IndicDoc Verify; its AP@50 on the question-paper labels is in Table XIII.</p>')
    a(N.figure(architecture_diagram(params), 'Architectures of the three detectors with input and output dimensions. '
               'Shaded blocks are the learned feature extractor and prediction layers.'))
    a(N.table('Model details', ['', 'DocLayout-YOLO', 'YOLOv8s', 'Faster R-CNN'], [
        ['Role', 'used by IndicDoc Verify', 'fine-tuned (proposed)', 'fine-tuned (baseline)'],
        ['Type', 'one-stage, NMS-free', 'one-stage, anchor-free', 'two-stage'],
        ['Parameters', f"{params['authors']:.1f} M", f"{params['proposed']:.1f} M", f"{params['baseline']:.1f} M"],
        ['Input', '1024 &times; 1024 &times; 3', '640 &times; 640 &times; 3', 'short side 640&ndash;1024'],
        ['Output', 'N &times; (box, 42 scores)', 'N &times; (box, 42 scores)', 'N &times; (box, 43 scores)'],
        ['Pre-training', 'synthetic documents, then full IndicDLP [18]', 'COCO', 'COCO'],
        ['Modification', 'none', '80-class head &rarr; 42 classes', 'predictor &rarr; 43 classes'],
        ['Loss', '&mdash;', 'CIoU + DFL + BCE', 'RPN + cross-entropy + smooth-L1'],
    ], align='llll', cls='small'))

    # ------------------------------------------------------------------ 9. tools
    a('<h2>9. Tools, Technologies and Environment</h2>')
    a(N.table('Tools, technologies and environment', ['Component', 'Details'], [
        ['Programming language', 'Python 3.11 (models, checks, app, API); JavaScript (Node) for the benchmark renderer'],
        ['DL framework', 'PyTorch 2 [15] with torchvision; Ultralytics 8 (YOLOv8) [13]; doclayout-yolo 0.0.4'],
        ['Development environment', 'VS Code and terminal, Git + GitHub, pytest (unit tests for every check)'],
        ['Libraries', 'pycocotools, OpenCV, Pillow, NumPy, pandas, Matplotlib, PyMuPDF; Tesseract 5 (tessdata_best, 12 scripts) [19]'],
        ['Hardware', 'Apple M1 Pro GPU (MPS) for training and layout evaluation; CPU container for the verification benchmark'],
        ['Dataset tools', 'Hugging Face Hub + fsspec; COCO / YOLO converters; Playwright and Noto fonts to render benchmark papers; tesseract.js'],
        ['Interface / deployment', 'Streamlit app (Verify Paper tab), FastAPI (POST /verify), Docker + docker-compose, offline'],
    ], align='ll', cls='split small'))

    # ------------------------------------------------------------------ 10. implementation
    a('<h2>10. Implementation Details</h2>')
    a(N.table('Implementation summary', ['Part', 'Implementation (file)'], [
        ['Layout training', f'COCO-pretrained YOLOv8s (batch 8, {y_epochs} epochs, {y_minutes:.0f} min) and Faster R-CNN '
         f'(SGD lr 0.005, momentum 0.9, weight decay 5&times;10<sup>&minus;4</sup>, cosine, batch {base_hist["batch_size"]}, '
         f'{b_epochs} epochs, {b_minutes:.0f} min); best checkpoint on {pages["val"]} validation pages '
         '(<span class="tt">ml/training/</span>)'],
        ['Inference', 'DocLayout-YOLO at its high-recall threshold, duplicate boxes merged (<span class="tt">ml/inference/predict.py</span>)'],
        ['OCR + number pass', 'Indic model + English re-read; digits replaced only where the Indic word has no Indic letters or the '
         'English reading is more confident; page-level Latin / Indian digit decision (<span class="tt">ml/ocr/</span>)'],
        ['Question parsing', 'question regions, layout-gap option splitting, marker stripping, marks, figures; merged '
         'numbers ("5.16" &rarr; 5, 16) split by sequence (<span class="tt">ml/verify/paper.py</span>)'],
        ['Checks', 'structure, parity (global alignment), repeats (character 3-gram cosine), format profiles '
         '(<span class="tt">ml/verify/checks.py, profiles.py</span>)'],
        ['Report and interface', 'HTML / JSON report with outlined pages; Streamlit tab; <span class="tt">POST /verify</span>'],
        ['Benchmark', '<span class="tt">scripts/verify_bench/</span> (generate, render, degrade, OCR), '
         '<span class="tt">scripts/evaluate_verify.py</span>'],
    ], align='ll', cls='split small'))
    a('''<div class="listing"><div class="lcap">Listing 1. Verifying a Hindi paper against its English source.</div><pre>
<span class="c"># every page: layout model, then OCR with the number pass</span>
out = predict_single(model, page, "doclayout", device, conf=high_recall)
doc = extract_document(page, out, "hin", number_pass=True)

<span class="c"># ml/verify: questions, checks, parity, repeats, report</span>
report = verify(hindi_pages, source_pages=english_pages, bank=earlier_questions)
print(report["summary"])            <span class="c"># {'questions': 18, 'errors': 1, 'warnings': 0, ...}</span>
html = to_html(report, {"paper": page_images})</pre></div>''')

    # ------------------------------------------------------------------ 11. setup
    a('<h2>11. Experimental Setup and Training</h2>')
    a(N.table('Experimental setup', ['Experiment', 'Setting'], [
        ['E1 Layout', f'3 detectors on {pages["test"]} held-out IndicDLP pages ({A["num_ground_truth"]:,} boxes): COCO mAP, P / R / F1, per label'],
        ['E2 Robustness of layout', 'blur, noise, JPEG, half resolution, faded print; Robustness Index = damaged / clean mAP@50'],
        ['E3 Verification', f'{n_pairs} paper pairs; layout boxes from the renderer, real Tesseract OCR; an error is found '
         'only if a finding of the matching kind points at the right question (&plusmn;1) or, for paper-level errors, at the paper'],
        ['E4 Ablations', 'without the number pass; without layout-based option splitting; noisy layout (boxes shifted by '
         'up to 8 px, 10% of question / option labels changed to "paragraph")'],
        ['Protocol', 'parser developed on a separate development set; test set run once'],
    ], align='ll', cls='split small'))
    a(N.figure(img(loss_png), 'Training curves of our two detectors (measured): (a) Faster R-CNN loss; (b) YOLOv8s '
               'loss; (c) YOLOv8s validation mAP per epoch.'))

    # ------------------------------------------------------------------ 12. results
    a('<h2>12. Results and Performance Evaluation</h2>')

    def at(mm, key):
        return mm['at_recommended_conf'][key]
    a(f'<p><b>RQ1: structure.</b> Table XII gives the layout results on the same {pages["test"]} test pages; Table XIII the '
      f'question-paper labels. The released model finds questions and options reliably '
      f'(AP@50 {min(qap.values()):.2f}&ndash;{max(qap.values()):.2f}; question-paper domain mAP@50 '
      f'{study["per_domain_authors"]["question papers"]:.2f}), which is why the checker is built on it.</p>')
    a(N.table(f'Layout results on {pages["test"]} test pages', ['Metric', 'DocLayout-YOLO', 'YOLOv8s (ours)', 'Faster R-CNN (ours)'], [
        ['mAP@50', f'<b>{f3(A["map50"])}</b>', f3(P['map50']), f3(B['map50'])],
        ['mAP@50:95', f'<b>{f3(A["map50_95"])}</b>', f3(P['map50_95']), f3(B['map50_95'])],
        ['Precision (balanced threshold)', f'<b>{f3(at(A, "precision"))}</b>', f3(at(P, 'precision')), f3(at(B, 'precision'))],
        ['Recall (balanced threshold)', f'<b>{f3(at(A, "recall"))}</b>', f3(at(P, 'recall')), f3(at(B, 'recall'))],
        ['F1 (balanced threshold)', f'<b>{f3(at(A, "f1"))}</b>', f3(at(P, 'f1')), f3(at(B, 'f1'))],
        ['Recall, high-recall threshold', f'<b>{f3(A["at_high_recall_conf"]["recall"])}</b>',
         f3(P['at_high_recall_conf']['recall']), f3(B['at_high_recall_conf']['recall'])],
        ['Inference time (s / page)', f3(A['inference_time']), f'<b>{f3(P["inference_time"])}</b>', f3(B['inference_time'])],
    ], align='lrrr', cls='small'))
    a(N.table('AP@50 on the question-paper labels', ['Label', 'DocLayout-YOLO', 'YOLOv8s', 'Faster R-CNN'],
              [[k, f3(qap[k]), f3(P['per_class_ap50'].get(k, 0)), f3(B['per_class_ap50'].get(k, 0))] for k in Q_LABELS],
              align='lrrr', width='75%', cls='small'))
    a(f'<p><b>RQ2: verification.</b> Table XIV summarises the test set. On born-digital papers (the normal pre-print '
      f'case) {pct(cl["found_at_place"])} of the {cl["errors_injected"]} injected errors are found at the right '
      f'question, with {cl["false_alarms_per_clean_pair"]:.1f} false alarms per clean pair; on scan-degraded pages '
      f'{pct(dg["found_at_place"])} with {dg["false_alarms_per_clean_pair"]:.1f}. Fig. 5 breaks the result down by '
      f'error type. The questions were recovered from the pages in {pct(vo["questions_read_ratio"])} of cases. The '
      f'only misses were two Hindi papers whose "maximum marks" line OCR read as "पूर्णाक" (anusvara dropped). After '
      f'the test run we widened that pattern; re-running the unchanged test set then finds '
      f'{pct(VP["overall"]["found_at_place"])} with {VP["overall"]["false_alarms_per_clean_pair"]:.2f} false alarms '
      f'per clean pair. We report the frozen numbers above and this post-hoc result separately.</p>')
    a(N.table(f'Verification results on the test set ({n_pairs} paper pairs)',
              ['Condition', 'Errors', 'Found at right question', 'Clean pairs', 'False alarms / clean pair'], [
        ['Born-digital (clean render)', cl['errors_injected'], pct(cl['found_at_place']), cl['clean_pairs'],
         f"{cl['false_alarms_per_clean_pair']:.2f}"],
        ['Scan-degraded', dg['errors_injected'], pct(dg['found_at_place']), dg['clean_pairs'],
         f"{dg['false_alarms_per_clean_pair']:.2f}"],
        '---',
        ['English &rarr; Hindi', vl['hi']['errors_injected'], pct(vl['hi']['found_at_place']), vl['hi']['clean_pairs'],
         f"{vl['hi']['false_alarms_per_clean_pair']:.2f}"],
        ['English &rarr; Tamil', vl['ta']['errors_injected'], pct(vl['ta']['found_at_place']), vl['ta']['clean_pairs'],
         f"{vl['ta']['false_alarms_per_clean_pair']:.2f}"],
        '---',
        ['<b>All</b>', vo['errors_injected'], f"<b>{pct(vo['found_at_place'])}</b>", vo['clean_pairs'],
         f"<b>{vo['false_alarms_per_clean_pair']:.2f}</b>"],
    ], align='lrrrr', cls='small'))
    a(N.figure(img(F / 'fig_verify_by_type.png', '88%'), 'Share of injected errors found at the right question, per '
               'error type, on born-digital and scan-degraded papers (test set).'))
    abl = [('Full system', V), ('Without number pass', V_NP), ('Without layout-based options', V_NG), ('Noisy layout', VN)]
    a(f'<p><b>RQ3 and RQ4: ablations and robustness.</b> Without the number pass, false alarms rise from '
      f'{vo["false_alarms_per_clean_pair"]:.1f} to {V_NP.get("overall", {}).get("false_alarms_per_clean_pair", 0):.1f} '
      f'per clean pair (Table XV, Fig. 6): Indic OCR of Latin digits is the main source of noise. With a noisy layout, '
      f'{pct(VN.get("overall", {}).get("found_at_place", 0))} of errors are still found, because the parser recovers '
      f'questions from their text when a label is wrong. The layout model itself keeps '
      f'{study["mean_ri"]["authors"]:.0%} of its accuracy on damaged scans (E2).</p>')
    a(N.table('Ablations on the test set', ['System', 'Found at right question', 'False alarms / clean pair'],
              [[n, pct(r['overall']['found_at_place']), f"{r['overall']['false_alarms_per_clean_pair']:.2f}"]
               for n, r in abl if r], align='lrr', width='75%', cls='small'))
    a(N.figure(img(F / 'fig_verify_ablation.png', '88%'), 'Ablations: errors found (left) and false alarms (right).'))

    # ------------------------------------------------------------------ 13. comparison
    a('<h2>13. Comparison with Baseline / Existing Methods</h2>')
    a(N.table('Layout detectors and published results', ['Method', 'Training data', 'mAP@50', 'mAP@50:95'], [
        ['Faster R-CNN (ours, baseline)', f'{pages["train"]:,} pages, {b_minutes:.0f} min', f3(B['map50']), f3(B['map50_95'])],
        ['YOLOv8s (ours, proposed)', f'{pages["train"]:,} pages, {y_minutes:.0f} min', f3(P['map50']), f3(P['map50_95'])],
        ['DocLayout-YOLO (released by [1])', '~96K pages', f'<b>{f3(A["map50"])}</b>', f'<b>{f3(A["map50_95"])}</b>'],
        '---',
        ['Published: DocLayout-YOLO [1]', 'full IndicDLP test set', '&asymp; 0.735', '0.545'],
    ], align='llrr', cls='small'))
    a(f'<p>With the same data and a similar budget, YOLOv8s beats the Faster R-CNN baseline on every metric and is '
      f'{B["inference_time"] / P["inference_time"]:.0f}&times; faster; the released model, trained on about 42&times; '
      f'more pages, is best and agrees with the published result, so IndicDoc Verify uses it. Table XVII compares the '
      f'system with existing ways of checking papers; none of them compares language versions of a paper question by '
      f'question, offline.</p>')
    a(N.table('IndicDoc Verify and existing approaches', ['Approach', 'Checks the paper', 'Compares versions', 'Indic, offline'], [
        ['Manual vetting (status quo)', 'yes, by hand', 'by hand', 'yes'],
        ['Answer-sheet evaluation tools', 'no (grades answers)', 'no', 'partly'],
        ['MT quality estimation [10]', 'no structure', 'sentence by sentence', 'GPU / cloud'],
        ['LLM back-translation [9]', 'no structure', 'with human review', 'cloud'],
        ['<b>IndicDoc Verify</b>', '<b>numbering, options, marks, repeats</b>', '<b>question by question</b>', '<b>12 languages, offline</b>'],
    ], align='llll', cls='small'))

    # ------------------------------------------------------------------ 14. samples
    a('<h2>14. Sample Outputs and Demonstration</h2>')
    a(N.figure(f'<div class="grid2">{img(SC / "08_verify_page.png")}{img(SC / "07_verify_report.png")}</div>',
               'Correct output. Left: a number changed in the Hindi version, found and outlined on the page. Right: '
               'the report with the verdict and the findings.'))
    a(N.figure(img(SC / '09_false_alarm.png', '80%'), 'Incorrect output: a false alarm on a scan-degraded page, where '
               'OCR lost part of an option value so the versions appear to differ.'))
    a(N.figure(f'<div class="grid2">{img(SC / "02_detection_tamil.png")}{img(PL / "authors_confusion_matrix.png")}</div>',
               'Layout model in the running app (Tamil page) and its confusion matrix: most errors are missed regions.'))

    # ------------------------------------------------------------------ 15. challenges
    a('<h2>15. Challenges and Limitations</h2>')
    a('<ol class="chal">'
      '<li><b>OCR of numbers.</b> The Hindi model reads "1947" as "947", "1" as "&#2404;" and units as Indian digits; '
      'markers run into values. The number pass, layout-based options and the page-level digit decision were needed; '
      'scan-degraded pages still give more false alarms.</li>'
      '<li><b>No real error data.</b> Erroneous pre-print papers are not public; the benchmark injects realistic '
      'errors into rendered papers, with oracle layout boxes (the noisy-layout run tests the layout dependence).</li>'
      '<li><b>Meaning.</b> Changes of meaning without a change of number or structure need a cross-lingual sentence '
      'model; the hook exists but is not evaluated. Number words and formulas are not compared yet.</li>'
      f'<li><b>Compute.</b> Our own detectors trained on {pages["train"]:,} pages for about an hour and stay below the '
      'released model; the product uses the released model.</li></ol>')

    # ------------------------------------------------------------------ 16. conclusion
    a('<h2>16. Conclusion and Future Scope</h2>')
    a(f'<p><b>Conclusion.</b> IndicDoc Verify checks an Indian question paper and its translations before printing. '
      f'A deep-learning layout detector recovers the questions and options (AP@50 '
      f'{min(qap.values()):.2f}&ndash;{max(qap.values()):.2f}); OCR with a number pass reads them; structure, parity, '
      f'repeat and format checks find {pct(cl["found_at_place"])} of injected errors at the right question on '
      f'born-digital papers ({pct(dg["found_at_place"])} on degraded scans) with {cl["false_alarms_per_clean_pair"]:.1f} '
      f'false alarms per clean pair. All four objectives were met; our own detectors did not reach the released model, '
      f'which we report openly.</p>')
    a('<p><b>Product.</b> Buyers are national exam agencies, UPSC and state commissions, school boards, university exam '
      'cells and test-series publishers. Because pre-print papers are confidential, the product is an offline licence '
      'per exam body; a hosted plan suits only non-confidential mock tests.</p>')
    a('<p><b>Future scope.</b></p><ul class="fut">'
      '<li>Pilot with a university exam cell or a state board on real pre-print papers; measure minutes saved.</li>'
      '<li>A cross-lingual meaning check (sentence embeddings or quality estimation [10]) as a warning for the vetter.</li>'
      '<li>Better Indic OCR behind the same interface; number words, formulas and answer-key consistency.</li>'
      '<li>More format profiles: bilingual circulars, forms, certificates; a question bank built from MILU-style data [16].</li></ul>')

    # ------------------------------------------------------------------ closing sections
    a('<h2>Expected Outcomes</h2>')
    a(N.table('Expected outcomes and their status', ['Expected outcome', 'Status', 'Evidence'], [
        ['Development of a working Deep Learning model addressing a real-world problem', 'Achieved',
         'Layout detectors + IndicDoc Verify in a working app and API (Sections 8&ndash;10, 14)'],
        ['Performance evaluation of implemented model', 'Achieved',
         'Layout mAP and per-label AP; errors found and false alarms on a benchmark; ablations (Sections 12&ndash;13)'],
        ['Improved understanding of research methodology and problem-solving skills', 'Achieved',
         'Research questions, separate development and test sets, honest failure cases'],
        ['Preparation of a research paper suitable for submission', 'Achieved', 'IEEE-format paper in <span class="tt">paper/</span>'],
    ], align='lll', cls='small'))
    a('<h2>Deliverables</h2>')
    a(N.table('Deliverables and their location in the repository', ['Deliverable', 'Location', 'Contents'], [
        ['Dataset (CSV / image dataset)', '<span class="tt">dataset/, scripts/verify_bench/</span>',
         f'IndicDLP subset as CSV ({total_pages:,} pages, {total_boxes:,} boxes); generator of the verification benchmark'],
        ['Project code with complete implementation', '<span class="tt">ml/ app/ backend/ scripts/</span>',
         'Training, evaluation, OCR, verification, web app, REST API, Docker'],
        ['GitHub repository with organized code and documentation', '<span class="tt">README.md, docs/</span>',
         'Product, dataset, architecture, evaluation and results documents; tests'],
        ['Presentation slides (Phase-1 and Phase-2)', '<span class="tt">docs/presentation/</span>', 'PDF and PPTX'],
        ['Final research paper', '<span class="tt">paper/</span>', 'IEEE-format paper'],
        ['Experiment report (this document)', '<span class="tt">docs/report/</span>', 'generated by <span class="tt">scripts/build_lab_report.py</span>'],
    ], align='lll', cls='split small'))
    a('<h2>Evaluation Criteria</h2>')
    a(N.table('Evaluation criteria and where each is addressed', ['Criterion', 'Marks', 'Where addressed'], [
        ['Relevance of problem to SDG', '2', 'Section 1, Tables I&ndash;II (SDG 4, 9, 10, 16)'],
        ['Quality of literature survey and research gap identification', '2', 'Sections 2&ndash;3'],
        ['Implementation of deep learning models', '2', 'Sections 7&ndash;11'],
        ['Performance evaluation and Frontend development', '2', 'Sections 12&ndash;14; Streamlit Verify tab, REST API'],
        ['Documentation and presentation quality', '2', 'Repository docs, Phase-1 / Phase-2 slides, this report'],
        ['Innovation and Research Paper', '5', 'Translation parity without translation, number pass, layout-based checking; IEEE paper'],
    ], align='lcl', cls='small'))
    a('<h2>References</h2><ol class="refs">' + ''.join(f'<li>{r}</li>' for r in REFS) + '</ol>')
    return page_html(''.join(s))


def page_html(body: str) -> str:
    fonts = ASSETS / 'fonts'
    face = ''.join(
        f"@font-face{{font-family:'CMU Serif';src:url('{uri(fonts / f'cmu-serif-{w}-{st}.woff2')}') format('woff2');"
        f"font-weight:{w};font-style:{'italic' if st == 'italic' else 'normal'};}}"
        for w in (500, 700) for st in ('roman', 'italic'))
    face += (f"@font-face{{font-family:'CMU Typewriter';src:url('{uri(fonts / 'cmu-typewriter-text-500-roman.woff2')}') "
             f"format('woff2');}}")
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>IndicDoc AI - Expt. 11 Report</title><style>{face}
@page {{ size: letter; margin: 0.95in 1in 0.9in 1in;
  @bottom-center {{ content: counter(page); font-family: 'CMU Serif'; font-size: 10.5pt; }} }}
html {{ font-family: 'CMU Serif', serif; font-weight: 500; font-size: 11pt; line-height: 1.36; color: #000;
  -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
body {{ margin: 0; background: #fff; }}
p {{ margin: 0 0 7pt; text-align: justify; hyphens: auto; }}
b, strong, th, h2, h3 {{ font-weight: 700; }}
.tt, pre {{ font-family: 'CMU Typewriter', monospace; }}
.tt {{ font-size: 0.92em; }}
.title {{ text-align: center; font-size: 20.7pt; font-weight: 700; line-height: 1.15; }}
.subtitle {{ text-align: center; font-size: 13pt; margin-top: 6pt; line-height: 1.3; }}
.expt {{ text-align: center; font-size: 12pt; margin: 14pt 0 12pt; }}
.rule {{ border-top: 1px solid #000; border-bottom: 1px solid #000; height: 3px; margin-bottom: 14pt; }}
table.meta {{ width: 100%; border-collapse: collapse; font-size: 10pt; margin: 0 0 6pt; }}
table.meta > tbody > tr > th {{ text-align: left; width: 1.35in; vertical-align: top; padding: 2.5pt 6pt 2.5pt 0; }}
table.meta > tbody > tr > td {{ padding: 2.5pt 0; vertical-align: top; }}
table.team {{ border-collapse: collapse; font-size: 10pt; }}
table.team th, table.team td {{ border: 0.6pt solid #000; padding: 1.5pt 10pt; text-align: left; }}
h2 {{ font-size: 14.4pt; margin: 16pt 0 8pt; break-after: avoid; line-height: 1.2; }}
ol, ul {{ margin: 0 0 8pt; padding-left: 26pt; }}
li {{ margin-bottom: 5pt; text-align: justify; }}
ol.phases li {{ margin-bottom: 7pt; }}
.abstract {{ font-size: 10pt; text-align: justify; margin: 14pt 0 4pt; }}
.kw {{ font-size: 10pt; }}
/* IEEE tables */
.tbl {{ margin: 10pt auto 12pt; break-inside: avoid; }}
.tbl.split {{ break-inside: auto; }}
.tbl tr {{ break-inside: avoid; }}
.tbl caption {{ caption-side: top; break-after: avoid; break-inside: avoid; }}
.tcap {{ text-align: center; font-size: 9pt; margin-bottom: 4pt; font-variant: small-caps; letter-spacing: 0.02em; }}
.tcap .tno {{ font-variant: normal; text-transform: uppercase; }}
.tbl table {{ border-collapse: collapse; margin: 0 auto; font-size: 9.5pt; line-height: 1.25; }}
.tbl.small table {{ font-size: 9pt; }}
.tbl.lit table {{ font-size: 8pt; }}
.tbl thead th {{ border-top: 1.2pt solid #000; border-bottom: 0.7pt solid #000; padding: 3pt 5pt; vertical-align: bottom; }}
.tbl td {{ padding: 2.4pt 5pt; vertical-align: top; text-align: left; }}
.tbl tbody tr:last-child td {{ border-bottom: 1.2pt solid #000; }}
.tbl tr.mid td {{ border-top: 0.6pt solid #000; padding: 0; height: 0; }}
/* IEEE figures */
figure {{ margin: 10pt 0 12pt; text-align: center; break-inside: avoid; }}
figure img, figure svg {{ display: block; margin: 0 auto; }}
figure svg {{ width: 100%; height: auto; }}
figcaption {{ font-size: 9pt; text-align: justify; margin-top: 5pt; line-height: 1.3; }}
.fno {{ font-weight: 500; }}
.grid4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8pt; }}
.grid4 img {{ height: 2.1in; object-fit: cover; object-position: top; border: 0.5pt solid #999; }}
.grid4 span {{ font-size: 8.5pt; font-style: italic; }}
.grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8pt; }}
.grid2 img {{ border: 0.5pt solid #999; }}
.listing {{ margin: 8pt 0 12pt; break-inside: avoid; }}
.lcap {{ font-size: 9pt; margin-bottom: 3pt; }}
pre {{ font-size: 8.4pt; line-height: 1.35; border-top: 0.8pt solid #000; border-bottom: 0.8pt solid #000;
  padding: 5pt 4pt; margin: 0; white-space: pre-wrap; }}
pre .c {{ color: #555; font-style: italic; }}
ol.refs {{ font-size: 9pt; padding-left: 0; list-style: none; counter-reset: r; }}
ol.refs li {{ counter-increment: r; position: relative; padding-left: 26pt; margin-bottom: 3pt; text-align: left; }}
ol.refs li::before {{ content: '[' counter(r) ']'; position: absolute; left: 0; }}
</style></head><body>{body}</body></html>'''


def find_chrome() -> str:
    candidates = [os.environ.get('CHROME'), '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
                  shutil.which('google-chrome'), shutil.which('chromium'), shutil.which('chromium-browser')]
    candidates += sorted(str(p) for p in Path('/opt/pw-browsers').glob('chromium-*/chrome-linux*/chrome'))
    for c in candidates:
        if c and Path(c).exists():
            return c
    sys.exit('Chrome/Chromium not found; set CHROME=/path/to/chrome')


def main():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        html_path = tmp / 'report.html'
        html_path.write_text(build_html(tmp))
        sandbox = ['--no-sandbox'] if hasattr(os, 'geteuid') and os.geteuid() == 0 else []  # Chrome refuses root otherwise
        subprocess.run([find_chrome(), *sandbox, '--headless=new', '--disable-gpu', '--no-pdf-header-footer',
                        '--allow-file-access-from-files', '--virtual-time-budget=10000',
                        f'--print-to-pdf={PDF_PATH}', html_path.as_uri()], check=True, capture_output=True, timeout=180)
    print(f'Report written to {PDF_PATH}')


if __name__ == '__main__':
    main()
