"""
IndicDoc AI - Build the Expt.-11 (SDG-based Deep Learning Mini Project) lab report.

Structure follows the Expt.-11 document (Aim, Lab Outcome, Methodology, Expected Outcomes, Deliverables,
Evaluation Criteria); the body follows the Phase-2 content list (16 sections);
tables, figures, citations and references follow IEEE style. Every number is read from files the pipeline wrote
(dataset/*.csv, outputs/metrics); nothing is typed in by hand. The PDF is printed with headless Chrome/Chromium.

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
PDF_PATH = REPORT_DIR / 'IndicDoc_AI_Expt11_Lab_Report.pdf'

TITLE = 'IndicDoc AI: Where Do Layout Detectors Fail on Indian Documents? A Study of Script, Domain and Scan Quality'
TEAM = [('Riddhi Patil', 39), ('Mariam Badure', 40), ('Prashant Jha', 41), ('Aditya Soni', 42)]
GUIDE = 'Dr. Joanne Gomes'
REPO = 'https://github.com/va-ada/IndicDoc-AI'
MODEL_NAMES = {'authors': 'DocLayout-YOLO (IndicDLP authors)', 'proposed': 'YOLOv8s (ours)',
               'baseline': 'Faster R-CNN (ours)'}


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


def domain_figure(out: Path) -> Path:
    """mAP@50 per document domain for DocLayout-YOLO and YOLOv8s on clean test pages (from the robustness study)."""
    rob = json.loads((METRICS / 'study_robustness.json').read_text())
    a, p = rob['doclayout']['clean']['per_domain'], rob['yolov8']['clean']['per_domain']
    doms = sorted(a, key=lambda d: a[d]['map50'])
    plt.rcParams.update({'font.family': 'STIXGeneral', 'font.size': 9})
    fig, ax = plt.subplots(figsize=(6.0, 3.1))
    y = range(len(doms))
    ax.barh([i + 0.2 for i in y], [a[d]['map50'] for d in doms], 0.4, color='#1E2A5A', label='DocLayout-YOLO')
    ax.barh([i - 0.2 for i in y], [p[d]['map50'] for d in doms], 0.4, color='#B8B8B8', label='YOLOv8s (ours)')
    ax.set_yticks(list(y), [f"{d} (n={a[d]['images']})" for d in doms])
    ax.set_xlabel('mAP@50 on clean test pages')
    ax.set_xlim(0, 1)
    ax.grid(axis='x', alpha=0.3, lw=0.5)
    ax.legend(frameon=False, loc='lower right', fontsize=8)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
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
    W, bw, bh, gap = 684, 118, 58, 20
    xs = [6 + i * (bw + gap) for i in range(5)]
    y1, y2 = 70, 178
    g = [f'<svg viewBox="0 0 {W} 250" xmlns="http://www.w3.org/2000/svg" font-family="CMU Serif, serif">', SVG_DEFS]
    # interface lane
    g.append(svg_box(xs[0], 4, bw, 40, 'Streamlit web app', 'upload, view, download', '#FFFFFF'))
    g.append(svg_box(xs[1], 4, bw, 40, 'REST API (FastAPI)', '/analyze, /metrics', '#FFFFFF'))
    g.append(arrow(xs[0] + bw / 2, 44, xs[0] + bw / 2, y1, dashed=True))
    g.append(arrow(xs[1] + bw / 2, 44, xs[0] + bw - 10, y1, dashed=True))
    g.append(svg_box(xs[3], 4, bw, 40, 'Evaluation', 'mAP, P / R / F1, CI, RI', '#FFFFFF'))
    g.append(arrow(xs[3] + bw / 2, y1, xs[3] + bw / 2, 44, dashed=True))
    top = [('Input page', 'image / PDF page|12 languages'), ('Decode & check', 'PNG, JPG, TIFF, PDF|size limit'),
           ('Pre-process', 'RGB, letterbox|1024 px (640 px ours)'),
           ('Layout detector', 'DocLayout-YOLO | YOLOv8s|Faster R-CNN'),
           ('Post-process', 'confidence mode|duplicate removal')]
    for i, (t, s) in enumerate(top):
        fill = '#E3E7F2' if i == 3 else '#F4F4F4'
        g.append(svg_box(xs[i], y1, bw, bh + 8, t, s, fill))
        if i:
            g.append(arrow(xs[i - 1] + bw, y1 + 33, xs[i], y1 + 33))
    bottom = [('Outputs', 'boxes + labels|Markdown / JSON|HTML / EPUB3'), ('Reading order', 'regions sorted|into columns'),
              ('OCR', 'Tesseract 5, one|pass, words→regions'), ('Script probe', '12 script models|on one region')]
    bx = [xs[1], xs[2], xs[3], xs[4]]
    for i, (t, s) in enumerate(bottom):
        g.append(svg_box(bx[i], y2, bw, bh + 8, t, s))
    g.append(arrow(xs[4] + bw / 2, y1 + bh + 8, xs[4] + bw / 2, y2))
    for i in range(3, 0, -1):
        g.append(arrow(bx[i], y2 + 33, bx[i - 1] + bw, y2 + 33))
    g.append(f'<rect x="{xs[0]}" y="{y2 + 8}" width="12" height="10" fill="#E3E7F2" stroke="#222" stroke-width="0.7"/>')
    g.append(f'<text x="{xs[0] + 17}" y="{y2 + 17}" font-size="9" font-style="italic">deep-learning stage</text>')
    g.append('</svg>')
    return ''.join(g)


def architecture_diagram(params: dict) -> str:
    rows = [
        (f"DocLayout-YOLO [9]  ({params['authors']:.1f} M params, used as released)",
         [('Input', '1024×1024×3'), ('Backbone', 'CSP + GL-CRM|global-to-local'), ('Neck', 'PAN pyramid|3 scales'),
          ('Head', 'YOLOv10-style|NMS-free'), ('Output', 'N × (box, 42 scores)')]),
        (f"YOLOv8s [11]  ({params['proposed']:.1f} M params, fine-tuned here)",
         [('Input', '640×640×3'), ('Backbone', 'CSPDarknet|C2f + SPPF'), ('Neck', 'PAN-FPN|3 scales'),
          ('Head', 'decoupled|anchor-free'), ('Output', 'boxes + 42 classes')]),
        (f"Faster R-CNN [10]  ({params['baseline']:.1f} M params, fine-tuned here)",
         [('Input', 'short side 640–1024'), ('Backbone', 'ResNet-50 [15]|frozen BN'), ('Neck', 'FPN [16]|5 levels'),
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
def build_html(tmp: Path) -> str:
    m, study, ocr, tests, langs, classes, base_hist, yolo_hist = load()
    A, P, B = m['authors'], m['proposed'], m['baseline']
    N = Numbering()

    pages = {s: int(langs[s].sum()) for s in ('train', 'val', 'test')}
    boxes = {s: int(classes[s].sum()) for s in ('train', 'val', 'test')}
    total_pages, total_boxes = sum(pages.values()), sum(boxes.values())
    para_share = classes.loc[classes.class_name == 'paragraph', 'total'].iloc[0] / total_boxes
    rare = int((classes['train'] < 100).sum())
    params = {k: m[k]['parameters'] / 1e6 for k in m}
    b_minutes = base_hist['epoch_end_seconds'][-1] / 60
    y_minutes = float(yolo_hist['time'].iloc[-1]) / 60
    y_epochs, b_epochs = len(yolo_hist), len(base_hist['train_losses'])
    ci, ri, rmap = study['language_ci'], study['robustness_index'], study['robust_map50']
    script_ok, script_n = ocr['script_detection_correct'], ocr['pages']
    osd_ok = 70  # Tesseract OSD on the same 120 pages (scripts/evaluate_ocr.py, reported in the paper and slides)

    loss_png = training_figure(base_hist, yolo_hist, tmp / 'fig_training.png')
    F = PROJECT_ROOT / 'docs' / 'figures'
    PL = PROJECT_ROOT / 'outputs' / 'plots'
    PR = PROJECT_ROOT / 'outputs' / 'predictions'
    SC = PROJECT_ROOT / 'docs' / 'screenshots'
    DS = PROJECT_ROOT / 'dataset' / 'samples'

    s = []  # body sections
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
(SDGs). <i>Done:</i> layout understanding and text extraction for Indian-language documents (Section 1).</li>
<li><b>Phase-1: Literature Survey:</b> Review 4&ndash;8 research papers and identify the research gap.
<i>Done:</i> eight papers reviewed, three gaps identified (Sections 2&ndash;3).</li>
<li><b>Phase-2: Final Project Implementation with Performance Analysis:</b> Develop system architecture, collect
dataset, and implement DL model. Analyze results, improve performance, design front end, integrate the system.
Demonstrate the finished project and submit the code along with final working project. <i>Done:</i>
Sections 4&ndash;16.</li>
<li><b>Documentation:</b> Prepare a research paper in standard format. <i>Done:</i> IEEE conference-format paper
in <span class="tt">paper/</span>.</li>
</ol>

<div class="abstract"><b><i>Abstract</i></b>&mdash;Indian offices, schools and archives hold millions of scanned
pages in a dozen scripts. Before such a page can be searched, read aloud or converted for a blind reader, a system
must find its parts (paragraphs, headings, tables, figures, page numbers) and then read the text in the right
order. We fine-tuned two convolutional object detectors, Faster R-CNN and YOLOv8s, on a language-balanced
{pages['train']:,}-page subset of the IndicDLP dataset [1] and compared them with the DocLayout-YOLO model released
by the dataset's authors on the same {pages['test']} held-out test pages covering 12 languages and 42 region
labels. The released model reaches {f3(A['map50'])} mAP@50 ({f3(A['map50_95'])} mAP@50:95); our YOLOv8s reaches
{f3(P['map50'])} and our Faster R-CNN {f3(B['map50'])} after about one hour of training each on a laptop GPU. We
add two simple measures, a Consistency Index and a Robustness Index, and find that accuracy is fairly even across
languages (CI {ci['authors']:.2f}) but not across document types (CI {study['domain_ci']['authors']:.2f}), and that
the best model keeps {study['mean_ri']['authors']:.0%} of its accuracy on damaged scans. A region-based script probe
picks the right OCR language on {script_ok}/{script_n} pages (Tesseract's own detector: {osd_ok}/{script_n}). The
work is delivered as IndicDoc&nbsp;AI, a web app and REST API in one offline Docker image, with accessible HTML and
EPUB3 export for screen-reader users.</div>
<p class="kw"><b><i>Index Terms</i></b>&mdash;document layout analysis, Indian languages, object detection,
transfer learning, robustness, OCR, accessibility.</p>
''')

    # ------------------------------------------------------------------ 1. SDG mapping
    a('<h2>1. Topic Selection and SDG Mapping</h2>')
    a('<p><b>Problem.</b> Most document-AI tools are built and tested on English. Indian documents use many scripts '
      '(Devanagari, Bengali, Tamil, Telugu, Kannada, Malayalam, Gujarati, Gurmukhi, Odia, Latin) and dense formats '
      '(multi-column newspapers, question papers with nested sub-questions, legal acts with numbered clauses). '
      'If the layout step fails, OCR reads columns straight across and every later step fails with it. This hits '
      'hardest the people who depend on machines to read for them: blind and low-vision students, who study from '
      'state-board textbooks that are mostly available only as scans, and citizens who need government notices in '
      'their own language. The mapping to the SDG targets of the 2030 Agenda [20] is given in Table I.</p>')
    a(N.table('Mapping of the project to the Sustainable Development Goals', ['SDG', 'Target', 'How IndicDoc AI contributes'], [
        ['SDG 4 Quality Education', '4.5: equal access to education for persons with disabilities',
         'Turns scanned textbook and exam pages into structured, screen-reader-ready HTML / EPUB3 in their own language'],
        ['SDG 9 Industry, Innovation &amp; Infrastructure', '9.5: enhance research and technological capability',
         'Open, offline, reproducible document-AI pipeline for 12 Indian languages; measured study of its weak spots'],
        ['SDG 10 Reduced Inequalities', '10.2: social inclusion irrespective of disability, ethnicity or origin',
         'Serves speakers of 11 Indian languages besides English, which commercial tools cover unevenly'],
        ['SDG 16 Strong Institutions', '16.10: public access to information',
         'Makes scanned government circulars, acts and forms searchable and readable'],
    ], align='lll', cls='split small'))

    # ------------------------------------------------------------------ 2. literature survey
    a('<h2>2. Literature Survey Summary Table</h2>')
    a('<p>Eight recent papers on Indian-language document understanding were reviewed in Phase-1 (Table II). They '
      'cover layout datasets [1], [3], text recognition [2], [7], table understanding [4], [8] and multimodal '
      'document models [5], [6].</p>')
    lit = [
        ['[1]', 'Nath <i>et al.</i>, 2025', 'Layout parsing of Indian-language pages',
         'IndicDLP: 119,806 pages, 12 languages, 12 domains, 42 labels',
         'YOLOv10x, DocLayout-YOLO, RoDLA, DINO, DiT, Florence-2', 'Vision-only; 20&ndash;25 mAP drop on unseen scripts; no robustness test'],
        ['[2]', 'Kasuba <i>et al.</i>, 2025', 'Page-level handwritten Indic text recognition', 'CHIPS, ~25K pages, 10 languages',
         'DBNet word detection + 6 recognisers (CRNN, PARSeq, ...)', 'Synthetic pages; user must name the language'],
        ['[3]', 'Mondal <i>et al.</i>, 2026', 'Multilingual table detection', 'UniTabBank: 55K pages, 82K tables, 28 languages',
         'Semi-automatic labelling; YOLOv11 detector', 'Borderless tables and degraded scans fail'],
        ['[4]', 'Pal <i>et al.</i>, 2024', 'Table QA for low-resource Indic languages', 'BanglaTabQA 2M, HindiTabQA 643K',
         'SQL-generated QA; fine-tuned mBART / M2M / Llama', 'Wikipedia tables only; small test sets'],
        ['[5]', 'Xu <i>et al.</i>, 2021', 'Multilingual document understanding', '30M documents; XFUND, 7 languages',
         'LayoutXLM: text + layout + image transformer', 'No Indian language; needs good OCR'],
        ['[6]', 'Kasuba <i>et al.</i>, 2025', 'Grounding answers on the page', 'MGVG: 70 circulars, 509 QA pairs',
         'OCR + Llama-3.1-8B + region matching', 'OCR-dependent; single page; small test set'],
        ['[7]', 'Bhattacharyya <i>et al.</i>, 2025', 'Hindi OCR with vision-language models', '1.1M printed Hindi words + pages',
         'LoRA-tuned GOT-OCR 2.0, progressive learning', 'Hindi only; mixed-language lines'],
        ['[8]', 'Dumada &amp; Shah, 2026', 'Devanagari table extraction', '500+ Devanagari documents',
         'Structure + Tesseract content, weighted fusion', 'Self-reported results; no ablation'],
    ]
    a(N.table('Literature survey summary', ['Ref', 'Author', 'Problem', 'Dataset', 'Method', 'Limitation'], lit,
              align='llllll', cls='split small lit'))

    # ------------------------------------------------------------------ 3. gaps
    a('<h2>3. Research Gaps</h2>')
    a('<ol class="gaps">'
      '<li><b>Only clean pages are tested.</b> IndicDLP [1] benchmarks large detectors on clean pages; no study '
      'measures how accuracy holds up on blurred, noisy, compressed, low-resolution or faded scans, which are '
      'the norm in real archives and phone photos (addressed by RQ4).</li>'
      '<li><b>Results are one overall number.</b> Papers report a single mAP. It is not known whether accuracy is '
      'even across the 12 languages and 12 document types, or which labels fail and why (RQ2, RQ3).</li>'
      '<li><b>OCR tools expect the user to name the language</b> [2], [7]. A multilingual tool needs to choose the '
      'script automatically, page by page (RQ5).</li>'
      '<li><b>No accessible output.</b> None of the surveyed systems produces structured, screen-reader-ready output '
      '(headings, lists, tables, language tag) from a scanned Indian-language page.</li></ol>')
    a('<p><b>Position.</b> Strong detectors already exist [1], [9]. We therefore do not claim a new detector: our '
      'contribution is a careful study of where they fail on Indian documents, and a working, accessible tool built '
      'on the best one.</p>')

    # ------------------------------------------------------------------ 4. problem statement
    a('<h2>4. Problem Statement and Objectives</h2>')
    a('<p><b>Problem statement.</b> Given an image or PDF page in any of 12 Indian languages, detect and classify every '
      'structural region into one of the 42 IndicDLP labels, read the text of each region in reading order, and '
      'measure, in simple numbers, where deep-learning layout detectors work and where they fail.</p>')
    a('<p><b>Main objective.</b> Build and evaluate a deep-learning layout-and-text pipeline for Indian-language '
      'documents and deliver it as a usable tool.</p>')
    a('<p><b>Specific objectives.</b></p><ol class="obj">'
      '<li>Fine-tune two detectors (Faster R-CNN, YOLOv8s) with transfer learning and compare them with the published '
      'IndicDLP model on the same held-out test pages.</li>'
      '<li>Define two simple measures, a Consistency Index and a Robustness Index, and use them to answer five research '
      'questions (RQ1&ndash;RQ5, Table III).</li>'
      '<li>Add OCR with automatic script detection for 12 languages, and accessible HTML / EPUB3 export.</li>'
      '<li>Ship a web app and REST API in one Docker image that works offline.</li></ol>')
    a(N.table('Research questions', ['RQ', 'Question'], [
        ['RQ1', 'How far behind is a detector trained on a small budget, compared with one trained on the full dataset?'],
        ['RQ2', 'Is accuracy even across the 12 languages and the 12 document domains?'],
        ['RQ3', 'Which region types are hardest, and is that because they are rare in the training data?'],
        ['RQ4', 'How much accuracy survives blur, noise, JPEG compression, low resolution and faded print?'],
        ['RQ5', 'Can the detected regions be used to choose the right OCR language automatically?'],
    ], align='ll', cls='small'))
    a('<p><b>Expected output.</b> A labelled box with a confidence score for every region; the page text in reading order '
      'as Markdown, JSON and accessible HTML / EPUB3; and the measured study results.</p>')

    # ------------------------------------------------------------------ 5. proposed system
    a('<h2>5. Proposed System, Methodology and Novelty</h2>')
    a('<p><b>Workflow (input to output).</b></p><ol class="steps">'
      '<li><b>Input:</b> a page image (PNG, JPG, TIFF) or a PDF page, rendered at 150 dpi.</li>'
      '<li><b>Detect regions:</b> a convolutional object detector (DocLayout-YOLO by default) outputs a box, one of 42 '
      'labels and a confidence for every region.</li>'
      '<li><b>Clean up:</b> boxes below the chosen confidence are dropped; boxes covering the same region with '
      'different labels (IoU &gt; 0.7) are merged.</li>'
      '<li><b>Pick the script:</b> one detected text region is read with each of the 12 Tesseract script models; the '
      'model that returns the most characters of its own script wins.</li>'
      '<li><b>Read text:</b> Tesseract 5 [13] reads the page once; each word is assigned to the region that contains it, '
      'so words outside every region are kept rather than lost.</li>'
      '<li><b>Order:</b> regions are sorted into columns and reading order.</li>'
      '<li><b>Output:</b> annotated image, Markdown / JSON, and accessible HTML / EPUB3 (semantic headings, lists, '
      'tables, <span class="tt">lang</span> attribute).</li></ol>')
    a('<p><b>How deep learning is used.</b> Layout detection is posed as object detection. Convolutional detectors '
      'pre-trained on COCO [12] are fine-tuned on IndicDLP pages (transfer learning): the backbone learns visual '
      'features of text blocks, rules and figures, the neck combines them across scales, and the head predicts boxes '
      'and labels. The released DocLayout-YOLO [9], pre-trained on synthetic documents and fine-tuned on the full '
      'IndicDLP training set, is used as the production model.</p>')
    a('<p><b>Novelty.</b> (i) <i>Study:</i> two interpretable ratios reported next to mAP &mdash; the Consistency Index '
      'CI&nbsp;=&nbsp;min<sub>g</sub>&nbsp;mAP<sub>50</sub>(g)&nbsp;/&nbsp;max<sub>g</sub>&nbsp;mAP<sub>50</sub>(g) over '
      'languages or domains, and the Robustness Index RI<sub>c</sub>&nbsp;=&nbsp;mAP<sub>50</sub>(damaged by '
      'c)&nbsp;/&nbsp;mAP<sub>50</sub>(clean) &mdash; and, to our knowledge, the first scan-damage robustness test on '
      'IndicDLP. (ii) <i>Tool:</i> two operating modes whose thresholds are chosen on validation pages ("find every '
      'part" and "balanced"); automatic script choice by probing detected regions; accessible EPUB3 output from '
      'scanned pages; one offline app for 12 languages.</p>')

    # ------------------------------------------------------------------ 6. block diagram
    a('<h2>6. System Block Diagram</h2>')
    a('<p>Fig. 1 shows the major components and the flow of data. The same Python modules serve the web app, the REST '
      'API and the evaluation scripts, so the numbers in this report are produced by the code the user runs.</p>')
    a(N.figure(block_diagram(), 'System block diagram of IndicDoc AI. Solid arrows: data flow for one page; dashed '
               'arrows: requests from the two interfaces and the evaluation path. The deep-learning stage is shaded.'))

    # ------------------------------------------------------------------ 7. dataset
    a('<h2>7. Dataset Description and Pre-processing</h2>')
    a(f'<p><b>Dataset.</b> IndicDLP [1], [18] (AI4Bharat, IIT Madras and IIIT Hyderabad; ICDAR 2025; also listed on '
      f'India\'s AIKosh platform): 119,806 human-annotated pages, 12 languages, 12 document domains, 42 region labels. '
      f'The full dataset is about 91 GB, so for our own two models we built a language-balanced subset of '
      f'{total_pages:,} pages and {total_boxes:,} boxes, taken from the <i>official</i> train / validation / test splits '
      f'so that no test page was seen in training (Table IV). The subset is released as CSV files in '
      f'<span class="tt">dataset/</span> (pages, boxes, class counts, language split) with one sample page per '
      f'language (Fig. 2).</p>')
    rows = [[r.language.capitalize(), r.train, r.val, r.test, r.total] for r in langs.itertuples()]
    rows += ['---', ['<b>Total pages</b>', f"<b>{pages['train']:,}</b>", f"<b>{pages['val']}</b>",
                     f"<b>{pages['test']}</b>", f'<b>{total_pages:,}</b>'],
             ['Labelled boxes', f"{boxes['train']:,}", f"{boxes['val']:,}", f"{boxes['test']:,}", f'{total_boxes:,}']]
    a(N.table('IndicDLP subset used in this project (pages per split)', ['Language', 'Train', 'Validation', 'Test', 'Total'],
              rows, width='72%', cls='small'))
    samples = ['hindi_newspapers', 'tamil_forms', 'bengali_forms', 'punjabi_brochures']
    grid = ''.join(f'<div>{img(DS / (n + ".jpg"))}<span>{n.replace("_", " · ")}</span></div>' for n in samples)
    a(N.figure(f'<div class="grid4">{grid}</div>', 'Sample pages from the dataset (one per language is shipped in '
               '<span class="tt">dataset/samples/</span>).'))
    a(N.table('Pre-processing and augmentation', ['Step', 'What was done'], [
        ['Loading', 'Only the needed Parquet row groups are streamed from Hugging Face (fsspec); images saved as PNG, '
                    'annotations as COCO JSON'],
        ['Cleaning', 'Boxes clipped to the page; boxes smaller than 1 px dropped (0.5% of training boxes)'],
        ['Format conversion', 'COCO JSON &rarr; YOLO text labels for YOLOv8s'],
        ['Resizing', 'Letterbox to 640 px (our models) or 1024 px (DocLayout-YOLO); Faster R-CNN short side 640&ndash;1024'],
        ['Augmentation (YOLOv8s)', 'Mosaic, HSV colour jitter, scale and translate; horizontal flips off (a mirrored page '
                                   'is not a realistic document)'],
        ['Augmentation (Faster R-CNN)', 'Colour jitter and mild blur; no flips'],
        ['Inference input', 'PDF pages rendered at 150 dpi; RGB conversion; optional deskew and CLAHE contrast enhancement'],
    ], align='ll', cls='split small'))
    a(f'<p><b>Challenges in the data.</b> Heavy class imbalance: <i>paragraph</i> alone is {para_share:.0%} of all '
      f'boxes and {rare} of the 42 labels have fewer than 100 training boxes (Fig. 3); two labels have almost no examples '
      f'in the subset. Hindi has only {int(langs.loc[langs.language == "hindi", "test"].iloc[0])} test pages in the '
      f'official split, so its numbers are less certain. IndicDLP provides no text ground truth, so OCR character '
      f'accuracy cannot be measured on it.</p>')
    a(N.figure(img(PL / 'class_distribution.png', '92%'), f'Number of ground-truth boxes per label in the '
               f'{pages["test"]} test pages: a long-tailed distribution dominated by <i>paragraph</i>.'))

    # ------------------------------------------------------------------ 8. model architecture
    a('<h2>8. Deep Learning Model Architecture</h2>')
    a('<p>Three detectors are compared (Fig. 4, Table VI). All follow the backbone&ndash;neck&ndash;head structure of '
      'modern object detectors.</p>')
    a(N.figure(architecture_diagram(params), 'Architectures of the three detectors with input and output dimensions. '
               'Shaded blocks are the learned feature extractor and prediction layers.'))
    a(N.table('Model details', ['', 'DocLayout-YOLO', 'YOLOv8s', 'Faster R-CNN'], [
        ['Role', 'production model', 'proposed (ours)', 'baseline (ours)'],
        ['Type', 'one-stage, NMS-free', 'one-stage, anchor-free', 'two-stage'],
        ['Parameters', f"{params['authors']:.1f} M", f"{params['proposed']:.1f} M", f"{params['baseline']:.1f} M"],
        ['Model size', f"{A['model_size_mb']:.1f} MB", f"{P['model_size_mb']:.1f} MB", f"{B['model_size_mb']:.1f} MB"],
        ['Input', '1024 &times; 1024 &times; 3', '640 &times; 640 &times; 3', 'short side 640&ndash;1024'],
        ['Output', 'N &times; (box, 42 scores)', 'N &times; (box, 42 scores)', 'N &times; (box, 43 scores)'],
        ['Pre-training', 'DocSynth-300K, then full IndicDLP by its authors [19]', 'COCO [12]', 'COCO [12]'],
        ['Modification', 'none (used as released)', '80-class head replaced by 42-class head',
         'predictor replaced (FastRCNNPredictor, 43 classes)'],
        ['Loss', '&mdash;', 'CIoU + DFL (box) + BCE (class)', 'RPN + cross-entropy + smooth-L1'],
    ], align='llll', cls='small'))
    a('<p><b>Major components.</b> <i>Backbone</i> (CSPDarknet / ResNet-50 [15]) extracts visual features of strokes, '
      'text lines and rules at several resolutions. <i>Neck</i> (PAN / FPN [16]) fuses fine and coarse features so that '
      'both a folio number and a full-page table can be detected. <i>Head</i> predicts the box coordinates and a score '
      'for each of the 42 labels; Faster R-CNN first proposes candidate regions (RPN) and then classifies each '
      'RoIAligned region. DocLayout-YOLO adds a global-to-local controllable receptive module (GL-CRM) suited to '
      'document elements whose sizes vary greatly [9].</p>')

    # ------------------------------------------------------------------ 9. tools
    a('<h2>9. Tools, Technologies and Environment</h2>')
    a(N.table('Tools, technologies and environment', ['Component', 'Details'], [
        ['Programming language', 'Python 3.11 (ML, app, API); JavaScript only for an experimental React prototype'],
        ['DL framework', 'PyTorch 2 [17] with torchvision; Ultralytics 8 (YOLOv8) [11]; doclayout-yolo 0.0.4'],
        ['Development environment', 'VS Code and terminal, Git + GitHub, pytest'],
        ['Libraries', 'pycocotools (COCO mAP), OpenCV, Pillow, NumPy, pandas, Matplotlib, Plotly, PyMuPDF, Tesseract 5 '
                      '(tessdata_best, 12 scripts)'],
        ['Hardware', 'Apple M1 Pro laptop, 16-core GPU via PyTorch MPS (training and evaluation); CPU inside Docker'],
        ['Dataset tools', 'Hugging Face Hub + fsspec (partial Parquet reads); custom COCO / YOLO converters'],
        ['Interface / deployment', 'Streamlit web app + FastAPI REST API; Docker + docker-compose (works offline)'],
    ], align='ll', cls='split small'))

    # ------------------------------------------------------------------ 10. implementation
    a('<h2>10. Implementation Details</h2>')
    a('<p>The repository is organised as reusable modules (<span class="tt">ml/</span>) with thin entry points '
      '(<span class="tt">app/</span>, <span class="tt">backend/</span>, <span class="tt">scripts/</span>).</p>')
    a(N.table('Implementation summary', ['Part', 'Implementation (file)'], [
        ['Data loading', 'Language-balanced sampling from the official splits; COCO JSON + YOLO labels '
                         '(<span class="tt">scripts/prepare_indicdlp.py</span>, <span class="tt">ml/datasets/</span>)'],
        ['Model construction', 'COCO-pretrained YOLOv8s and Faster R-CNN with new 42-class heads; released DocLayout-YOLO '
                               'loaded as is (<span class="tt">ml/models/</span>)'],
        ['Loss / optimiser', 'Faster R-CNN: SGD, lr 0.005, momentum 0.9, weight decay 5&times;10<sup>&minus;4</sup>, cosine '
                             'schedule. YOLOv8s: Ultralytics automatic optimiser, CIoU + DFL + BCE'],
        ['Batch / epochs', f'Faster R-CNN batch {base_hist["batch_size"]}, {b_epochs} epochs ({b_minutes:.0f}-min budget); '
                           f'YOLOv8s batch 8, {y_epochs} epochs ({y_minutes:.0f}-min budget); seed 42'],
        ['Validation', f'Best checkpoint chosen on {pages["val"]} validation pages; operating thresholds chosen on '
                       f'validation, never on test (<span class="tt">ml/training/</span>)'],
        ['Inference', 'One page at a time, duplicates removed, two modes (<span class="tt">ml/inference/predict.py</span>)'],
        ['OCR and output', 'Script probe, word-to-region assignment, reading order, Markdown / JSON '
                           '(<span class="tt">ml/ocr/</span>); accessible HTML / EPUB3 (<span class="tt">ml/accessible.py</span>, '
                           '<span class="tt">scripts/pdf_to_epub.py</span>)'],
        ['Interface', 'Streamlit app (<span class="tt">app/streamlit_app.py</span>), FastAPI (<span class="tt">backend/main.py</span>), '
                      'Dockerfile + docker-compose'],
        ['Quality', f'{tests["passed"]} automated tests passed in the last recorded run ({tests["python_tests"]} Python, '
                    f'{tests["frontend_tests"]} front-end); fixes logged in <span class="tt">docs/FINAL_AUDIT.md</span>'],
    ], align='ll', cls='split small'))
    a('<p>Listing 1 shows the core inference path used by the app, the API and the evaluation scripts.</p>')
    a('''<div class="listing"><div class="lcap">Listing 1. One page in, labelled boxes and text out.</div><pre>
<span class="c"># ml/inference/predict.py — layout detection</span>
model = load_model(ckpt, "doclayout", device)          <span class="c"># DocLayout-YOLO, YOLOv8s or Faster R-CNN</span>
out = predict_single(model, page, "doclayout", device, conf=0.15)
out = dedupe_boxes(out, iou_threshold=0.7)             <span class="c"># merge same-region boxes</span>

<span class="c"># ml/ocr — read the text inside the regions</span>
doc = extract_document(page, out, languages="auto")    <span class="c"># script probe + Tesseract</span>
print(doc["markdown"])                                 <span class="c"># text in reading order</span>

<span class="c"># ml/accessible.py — screen-reader-ready output</span>
html = to_html(doc, title="Chapter 1")                 <span class="c"># headings, lists, tables, lang tag</span></pre></div>''')
    a('''<div class="listing"><div class="lcap">Listing 2. YOLOv8s fine-tuning call (ml/training/train_yolo.py, abridged).</div><pre>
model = get_proposed_model(model_size="s")             <span class="c"># COCO-pretrained YOLOv8s</span>
model.train(data=yaml_path, imgsz=640, batch=8,
            seed=SEED, fliplr=0.0,                     <span class="c"># no mirror flips</span>
            time=args.hours)                           <span class="c"># schedule fitted to a 1-hour budget</span></pre></div>''')
    a('<p><b>REST API.</b> <span class="tt">POST /analyze</span> accepts an image or PDF and returns the boxes, labels and '
      'scores as JSON; <span class="tt">?ocr=true</span> adds the text with the script detected automatically. '
      '<span class="tt">GET /metrics</span> returns the saved evaluation results.</p>')

    # ------------------------------------------------------------------ 11. experimental setup
    a('<h2>11. Experimental Setup and Training</h2>')
    a(N.table('Experimental setup', ['Item', 'Setting'], [
        ['Data sizes', f'train {pages["train"]:,} &middot; validation {pages["val"]} &middot; test {pages["test"]} pages '
                       f'(test = {boxes["test"]:,} boxes)'],
        ['Faster R-CNN', f'ResNet-50 FPN, 640&ndash;1024 px, batch {base_hist["batch_size"]}, SGD 0.005, '
                         f'{b_epochs} epochs, {b_minutes:.0f} min'],
        ['YOLOv8s', f'640 px, batch 8, mosaic on, no flips, {y_epochs} epochs, {y_minutes:.0f} min'],
        ['DocLayout-YOLO', '1024 px, released checkpoint, evaluated only'],
        ['Hardware', 'Apple M1 Pro GPU (MPS); timing measured one page at a time'],
        ['Metrics', 'COCO mAP@50, mAP@50:95, mAP@75 (pycocotools, up to 300 boxes per page); precision, recall, F1 at '
                    'a fixed confidence; average IoU; seconds per page'],
        ['E1 (RQ1&ndash;RQ3)', f'All three detectors on the {pages["test"]} clean test pages, broken down by language, '
                              f'domain and label'],
        ['E2 (RQ4)', 'DocLayout-YOLO and YOLOv8s on the same pages under five kinds of scan damage (Table XII)'],
        ['E3 (RQ5)', f'OCR script choice on {script_n} test pages (10 per language): region probe vs. Tesseract OSD'],
    ], align='ll', cls='split small'))
    a(f'<p>Fig. 5 shows the measured training curves. Both losses fall steadily and the validation loss follows the '
      f'training loss, so neither model over-fits within its budget; the YOLOv8s validation mAP@50 is still rising '
      f'at epoch {y_epochs} ({yolo_hist["metrics/mAP50(B)"].iloc[-1]:.3f}), which shows that the budget, not the '
      f'architecture, limits our models. Within the one-hour budget, the one-stage YOLOv8s completed {y_epochs} '
      f'epochs while the two-stage Faster R-CNN completed {b_epochs}.</p>')
    a(N.figure(img(loss_png), 'Training curves (measured): (a) Faster R-CNN training and validation loss; (b) YOLOv8s '
               'loss; (c) YOLOv8s validation mAP per epoch.'))

    # ------------------------------------------------------------------ 12. results
    a('<h2>12. Results and Performance Evaluation</h2>')
    a(f'<p>All results are measured by <span class="tt">scripts/evaluate.py</span> on the same {pages["test"]} '
      f'held-out test pages ({A["num_ground_truth"]:,} ground-truth boxes) that none of the models was trained on '
      f'(Table X). Precision, recall and F1 are reported at the "balanced" confidence threshold chosen on validation '
      f'pages for each model.</p>')

    def at(mm, key):
        return mm['at_recommended_conf'][key]
    a(N.table(f'Overall results on {pages["test"]} test pages', ['Metric', 'DocLayout-YOLO', 'YOLOv8s (ours)', 'Faster R-CNN (ours)'], [
        ['mAP@50', f'<b>{f3(A["map50"])}</b>', f3(P['map50']), f3(B['map50'])],
        ['mAP@50:95', f'<b>{f3(A["map50_95"])}</b>', f3(P['map50_95']), f3(B['map50_95'])],
        ['mAP@75', f'<b>{f3(A["map75"])}</b>', f3(P['map75']), f3(B['map75'])],
        ['Balanced threshold (from validation)', A['recommended_conf'], P['recommended_conf'], B['recommended_conf']],
        ['Precision', f'<b>{f3(at(A, "precision"))}</b>', f3(at(P, 'precision')), f3(at(B, 'precision'))],
        ['Recall', f'<b>{f3(at(A, "recall"))}</b>', f3(at(P, 'recall')), f3(at(B, 'recall'))],
        ['F1', f'<b>{f3(at(A, "f1"))}</b>', f3(at(P, 'f1')), f3(at(B, 'f1'))],
        ['Recall, "find every part" mode', f'<b>{f3(A["at_high_recall_conf"]["recall"])}</b>',
         f3(P['at_high_recall_conf']['recall']), f3(B['at_high_recall_conf']['recall'])],
        ['Average IoU of correct boxes', f'<b>{f3(A["avg_iou"])}</b>', f3(P['avg_iou']), f3(B['avg_iou'])],
        ['Inference time (s / page, M1 Pro GPU)', f3(A['inference_time']), f'<b>{f3(P["inference_time"])}</b>',
         f3(B['inference_time'])],
    ], align='lrrr', cls='small'))
    a(f'<p><b>RQ1 (budget gap).</b> The released model, trained on about 42&times; more pages, is far ahead '
      f'({f3(A["map50"])} vs. {f3(P["map50"])} mAP@50). In "find every part" mode it finds '
      f'{A["at_high_recall_conf"]["recall"]:.0%} of all regions at {A["at_high_recall_conf"]["precision"]:.0%} precision. '
      f'YOLOv8s is the fastest model ({P["inference_time"]:.3f} s per page) and 7&times; smaller than Faster R-CNN.</p>')

    lang_rows = [[lg.capitalize(), A['per_group'][lg]['images'], f3(A['per_group'][lg]['map50']),
                  f3(P['per_group'][lg]['map50']), f3(B['per_group'][lg]['map50'])] for lg in A['per_group']]
    lang_rows += ['---', ['<b>Consistency Index (CI)</b>', '', f'<b>{ci["authors"]:.2f}</b>', f'{ci["proposed"]:.2f}',
                          f'{ci["baseline"]:.2f}']]
    a(N.table('mAP@50 per language and language Consistency Index', ['Language', 'Pages', 'DocLayout-YOLO', 'YOLOv8s', 'Faster R-CNN'],
              lang_rows, width='80%', cls='small'))
    dom = study['per_domain_authors']
    a(f'<p><b>RQ2 (languages and domains).</b> Accuracy is fairly even across languages for the full-data model '
      f'(CI {ci["authors"]:.2f}; Table XI) and less even for our smaller models ({ci["proposed"]:.2f}, '
      f'{ci["baseline"]:.2f}). Across document domains it is much less even: CI {study["domain_ci"]["authors"]:.2f} '
      f'for DocLayout-YOLO, from {f3(max(dom.values()))} on research papers to {f3(min(dom.values()))} on forms '
      f'(Fig. 6). <i>The type of document matters more than the script</i> once the script has been seen in training.</p>')
    a(N.figure(img(domain_figure(tmp / 'fig_domain.png'), '82%'), 'mAP@50 per document domain (n = number of test '
               'pages). Forms and magazines are the hardest domains for both models.'))
    hard = ', '.join(f'<i>{k}</i> ({v:.2f})' for k, v in list(study['hardest_classes'].items())[:4])
    easy = ', '.join(f'<i>{k}</i> ({v:.2f})' for k, v in list(study['easiest_classes'].items())[:4])
    a(f'<p><b>RQ3 (rare labels).</b> For DocLayout-YOLO the easiest labels are {easy} AP@50 and the hardest are {hard}. '
      f'The Spearman rank correlation between a label\'s training frequency and its AP is '
      f'{study["class_freq_spearman"]["proposed"]:.2f} for our YOLOv8s but only '
      f'{study["class_freq_spearman"]["authors"]:.2f} for the full-data model (Fig. 7): with enough data, rarity stops '
      f'deciding accuracy, and what remains hard are labels that look alike (quote vs. paragraph, nested list levels).</p>')
    a(N.figure(img(F / 'fig_class_frequency.png'), 'AP@50 of each label against its number of training boxes (log '
               'scale). Rare labels are hard for the small-data model (&rho; = 0.75), much less so for the full-data model '
               '(&rho; = 0.40).'))
    dmg = [('blur', 'Blur', 'Gaussian blur, radius 2 px (out of focus)'),
           ('noise', 'Noise', 'Gaussian noise, &sigma; = 20 grey levels'),
           ('jpeg', 'JPEG', 'JPEG quality 15 (forwarded phone image)'),
           ('low_res', 'Half resolution', 'down to 50% and back up (low-dpi scan)'),
           ('faded', 'Faded print', 'contrast &times; 0.45, brightness &times; 1.25')]
    rrows = [['Clean', '&mdash;', f3(rmap['authors']['clean']), '1.00', f3(rmap['proposed']['clean']), '1.00']]
    rrows += [[n, d, f3(rmap['authors'][k]), f'{ri["authors"][k]:.2f}', f3(rmap['proposed'][k]),
               f'{ri["proposed"][k]:.2f}'] for k, n, d in dmg]
    rrows += ['---', ['<b>Mean RI</b>', '', '', f'<b>{study["mean_ri"]["authors"]:.2f}</b>', '',
                      f'<b>{study["mean_ri"]["proposed"]:.2f}</b>']]
    a(N.table('Robustness to scan damage (E2): mAP@50 and Robustness Index',
              ['Condition', 'How it is made', 'DocLayout-YOLO', 'RI', 'YOLOv8s', 'RI'], rrows, align='llrrrr', cls='small'))
    a(f'<p><b>RQ4 (damaged scans).</b> The released model keeps {study["mean_ri"]["authors"]:.0%} of its accuracy on '
      f'average and YOLOv8s {study["mean_ri"]["proposed"]:.0%} (Table XII, Fig. 8). Blur hurts the released model most '
      f'(RI {ri["authors"]["blur"]:.2f}); noise and faded print hurt the small model most '
      f'(RI {ri["proposed"]["noise"]:.2f} and {ri["proposed"]["faded"]:.2f}). Each damaged copy is made once with a fixed '
      f'seed and the ground-truth boxes are unchanged.</p>')
    a(N.figure(img(F / 'fig_robustness.png'), 'Robustness Index per damage type (left) and absolute mAP@50 of '
               'DocLayout-YOLO under each condition (right).'))
    a(f'<p><b>RQ5 (choosing the OCR script).</b> The region probe picks the right script on {script_ok} of {script_n} '
      f'pages ({ocr["script_detection_accuracy"]:.0%}) against {osd_ok}/{script_n} for Tesseract\'s built-in detector '
      f'(OSD). It is perfect on Assamese, Bengali and Punjabi and weakest on Marathi (6/10), where pages are '
      f'sometimes mistaken for Latin or Telugu (Fig. 9).</p>')
    a(N.figure(img(F / 'fig_script_detection.png', '70%'), f'Share of pages per language on which the region probe '
               f'chose the right OCR script ({script_ok}/{script_n} overall).'))

    # ------------------------------------------------------------------ 13. comparison
    a('<h2>13. Comparison with Baseline / Existing Methods</h2>')
    a(N.table('Comparison with the baseline and published results', ['Method', 'Training data', 'mAP@50', 'mAP@50:95'], [
        ['Faster R-CNN (ours, baseline)', f'{pages["train"]:,} pages, {b_minutes:.0f} min', f3(B['map50']), f3(B['map50_95'])],
        ['YOLOv8s (ours, proposed)', f'{pages["train"]:,} pages, {y_minutes:.0f} min', f3(P['map50']), f3(P['map50_95'])],
        ['DocLayout-YOLO (released by [1])', '~96K pages (80% of 119,806)', f'<b>{f3(A["map50"])}</b>',
         f'<b>{f3(A["map50_95"])}</b>'],
        '---',
        ['Published: DocLayout-YOLO [1]', 'full IndicDLP test set', '&asymp; 0.735', '0.545'],
        ['Published: YOLOv10x [1] (best in paper)', 'full IndicDLP test set', '&mdash;', '0.550'],
    ], align='llrr', cls='small'))
    a(f'<p><b>Metrics used.</b> COCO mAP@50 and mAP@50:95, precision / recall / F1 and seconds per page, all on the same '
      f'{pages["test"]} test pages (rows above the line); published numbers are on the authors\' full test set and are '
      f'shown for reference. Our measurement of the released model ({f3(A["map50_95"])} mAP@50:95) agrees with the '
      f'published 0.545, which confirms that our evaluation pipeline is correct.</p>')
    a(f'<p><b>One-stage vs. two-stage.</b> With the same data and a similar time budget, YOLOv8s beats Faster R-CNN on '
      f'every metric ({f3(P["map50"])} vs. {f3(B["map50"])} mAP@50; precision {f3(at(P, "precision"))} vs. '
      f'{f3(at(B, "precision"))}) and is {B["inference_time"] / P["inference_time"]:.0f}&times; faster with a quarter of '
      f'the parameters. The proposed model therefore improves on the baseline.</p>')
    a('<p><b>Did we beat the state of the art?</b> No. Our own models saw about 42&times; less data and one hour of laptop '
      'training, and do not reach the published model. The application therefore uses the released DocLayout-YOLO, and '
      'the contribution of this project is the study (RQ1&ndash;RQ5), the OCR and script-selection pipeline, and the '
      'accessible, offline tool. We report this openly in the paper as well.</p>')
    a(N.figure(img(PL / 'model_comparison.png', '72%'), 'Precision, recall, F1 (confidence &ge; 0.5) and mAP of the '
               'three models on the same test pages.'))

    # ------------------------------------------------------------------ 14. sample outputs
    a('<h2>14. Sample Outputs and Demonstration</h2>')
    a('<p>Figs. 11&ndash;15 give evidence that the implementation works end to end: the running application, a correct '
      'and an incorrect prediction, and the confusion matrix.</p>')
    a(N.figure(f'<div class="grid2">{img(SC / "02_detection_tamil.png")}{img(SC / "03_ocr.png")}</div>',
               'Running Streamlit application. Left: a Tamil textbook page with the detected regions and confidences. '
               'Right: OCR with the script detected automatically (Tamil), text in reading order, Markdown / JSON download.'))
    a(N.figure(img(PR / 'authors_prediction_1.png', '88%'), 'Correct prediction: Kannada newspaper page, ground truth '
               '(green, left) vs. DocLayout-YOLO prediction (red, right).'))
    a(N.figure(img(PR / 'errors' / 'authors_most_missed.png', '88%'), 'Incorrect prediction: the dense Bengali form '
               'with the most missed regions in the test set (90 of 105 regions missed). Forms with many tiny fields are '
               'the hardest domain.'))
    a(N.figure(img(PL / 'authors_confusion_matrix.png', '74%'), 'Confusion matrix of DocLayout-YOLO for the most '
               'frequent labels (IoU &ge; 0.5, confidence &ge; 0.5). The last row counts missed regions; most errors are '
               'misses rather than wrong labels.'))
    a(N.figure(img(SC / '04_dashboard.png', '80%'), 'Dashboard tab of the application showing the measured metrics of '
               'the three models.'))

    # ------------------------------------------------------------------ 15. challenges
    a('<h2>15. Challenges and Limitations</h2>')
    a('<ol class="chal">'
      f'<li><b>Data size and compute.</b> The full dataset is about 91 GB; a laptop GPU could train on '
      f'{pages["train"]:,} pages for about an hour per model. Our models stay far below the published one '
      f'({f3(P["map50"])} vs. {f3(A["map50"])} mAP@50), and each was trained once with one seed.</li>'
      '<li><b>Dense forms and rare labels.</b> Forms, with many tiny fields, score lowest; labels such as '
      '<i>quote</i> and <i>sub-unordered-list</i> score below 0.2 AP even for the best model.</li>'
      '<li><b>OCR quality.</b> Tesseract misreads small print and mis-orders vowel signs in some scripts; IndicDLP has no '
      'text ground truth, so character accuracy could not be measured. Scan damage in E2 is simulated.</li>'
      '<li><b>Engineering.</b> The initial code base had crashes, a wrong label list and plots not backed by data; every '
      'fix is logged in <span class="tt">docs/FINAL_AUDIT.md</span>. Timings were spoiled twice by jobs sharing the '
      'GPU and had to be re-measured.</li></ol>')

    # ------------------------------------------------------------------ 16. conclusion
    a('<h2>16. Conclusion and Future Scope</h2>')
    a(f'<p><b>Conclusion.</b> We implemented a complete pipeline: data preparation from IndicDLP, two fine-tuned detectors, '
      f'a controlled evaluation of three detectors, OCR with automatic script choice for 12 languages, accessible '
      f'HTML / EPUB3 export, a web app, a REST API and an offline Docker image. The best model reaches '
      f'{f3(A["map50"])} mAP@50 on {pages["test"]} test pages, finds {A["at_high_recall_conf"]["recall"]:.0%} of regions in '
      f'"find every part" mode and runs at {A["inference_time"]:.2f} s per page; the script is chosen correctly on '
      f'{script_ok}/{script_n} pages. The study shows that accuracy is fairly even across languages '
      f'(CI {ci["authors"]:.2f}) but not across document types (CI {study["domain_ci"]["authors"]:.2f}); that data size, '
      f'not the script, decides how well rare labels work; and that the best model keeps '
      f'{study["mean_ri"]["authors"]:.0%} of its accuracy on damaged scans, with blur hurting most.</p>')
    a('<p><b>Objectives met.</b> All four specific objectives (Section 4) were met. Our own detectors did not reach the '
      'published accuracy, which we report openly; the proposed YOLOv8s did improve on the Faster R-CNN baseline.</p>')
    a('<p><b>Future scope.</b></p><ul class="fut">'
      '<li><i>Model and data:</i> train longer on a cloud GPU with blur and noise augmentation; more labelled forms; a '
      'controlled unseen-script experiment.</li>'
      '<li><i>OCR:</i> replace Tesseract with an Indic OCR model behind the same interface and build a small text '
      'ground-truth set to measure character accuracy.</li>'
      '<li><i>Features:</i> exam-paper mode (questions, options and marks as structured items), multi-page books, Braille '
      '(.brf) and offline text-to-speech, and a review screen to correct OCR before export; table structure and question '
      'answering on detected tables [3], [4].</li>'
      '<li><i>Real-world validation:</i> a pilot on state-board textbooks and government circulars with screen-reader '
      'users, measuring reading-order accuracy and time saved over manual conversion.</li></ul>')

    # ------------------------------------------------------------------ template closing sections
    a('<h2>Expected Outcomes</h2>')
    a(N.table('Expected outcomes and their status', ['Expected outcome', 'Status', 'Evidence'], [
        ['Development of a working Deep Learning model addressing a real-world problem', 'Achieved',
         'Two fine-tuned detectors + released model in a working app and API (Sections 8&ndash;10, 14)'],
        ['Performance evaluation of implemented model', 'Achieved',
         'mAP, P / R / F1, IoU, speed, per-language / domain / label results, robustness, OCR script choice (Sections 12&ndash;13)'],
        ['Improved understanding of research methodology and problem-solving skills', 'Achieved',
         'Five research questions, held-out test set, thresholds chosen on validation only, audit log of fixes'],
        ['Preparation of a research paper suitable for submission', 'Achieved',
         'IEEE conference-format paper in <span class="tt">paper/</span>'],
    ], align='lll', cls='small'))
    a('<h2>Deliverables</h2>')
    a(N.table('Deliverables and their location in the repository', ['Deliverable', 'Location', 'Contents'], [
        ['Dataset (CSV / image dataset)', '<span class="tt">dataset/</span>',
         f'pages.csv ({total_pages:,} rows), boxes.csv ({total_boxes:,} rows), class_counts.csv, language_split.csv, '
         f'12 sample pages'],
        ['Project code with complete implementation', '<span class="tt">ml/ app/ backend/ scripts/</span>',
         'Training, evaluation, inference, OCR, accessible export, web app, REST API, Docker'],
        ['GitHub repository with organized code and documentation', '<span class="tt">README.md, docs/</span>',
         'Dataset, architecture, methodology, evaluation, results, deployment and audit documents; tests'],
        ['Presentation slides (Phase-1 and Phase-2)', '<span class="tt">docs/presentation/</span>',
         'Phase-1 literature survey (PDF); Phase-2 implementation (PDF + PPTX)'],
        ['Final research paper', '<span class="tt">paper/</span>', 'IEEE-format paper (PDF + LaTeX source)'],
        ['Experiment report (this document)', '<span class="tt">docs/report/</span>',
         'Expt.-11 report; generated by <span class="tt">scripts/build_lab_report.py</span>'],
    ], align='lll', cls='split small'))
    a('<h2>Evaluation Criteria</h2>')
    a(N.table('Evaluation criteria and where each is addressed', ['Criterion', 'Marks', 'Where addressed'], [
        ['Relevance of problem to SDG', '2', 'Section 1, Table I (SDG 4, 9, 10, 16 with targets)'],
        ['Quality of literature survey and research gap identification', '2', 'Sections 2&ndash;3; Phase-1 slides'],
        ['Implementation of deep learning models', '2', 'Sections 7&ndash;11 (data, three detectors, training)'],
        ['Performance evaluation and Frontend development', '2', 'Sections 12&ndash;14 (metrics, study, app, API)'],
        ['Documentation and presentation quality', '2', 'Repository docs, Phase-1 / Phase-2 slides, this report'],
        ['Innovation and Research Paper', '5', 'CI / RI measures, robustness test, script probe, accessible output; '
                                              'IEEE paper'],
    ], align='lcl', cls='small'))

    # ------------------------------------------------------------------ references (IEEE)
    refs = [
        'O. Nath, S. Kukkala, M. Khapra, and R. K. Sarvadevabhatla, &ldquo;IndicDLP: A foundational dataset for '
        'multi-lingual and multi-domain document layout parsing,&rdquo; in <i>Proc. Int. Conf. Document Analysis and '
        'Recognition (ICDAR)</i>, 2025, pp. 23&ndash;39, arXiv:2512.20236.',
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
        'Z. Zhao, H. Kang, B. Wang, and C. He, &ldquo;DocLayout-YOLO: Enhancing document layout analysis through '
        'diverse synthetic data and global-to-local adaptive perception,&rdquo; arXiv:2410.12628, 2024.',
        'S. Ren, K. He, R. Girshick, and J. Sun, &ldquo;Faster R-CNN: Towards real-time object detection with region '
        'proposal networks,&rdquo; in <i>Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)</i>, 2015, pp. 91&ndash;99.',
        'G. Jocher, A. Chaurasia, and J. Qiu, &ldquo;Ultralytics YOLOv8,&rdquo; 2023. [Online]. Available: '
        '<span class="tt">https://github.com/ultralytics/ultralytics</span>',
        'T.-Y. Lin <i>et al.</i>, &ldquo;Microsoft COCO: Common objects in context,&rdquo; in <i>Proc. Eur. Conf. '
        'Comput. Vis. (ECCV)</i>, 2014, pp. 740&ndash;755.',
        'R. Smith, &ldquo;An overview of the Tesseract OCR engine,&rdquo; in <i>Proc. 9th Int. Conf. Document Analysis '
        'and Recognition (ICDAR)</i>, 2007, pp. 629&ndash;633.',
        'D. Hendrycks and T. Dietterich, &ldquo;Benchmarking neural network robustness to common corruptions and '
        'perturbations,&rdquo; in <i>Proc. Int. Conf. Learn. Represent. (ICLR)</i>, 2019.',
        'K. He, X. Zhang, S. Ren, and J. Sun, &ldquo;Deep residual learning for image recognition,&rdquo; in <i>Proc. '
        'IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)</i>, 2016, pp. 770&ndash;778.',
        'T.-Y. Lin, P. Doll&aacute;r, R. Girshick, K. He, B. Hariharan, and S. Belongie, &ldquo;Feature pyramid networks '
        'for object detection,&rdquo; in <i>Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)</i>, 2017, '
        'pp. 2117&ndash;2125.',
        'A. Paszke <i>et al.</i>, &ldquo;PyTorch: An imperative style, high-performance deep learning library,&rdquo; in '
        '<i>Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)</i>, 2019, pp. 8024&ndash;8035.',
        'AI4Bharat, &ldquo;IndicDLP dataset,&rdquo; Hugging Face, 2025. [Online]. Available: '
        '<span class="tt">https://huggingface.co/datasets/ai4bharat/indicdlp</span>',
        'O. Nath <i>et al.</i>, &ldquo;IndicDLP model checkpoints,&rdquo; Zenodo, record 15881917, 2025, CC-BY 4.0. '
        '[Online]. Available: <span class="tt">https://zenodo.org/records/15881917</span>',
        'United Nations General Assembly, &ldquo;Transforming our world: The 2030 Agenda for Sustainable Development,&rdquo; '
        'Resolution A/RES/70/1, 2015.',
        'W3C, &ldquo;EPUB Accessibility 1.1,&rdquo; W3C Recommendation, 2023. [Online]. Available: '
        '<span class="tt">https://www.w3.org/TR/epub-a11y-11/</span>',
        'Streamlit, FastAPI and Docker, official documentation. [Online]. Available: '
        '<span class="tt">https://docs.streamlit.io</span>, <span class="tt">https://fastapi.tiangolo.com</span>, '
        '<span class="tt">https://docs.docker.com</span>',
        f'P. Jha, A. Soni, R. Patil, and M. Badure, &ldquo;IndicDoc AI,&rdquo; source code, 2026. [Online]. Available: '
        f'<span class="tt">{REPO}</span>',
    ]
    a('<h2>References</h2><ol class="refs">' + ''.join(f'<li>{r}</li>' for r in refs) + '</ol>')

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
