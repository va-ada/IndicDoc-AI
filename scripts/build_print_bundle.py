"""
IndicDoc AI - the whole submission as one print-ready PDF (A4, for double-sided printing).

    Cover and contents (one sheet)
    Part 1  Dataset                          IndicDLP subset (dataset/*.csv, samples) and the verification benchmark
    Part 2  Project code and implementation  architecture, structure, API, tests, screenshots, source listings
    Part 3  GitHub repository                the repository page (files, branches, commits)
    Part 4  Phase-1 presentation             literature survey deck, two slides per page
    Part 5  Phase-2 presentation             implementation deck, two slides per page
    Part 6  Research paper                   IEEE paper on IndicDoc Verify, scaled from US Letter to A4

Every part starts on the front of a new sheet: a part with an odd number of pages is followed by a page marked
"intentionally left blank", so one double-sided print separates cleanly into the parts. Every page carries a
running footer (part and page number) and the PDF has one bookmark per part.

The repository is private, so its GitHub page cannot be captured without the owner's sign-in. By default Part 3 is
built from the repository itself (git); pass browser screenshots of the GitHub page to print those instead.

Usage:
    python scripts/build_print_bundle.py                       # -> docs/print/IndicDoc_AI_Complete_Submission.pdf
    python scripts/build_print_bundle.py --repo-screenshot github_page.png [more.png ...]

Needs Chrome/Chromium (as the other document builders) and pip install pypdf pygments qrcode.
"""
import argparse
import html
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import date
from io import BytesIO
from pathlib import Path

import pandas as pd
import qrcode
import qrcode.image.svg
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_for_filename
from pypdf import PageObject, PdfReader, PdfWriter, Transformation

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from build_lab_report import GUIDE, REPO, TEAM, TITLE, block_diagram, find_chrome  # noqa: E402

OUT = ROOT / 'docs' / 'print' / 'IndicDoc_AI_Complete_Submission.pdf'
FONTS = ROOT / 'docs' / 'report' / 'assets' / 'fonts'
METRICS = ROOT / 'outputs' / 'metrics'
FIG, SHOTS, DATA = ROOT / 'docs' / 'figures', ROOT / 'docs' / 'screenshots', ROOT / 'dataset'
PHASE1 = ROOT / 'docs' / 'presentation' / 'IndicDoc_AI_Phase1_Literature_Survey.pdf'
PHASE2 = ROOT / 'docs' / 'presentation' / 'IndicDoc_AI_Phase2_Implementation.pdf'
PAPER = ROOT / 'paper' / 'IndicDoc_Verify_Research_Paper.pdf'
A4W, A4H = 595.276, 841.89  # points

# As listed on GitHub (repository settings)
REPO_DESCRIPTION = ('Layout detection + OCR for multilingual Indian documents (IndicDLP): DocLayout-YOLO, YOLOv8, '
                    'Faster R-CNN, Tesseract; Streamlit app + FastAPI in Docker')

# Source files printed in Part 2, in reading order, with what each does
CODE = [
    ('ml/verify/__init__.py', 'Entry point: verify() parses the paper, runs every check and returns the report'),
    ('ml/verify/text.py', 'Text helpers: Indian digits, numerals, question numbers, marks, option splitting, similarity'),
    ('ml/verify/paper.py', 'Builds the questions (number, stem, options, marks) from the layout regions and OCR words'),
    ('ml/verify/checks.py', 'Structure checks, alignment of two language versions (translation parity), repeats'),
    ('ml/verify/profiles.py', 'Format profiles: any document type described as JSON rules'),
    ('ml/verify/profiles/question_paper.json', 'The question-paper profile'),
    ('ml/verify/report.py', 'The report: findings outlined on the pages, HTML output'),
    ('ml/ocr/__init__.py', 'Tesseract OCR of the detected regions, script detection, the number pass, reading order'),
    ('ml/inference/predict.py', 'Loads a layout detector (DocLayout-YOLO, YOLOv8, Faster R-CNN) and predicts regions'),
    ('ml/models/proposed_yolov8.py', 'YOLOv8s model builder and COCO-to-YOLO conversion'),
    ('ml/training/train_yolo.py', 'Fine-tuning YOLOv8s on the IndicDLP subset'),
    ('ml/models/baseline_frcnn.py', 'Faster R-CNN (ResNet-50 FPN) baseline builder'),
    ('backend/main.py', 'FastAPI REST API: /analyze (layout + OCR) and /verify (pre-print check)'),
]

# Phase-1 deck: two lines whose word spaces were lost when the slides were exported to images.
# (page, left, top, width, height, text, css) in slide points (960 x 540); the patch repaints the line.
PHASE1_PATCHES = [
    (0, 50, 122, 512, 30, 'Document Layout Parsing for Multilingual Indian Documents',
     'left:54.6pt;top:124.6pt;font-size:18pt;color:#CBD5E1;'),
    (18, 350, 265, 270, 30, 'Questions and discussion',
     'left:0;width:960pt;text-align:center;top:267.8pt;font-size:17.4pt;color:#D9A441;'),
]
SLIDE_BG = '#112848'  # the slides' background as Chrome prints it


# ----------------------------------------------------------------------------------------------------- helpers
def esc(s) -> str:
    return html.escape(str(s))


def uri(p: Path) -> str:
    return Path(p).resolve().as_uri()


def font_faces() -> str:
    face = ''.join(
        f"@font-face{{font-family:'CMU Serif';src:url('{uri(FONTS / f'cmu-serif-{w}-{st}.woff2')}') format('woff2');"
        f"font-weight:{w};font-style:{'italic' if st == 'italic' else 'normal'};}}"
        for w in (500, 700) for st in ('roman', 'italic'))
    return face


GOOGLE_FONTS = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@400'
                '&family=Noto+Sans+Tamil:wght@400&display=block">')

BASE_CSS = '''
@page { size: A4; margin: 16mm 16mm 19mm 16mm; }
* { box-sizing: border-box; }
html, body { margin: 0; background: #fff; color: #111; }
body { font-family: 'CMU Serif', 'Noto Sans Devanagari', 'Noto Sans Tamil', serif; font-size: 10.4pt; line-height: 1.38;
       font-weight: 500; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
h1 { font-size: 21pt; margin: 0 0 2pt; font-weight: 700; letter-spacing: -0.2pt; }
.kicker { font-family: Inter, sans-serif; font-size: 8pt; letter-spacing: 1.6pt; text-transform: uppercase;
          color: #8A6A1E; font-weight: 600; margin-bottom: 4pt; }
.lede { font-size: 11pt; color: #333; margin: 0 0 10pt; }
.rule { border-top: 1.4pt solid #14284B; margin: 6pt 0 12pt; }
h2 { font-size: 13pt; margin: 14pt 0 5pt; font-weight: 700; break-after: avoid; color: #14284B; }
h3 { font-size: 11pt; margin: 10pt 0 4pt; font-weight: 700; break-after: avoid; }
p { margin: 0 0 6pt; text-align: justify; hyphens: auto; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 10pt; font-size: 9.2pt; break-inside: auto; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th, td { padding: 2.6pt 5pt; text-align: left; vertical-align: top; }
thead th { border-top: 1.1pt solid #111; border-bottom: 0.7pt solid #111; font-weight: 700; }
tbody tr:last-child td { border-bottom: 1.1pt solid #111; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
tr.tot td { border-top: 0.6pt solid #111; font-weight: 700; }
.cap { font-size: 8.8pt; color: #333; margin: 3pt 0 10pt; text-align: center; }
.cap b { color: #111; }
.tt, code { font-family: 'DejaVu Sans Mono', 'Noto Sans Devanagari', monospace; font-size: 8.3pt; }
pre.cmd { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.2pt; background: #F4F5F8; border: 0.6pt solid #D5D9E2;
          padding: 6pt 8pt; margin: 4pt 0 10pt; white-space: pre-wrap; break-inside: avoid; line-height: 1.45; }
.grid { display: grid; gap: 8pt 10pt; }
.grid figure { margin: 0; break-inside: avoid; }
.grid img { width: 100%; border: 0.6pt solid #BBB; display: block; }
.grid figcaption { font-size: 8.4pt; text-align: center; margin-top: 2pt; color: #333; }
.shot { break-inside: avoid; margin: 4pt 0 10pt; }
.shot img { width: 100%; border: 0.6pt solid #BBB; display: block; }
.newpage { break-before: page; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 0 14pt; }
.note { font-size: 9pt; color: #333; border-left: 2pt solid #D9A441; padding: 2pt 0 2pt 7pt; margin: 6pt 0 10pt; }
'''


def page_html(title: str, body: str, css: str = '') -> str:
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{esc(title)}</title>{GOOGLE_FONTS}'
            f'<style>{font_faces()}{BASE_CSS}{css}</style></head><body>{body}</body></html>')


def print_pdf(doc: str, out: Path, tmp: Path) -> Path:
    src = tmp / (out.stem + '.html')
    src.write_text(doc, encoding='utf-8')
    sandbox = ['--no-sandbox'] if hasattr(os, 'geteuid') and os.geteuid() == 0 else []
    subprocess.run([find_chrome(), *sandbox, '--headless=new', '--disable-gpu', '--no-pdf-header-footer',
                    '--allow-file-access-from-files', '--virtual-time-budget=15000', f'--print-to-pdf={out}',
                    src.as_uri()], check=True, capture_output=True, timeout=300)
    return out


def table(head, rows, num=(), cls='', total=None) -> str:
    th = ''.join(f'<th class="{"n" if i in num else ""}">{h}</th>' for i, h in enumerate(head))
    body = ''.join('<tr>' + ''.join(f'<td class="{"n" if i in num else ""}">{c}</td>' for i, c in enumerate(r)) + '</tr>'
                   for r in rows)
    if total:
        body += '<tr class="tot">' + ''.join(f'<td class="{"n" if i in num else ""}">{c}</td>'
                                             for i, c in enumerate(total)) + '</tr>'
    return f'<table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'


def git(*args) -> str:
    return subprocess.run(['git', '-C', str(ROOT), *args], check=True, capture_output=True, text=True).stdout.strip()


def fmt(n) -> str:
    return f'{int(n):,}'


def header(part: int, title: str, lede: str) -> str:
    return (f'<div class="kicker">Part {part}</div><h1>{esc(title)}</h1><p class="lede">{lede}</p>'
            '<div class="rule"></div>')


# ------------------------------------------------------------------------------------------------ Part 1: dataset
def class_chart(classes: pd.DataFrame, tmp: Path) -> str:
    """Bar chart of training boxes per label (log scale), question-paper labels highlighted."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    c = classes.sort_values('train', ascending=False)
    q = {'first-level-question', 'second-level-question', 'third-level-question', 'options', 'answer'}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7})
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    ax.bar(range(len(c)), c.train.clip(lower=0.8), color=['#D9A441' if n in q else '#14284B' for n in c.class_name], width=0.75)
    ax.set_yscale('log')
    ax.set_xticks(range(len(c)), c.class_name, rotation=90, fontsize=6)
    ax.set_xlim(-0.7, len(c) - 0.3)
    ax.set_ylabel('training boxes (log scale)')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.grid(axis='y', alpha=0.3, lw=0.5)
    fig.tight_layout()
    out = tmp / 'class_counts.png'
    fig.savefig(out, dpi=250)
    plt.close(fig)
    return uri(out)


def dataset_part(tmp: Path) -> str:
    pages = pd.read_csv(DATA / 'pages.csv')
    classes = pd.read_csv(DATA / 'class_counts.csv')
    langs = pd.read_csv(DATA / 'language_split.csv')
    boxes_head = pd.read_csv(DATA / 'boxes.csv', nrows=3)
    splits = ('train', 'val', 'test')
    n_pages = {s: int((pages.split == s).sum()) for s in splits}
    n_boxes = {s: int(classes[s].sum()) for s in splits}

    s = [header(1, 'Dataset', 'The IndicDLP subset used to train and test the layout detectors, the data files '
                'delivered with the project, and the benchmark used to measure IndicDoc Verify.')]
    a = s.append
    a('<h2>1.1 Source: IndicDLP</h2>')
    a('<p>IndicDLP is a human-labelled dataset for document layout parsing in Indian languages, released by IIT Madras '
      'and IIIT Hyderabad at ICDAR 2025. Each page is labelled with rectangular regions of 42 kinds, including the '
      'question-paper labels this project depends on: first-, second- and third-level questions, options and answers.</p>')
    a(table(['Property', 'Details'], [
        ('Paper', 'O. Nath, S. Kukkala, M. Khapra, R. K. Sarvadevabhatla, &ldquo;IndicDLP: A foundational dataset for '
                  'multi-lingual and multi-domain document layout parsing,&rdquo; ICDAR 2025, LNCS, pp. 23&ndash;39'),
        ('Download', '<span class="tt">huggingface.co/datasets/ai4bharat/indicdlp</span> (gated; also listed on '
                     'AIKosh, the IndiaAI dataset platform)'),
        ('Size', '119,806 pages; Parquet shards (train 73.3 GB, validation 8.9 GB, test 8.7 GB)'),
        ('Annotation', 'PNG page image; boxes <span class="tt">[x, y, width, height]</span> in pixels; class ids 0&ndash;41'),
        ('Languages (12)', 'Assamese, Bengali, English, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, '
                           'Tamil, Telugu'),
        ('Domains (12)', 'acts &amp; rules, brochures, forms, magazines, manuals, newspapers, notices, novels, question '
                         'papers, research papers, syllabi, textbooks'),
        ('Licence', 'MIT (Hugging Face dataset page)'),
    ]))

    a('<h2>1.2 Subset used in this project</h2>')
    a(f'<p>A language-balanced subset is drawn from the official splits by <span class="tt">scripts/prepare_indicdlp.py'
      f'</span>, so no test page was seen in training, by us or by the dataset authors. It has '
      f'{fmt(sum(n_pages.values()))} pages and {fmt(sum(n_boxes.values()))} labelled regions. Boxes under 1 px after '
      'clipping to the page were dropped (165 of 33,832 training boxes, 0.5%); 15 training pages without labelled '
      'elements are kept as background examples.</p>')
    a(table(['Split', 'Pages', 'Boxes', 'Boxes per page'],
            [(n, fmt(n_pages[k]), fmt(n_boxes[k]), f'{n_boxes[k] / n_pages[k]:.1f}')
             for k, n in zip(splits, ('Train', 'Validation', 'Test'))], num=(1, 2, 3),
            total=('Total', fmt(sum(n_pages.values())), fmt(sum(n_boxes.values())),
                   f'{sum(n_boxes.values()) / sum(n_pages.values()):.1f}')))
    lang_rows = [(r.language.capitalize(), r.train, r.val, r.test, r.total) for r in langs.itertuples()]
    dom = pd.crosstab(pages.domain, pages.split)[list(splits)]
    dom_rows = [(d, *(int(dom.loc[d, k]) for k in splits), int(dom.loc[d].sum())) for d in dom.index]
    a('<div class="two" style="break-inside:avoid"><div><h3>Pages by language</h3>')
    a(table(['Language', 'Train', 'Val', 'Test', 'Total'], lang_rows, num=(1, 2, 3, 4),
            total=('Total', *(int(langs[k].sum()) for k in (*splits, 'total')))))
    a('</div><div><h3>Pages by domain</h3>')
    a(table(['Domain', 'Train', 'Val', 'Test', 'Total'], dom_rows, num=(1, 2, 3, 4),
            total=('Total', *(int(dom[k].sum()) for k in splits), int(dom.values.sum()))))
    a('</div></div>')

    a('<h2>1.3 Data files delivered</h2>')
    a('<p>The folder <span class="tt">dataset/</span> of the repository describes every page and box used, so the '
      'subset can be rebuilt exactly; the 2.3 GB of images are rebuilt from Hugging Face by '
      '<span class="tt">scripts/prepare_indicdlp.py</span> and are not stored in git.</p>')
    a(table(['File', 'One row per', 'Columns'], [
        ('<span class="tt">pages.csv</span>', f'page ({fmt(len(pages))})', 'split, file_name, language, domain, width, height, num_boxes'),
        ('<span class="tt">boxes.csv</span>', f'labelled region ({fmt(sum(n_boxes.values()))})',
         'split, file_name, class_id, class_name, x, y, width, height (pixels, top-left origin)'),
        ('<span class="tt">class_counts.csv</span>', f'label ({len(classes)})', 'class_id, class_name, train, val, test, total'),
        ('<span class="tt">language_split.csv</span>', f'language ({len(langs)})', 'language, train, val, test, total'),
        ('<span class="tt">samples/</span>', 'language (12)', 'one real test page with its ground-truth boxes drawn'),
    ]))
    a('<h3>First rows of pages.csv</h3>')
    a(table(list(pages.columns), pages.head(4).values.tolist(), num=(4, 5, 6)))
    a('<h3>First rows of boxes.csv</h3>')
    a(table(list(boxes_head.columns), [[f'{v:.1f}' if isinstance(v, float) else v for v in r]
                                       for r in boxes_head.values.tolist()], num=(2, 4, 5, 6, 7)))

    a('<h2 class="newpage">1.4 The 42 layout labels</h2>')
    a('<p>Labelled regions per label and split (class-id order of the authors&rsquo; <span class="tt">indic.yaml</span>). '
      'The question-paper labels used by IndicDoc Verify are in bold.</p>')
    qlabels = {'first-level-question', 'second-level-question', 'third-level-question', 'options', 'answer'}
    rows = [(r.class_id, f'<b>{r.class_name}</b>' if r.class_name in qlabels else r.class_name, fmt(r.train),
             fmt(r.val), fmt(r.test), fmt(r.total)) for r in classes.itertuples()]
    half = (len(rows) + 1) // 2
    a('<div class="two">' + ''.join(f'<div>{table(["id", "Label", "Train", "Val", "Test", "Total"], part, num=(0, 2, 3, 4, 5))}</div>'
                                    for part in (rows[:half], rows[half:])) + '</div>')
    a(f'<img src="{class_chart(classes, tmp)}" style="width:100%;margin-top:2pt">')
    a('<div class="cap"><b>Figure 1.1.</b> Training boxes per label, most frequent first (log scale); question-paper '
      'labels in gold. <i>paragraph</i> alone is a third of the training boxes; several labels have fewer than 20.</div>')

    samples = sorted(DATA.glob('samples/*.jpg'))
    for k in range(0, len(samples), 6):
        a(f'<h2 class="newpage">1.5 Sample test pages{" (continued)" if k else ""}</h2>')
        if not k:
            a('<p>One real IndicDLP test page per language with its ground-truth regions drawn (one colour per label).</p>')
        a('<div class="grid" style="grid-template-columns:repeat(3,1fr)">')
        for f in samples[k:k + 6]:
            lang, _, dom_ = f.stem.partition('_')
            a(f'<figure><img src="{uri(f)}"><figcaption>{lang.capitalize()}, {dom_.replace("_", " ")}</figcaption></figure>')
        a('</div>')

    bench = json.loads((METRICS / 'verify_benchmark_oracle.json').read_text())
    counts = pd.Series([c['type'] for c in bench['cases']]).value_counts()
    a('<h2 class="newpage">1.6 Verification benchmark</h2>')
    a('<p>No public dataset has question papers with known errors in their translations, so the benchmark was built '
      'for this project (<span class="tt">scripts/verify_bench/</span>). Papers of 18 questions are composed from '
      'parallel English, Hindi and Tamil questions; each pair is an English source and a Hindi or Tamil version. '
      'One error of a kind that has happened in real exams is injected into the version, or none (a clean pair). The '
      'pages are rendered at 200 dpi with Noto fonts; half are degraded like scans (blur, noise, JPEG at quality 55), '
      'and every page is read with real OCR (Tesseract).</p>')
    rows = [(esc(bench['labels'][t]), int(counts.get(t, 0))) for t in bench['labels']]
    rows.append(('No error (clean pair)', int(counts.get('clean', 0))))
    a('<div class="two"><div>')
    a(table(['Injected error', 'Test pairs'], rows, num=(1,), total=('Total', int(counts.sum()))))
    a('</div><div>')
    a(table(['Set', 'Pairs', 'Use'], [('Development (seed 7)', 48, 'building the parser and the checks'),
                                       ('Test (seed 42)', bench['overall']['pairs'], 'run once, results reported')], num=(1,)))
    a(f'<p style="font-size:9.4pt">An error counts as found only when a finding of the matching kind points at the '
      f'right question. On the test set {100 * bench["overall"]["found_at_place"]:.1f}% of the '
      f'{bench["overall"]["errors_injected"]} injected errors were found, with '
      f'{bench["overall"]["false_alarms_per_clean_pair"]:.2f} false alarms per clean pair (Part 6).</p>')
    a('</div></div>')
    a('<div class="grid" style="grid-template-columns:repeat(3,1fr)">')
    for f, c in (('bench_en', 'English source'), ('bench_hi', 'Hindi version'), ('bench_ta', 'Tamil version, scan-degraded')):
        a(f'<figure><img src="{uri(FIG / (f + ".jpg"))}"><figcaption>{c}</figcaption></figure>')
    a('</div>')
    return page_html('Part 1 - Dataset', ''.join(s))


# ---------------------------------------------------------------------------------- Part 2: code and implementation
def code_lines(rel: str) -> int:
    return len((ROOT / rel).read_text(encoding='utf-8').splitlines())


def implementation_part() -> str:
    tests = json.loads((METRICS / 'test_suite.json').read_text())
    collected = subprocess.run([sys.executable, '-m', 'pytest', '--collect-only', '-q', '-p', 'no:cacheprovider'],
                               cwd=ROOT, capture_output=True, text=True).stdout
    per_file = pd.Series([ln.split('::')[0] for ln in collected.splitlines() if '::' in ln]).value_counts().sort_index()
    purpose = {
        'tests/test_verify.py': 'question parsing, structure checks, translation parity, repeats, profiles',
        'tests/test_ocr.py': 'OCR of regions, script detection, the number pass, reading order',
        'tests/test_backend.py': 'REST API: /analyze, /metrics, /verify, upload limits',
        'tests/test_app.py': 'Streamlit app: tabs, upload, results (AppTest)',
        'tests/test_pipeline.py': 'models, datasets, training and inference on synthetic pages',
        'tests/test_coco_eval.py': 'COCO mAP, precision/recall, confusion matrix',
        'tests/test_prepare_indicdlp.py': 'IndicDLP conversion and cleaning',
        'tests/test_accessible.py': 'accessible HTML/EPUB export',
    }

    s = [header(2, 'Project Code and Implementation',
                'How IndicDoc Verify is built: the architecture, the code base, how to run it, the API, the tests, the '
                'working product, and the source code of its core modules.')]
    a = s.append
    a('<h2>2.1 System architecture</h2>')
    a('<p>IndicDoc Verify checks an Indian question paper, and its translations, before it is printed. A deep-learning '
      'layout detector (DocLayout-YOLO, fine-tuned on IndicDLP by its authors) finds the questions, sub-questions, '
      'options and answers on each page. Tesseract reads the text with the page script&rsquo;s own model, and a '
      'second <i>number pass</i> with the English model must agree on every digit. The questions are built from the '
      'layout and checked for structure, parity with the source version, repeats from earlier papers and format. '
      'Everything runs offline in one Docker image.</p>')
    a(f'<div style="margin:6pt 0 2pt">{block_diagram()}</div>')
    a('<div class="cap"><b>Figure 2.1.</b> Processing pipeline of IndicDoc Verify.</div>')
    a('<h2>2.2 Technology stack</h2>')
    a(table(['Layer', 'Technology'], [
        ('Layout detection', 'DocLayout-YOLO (IndicDLP authors&rsquo; weights); YOLOv8s and Faster R-CNN fine-tuned here '
                             '(Ultralytics, PyTorch, torchvision)'),
        ('OCR', 'Tesseract 5 with <span class="tt">tessdata_best</span> models for 12 languages (pytesseract)'),
        ('Verification', 'Python 3.11: question parser, checks, Needleman&ndash;Wunsch alignment of versions, JSON '
                         'format profiles, HTML/JSON report'),
        ('Documents', 'PyMuPDF (PDF pages at 200 dpi), OpenCV and Pillow (pre-processing)'),
        ('Interfaces', 'Streamlit web app (port 8501); FastAPI REST API (port 8000)'),
        ('Evaluation', 'pycocotools (COCO mAP), the verification benchmark (Playwright rendering, tesseract.js)'),
        ('Deployment, tests', 'Docker and docker-compose; pytest; node test runner for the React prototype'),
    ]))

    a('<h2 class="newpage">2.3 Repository structure</h2>')
    tree = [
        ('app/streamlit_app.py', 'Streamlit web app: Verify Paper, Analyze, Dashboard, Model Insights, Evaluation, About'),
        ('backend/main.py', 'FastAPI REST API'),
        ('ml/config.py', 'paths, the 42 IndicDLP labels, defaults, device selection'),
        ('ml/datasets/', 'COCO dataset loader, synthetic sample generator'),
        ('ml/preprocessing/', 'transforms, deskew/CLAHE, PDF page rendering'),
        ('ml/models/', 'Faster R-CNN and YOLOv8 builders, COCO-to-YOLO converter'),
        ('ml/training/', 'train_baseline.py (Faster R-CNN), train_yolo.py (YOLOv8s)'),
        ('ml/evaluation/', 'COCO mAP, precision/recall/F1, confusion matrix, per-language results'),
        ('ml/inference/', 'load a checkpoint and predict regions'),
        ('ml/ocr/', 'OCR of detected regions, script detection, number pass, reading order'),
        ('ml/verify/', 'IndicDoc Verify: parsing, checks, parity, repeats, profiles, report'),
        ('scripts/', 'data preparation, evaluation, benchmark, and the builders of every document'),
        ('scripts/verify_bench/', 'benchmark generator, renderer, degradation, OCR'),
        ('tests/', 'pytest suite'),
        ('dataset/', 'the subset as CSV files, sample pages'),
        ('outputs/', 'trained models, metrics, plots'),
        ('docs/, paper/', 'documentation, report, presentations, research paper'),
        ('Dockerfile, docker-compose.yml', 'one image with the app, the API, the models and Tesseract'),
    ]
    a(table(['Path', 'Contents'], [(f'<span class="tt">{esc(p)}</span>', c) for p, c in tree]))
    py_files = [f for f in git('ls-files').splitlines() if f.endswith('.py')]
    total_py = sum(code_lines(f) for f in py_files)
    a(f'<p>The repository has {len(py_files)} Python files with {fmt(total_py)} lines in all. The core modules '
      'printed in Section 2.8 are:</p>')
    a(table(['File', 'Lines', 'Role'], [(f'<span class="tt">{esc(f)}</span>', code_lines(f), esc(d)) for f, d in CODE],
            num=(1,)))

    a('<h2>2.4 Running the system</h2>')
    a('<h3>With Docker (recommended)</h3>')
    a('<pre class="cmd">docker compose up --build -d\n# web app  http://localhost:8501\n'
      '# REST API http://localhost:8000   (interactive docs at /docs)</pre>')
    a('<h3>Local setup</h3>')
    a('<pre class="cmd">python3.11 -m venv venv &amp;&amp; source venv/bin/activate\npip install -r requirements.txt\n'
      'python scripts/download_models.py      # DocLayout-YOLO + Tesseract language models (~300 MB, once)\n'
      'brew install tesseract                 # or: apt install tesseract-ocr\n\n'
      'streamlit run app/streamlit_app.py     # web app\nuvicorn backend.main:app --port 8000   # REST API\n'
      'python -m pytest tests/                # test suite</pre>')
    a('<h3>Reproducing the results</h3>')
    a('<pre class="cmd">python scripts/prepare_indicdlp.py                 # IndicDLP subset -> data/raw/indicdlp\n'
      'python -m ml.training.train_yolo                   # fine-tune YOLOv8s\n'
      'python -m ml.training.train_baseline               # fine-tune Faster R-CNN\n'
      'python scripts/evaluate.py                         # layout mAP on the test pages\n'
      'python scripts/evaluate_verify.py --layout oracle  # verification benchmark</pre>')
    a('<h2>2.5 REST API</h2>')
    a(table(['Method and path', 'Input', 'Output'], [
        ('<span class="tt">GET /</span>', '&mdash;', 'service status'),
        ('<span class="tt">POST /analyze</span>', 'page image or PDF; <span class="tt">ocr</span>, '
         '<span class="tt">ocr_lang</span>', 'regions (label, box, confidence) and their text'),
        ('<span class="tt">GET /metrics</span>', '&mdash;', 'evaluation metrics of the layout models'),
        ('<span class="tt">GET /verify/profiles</span>', '&mdash;', 'available format profiles'),
        ('<span class="tt">POST /verify</span>', 'paper (PDF/images); optional <span class="tt">source</span> version, '
         'question <span class="tt">bank</span>, <span class="tt">lang</span>, <span class="tt">source_lang</span>, '
         '<span class="tt">bilingual</span>, <span class="tt">profile</span>', 'verdict and findings (question, page, box) '
         'as JSON, or the HTML report with the pages outlined'),
    ]))
    a('<pre class="cmd">curl -F file=@paper_hi.pdf -F source=@paper_en.pdf \\\n'
      '     "http://localhost:8000/verify?lang=hin&amp;source_lang=eng&amp;output=html" &gt; report.html</pre>')

    a('<h2>2.6 Testing</h2>')
    a(f'<p>The test suite was run on {date.fromisoformat(tests["run_at"][:10]).strftime("%d %B %Y")}: '
      f'<b>{tests["python_tests"]} passed, {tests["python_skipped"]} skipped, {tests["failed"]} failed</b>, plus '
      f'{tests["frontend_tests"]} passing tests of the React prototype. The skipped tests need the Tesseract binary '
      'and the trained checkpoints, which were not installed on that machine; they run in the Docker image.</p>')
    a(table(['Test file', 'Tests', 'What is tested'],
            [(f'<span class="tt">{esc(f)}</span>', int(n), esc(purpose.get(f, ''))) for f, n in per_file.items()],
            num=(1,), total=('Total', int(per_file.sum()), '')))

    a('<h2 class="newpage">2.7 The working product</h2>')
    shots = [('07_verify_report.png', 'Verification report: verdict, findings with question and page'),
             ('08_verify_page.png', 'A changed number found in the Hindi version (Q-level finding outlined on the page)'),
             ('01_upload.png', 'Streamlit app: uploading a document'),
             ('02_detection.png', 'Layout regions found by DocLayout-YOLO'),
             ('03_ocr.png', 'Text read from the regions (OCR), language detected per region'),
             ('04_dashboard.png', 'Dashboard of the analysed document')]
    for k, (f, c) in enumerate(shots, 1):
        a(f'<div class="shot"><img src="{uri(SHOTS / f)}"><div class="cap"><b>Figure 2.{k + 1}.</b> {esc(c)}</div></div>')

    fmt_ = HtmlFormatter(linenos='inline', cssclass='hl', style='default')
    a('<h2 class="newpage">2.8 Source code of the core modules</h2>')
    a('<p>The full code base is in the repository (Part 3). Below are the modules that make up IndicDoc Verify, the OCR '
      'stage, the layout models and the API, as they are in the repository.</p>')
    for rel, desc in CODE:
        src = (ROOT / rel).read_text(encoding='utf-8')
        a(f'<div class="file"><div class="fhead">'
          f'<span class="tt fname">{esc(rel)}</span><span class="fmeta">{len(src.splitlines())} lines &middot; '
          f'{esc(desc)}</span></div>{highlight(src, get_lexer_for_filename(rel), fmt_)}</div>')
    css = fmt_.get_style_defs('.hl') + '''
.file { margin: 10pt 0 6pt; }
.fhead { border-top: 1.1pt solid #14284B; border-bottom: 0.5pt solid #9AA3B5; padding: 3pt 0; margin-bottom: 3pt;
         display: flex; justify-content: space-between; gap: 10pt; break-after: avoid; background: #fff; }
.fname { font-weight: bold; font-size: 8.8pt; color: #14284B; }
.fmeta { font-size: 8.4pt; color: #444; text-align: right; }
.hl { background: #fff !important; }
.hl pre { font-family: 'DejaVu Sans Mono', 'Noto Sans Devanagari', 'Noto Sans Tamil', monospace; font-size: 6.9pt;
          line-height: 1.32; margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; }
.hl .linenos { color: #9AA3B5; padding-right: 6pt; user-select: none; }
'''
    return page_html('Part 2 - Project Code and Implementation', ''.join(s), css)


# --------------------------------------------------------------------------------------- Part 3: GitHub repository
def repo_part(screenshots) -> str:
    if screenshots:
        body = header(3, 'GitHub Repository', f'<span class="tt">{REPO}</span>')
        body += ''.join(f'<div class="shot{" newpage" if k else ""}"><img src="{uri(p)}"></div>'
                        for k, p in enumerate(screenshots))
        return page_html('Part 3 - GitHub Repository', body)

    branch = git('rev-parse', '--abbrev-ref', 'HEAD')
    n_commits = git('rev-list', '--count', 'HEAD')
    head = git('log', '-1', '--format=%h|%s|%ad', '--date=format:%d %b %Y')
    top = git('ls-tree', '--name-only', 'HEAD').splitlines()
    dirs = set(git('ls-tree', '-d', '--name-only', 'HEAD').splitlines())
    entries = []
    for name in sorted(top, key=lambda n: (n not in dirs, n.lower())):
        msg, when = git('log', '-1', '--format=%s|%ad', '--date=format:%d %b %Y', '--', name).split('|', 1)
        entries.append((name, name in dirs, msg if len(msg) <= 52 else msg[:50].rstrip(' ,:;') + '\u2026', when))
    commits = [ln.split('|', 3) for ln in git('log', '-10', '--format=%h|%an|%ad|%s', '--date=format:%d %b %Y').splitlines()]
    sizes = {}
    ext_lang = {'.py': 'Python', '.js': 'JavaScript', '.jsx': 'JavaScript', '.tex': 'TeX', '.html': 'HTML',
                '.css': 'CSS', '.ipynb': 'Jupyter Notebook'}
    for f in git('ls-files').splitlines():
        lang = ext_lang.get(Path(f).suffix)
        if lang and (ROOT / f).is_file() and not f.startswith('docs/presentation/'):
            sizes[lang] = sizes.get(lang, 0) + (ROOT / f).stat().st_size
    total = sum(sizes.values())
    colours = {'Python': '#3572A5', 'JavaScript': '#F1E05A', 'TeX': '#3D6117', 'HTML': '#E34C26', 'CSS': '#563D7C',
               'Jupyter Notebook': '#DA5B0B'}
    langs = sorted(sizes.items(), key=lambda kv: -kv[1])
    bar = ''.join(f'<span style="width:{100 * v / total:.2f}%;background:{colours[k]}"></span>' for k, v in langs)
    legend = ''.join(f'<div><i style="background:{colours[k]}"></i>{k} <b>{100 * v / total:.1f}%</b></div>'
                     for k, v in langs if v / total >= 0.005)
    qr = qrcode.make(REPO, image_factory=qrcode.image.svg.SvgPathImage, box_size=10, border=1)
    buf = BytesIO()
    qr.save(buf)
    qr_svg = buf.getvalue().decode()
    qr_svg = qr_svg[qr_svg.find('<svg'):]

    readme = (ROOT / 'README.md').read_text(encoding='utf-8').split('<!-- VERIFY:START -->')[0]
    readme = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', readme)

    def md_inline(t):
        t = esc(t)
        t = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', t)
        t = re.sub(r'\*(.+?)\*', r'<i>\1</i>', t)
        return re.sub(r'`([^`]+)`', r'<code>\1</code>', t)
    blocks, rd = [b.strip() for b in readme.split('\n\n') if b.strip()], []
    for b in blocks:
        if b.startswith('# '):
            rd.append(f'<div class="rd-h1">{md_inline(b[2:])}</div>')
        elif b.startswith('- '):
            items = re.split(r'\n- ', b[2:])
            rd.append('<ul>' + ''.join(f'<li>{md_inline(" ".join(i.split()))}</li>' for i in items) + '</ul>')
        else:
            rd.append(f'<p>{md_inline(" ".join(b.split()))}</p>')

    rows = ''.join(f'<tr><td class="fn"><span class="ic {"d" if d else "f"}"></span>{esc(n)}</td>'
                   f'<td class="msg">{esc(m)}</td><td class="when">{w}</td></tr>' for n, d, m, w in entries)
    commit_rows = ''.join(f'<tr><td class="tt">{h}</td><td class="msg">{esc(m)}</td><td>{esc(au)}</td>'
                          f'<td class="when">{w}</td></tr>' for h, au, w, m in commits)
    h_sha, h_msg, h_date = head.split('|')
    body = header(3, 'GitHub Repository',
                  f'The project&rsquo;s code repository on GitHub, <span class="tt">{REPO.replace("https://", "")}</span>.')
    body += f'''
<div class="gh">
  <div class="gh-top"><span class="gh-name">va-ada / <b>IndicDoc-AI</b></span><span class="badge">Private</span></div>
  <div class="gh-bar"><span class="pill">branch: {esc(branch)}</span><span class="muted">2 branches</span>
    <span class="muted">{n_commits} commits</span><span class="url tt">{REPO}.git</span></div>
  <div class="gh-cols">
    <div class="gh-main">
      <table class="files"><thead><tr><th colspan="3"><span class="tt">{h_sha}</span> {esc(h_msg)}
        <span class="when">{h_date}</span></th></tr></thead><tbody>{rows}</tbody></table>
    </div>
    <div class="gh-side">
      <div class="side-h">About</div>
      <p class="about">{esc(REPO_DESCRIPTION)}</p>
      <div class="side-h">Languages</div>
      <div class="lbar">{bar}</div><div class="lleg">{legend}</div>
      <div class="side-h">Branches</div>
      <div class="br"><span class="tt">main</span><br><span class="tt">{esc(branch)}</span> (this submission)</div>
      <div class="side-h">Open the repository</div>
      <div class="qr">{qr_svg}</div>
    </div>
  </div>
</div>
<h2>Latest commits on <span class="tt" style="font-size:11pt">{esc(branch)}</span></h2>
<table class="commits"><thead><tr><th>Commit</th><th>Message</th><th>Author</th><th>Date</th></tr></thead>
<tbody>{commit_rows}</tbody></table>
<h2 class="newpage">README.md</h2>
<div class="readme">{"".join(rd)}</div>
<p class="note">Repository details taken from git and GitHub on {date.today().strftime("%d %B %Y")}. Clone with
<span class="tt">git clone {REPO}.git</span> (access granted by the owner).</p>
'''
    css = '''
.gh { border: 0.8pt solid #D0D7DE; border-radius: 6pt; padding: 10pt 12pt; font-family: Inter, sans-serif;
      font-size: 8.6pt; font-weight: 400; }
.gh-top { display: flex; align-items: center; gap: 8pt; font-size: 14pt; color: #0969DA; }
.badge { font-size: 7.4pt; border: 0.7pt solid #8C959F; color: #57606A; border-radius: 9pt; padding: 1pt 6pt; }
.gh-bar { display: flex; gap: 12pt; align-items: center; margin: 9pt 0 9pt; color: #24292F; }
.pill { border: 0.7pt solid #D0D7DE; background: #F6F8FA; border-radius: 5pt; padding: 2pt 7pt; font-weight: 600; }
.muted { color: #57606A; } .url { margin-left: auto; color: #57606A; font-size: 7.6pt; }
.gh-cols { display: grid; grid-template-columns: 1fr 150pt; gap: 14pt; }
table.files { font-size: 8.2pt; margin: 0; border: 0.7pt solid #D0D7DE; }
table.files thead th { background: #F6F8FA; border: none; border-bottom: 0.7pt solid #D0D7DE; font-weight: 400; }
table.files td { border-top: 0.5pt solid #E5E9EE; padding: 2.4pt 5pt; }
table.files tbody tr:last-child td { border-bottom: none; }
td.fn { white-space: nowrap; color: #24292F; } td.msg { color: #57606A; white-space: nowrap; } .when { color: #57606A; white-space: nowrap; text-align: right; }
thead .when { float: right; }
.ic { display: inline-block; width: 9pt; height: 7.5pt; margin-right: 5pt; vertical-align: -0.5pt; border-radius: 1pt; }
.ic.d { background: #54AEFF; } .ic.f { border: 0.8pt solid #8C959F; height: 9pt; width: 7.5pt; }
.side-h { font-weight: 600; font-size: 9pt; margin: 0 0 3pt; color: #24292F; }
.about { font-size: 8.4pt; text-align: left; color: #24292F; margin-bottom: 9pt; }
.lbar { display: flex; height: 6pt; border-radius: 3pt; overflow: hidden; margin: 3pt 0 4pt; }
.lbar span { display: block; height: 100%; }
.lleg { font-size: 7.8pt; color: #57606A; margin-bottom: 9pt; }
.lleg i { display: inline-block; width: 6pt; height: 6pt; border-radius: 3pt; margin-right: 4pt; }
.lleg b { color: #24292F; font-weight: 600; }
.br { font-size: 8pt; margin-bottom: 9pt; line-height: 1.6; }
.qr svg { width: 92pt; height: 92pt; }
table.commits td.msg { width: 60%; } table.commits td { font-size: 8.8pt; }
.readme { border: 0.8pt solid #D0D7DE; border-radius: 6pt; padding: 10pt 14pt; font-family: Inter, sans-serif;
          font-size: 8.8pt; font-weight: 400; line-height: 1.5; }
.readme p { text-align: left; } .readme ul { margin: 0 0 6pt; padding-left: 14pt; }
.rd-h1 { font-size: 15pt; font-weight: 600; border-bottom: 0.7pt solid #D0D7DE; padding-bottom: 4pt; margin-bottom: 7pt; }
'''
    return page_html('Part 3 - GitHub Repository', body, css)


# ------------------------------------------------------------------------------------------------ cover and stamps
def cover_html(parts, starts, total) -> str:
    team = ''.join(f'<tr><td>{esc(n)}</td><td class="n">{r}</td></tr>' for n, r in TEAM)
    name, subtitle = TITLE.split(': ', 1)
    rows = ''.join(f'<tr><td class="pn">{k}</td><td><b>{esc(p["title"])}</b><br><span class="d">{esc(p["desc"])}</span>'
                   f'</td><td class="n">{starts[k - 1]}&ndash;{starts[k - 1] + p["pages"] - 1}</td></tr>'
                   for k, p in enumerate(parts, 1))
    css = '''
@page { size: A4; margin: 0; }
.sheet { width: 210mm; height: 297mm; padding: 26mm 22mm 20mm; position: relative; overflow: hidden; break-after: page; }
.sheet:last-child { break-after: auto; }
.band { position: absolute; left: 0; top: 0; width: 100%; height: 9mm; background: #14284B; }
.band2 { position: absolute; left: 0; top: 9mm; width: 100%; height: 1.4mm; background: #D9A441; }
.big { font-size: 40pt; font-weight: 700; letter-spacing: -0.6pt; margin: 34mm 0 4pt; color: #14284B; }
.sub { font-size: 15pt; line-height: 1.3; margin: 0 0 10pt; color: #222; }
.ex { font-family: Inter, sans-serif; font-size: 9pt; letter-spacing: 1.8pt; text-transform: uppercase; color: #8A6A1E;
      font-weight: 600; margin-bottom: 18mm; }
table.meta { width: 100%; font-size: 10.5pt; margin: 0; }
table.meta > tbody > tr > th { width: 34mm; font-weight: 700; padding: 5pt 0; border: none; vertical-align: top; }
table.meta > tbody > tr > td { padding: 5pt 0; border: none; }
table.meta tbody tr:last-child td { border: none; }
table.team { width: 70mm; margin: 0; font-size: 10pt; }
table.team th { border-top: none !important; } table.team td { padding: 1.5pt 5pt; }
.foot { position: absolute; bottom: 18mm; left: 22mm; right: 22mm; font-size: 9pt; color: #555;
        border-top: 0.6pt solid #999; padding-top: 6pt; }
table.toc { font-size: 10.5pt; } table.toc td { padding: 6pt 5pt; border-bottom: 0.4pt solid #DDD; }
table.toc td.pn { font-size: 16pt; font-weight: 700; color: #14284B; width: 12mm; }
table.toc .d { font-size: 9pt; color: #444; }
.howto { margin-top: 14pt; font-size: 9.6pt; border: 0.7pt solid #C9CED8; background: #F6F7FA; padding: 9pt 12pt; }
.howto b { color: #14284B; }
'''
    body = f'''
<div class="sheet"><div class="band"></div><div class="band2"></div>
<div class="big">{esc(name)}</div><div class="sub">{esc(subtitle)}</div>
<div class="ex">Expt.-11 &middot; SDG-Based Deep Learning Mini Project &middot; Complete submission</div>
<table class="meta"><tbody>
<tr><th>Team</th><td><table class="team"><thead><tr><th>Name</th><th class="n">Roll No.</th></tr></thead><tbody>{team}</tbody></table></td></tr>
<tr><th>Project guide</th><td>{esc(GUIDE)}</td></tr>
<tr><th>SDGs</th><td>SDG 4 Quality Education &middot; SDG 9 Industry, Innovation and Infrastructure &middot; SDG 10
Reduced Inequalities &middot; SDG 16 Peace, Justice and Strong Institutions</td></tr>
<tr><th>Repository</th><td><span class="tt">{REPO.replace("https://", "")}</span></td></tr>
<tr><th>Date</th><td>{date.today().strftime("%d %B %Y")}</td></tr>
</tbody></table>
<div class="foot">This volume contains the dataset, the code and implementation, the repository, both phase
presentations and the research paper ({total} pages).</div></div>
<div class="sheet"><h1 style="margin-top:10mm">Contents</h1><div class="rule"></div>
<table class="toc"><thead><tr><th>Part</th><th>Contents</th><th class="n">Pages</th></tr></thead><tbody>{rows}</tbody></table>
<div class="howto"><b>Printing.</b> Print the whole file double-sided on A4 (flip on long edge). Every part begins on
the front of a new sheet, so the printed stack separates into the parts; pages marked &ldquo;intentionally left
blank&rdquo; keep it that way. The presentations are printed two slides per page.</div></div>'''
    return page_html('IndicDoc AI - Cover', body, css)


def stamp_html(pages) -> str:
    """One A4 page per output page: footer, blank-page note, frames around slides."""
    out = []
    for pg in pages:
        inner = ''
        if pg.get('footer'):
            inner += (f'<div class="ft"><span>IndicDoc AI &middot; {esc(pg["footer"])}</span>'
                      f'<span>Page {pg["no"]} of {pg["total"]}</span></div>')
        if pg.get('blank'):
            inner += '<div class="bl">This page is intentionally left blank.</div>'
        for x, y, w, h in pg.get('frames', []):
            inner += f'<div class="fr" style="left:{x}pt;top:{y}pt;width:{w}pt;height:{h}pt"></div>'
        if pg.get('label'):
            inner += f'<div class="lb">{esc(pg["label"])}</div>'
        out.append(f'<div class="pg">{inner}</div>')
    css = '''
@page { size: A4; margin: 0; }
html, body { margin: 0; background: transparent !important; }
.pg { width: 210mm; height: 297mm; position: relative; break-after: page; overflow: hidden; }
.pg:last-child { break-after: auto; }
.ft { position: absolute; left: 16mm; right: 16mm; bottom: 8.5mm; display: flex; justify-content: space-between;
      font-family: Inter, sans-serif; font-size: 7.4pt; color: #555; border-top: 0.5pt solid #BBB; padding-top: 3pt; }
.bl { position: absolute; top: 48%; width: 100%; text-align: center; font-family: Inter, sans-serif; font-size: 9pt;
      color: #888; font-style: italic; }
.fr { position: absolute; border: 0.6pt solid #9AA3B5; }
.lb { position: absolute; left: 16mm; top: 9mm; font-family: Inter, sans-serif; font-size: 7.4pt; color: #555;
      letter-spacing: 0.6pt; text-transform: uppercase; }
'''
    return (f'<!doctype html><html><head><meta charset="utf-8"><style>{css}</style></head><body>'
            f'{"".join(out)}</body></html>')


def phase1_patch_html() -> str:
    pages = []
    for _, x, y, w, h, text, style in PHASE1_PATCHES:
        pages.append(f'<div class="pg"><div style="position:absolute;left:{x}pt;top:{y}pt;width:{w}pt;height:{h}pt;'
                     f'background:{SLIDE_BG}"></div><div class="t" style="{style}">{esc(text)}</div></div>')
    css = ('@page{size:960pt 540pt;margin:0}html,body{margin:0;background:transparent!important}'
           '.pg{width:960pt;height:540pt;position:relative;break-after:page;overflow:hidden}'
           '.pg:last-child{break-after:auto}.t{position:absolute;font-family:Inter,sans-serif;font-weight:400;'
           'line-height:1;white-space:nowrap;-webkit-print-color-adjust:exact}')
    return f'<!doctype html><html><head><meta charset="utf-8"><style>{css}</style></head><body>{"".join(pages)}</body></html>'


# --------------------------------------------------------------------------------------------------- composition
def placed(src: PageObject, scale: float, tx: float, ty: float):
    return src, Transformation().scale(scale).translate(tx, ty)


def a4_pages_of(reader: PdfReader):
    """Pages already printed on A4 by Chrome."""
    return [[placed(p, 1, 0, 0)] for p in reader.pages]


def two_up(reader: PdfReader):
    """Two slides per A4 portrait page, framed; returns (placements, frames in top-left pt)."""
    out = []
    margin_x, w = 40.0, A4W - 80.0
    for k in range(0, len(reader.pages), 2):
        pl, frames = [], []
        for j, p in enumerate(reader.pages[k:k + 2]):
            sw, sh = float(p.mediabox.width), float(p.mediabox.height)
            s = w / sw
            h = sh * s
            gap = 40.0
            top = (A4H - 2 * h - gap) / 2 - 6 + j * (h + gap)
            pl.append(placed(p, s, margin_x - float(p.mediabox.left) * s, A4H - top - h - float(p.mediabox.bottom) * s))
            frames.append((margin_x, top, w, h))
        out.append((pl, frames))
    return out


def fit_a4(reader: PdfReader):
    out = []
    for p in reader.pages:
        sw, sh = float(p.mediabox.width), float(p.mediabox.height)
        s = min(A4W / sw, (A4H - 34) / sh)
        tx = (A4W - sw * s) / 2 - float(p.mediabox.left) * s
        ty = 34 + (A4H - 34 - sh * s) / 2 - float(p.mediabox.bottom) * s
        out.append([placed(p, s, tx, ty)])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--repo-screenshot', nargs='+', type=Path, help='browser screenshots of the GitHub page for Part 3')
    ap.add_argument('--out', type=Path, default=OUT)
    args = ap.parse_args()
    for f in (PHASE1, PHASE2, PAPER):
        if not f.exists():
            sys.exit(f'missing {f}')

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        print('building parts 1-3 ...')
        built = {
            'dataset': print_pdf(dataset_part(tmp), tmp / 'part1.pdf', tmp),
            'impl': print_pdf(implementation_part(), tmp / 'part2.pdf', tmp),
            'repo': print_pdf(repo_part(args.repo_screenshot), tmp / 'part3.pdf', tmp),
        }
        phase1 = PdfReader(PHASE1)
        patch = PdfReader(print_pdf(phase1_patch_html(), tmp / 'patch.pdf', tmp))
        for (page, *_), pp in zip(PHASE1_PATCHES, patch.pages):
            phase1.pages[page].merge_page(pp)

        parts = [
            dict(title='Dataset', desc='IndicDLP subset: splits, languages, domains, the 42 labels, data files, sample '
                 'pages; the verification benchmark', pages_of=a4_pages_of(PdfReader(built['dataset']))),
            dict(title='Project Code and Implementation', desc='Architecture, technology stack, repository structure, '
                 'running the system, REST API, tests, the working product, source code of the core modules',
                 pages_of=a4_pages_of(PdfReader(built['impl']))),
            dict(title='GitHub Repository', desc='Files, branches, languages and commits of the repository; README',
                 pages_of=a4_pages_of(PdfReader(built['repo']))),
            dict(title='Phase-1 Presentation: Literature Survey', desc=f'{len(phase1.pages)} slides, two per page',
                 pages_of=two_up(phase1), slides=True),
            dict(title='Phase-2 Presentation: Implementation', desc=f'{len(PdfReader(PHASE2).pages)} slides, two per page',
                 pages_of=two_up(PdfReader(PHASE2)), slides=True),
            dict(title='Research Paper', desc='IndicDoc Verify: layout-aware pre-print checking of Indian question '
                 'papers and their translations (IEEE format)', pages_of=fit_a4(PdfReader(PAPER))),
        ]
        cover_pages = 2
        starts, n = [], cover_pages
        for p in parts:
            p['pages'] = len(p['pages_of'])
            p['padded'] = p['pages'] + p['pages'] % 2
            starts.append(n + 1)
            n += p['padded']
        total = n
        cover = PdfReader(print_pdf(cover_html(parts, starts, total), tmp / 'cover.pdf', tmp))
        assert len(cover.pages) == cover_pages, len(cover.pages)

        # page plan: (placements, stamp info)
        plan = [([placed(cover.pages[0], 1, 0, 0)], {}),
                ([placed(cover.pages[1], 1, 0, 0)], {'footer': 'Contents'})]
        for k, p in enumerate(parts, 1):
            footer = f'Part {k} – {p["title"]}'
            for j, item in enumerate(p['pages_of']):
                if p.get('slides'):
                    pl, frames = item
                    first = j * 2 + 1
                    last = min(first + 1, first + len(pl) - 1)
                    plan.append((pl, {'footer': footer, 'frames': frames,
                                      'label': f'Part {k} · {p["title"]} · slides {first}–{last}'
                                      if last > first else f'Part {k} · {p["title"]} · slide {first}'}))
                else:
                    plan.append((item, {'footer': footer}))
            if p['padded'] > p['pages']:
                plan.append(([], {'blank': True, 'footer': footer}))
        assert len(plan) == total
        for no, (_, st) in enumerate(plan, 1):
            st.update(no=no, total=total)
        stamps = PdfReader(print_pdf(stamp_html([st for _, st in plan]), tmp / 'stamps.pdf', tmp))
        assert len(stamps.pages) == total, (len(stamps.pages), total)

        writer = PdfWriter()
        for (placements, _), stamp in zip(plan, stamps.pages):
            page = writer.add_blank_page(A4W, A4H)
            for src, tf in placements:
                page.merge_transformed_page(src, tf)
            page.merge_page(stamp)
        writer.add_outline_item('Cover and contents', 0)
        for k, (p, s) in enumerate(zip(parts, starts), 1):
            writer.add_outline_item(f'Part {k}: {p["title"]}', s - 1)
        writer.add_metadata({'/Title': 'IndicDoc AI - Complete Submission', '/Author': ', '.join(n for n, _ in TEAM),
                             '/Subject': TITLE})
        writer.compress_identical_objects()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'wb') as fh:
            writer.write(fh)

    print(f'{args.out}  ({total} pages, {args.out.stat().st_size / 1e6:.1f} MB)')
    for k, (p, s) in enumerate(zip(parts, starts), 1):
        print(f'  Part {k}: {p["title"]:<42} pages {s}-{s + p["pages"] - 1}'
              f'{"  + 1 blank" if p["padded"] > p["pages"] else ""}')


if __name__ == '__main__':
    main()
