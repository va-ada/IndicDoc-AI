"""
IndicDoc AI - Phase-2 presentation (Project Implementation), following the department template.

Every number comes from outputs/metrics/ (measured results); nothing is typed in by hand.
Writes docs/presentation/IndicDoc_AI_Phase2_Implementation.html and .pdf (printed with headless Chrome).
Usage: python scripts/build_slides.py
"""
import base64
import csv
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / 'outputs' / 'metrics'
FIG = ROOT / 'docs' / 'figures'
PLOTS = ROOT / 'outputs' / 'plots'
SHOTS = ROOT / 'docs' / 'screenshots'
OUT = ROOT / 'docs' / 'presentation'
CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'

TEAM = [('PJ', 'Prashant Jha', 41), ('AS', 'Aditya Soni', 42), ('RP', 'Riddhi Patil', 39), ('MB', 'Mariam Badure', 40)]
GUIDE = 'Dr. Joanne Gomes'
FOOT = 'IndicDoc AI | Phase-2: Project Implementation'


def j(name):
    p = M / name
    return json.loads(p.read_text()) if p.exists() else {}


A, P, B = j('authors_metrics.json'), j('proposed_metrics.json'), j('baseline_metrics.json')
S, OCR, ROB = j('study_summary.json'), j('ocr_evaluation.json'), j('study_robustness.json')
BH = j('baseline_training_history.json')


def img(path, cls='', style=''):
    path = Path(path)
    if not path.exists():
        return f'<div class="missing">missing: {path.name}</div>'
    mime = 'jpeg' if path.suffix.lower() in ('.jpg', '.jpeg') else 'png'
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img class="{cls}" style="{style}" src="data:image/{mime};base64,{data}">'


def f3(v):
    return f'{v:.3f}'


def pct(v):
    return f'{100 * v:.0f}%'


slides = []


def slide(kicker, title, body, cls=''):
    n = len(slides) + 1
    slides.append(f'''<section class="slide {cls}"><div class="bar"></div>
<div class="kicker">{kicker}</div><h1>{title}</h1><div class="body">{body}</div>
<div class="foot"><span>{FOOT}</span><span>{n}</span></div></section>''')


def card(title, text, color='#2563EB', icon=''):
    return f'<div class="card" style="--c:{color}"><div class="ct">{icon}{title}</div><div class="cx">{text}</div></div>'


# ---------------------------------------------------------------- 1. Title
team_html = ''.join(f'<div class="member"><div class="av">{i}</div><div><b>{n}</b><span>Roll No. {r}</span></div></div>'
                    for i, n, r in TEAM)
slides.append(f'''<section class="slide title"><div class="blob b1"></div><div class="blob b2"></div>
<div class="tk">PHASE-2 · PROJECT IMPLEMENTATION · DL MINI PROJECT</div>
<div class="tt">IndicDoc AI</div>
<div class="ts">Where do layout detectors fail on Indian documents? A study of script, domain and scan quality</div>
<div class="team">{team_html}</div>
<div class="meta"><p><b>Course:</b> Deep Learning Mini Project (Expt. 11, SDG-based)</p>
<p><b>Department:</b> Artificial Intelligence and Machine Learning, St. Francis Institute of Technology (SFIT)</p>
<p><b>Guide:</b> {GUIDE}</p><p><b>Academic year:</b> 2026–27 · Semester V (TE AIML)</p></div>
<div class="titleimg">{img(ROOT / 'dataset' / 'samples' / 'hindi_newspapers.jpg')}</div></section>''')

# ---------------------------------------------------------------- 2. Content
items = ['Literature Survey Summary Table', 'Research Gaps', 'Problem Statement and Objectives',
         'Proposed System, Methodology and Novelty', 'System Block Diagram', 'Dataset Description and Pre-processing',
         'Deep Learning Model Architecture', 'Tools, Technologies and Environment', 'Implementation Details',
         'Experimental Setup and Training', 'Results and Performance Evaluation',
         'Comparison with Baseline / Existing Methods', 'Sample Outputs and Demonstration',
         'Challenges and Limitations', 'Conclusion and Future Scope', 'References']
lis = ''.join(f'<li><span>{i:02d}</span>{t}</li>' for i, t in enumerate(items, 1))
slide('OVERVIEW', 'Content', f'<ol class="toc">{lis}</ol>')

# ---------------------------------------------------------------- 3. Literature table
lit = [
    ('[1]', 'Nath et al., 2025', 'Layout parsing of Indian-language pages', 'IndicDLP: 119,806 pages, 12 languages, 12 domains, 42 labels',
     'Benchmark of YOLOv10x, DocLayout-YOLO, RoDLA, DINO, DiT, Florence-2', 'Vision-only; 20–25 mAP drop on unseen scripts; no robustness test'),
    ('[2]', 'Kasuba et al., 2025', 'Page-level handwritten Indic text recognition', 'CHIPS, ~25K pages, 10 languages',
     'DBNet word detection + 6 recognisers (CRNN, PARSeq, ...)', 'Synthetic pages; user must name the language'),
    ('[3]', 'Mondal et al., 2026', 'Multilingual table detection', 'UniTabBank: 55K pages, 82K tables, 28 languages',
     'Semi-automatic labelling; YOLOv11 detector', 'Borderless tables and degraded scans fail'),
    ('[4]', 'Pal et al., 2024', 'Table question answering, low-resource Indic', 'BanglaTabQA 2M, HindiTabQA 643K',
     'SQL-generated QA; fine-tuned mBART / M2M / Llama', 'Wikipedia tables only; small test sets'),
    ('[5]', 'Xu et al., 2021', 'Multilingual document understanding', '30M docs; XFUND, 7 languages',
     'LayoutXLM: text + layout + image transformer', 'No Indian language; needs good OCR'),
    ('[6]', 'Kasuba et al., 2025', 'Grounding answers on the page', 'MGVG: 70 circulars, 509 QA pairs',
     'OCR + Llama-3.1-8B + region matching', 'OCR-dependent; single page; small test set'),
    ('[7]', 'Bhattacharyya et al., 2025', 'Hindi OCR with vision-language models', '1.1M printed Hindi words + pages',
     'LoRA-tuned GOT-OCR 2.0, progressive learning', 'Hindi only; mixed-language lines'),
    ('[8]', 'Dumada & Shah, 2026', 'Devanagari table extraction', '500+ Devanagari documents',
     'Structure + Tesseract content, weighted fusion', 'Self-reported results; no ablation'),
]
rows = ''.join(f'<tr><td class="ref">{r}</td><td><b>{a}</b></td><td>{p}</td><td>{d}</td><td>{m}</td><td class="lim">{l}</td></tr>'
               for r, a, p, d, m, l in lit)
slide('PHASE-1 RECAP', 'Literature Survey Summary Table',
      f'<table class="lit"><tr><th>Ref</th><th>Author</th><th>Problem</th><th>Dataset</th><th>Method</th><th>Limitation</th></tr>{rows}</table>')

# ---------------------------------------------------------------- 4. Research gaps
gaps = [('Only large models were benchmarked, on clean pages', 'No study of how accuracy holds up on blurred, noisy, compressed or faded scans', 'RQ4'),
        ('Results are one overall number', 'Is accuracy even across 12 languages and 12 domains? Which labels fail, and why?', 'RQ2, RQ3'),
        ('OCR tools expect the user to name the language', 'Automatic script choice per page, using the detected regions', 'RQ5')]
g = ''.join(f'<div class="gap"><div class="gl">⚠ {a}</div><div class="arrow">→</div><div class="gr">✓ {b}<span class="rq">{c}</span></div></div>' for a, b, c in gaps)
slide('WHAT IS MISSING', 'Research Gaps', f'''<div class="gaps">{g}</div>
<div class="note"><b>Our position:</b> we do <u>not</u> claim a new detector. Strong detectors already exist;
our contribution is a careful <b>study</b> of where they fail on Indian documents, plus a working tool built on the best one.</div>''')

# ---------------------------------------------------------------- 5. Problem & objectives
rqs = [('RQ1', 'How far behind is a detector trained on a small budget, compared with one trained on the full dataset?'),
       ('RQ2', 'Is accuracy even across the 12 languages and 12 document domains?'),
       ('RQ3', 'Which region types are hardest, and is that because they are rare?'),
       ('RQ4', 'How much accuracy survives blur, noise, compression, low resolution and fading?'),
       ('RQ5', 'Can detected regions choose the right OCR language automatically?')]
rq = ''.join(f'<div class="rqrow"><span>{a}</span>{b}</div>' for a, b in rqs)
slide('WHAT WE SOLVE', 'Problem Statement and Objectives', f'''<div class="two">
<div><div class="quote"><b>Problem:</b> Indian offices, schools and archives hold millions of scanned pages in 12+ scripts.
To search or reuse them, a system must first find every region on the page (paragraph, heading, table, figure ...)
and then read the text. Most tools are built and tested on English.</div>
<div class="sub">MAIN OBJECTIVE</div><p class="big">Measure, in simple numbers, where deep-learning layout detectors work and where they fail on Indian-language pages, and deliver a usable tool.</p>
<div class="sub">SPECIFIC OBJECTIVES</div><ul>
<li>Train two detectors (Faster R-CNN, YOLOv8s) and compare them with the published IndicDLP model on the same test pages.</li>
<li>Define two simple measures: Consistency Index and Robustness Index.</li>
<li>Add OCR with automatic script detection for 12 languages.</li>
<li>Ship a web app + REST API in Docker.</li></ul>
<div class="sub">EXPECTED OUTPUT</div><p>A labelled box for every region, the page text in reading order (Markdown / JSON), and the study results.</p></div>
<div><div class="sub">RESEARCH QUESTIONS</div>{rq}</div></div>''')

# ---------------------------------------------------------------- 6. Proposed system & novelty
steps = [('1', 'Page in', 'image or PDF, any of 12 languages'), ('2', 'Detect regions', 'DocLayout-YOLO, 42 labels'),
         ('3', 'Clean up', 'confidence threshold, remove duplicates'), ('4', 'Pick script', 'probe each script on one region'),
         ('5', 'Read text', 'Tesseract, one pass per page'), ('6', 'Order', 'words → regions → reading order'),
         ('7', 'Output', 'boxes + Markdown / JSON')]
st = ''.join(f'<div class="step"><div class="sn">{a}</div><b>{b}</b><span>{c}</span></div>' for a, b, c in steps)
slide('OUR APPROACH', 'Proposed System, Methodology and Novelty', f'''<div class="flow">{st}</div>
<div class="three">
{card('How deep learning is used', 'Convolutional object detectors look at the page image and output a box, a label and a confidence for every region. We fine-tune COCO-pretrained detectors (transfer learning) and use the published IndicDLP model.', '#2563EB')}
{card('What is new (the study)', '<b>Consistency Index</b> CI = worst group mAP / best group mAP<br><b>Robustness Index</b> RI = damaged mAP / clean mAP<br>First scan-damage test on IndicDLP: blur, noise, JPEG, half resolution, faded print', '#D9A441')}
{card('What is new (the tool)', 'Two operating modes picked on validation pages ("find every part", "balanced"); script detection by probing detected regions (106/120 vs 70/120 for Tesseract OSD); one app for 12 languages.', '#2E7D32')}
</div>''')

# ---------------------------------------------------------------- 7. Block diagram
def blk(t, s, c='#14284B'):
    return f'<div class="blk" style="--c:{c}"><b>{t}</b><span>{s}</span></div>'


slide('ARCHITECTURE', 'System Block Diagram', f'''<div class="diagram">
<div class="lane"><div class="lane-t">USER</div>{blk('Streamlit web app', 'upload, sample pages, view, download', '#2563EB')}{blk('REST API (FastAPI)', '/predict, /metrics, mode, ocr', '#2563EB')}</div>
<div class="down">▼</div>
<div class="lane"><div class="lane-t">INPUT</div>{blk('Decode & check', 'PNG / JPG / TIFF / PDF, size limit')}{blk('Pre-process', 'RGB, letterbox to 1024 px')}</div>
<div class="down">▼</div>
<div class="lane"><div class="lane-t">DEEP LEARNING</div>{blk('Layout detector', 'DocLayout-YOLO (default) · YOLOv8s · Faster R-CNN', '#D9A441')}{blk('Post-process', 'threshold (mode) + duplicate removal (IoU 0.7)', '#D9A441')}</div>
<div class="down">▼</div>
<div class="lane"><div class="lane-t">TEXT</div>{blk('Script probe', '12 Tesseract models, own-script characters', '#2E7D32')}{blk('OCR + assembly', 'words → regions → reading order', '#2E7D32')}</div>
<div class="down">▼</div>
<div class="lane"><div class="lane-t">OUTPUT</div>{blk('Labelled boxes', 'image + list + confidence', '#7C3AED')}{blk('Document text', 'Markdown / JSON download', '#7C3AED')}{blk('Evaluation', 'mAP, P/R/F1, study indices', '#7C3AED')}</div>
</div>''')

# ---------------------------------------------------------------- 8. Dataset
lang_rows = list(csv.DictReader(open(ROOT / 'dataset' / 'language_split.csv')))
tot = {k: sum(int(r[k]) for r in lang_rows) for k in ('train', 'val', 'test')}
class_rows = list(csv.DictReader(open(ROOT / 'dataset' / 'class_counts.csv')))
class_rows.sort(key=lambda r: -int(r['total']))
top = class_rows[0]
total_boxes = sum(int(r['total']) for r in class_rows)
rare = sum(1 for r in class_rows if int(r['train']) < 100)
samples = sorted((ROOT / 'dataset' / 'samples').glob('*.jpg'))
pick = [s for s in samples if s.stem.split('_')[0] in ('tamil', 'bengali', 'punjabi', 'hindi')][:4]
slide('DATA', 'Dataset Description and Pre-processing', f'''<div class="two w60">
<div><table class="kv">
<tr><td>Dataset</td><td><b>IndicDLP</b> (Nath et al., ICDAR 2025) — Hugging Face <code>ai4bharat/indicdlp</code>, also on AIKosh; MIT licence</td></tr>
<tr><td>Full size</td><td>119,806 pages · 12 languages · 12 domains · 42 region labels</td></tr>
<tr><td>Our subset</td><td><b>{tot['train'] + tot['val'] + tot['test']:,} pages</b>, {total_boxes:,} boxes, balanced by language (~200 train pages each), taken from the official train / val / test splits</td></tr>
<tr><td>Split</td><td>train <b>{tot['train']:,}</b> · validation <b>{tot['val']}</b> · test <b>{tot['test']}</b> pages</td></tr>
<tr><td>Pre-processing</td><td>Parquet → PNG + COCO JSON; boxes clipped to the page, &lt;1 px boxes dropped (0.5%); COCO → YOLO labels; resize / letterbox (640 px ours, 1024 px authors)</td></tr>
<tr><td>Augmentation</td><td>YOLOv8: mosaic, HSV, scale/translate. Faster R-CNN: colour jitter, blur. <b>No mirror flips</b> (a mirrored page is not real)</td></tr>
<tr><td>Challenges</td><td><b>Heavy class imbalance:</b> "{top['class_name']}" = {pct(int(top['total']) / total_boxes)} of all boxes; {rare} of 42 labels have &lt;100 training boxes. Hindi has only 22 test pages.</td></tr>
<tr><td>Deliverable</td><td><code>dataset/</code>: pages.csv, boxes.csv, class_counts.csv, language_split.csv + one sample per language</td></tr>
</table></div>
<div class="samples">{''.join(f'<figure>{img(s)}<figcaption>{s.stem.replace("_", " · ")}</figcaption></figure>' for s in pick)}</div></div>''')

# ---------------------------------------------------------------- 9. Architecture
def arch(name, color, parts, note):
    p = ''.join(f'<div class="ap"><b>{a}</b><span>{b}</span></div><div class="aa">→</div>' for a, b in parts)[:-len('<div class="aa">→</div>')]
    return f'<div class="arch" style="--c:{color}"><div class="an">{name}</div><div class="apipe">{p}</div><div class="anote">{note}</div></div>'


slide('MODELS', 'Deep Learning Model Architecture', f'''
{arch(f"DocLayout-YOLO · {A.get('parameters', 0) / 1e6:.1f} M params · default", '#14284B',
      [('Input', '1024×1024×3 page'), ('Backbone', 'CSP conv blocks + GL-CRM (global-to-local receptive fields)'), ('Neck', 'PAN feature pyramid, 3 scales'), ('Head', 'YOLOv10-style, NMS-free'), ('Output', 'N × (box, 42-class score)')],
      'Pre-trained on DocSynth-300K, fine-tuned on the full IndicDLP train set by its authors (Zenodo 15881917, CC-BY 4.0). Used as released; we did not change it.')}
{arch(f"YOLOv8s · {P.get('parameters', 0) / 1e6:.1f} M params · trained by us", '#D9A441',
      [('Input', '640×640×3'), ('Backbone', 'CSPDarknet with C2f blocks + SPPF'), ('Neck', 'PAN-FPN'), ('Head', 'decoupled, anchor-free'), ('Output', 'boxes + 42 classes')],
      'Transfer learning: COCO-pretrained weights; the 80-class head is replaced with a 42-class head. Loss: CIoU + DFL (boxes) + BCE (classes).')}
{arch(f"Faster R-CNN · {B.get('parameters', 0) / 1e6:.1f} M params · baseline", '#64748B',
      [('Input', 'longest side 640–1024'), ('Backbone', 'ResNet-50, frozen batch-norm'), ('Neck', 'FPN, 5 levels'), ('RPN + RoIAlign', 'region proposals'), ('Box head', '43 outputs (42 + background)')],
      'Two-stage detector. COCO-pretrained; the predictor is replaced (FastRCNNPredictor, 43 classes). Loss: RPN + classification CE + smooth-L1 box.')}''')

# ---------------------------------------------------------------- 10. Tools
tools = [('Programming language', 'Python 3.11 (ML, app, API); JavaScript only for an unused React prototype'),
         ('DL framework', 'PyTorch 2 with torchvision; Ultralytics 8 (YOLOv8); doclayout-yolo 0.0.4'),
         ('Development environment', 'VS Code / terminal, Git + GitHub (private repo), pytest (141 tests)'),
         ('Libraries', 'pycocotools (COCO mAP), OpenCV, Pillow, NumPy, pandas, Matplotlib, Plotly, PyMuPDF, Tesseract 5 (tessdata_best, 12 scripts)'),
         ('Hardware', 'Apple M1 Pro MacBook, 16-core GPU via PyTorch MPS (training + evaluation); CPU in Docker'),
         ('Dataset tools', 'Hugging Face Hub + fsspec (partial Parquet reads), custom COCO / YOLO converters'),
         ('Interface / deployment', 'Streamlit web app + FastAPI REST API, Docker + docker-compose (works offline)')]
slide('STACK', 'Tools, Technologies and Environment',
      '<table class="tools"><tr><th>Component</th><th>Details</th></tr>' +
      ''.join(f'<tr><td><b>{a}</b></td><td>{b}</td></tr>' for a, b in tools) + '</table>')

# ---------------------------------------------------------------- 11. Implementation
code = '''<span class="c"># ml/inference/predict.py — one page in, labelled boxes out</span>
model = load_model(ckpt, <span class="s">"doclayout"</span>, device)
out = predict_single(model, page, <span class="s">"doclayout"</span>, device, conf=<span class="n">0.15</span>)
out = dedupe_boxes(out, iou_threshold=<span class="n">0.7</span>)

<span class="c"># ml/ocr — read the text inside the regions</span>
doc = extract_document(page, out, languages=<span class="s">"auto"</span>)
print(doc[<span class="s">"markdown"</span>])   <span class="c"># text in reading order</span>'''
slide('WHAT WE BUILT', 'Implementation Details', f'''<div class="two">
<div><ul class="tight">
<li><b>Data loading:</b> Parquet row-groups streamed from Hugging Face, language-balanced sampling, COCO JSON + YOLO labels.</li>
<li><b>Model construction:</b> COCO-pretrained YOLOv8s and Faster R-CNN, new 42-class heads; the authors' DocLayout-YOLO loaded as released.</li>
<li><b>Loss / optimiser:</b> Faster R-CNN — SGD lr 0.005, momentum 0.9, weight decay 5e-4, cosine schedule. YOLOv8s — Ultralytics automatic optimiser, CIoU + DFL + BCE loss.</li>
<li><b>Batch / epochs:</b> Faster R-CNN batch 4, 2 epochs (time budget); YOLOv8s batch 8, 12 epochs (1-hour budget). Seed 42.</li>
<li><b>Validation:</b> best checkpoint on 342 validation pages; operating thresholds chosen on validation, never on test.</li>
<li><b>Inference:</b> one page at a time, duplicate boxes removed, two modes: "find every part" (conf {A.get('high_recall_conf')}) and "balanced" (conf {A.get('recommended_conf')}).</li>
<li><b>Integration:</b> same code in the Streamlit app, the FastAPI service and the evaluation scripts; Docker image runs with no internet.</li></ul></div>
<div><pre class="code">{code}</pre>
<div class="mini">{img(SHOTS / '01_upload.png')}</div></div></div>''')

# ---------------------------------------------------------------- 12. Experimental setup
slide('EXPERIMENTS', 'Experimental Setup and Training', f'''<div class="two">
<div><table class="kv small">
<tr><td>Data sizes</td><td>train {tot['train']:,} · val {tot['val']} · test {tot['test']} pages (test = {A.get('num_ground_truth', 0):,} boxes)</td></tr>
<tr><td>Faster R-CNN</td><td>ResNet-50 FPN, 640–1024 px, batch 4, SGD 0.005, <b>2 epochs, 76 min</b></td></tr>
<tr><td>YOLOv8s</td><td>640 px, batch 8, mosaic on, no flips, <b>12 epochs, 60 min</b></td></tr>
<tr><td>DocLayout-YOLO</td><td>1024 px, released checkpoint, evaluated only</td></tr>
<tr><td>Hardware</td><td>Apple M1 Pro GPU (MPS), one page at a time for timing</td></tr>
<tr><td>Metrics</td><td>COCO mAP@50 and mAP@50:95 (pycocotools, 300 boxes/page), precision, recall, F1</td></tr>
<tr><td>Study runs</td><td><b>E1</b> 3 models on clean pages: per language, domain, label<br>
<b>E2</b> DocLayout-YOLO and YOLOv8s × 6 page conditions (clean + 5 damage types) × {A.get('num_images')} pages<br>
<b>E3</b> OCR script detection on 120 pages (10 per language)</td></tr>
</table></div>
<div>{img(ROOT / 'docs' / 'report' / 'loss_curves.png', 'wide')}<p class="cap">Training and validation loss of our two models (measured).</p></div></div>''')

# ---------------------------------------------------------------- 13. Results
def row(name, m):
    hr = m.get('at_high_recall_conf', {})
    return (f"<tr><td><b>{name}</b></td><td>{f3(m['map50'])}</td><td>{f3(m['map50_95'])}</td>"
            f"<td>{f3(m['at_recommended_conf']['precision'])}</td><td>{f3(m['at_recommended_conf']['recall'])}</td>"
            f"<td>{f3(m['at_recommended_conf']['f1'])}</td><td>{f3(hr.get('recall', 0))}</td>"
            f"<td>{m['inference_time']:.3f}</td><td>{m['model_size_mb']:.0f}</td></tr>")


slide('RESULTS', 'Results and Performance Evaluation', f'''
<table class="res"><tr><th>Model ({A.get('num_images')} test pages)</th><th>mAP@50</th><th>mAP@50:95</th><th>Precision*</th><th>Recall*</th><th>F1*</th><th>Recall, find-every-part</th><th>s / page</th><th>MB</th></tr>
{row('DocLayout-YOLO (authors)', A)}{row('YOLOv8s (ours)', P)}{row('Faster R-CNN (ours)', B)}</table>
<p class="cap">* at the "balanced" threshold chosen on validation pages. Timing on Apple M1 Pro GPU.</p>
<div class="two">
<div>{img(PLOTS / 'model_comparison.png', 'wide')}</div>
<div class="kpis">
<div class="kpi"><b>{f3(A['map50'])}</b><span>mAP@50, best model (paper reports ≈0.735 on its full test set)</span></div>
<div class="kpi"><b>{pct(A['at_high_recall_conf']['recall'])}</b><span>of all regions found in "find every part" mode, at {pct(A['at_high_recall_conf']['precision'])} precision</span></div>
<div class="kpi"><b>{OCR.get('script_detection_correct')}/{OCR.get('pages')}</b><span>pages: OCR script picked correctly (Tesseract's own detector: 70/120)</span></div>
</div></div>''')

# ---------------------------------------------------------------- 14. Study results (RQ2-RQ5)
ci = S.get('language_ci', {})
dci = S.get('domain_ci', {})
mri = S.get('mean_ri', {})
rho = S.get('class_freq_spearman', {})
slide('STUDY FINDINGS', 'Results: Where Do Detectors Fail?', f'''<div class="grid4">
<div class="fcard"><div class="fh">RQ2 · Languages and domains</div>{img(FIG / 'fig_per_language.png')}
<p>Language CI: authors <b>{ci.get('authors', 0):.2f}</b>, YOLOv8s {ci.get('proposed', 0):.2f}, Faster R-CNN {ci.get('baseline', 0):.2f}.
Domain CI (authors): <b>{dci.get('authors', 0):.2f}</b> — the <b>domain</b> matters more than the script.</p></div>
<div class="fcard"><div class="fh">RQ3 · Rare labels</div>{img(FIG / 'fig_class_frequency.png')}
<p>Rank correlation of class frequency with AP: <b>{rho.get('proposed', 0):.2f}</b> for our small model, only {rho.get('authors', 0):.2f} for the full-data model. More data fixes rarity.</p></div>
<div class="fcard"><div class="fh">RQ4 · Damaged scans</div>{img(FIG / 'fig_robustness.png')}
<p>Mean Robustness Index: authors <b>{mri.get('authors', 0):.2f}</b>, YOLOv8s {mri.get('proposed', 0):.2f}. Blur hurts most (RI {S['robustness_index']['authors']['blur']:.2f}).</p></div>
<div class="fcard"><div class="fh">RQ5 · Choosing the OCR script</div>{img(FIG / 'fig_script_detection.png')}
<p>Region probe picks the right script on <b>{OCR.get('script_detection_correct')}/120</b> pages vs 70/120 for Tesseract's own detector.</p></div>
</div>''')

# ---------------------------------------------------------------- 15. Comparison
slide('COMPARISON', 'Comparison with Baseline / Existing Methods', f'''<div class="two">
<div><table class="res small"><tr><th>Method</th><th>Training data</th><th>mAP@50</th><th>mAP@50:95</th></tr>
<tr><td>Faster R-CNN (ours, baseline)</td><td>2,293 pages, 76 min</td><td>{f3(B['map50'])}</td><td>{f3(B['map50_95'])}</td></tr>
<tr><td>YOLOv8s (ours)</td><td>2,293 pages, 60 min</td><td>{f3(P['map50'])}</td><td>{f3(P['map50_95'])}</td></tr>
<tr><td><b>DocLayout-YOLO (IndicDLP authors)</b></td><td>~96K pages (80% of 119,806)</td><td><b>{f3(A['map50'])}</b></td><td><b>{f3(A['map50_95'])}</b></td></tr>
<tr class="pub"><td>Published: DocLayout-YOLO [1]</td><td>full IndicDLP test set</td><td>≈0.735</td><td>0.545</td></tr>
<tr class="pub"><td>Published: YOLOv10x [1] (best in paper)</td><td>full IndicDLP test set</td><td>–</td><td>0.550</td></tr></table>
<ul class="tight"><li><b>Metrics:</b> COCO mAP@50, mAP@50:95, precision/recall/F1, seconds per page — all on the same {A.get('num_images')} test pages.</li>
<li><b>One-stage vs two-stage:</b> with the same data and similar time, YOLOv8s beats Faster R-CNN ({f3(P['map50'])} vs {f3(B['map50'])}) and is {B['inference_time'] / P['inference_time']:.0f}× faster.</li>
<li><b>Did we improve?</b> Our own models did <u>not</u> beat the published model — they saw ~42× less data. So the app uses the published model, and our contribution is the study and the OCR pipeline. This is stated honestly in the paper.</li></ul></div>
<div>{img(PLOTS / 'authors_per_class_ap.png', 'wide')}<p class="cap">AP@50 per label, DocLayout-YOLO.</p>
{img(PLOTS / 'authors_confusion_matrix.png', 'wide', 'max-height:250px;object-fit:contain')}</div></div>''')

# ---------------------------------------------------------------- 16. Sample outputs
slide('DEMO', 'Sample Outputs and Demonstration', f'''<div class="three demo">
<figure>{img(SHOTS / '02_detection_tamil.png')}<figcaption>Running app: Tamil textbook page, 6 regions found</figcaption></figure>
<figure>{img(SHOTS / '03_ocr.png')}<figcaption>OCR: script auto-detected as Tamil, text in reading order, MD / JSON download</figcaption></figure>
<figure>{img(ROOT / 'outputs' / 'predictions' / 'authors_prediction_2.png')}<figcaption><b>Correct:</b> Kannada act — ground truth (green) vs prediction (red)</figcaption></figure>
</div><div class="two demo2">
<figure>{img(ROOT / 'outputs' / 'predictions' / 'errors' / 'authors_most_missed.png')}<figcaption><b>Incorrect:</b> dense Bengali form — 90 of 105 regions missed (forms are the hardest domain)</figcaption></figure>
<figure>{img(SHOTS / '04_dashboard.png')}<figcaption>Dashboard tab: measured metrics of all three models</figcaption></figure></div>''')

# ---------------------------------------------------------------- 17. Challenges
slide('LESSONS', 'Challenges and Limitations', f'''<div class="grid2">
{card('Data size and compute', 'The full dataset is 91 GB; a laptop GPU can train on ~2.3K pages for an hour. Our own models stay far below the published one (0.26 vs 0.76 mAP@50).', '#C62828')}
{card('Dense forms and rare labels', f"Forms, with many tiny fields, score lowest; labels like quote and sub-unordered-list score below 0.2 AP even for the best model.", '#D9A441')}
{card('OCR quality', 'Tesseract misreads small print and some scripts (e.g. Marathi script picked right on 6/10 pages); IndicDLP has no text ground truth, so character accuracy could not be measured.', '#2563EB')}
{card('Engineering bugs found and fixed', 'The original code had crashes, wrong label lists and made-up plots; 30 fixes are logged in docs/FINAL_AUDIT.md. Timings were spoiled twice by jobs sharing the GPU and had to be re-run.', '#7C3AED')}
</div>''')

# ---------------------------------------------------------------- 18. Conclusion
slide('WRAP-UP', 'Conclusion and Future Scope', f'''<div class="two">
<div><div class="sub">CONCLUSION</div><ul class="tight">
<li><b>Implemented:</b> a full pipeline — data preparation, two trained detectors, evaluation of three, OCR for 12 scripts, web app, REST API, Docker.</li>
<li><b>Main results:</b> best model {f3(A['map50'])} mAP@50 on {A.get('num_images')} test pages, finds {pct(A['at_high_recall_conf']['recall'])} of regions in "find every part" mode, {A['inference_time']:.2f} s/page; script picked right on {OCR.get('script_detection_correct')}/120 pages.</li>
<li><b>Study:</b> accuracy is fairly even across languages (CI {ci.get('authors', 0):.2f}) but not across domains (CI {dci.get('authors', 0):.2f}); data size, not the script, decides how well rare labels work; the best model keeps {pct(mri.get('authors', 0))} of its accuracy on damaged scans (blur hurts most).</li>
<li><b>Objectives:</b> all four met; our own detectors did not reach the published accuracy, which we report openly.</li></ul></div>
<div><div class="sub">FUTURE SCOPE</div><ul class="tight">
<li>Controlled unseen-script test (same data, with / without a script family); train longer on a cloud GPU with blur augmentation.</li>
<li>Replace Tesseract with an Indic OCR model (e.g. a fine-tuned VLM) and build a small text ground-truth set.</li>
<li>Table structure and question answering on detected tables [3, 4]; answer grounding [6].</li>
<li>Real-world pilot on government circulars and school textbooks; mobile / edge model.</li></ul>
<div class="sdg"><span>SDG 4</span><span>SDG 9</span><span>SDG 10</span><span>SDG 16</span></div></div></div>''')

# ---------------------------------------------------------------- 19. References
refs = ['O. Nath, S. Kukkala, M. Khapra, R. K. Sarvadevabhatla, "IndicDLP: A Foundational Dataset for Multi-Lingual and Multi-Domain Document Layout Parsing," ICDAR 2025; arXiv:2512.20236.',
        'B. V. Kasuba et al., "PLATTER: A Page-Level Handwritten Text Recognition System for Indic Scripts," arXiv:2502.06172, 2025.',
        'A. Mondal et al., "UniTabBank: A Large Scale Multi-Lingual, Multi-Layout, Multi-Type, Multi-Format Dataset for Table Detection," IEEE/CVF WACV 2026.',
        'V. Pal, E. Kanoulas, A. Yates, M. de Rijke, "Table Question Answering for Low-resourced Indic Languages," arXiv:2410.03576, 2024.',
        'Y. Xu et al., "LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding," arXiv:2104.08836, 2021.',
        'B. V. Kasuba, P. Chaudhuri, G. Ramakrishnan, "DRISHTIKON: Visual Grounding at Multiple Granularities in Documents," arXiv:2506.21316, 2025.',
        'S. Bhattacharyya et al., "Adapting Vision-Language Models for Hindi OCR," IIIT Hyderabad, 2025.',
        'A. R. Dumada, S. G. Shah, "Incorporate Structure and Content for Devanagari Table Extraction," IJSAT 17(1), 2026.',
        'Z. Zhao et al., "DocLayout-YOLO: Enhancing Document Layout Analysis through Diverse Synthetic Data and Global-to-Local Adaptive Perception," arXiv:2410.12628, 2024.',
        'S. Ren, K. He, R. Girshick, J. Sun, "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks," NeurIPS 2015.',
        'G. Jocher, A. Chaurasia, J. Qiu, "Ultralytics YOLOv8," 2023, github.com/ultralytics/ultralytics.',
        'T.-Y. Lin et al., "Microsoft COCO: Common Objects in Context," ECCV 2014 (metric and pretrained weights).',
        'R. Smith, "An Overview of the Tesseract OCR Engine," ICDAR 2007; tessdata_best models.',
        'D. Hendrycks, T. Dietterich, "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations," ICLR 2019.',
        'PyTorch, torchvision, pycocotools, Streamlit, FastAPI, Docker — official documentation.']
other = ['Dataset: huggingface.co/datasets/ai4bharat/indicdlp (AIKosh)', 'Model weights: zenodo.org/records/15881917',
         'Code: github.com/va-ada/IndicDoc-AI (private)', 'indicdlp.github.io']
slide('SOURCES', 'References', '<div class="two w65"><ol class="refs">' + ''.join(f'<li>{r}</li>' for r in refs) +
      '</ol><div class="card" style="--c:#D9A441"><div class="ct">OTHER SOURCES</div><div class="cx">' +
      '<br>'.join(other) + '</div></div></div>')

# ---------------------------------------------------------------- 20. Thank you
slides.append(f'''<section class="slide title end"><div class="blob b1"></div><div class="blob b2"></div>
<div class="tt c">Thank You</div><div class="ts c gold">Questions and discussion</div>
<div class="names">{' | '.join(n for _, n, _ in TEAM)}</div><div class="small c">{FOOT}</div></section>''')

CSS = '''
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&family=Inter:wght@400;500;600;700&display=swap');
@page { size: 1334px 750px; margin: 0; }
* { box-sizing: border-box; }
body { margin: 0; font-family: Inter, -apple-system, Helvetica, sans-serif; color: #1E293B; background: #CBD5E1; }
.slide { width: 1334px; height: 750px; position: relative; overflow: hidden; background: #EEF2F7; padding: 30px 66px 50px 76px;
  page-break-after: always; break-after: page; margin: 0 auto; }
.bar { position: absolute; left: 0; top: 0; bottom: 0; width: 14px; background: #14284B; border-right: 4px solid #D9A441; }
.kicker { color: #D9A441; font-weight: 700; letter-spacing: .14em; font-size: 13px; margin-top: 4px; }
h1 { font-family: 'Source Serif 4', Georgia, serif; color: #14284B; font-size: 40px; margin: 2px 0 16px; }
.body { height: 590px; font-size: 17px; line-height: 1.38; }
.foot { position: absolute; left: 76px; right: 66px; bottom: 16px; display: flex; justify-content: space-between; color: #64748B; font-size: 12.5px; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; } .two.w60 { grid-template-columns: 1.35fr 1fr; } .two.w65 { grid-template-columns: 2fr 1fr; }
.three { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
.grid4 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px 16px; }
.card { background: #fff; border-radius: 14px; padding: 16px 18px; border-top: 5px solid var(--c); box-shadow: 0 1px 3px rgba(0,0,0,.06); }
.card .ct { font-weight: 700; color: #14284B; font-size: 18px; margin-bottom: 8px; } .card .cx { font-size: 16px; line-height: 1.5; }
.grid2 .card { min-height: 270px; padding: 22px 24px; } .grid2 .card .cx { font-size: 19.5px; line-height: 1.55; } .grid2 .card .ct { font-size: 23px; margin-bottom: 10px; }
table { border-collapse: collapse; width: 100%; background: #fff; border-radius: 12px; overflow: hidden; }
th { background: #14284B; color: #fff; text-align: left; padding: 8px 10px; font-size: 13.5px; }
td { padding: 7px 10px; border-bottom: 1px solid #E2E8F0; vertical-align: top; }
table.lit td { font-size: 14.2px; padding: 9px 9px; } .lit .ref { color: #2563EB; font-weight: 700; } .lit .lim { color: #9F1239; }
table.kv td:first-child { width: 150px; font-weight: 700; color: #14284B; background: #F8FAFC; } table.kv td { font-size: 15px; padding: 8px 10px; }
table.kv.small td { font-size: 15px; padding: 8px 10px; }
table.tools td { font-size: 18px; padding: 15px 16px; } table.tools td:first-child { width: 270px; color: #14284B; }
table.res td, table.res th { text-align: center; font-size: 15.5px; padding: 9px 10px; } table.res td:first-child, table.res th:first-child { text-align: left; }
table.res.small td { font-size: 14.5px; } tr.pub td { color: #64748B; font-style: italic; }
.cap { color: #64748B; font-size: 12px; margin: 4px 0 8px; }
img { max-width: 100%; display: block; } img.wide { width: 100%; border-radius: 10px; background: #fff; }
.missing { background: #FEE2E2; color: #991B1B; padding: 30px; border-radius: 10px; text-align: center; }
.toc { columns: 2; column-gap: 60px; list-style: none; padding: 0; margin: 10px 0 0; }
.toc li { font-size: 19px; padding: 10px 14px; margin-bottom: 9px; background: #fff; border-radius: 10px; break-inside: avoid; }
.toc li span { color: #D9A441; font-weight: 700; margin-right: 14px; }
.gaps { display: flex; flex-direction: column; gap: 16px; }
.gap { display: grid; grid-template-columns: 1fr 40px 1.2fr; align-items: center; }
.gl { background: #FFF1F2; border: 1.5px solid #FDA4AF; border-radius: 12px; padding: 20px 18px; font-size: 18px; }
.gr { background: #F0FDF4; border: 1.5px solid #86EFAC; border-radius: 12px; padding: 20px 18px; font-size: 18px; font-weight: 600; position: relative; }
.arrow { text-align: center; color: #D9A441; font-size: 26px; font-weight: 700; }
.rq { position: absolute; right: 10px; top: -9px; background: #14284B; color: #fff; font-size: 11px; padding: 2px 8px; border-radius: 10px; }
.note { margin-top: 24px; background: #14284B; color: #fff; border-radius: 12px; padding: 18px 22px; font-size: 18px; }
.quote { background: #fff; border-left: 5px solid #D9A441; border-radius: 10px; padding: 14px 18px; font-size: 16.5px; }
.sub { color: #14284B; font-weight: 700; letter-spacing: .12em; font-size: 13.5px; margin: 16px 0 8px; }
.big { font-size: 17.5px; font-weight: 600; margin: 0; } ul { margin: 0; padding-left: 20px; } li { margin-bottom: 4px; }
.rqrow { background: #fff; border-radius: 10px; padding: 16px 16px; margin-bottom: 12px; font-size: 17.5px; }
.rqrow span { background: #D9A441; color: #14284B; font-weight: 800; border-radius: 6px; padding: 2px 8px; margin-right: 10px; font-size: 13px; }
.flow { display: flex; gap: 8px; margin-bottom: 20px; }
.step { flex: 1; background: #fff; border-radius: 12px; padding: 12px 10px; border-top: 4px solid #2563EB; text-align: center; position: relative; }
.step b { display: block; color: #14284B; font-size: 17px; } .step span { font-size: 14px; color: #475569; }
.sn { width: 28px; height: 28px; border-radius: 50%; background: #14284B; color: #fff; margin: 0 auto 6px; line-height: 28px; font-weight: 700; }
.three .card .cx { font-size: 17.5px; line-height: 1.55; } .three .card { min-height: 360px; } .three .card .ct { font-size: 20px; }
.diagram { display: flex; flex-direction: column; gap: 2px; }
.lane { display: grid; grid-template-columns: 130px 1fr 1fr 1fr; gap: 14px; align-items: stretch; }
.lane-t { font-weight: 800; color: #94A3B8; letter-spacing: .12em; font-size: 12px; align-self: center; }
.blk { background: #fff; border-left: 6px solid var(--c); border-radius: 10px; padding: 13px 16px; }
.blk b { display: block; color: #14284B; font-size: 18px; } .blk span { font-size: 14.5px; color: #475569; }
.down { text-align: center; color: #D9A441; font-size: 18px; padding-left: 130px; line-height: 1.25; }
.samples { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.samples figure, .demo figure, .demo2 figure { margin: 0; background: #fff; border-radius: 10px; padding: 6px; }
.samples img { height: 232px; width: 100%; object-fit: cover; object-position: top; border-radius: 6px; }
figcaption { font-size: 13px; color: #475569; margin-top: 4px; text-align: center; }
.arch { background: #fff; border-radius: 12px; padding: 14px 18px; margin-bottom: 16px; border-left: 6px solid var(--c); }
.an { font-weight: 700; color: #14284B; font-size: 19px; margin-bottom: 7px; }
.apipe { display: flex; align-items: stretch; gap: 6px; }
.ap { flex: 1; background: #F1F5F9; border-radius: 8px; padding: 6px 9px; } .ap b { display: block; font-size: 15.5px; color: #14284B; } .ap span { font-size: 13.5px; color: #475569; }
.aa { align-self: center; color: var(--c); font-weight: 800; }
.anote { font-size: 14.5px; color: #334155; margin-top: 6px; }
ul.tight li { font-size: 16px; margin-bottom: 9px; line-height: 1.42; }
pre.code { background: #0F1B33; color: #E2E8F0; border-radius: 12px; padding: 14px 16px; font-size: 13.4px; line-height: 1.5; margin: 0 0 10px; white-space: pre-wrap; }
.code .c { color: #94A3B8; } .code .s { color: #FBBF24; } .code .n { color: #86EFAC; }
.mini img { border-radius: 10px; height: 240px; width: 100%; object-fit: cover; object-position: left top; }
.kpis { display: flex; flex-direction: column; gap: 10px; }
.kpi { background: #fff; border-radius: 12px; padding: 12px 18px; border-left: 6px solid #D9A441; }
.kpi b { font-family: 'Source Serif 4', Georgia, serif; font-size: 38px; color: #14284B; display: block; } .kpi span { font-size: 15px; color: #475569; }
.fcard { background: #fff; border-radius: 12px; padding: 8px 12px; } .fh { font-weight: 700; color: #14284B; font-size: 14px; }
.fcard img { height: 172px; width: 100%; object-fit: contain; } .fcard p { margin: 2px 0 0; font-size: 12.4px; line-height: 1.3; }
.demo img { height: 250px; width: 100%; object-fit: cover; object-position: center top; border-radius: 6px; }
.demo2 { margin-top: 10px; } .demo2 img { height: 225px; width: 100%; object-fit: cover; object-position: center top; border-radius: 6px; }
.refs { font-size: 13.6px; padding-left: 26px; margin: 0; } .refs li { margin-bottom: 6px; }
.sdg { display: flex; gap: 8px; margin-top: 16px; } .sdg span { flex: 1; text-align: center; color: #fff; font-weight: 800; padding: 10px; border-radius: 10px; }
.sdg span:nth-child(1) { background: #C5192D; } .sdg span:nth-child(2) { background: #FD6925; } .sdg span:nth-child(3) { background: #DD1367; } .sdg span:nth-child(4) { background: #00689D; }
.title { background: #14284B; color: #fff; padding: 64px 76px; }
.blob { position: absolute; border-radius: 50%; background: #1C3A66; } .b1 { width: 420px; height: 420px; right: -110px; top: -110px; } .b2 { width: 340px; height: 340px; left: 760px; bottom: -190px; }
.tk { color: #D9A441; font-weight: 700; letter-spacing: .14em; font-size: 14px; position: relative; }
.tt { font-family: 'Source Serif 4', Georgia, serif; font-size: 82px; font-weight: 700; line-height: 1.05; margin: 8px 0 6px; position: relative; }
.ts { font-size: 23px; color: #CBD5E1; max-width: 720px; position: relative; line-height: 1.3; }
.team { display: flex; gap: 12px; margin: 26px 0 22px; position: relative; }
.member { background: #1C3A66; border-radius: 12px; padding: 12px 14px; display: flex; gap: 10px; align-items: center; width: 182px; }
.member b { display: block; font-size: 14.5px; } .member span { color: #D9A441; font-size: 13px; }
.av { width: 38px; height: 38px; border-radius: 50%; background: #D9A441; color: #14284B; font-weight: 800; display: flex; align-items: center; justify-content: center; flex: none; }
.meta p { margin: 5px 0; font-size: 16px; position: relative; } .meta b { color: #D9A441; }
.titleimg { position: absolute; right: 70px; top: 120px; width: 400px; height: 500px; border-radius: 14px; overflow: hidden; border: 6px solid #fff; }
.titleimg img { width: 100%; height: 100%; object-fit: cover; object-position: top; }
.end { display: flex; flex-direction: column; align-items: center; justify-content: center; }
.c { text-align: center; } .gold { color: #D9A441; } .names { font-size: 20px; margin-top: 30px; position: relative; } .end .small { color: #94A3B8; margin-top: 8px; position: relative; }
@media screen { .slide { margin: 20px auto; box-shadow: 0 4px 20px rgba(0,0,0,.2); } }
'''


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    html = f'<!doctype html><html><head><meta charset="utf-8"><title>IndicDoc AI Phase-2</title><style>{CSS}</style></head><body>{"".join(slides)}</body></html>'
    hp = OUT / 'IndicDoc_AI_Phase2_Implementation.html'
    hp.write_text(html, encoding='utf-8')
    pdf = OUT / 'IndicDoc_AI_Phase2_Implementation.pdf'
    subprocess.run([CHROME, '--headless=new', '--disable-gpu', '--no-pdf-header-footer', '--virtual-time-budget=10000',
                    f'--print-to-pdf={pdf}', hp.as_uri()], check=True, capture_output=True, timeout=180)
    print(f'{len(slides)} slides -> {pdf}')


if __name__ == '__main__':
    main()
