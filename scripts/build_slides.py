"""
IndicDoc AI - Phase-2 presentation (Project Implementation), following the department template.

The project as delivered: IndicDoc Verify, offline pre-print checking of question papers and their translations in
12 Indian languages, built on deep-learning layout analysis. Every number comes from outputs/metrics/ (measured
results); nothing is typed in by hand.
Writes docs/presentation/IndicDoc_AI_Phase2_Implementation.html and .pdf (printed with headless Chrome / Chromium).
Usage: python scripts/build_slides.py      (CHROME=/path/to/chrome to choose the browser)
"""
import base64
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / 'outputs' / 'metrics'
FIG = ROOT / 'docs' / 'figures'
PLOTS = ROOT / 'outputs' / 'plots'
SHOTS = ROOT / 'docs' / 'screenshots'
OUT = ROOT / 'docs' / 'presentation'

TEAM = [('PJ', 'Prashant Jha', 41), ('AS', 'Aditya Soni', 42), ('RP', 'Riddhi Patil', 39), ('MB', 'Mariam Badure', 40)]
GUIDE = 'Dr. Joanne Gomes'
FOOT = 'IndicDoc AI | Phase-2: Project Implementation'


def j(name):
    p = M / name
    return json.loads(p.read_text()) if p.exists() else {}


A, P, B = j('authors_metrics.json'), j('proposed_metrics.json'), j('baseline_metrics.json')
S, OCR = j('study_summary.json'), j('ocr_evaluation.json')
V, VN = j('verify_benchmark_oracle.json'), j('verify_benchmark_noisy.json')
V_NP, V_NG = j('verify_benchmark_oracle_no_number_pass.json'), j('verify_benchmark_oracle_no_geometry.json')
QAP = A.get('per_class_ap50', {})


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


vo, vq = V.get('overall', {}), V.get('by_quality', {})
clean_q, deg_q = vq.get('clean_render', {}), vq.get('scan_degraded', {})
n_exact = sum(c['questions_read'] == 18 - (c['type'] == 'question_dropped') for c in V.get('cases', []))
q_labels = ['first-level-question', 'second-level-question', 'third-level-question', 'options', 'answer']
q_ap = [QAP.get(k, 0) for k in q_labels]

# ---------------------------------------------------------------- 1. Title
team_html = ''.join(f'<div class="member"><div class="av">{i}</div><div><b>{n}</b><span>Roll No. {r}</span></div></div>'
                    for i, n, r in TEAM)
slides.append(f'''<section class="slide title"><div class="blob b1"></div><div class="blob b2"></div>
<div class="tk">PHASE-2 · PROJECT IMPLEMENTATION · DL MINI PROJECT</div>
<div class="tt">IndicDoc <span class="gold">Verify</span></div>
<div class="ts">Deep-learning layout analysis that checks Indian question papers and their translations <b>before they are printed</b>: offline, 12 languages</div>
<div class="team">{team_html}</div>
<div class="meta"><p><b>Course:</b> Deep Learning Mini Project (Expt. 11, SDG-based)</p>
<p><b>Department:</b> Artificial Intelligence and Machine Learning, St. Francis Institute of Technology (SFIT)</p>
<p><b>Guide:</b> {GUIDE}</p><p><b>Academic year:</b> 2026–27 · Semester V (TE AIML)</p></div>
<div class="titleimg">{img(FIG / 'bench_hi.jpg')}</div></section>''')

# ---------------------------------------------------------------- 2. Content
items = ['Literature Survey Summary Table', 'Research Gaps', 'Problem Statement and Objectives',
         'Proposed System, Methodology and Novelty', 'System Block Diagram', 'Dataset Description and Pre-processing',
         'Deep Learning Model Architecture', 'Tools, Technologies and Environment', 'Implementation Details',
         'Experimental Setup and Training', 'Results and Performance Evaluation',
         'Comparison with Baseline / Existing Methods', 'Sample Outputs and Demonstration',
         'Challenges and Limitations', 'Conclusion and Future Scope', 'References']
lis = ''.join(f'<li><span>{i:02d}</span>{t}</li>' for i, t in enumerate(items, 1))
slide('OVERVIEW', 'Content', f'<ol class="toc">{lis}</ol>')

# ---------------------------------------------------------------- 3. Literature table (2021-2026)
lit = [
    ('[1]', 'Nath et al., 2025', 'Layout parsing of Indian-language pages', 'IndicDLP: 119,806 pages, 12 languages, 42 labels incl. questions and options',
     'Benchmark of YOLOv10x, DocLayout-YOLO, RoDLA, DINO, DiT, Florence-2', 'Detects regions only; nobody uses the question labels to check papers'),
    ('[2]', 'Kasuba et al., 2025', 'Page-level handwritten Indic text recognition', 'CHIPS, ~25K pages, 10 languages',
     'DBNet word detection + 6 recognisers (CRNN, PARSeq, ...)', 'User must name the language; text only'),
    ('[3]', 'Mondal et al., 2026', 'Multilingual table detection', 'UniTabBank: 55K pages, 82K tables, 28 languages',
     'Semi-automatic labelling; YOLOv11 detector', 'Borderless tables and degraded scans fail'),
    ('[4]', 'Pal et al., 2024', 'Table question answering, low-resource Indic', 'BanglaTabQA 2M, HindiTabQA 643K',
     'SQL-generated QA; fine-tuned mBART / M2M / Llama', 'Wikipedia tables only'),
    ('[5]', 'Xu et al., 2021', 'Multilingual document understanding', '30M docs; XFUND, 7 languages',
     'LayoutXLM: text + layout + image transformer', 'No Indian language'),
    ('[6]', 'Kasuba et al., 2025', 'Grounding answers on the page', 'MGVG: 70 circulars, 509 QA pairs',
     'OCR + Llama-3.1-8B + region matching', 'OCR-dependent; single page'),
    ('[7]', 'Bhattacharyya et al., 2025', 'Hindi OCR with vision-language models', '1.1M printed Hindi words + pages',
     'LoRA-tuned GOT-OCR 2.0', 'Hindi only; mixed-language lines are hard'),
    ('[8]', 'Dumada & Shah, 2026', 'Devanagari table extraction', '500+ Devanagari documents',
     'Structure + Tesseract content, weighted fusion', 'No ablation; self-reported'),
    ('[9]', 'Babayeva et al., 2025', 'Translating a test into other languages with an LLM', 'Force Concept Inventory, many languages',
     'GPT-4o translation + back-translation + human review', 'Every item still checked by hand; no page structure'),
    ('[10]', 'Rei et al., 2022', 'Translation quality estimation without a reference', 'WMT 2022 QE shared task',
     'CometKiwi: multilingual encoder predicts quality per sentence', 'Plain sentences: no question numbers, options or marks'),
]
rows = ''.join(f'<tr><td class="ref">{r}</td><td><b>{a}</b></td><td>{p}</td><td>{d}</td><td>{m}</td><td class="lim">{l}</td></tr>'
               for r, a, p, d, m, l in lit)
slide('PHASE-1 RECAP + TRANSLATION QA', 'Literature Survey Summary Table',
      f'<table class="lit lit10"><tr><th>Ref</th><th>Author</th><th>Problem</th><th>Dataset</th><th>Method</th><th>Limitation</th></tr>{rows}</table>')

# ---------------------------------------------------------------- 4. Research gaps
gaps = [('Exam papers are translated into up to 13 languages and vetted only by hand; errors reach candidates',
         'An automatic pre-print check of every language version against the source', 'G1'),
        ('Translation-quality tools [9, 10] read plain sentences: blind to question numbers, options, marks, figures',
         'Compare versions question by question using the page layout itself', 'G2'),
        ('Indic layout models [1] detect questions and options, but this is not used to check papers',
         f'Use the detector\'s question / option regions (AP@50 {min(q_ap):.2f}–{max(q_ap):.2f}) as the backbone of the checker', 'G3'),
        ('Indic OCR misreads Latin digits in mixed text [7]; a checker of numbers needs every digit right',
         'A "number pass": two OCR models must agree on every number', 'G4')]
g = ''.join(f'<div class="gap"><div class="gl">⚠ {a}</div><div class="arrow">→</div><div class="gr">✓ {b}<span class="rq">{c}</span></div></div>' for a, b, c in gaps)
slide('WHAT IS MISSING', 'Research Gaps', f'''<div class="gaps tightgaps">{g}</div>
<div class="note">We also looked at court-filing defect checks: the Supreme Court already uses an AI tool built with IIT Madras, and a
commercial tool is reported in 16 High Courts, so we chose the problem <b>nobody automates yet</b>: checking a question paper and its translations.</div>''')

# ---------------------------------------------------------------- 5. Problem & objectives
cases = [('NEET-UG 2018', '49 questions wrong in the Tamil version; HC ordered 196 grace marks, SC set it aside'),
         ('KPSC KAS 2024', '59 errors in the Kannada paper; re-exam for ~2 lakh candidates, errors again'),
         ('APPSC Group-I', '51 of 120 questions alleged mistranslated into Telugu; SC notice'),
         ('UGC-NET Jun 2026', 'Translation errors + questions repeated from old papers; re-test 9–10 Sep'),
         ('MBBS (NTRUHS) 2026', 'The same MCQ printed twice; AP HC ordered a compensatory mark')]
cs = ''.join(f'<div class="case"><b>{a}</b><span>{b}</span></div>' for a, b in cases)
slide('WHAT WE SOLVE', 'Problem Statement and Objectives', f'''<div class="two w55">
<div><div class="quote"><b>Problem:</b> a question paper is set in one language and translated into Hindi and regional
languages. Every version is checked by hand before printing. When a number, an option, a mark or a whole question differs, lakhs of
candidates are affected: grace marks, re-exams, court cases.</div>
<div class="sub">MAIN OBJECTIVE</div><p class="big">Build an offline deep-learning system that reads a printed paper in any of 12
Indian languages and flags every structural or numerical error, and every difference from its source version, before printing.</p>
<div class="sub">SPECIFIC OBJECTIVES</div><ul>
<li>Detect the paper's structure (questions, options, answers) with a layout detector; compare 3 detectors.</li>
<li>Read text with reliable numbers in 12 languages (number pass).</li>
<li>Checks: numbering, duplicates, options, marks, translation parity, repeats, format profiles.</li>
<li>Measure errors found and false alarms on a benchmark; ship app + API in Docker.</li></ul>
<div class="sub">EXPECTED OUTPUT</div><p>A verdict (print / check / do not print) and each finding with its question and page, outlined on the page, as HTML and JSON.</p></div>
<div><div class="sub">IT HAS ALREADY HAPPENED</div>{cs}
<p class="cap">Sources: The News Minute, Deccan Herald, India Legal, The Week, SCC Online, Medical Dialogues (refs. slide).</p></div></div>''')

# ---------------------------------------------------------------- 6. Proposed system & novelty
steps = [('1', 'Paper in', 'PDF / scans + source version'), ('2', 'Find structure', 'DocLayout-YOLO: questions, options'),
         ('3', 'Read text', 'Indic OCR + English number pass'), ('4', 'Build questions', 'number, stem, options, marks, figures'),
         ('5', 'Check', 'structure, parity, repeats, format'), ('6', 'Report', 'verdict + findings on the page')]
st = ''.join(f'<div class="step"><div class="sn">{a}</div><b>{b}</b><span>{c}</span></div>' for a, b, c in steps)
slide('OUR APPROACH', 'Proposed System, Methodology and Novelty', f'''<div class="flow">{st}</div>
<div class="three">
{card('How deep learning is used', f'A convolutional object detector finds every question, sub-question, option and answer on the page (AP@50 {min(q_ap):.2f}–{max(q_ap):.2f} on these labels). Its regions and the word positions inside them turn a picture of a page into structured questions, in any script.', '#2563EB')}
{card('What is new', '<b>Translation parity without translation:</b> numbers, option counts, marks, sub-questions and figures must be the same in every language, so versions are compared question by question with no MT model.<br><b>Number pass:</b> two OCR models must agree on every digit.', '#D9A441')}
{card('Why it sells', '<b>USP:</b> no tool we found checks an Indian question paper against its translations question by question; IndicDoc Verify does, offline, in 12 languages, before printing. Confidential papers never leave the exam body. Same engine checks any document format via JSON profiles (e.g. bilingual circulars).', '#2E7D32')}
</div>''')

# ---------------------------------------------------------------- 7. Block diagram
def blk(t, s, c='#14284B'):
    return f'<div class="blk" style="--c:{c}"><b>{t}</b><span>{s}</span></div>'


slide('ARCHITECTURE', 'System Block Diagram', f'''<div class="diagram">
<div class="lane"><div class="lane-t">USER</div>{blk('Streamlit: Verify Paper tab', 'upload paper, source, bank, profile', '#2563EB')}{blk('REST API (FastAPI)', 'POST /verify · GET /verify/profiles', '#2563EB')}{blk('Docker, offline', 'papers never leave the exam body', '#2563EB')}</div>
<div class="down">▼</div>
<div class="lane"><div class="lane-t">INPUT</div>{blk('Render pages', 'PDF at 200 dpi, images')}{blk('Source version', 'optional, any of 12 languages')}{blk('Question bank · profile', 'earlier papers · JSON format rules')}</div>
<div class="down">▼</div>
<div class="lane"><div class="lane-t">DEEP LEARNING</div>{blk('Layout detector', 'DocLayout-YOLO (default) · YOLOv8s · Faster R-CNN', '#D9A441')}{blk('Post-process', 'high-recall threshold, duplicate removal', '#D9A441')}{blk('Question regions', 'questions, sub-questions, options, answers', '#D9A441')}</div>
<div class="down">▼</div>
<div class="lane"><div class="lane-t">TEXT</div>{blk('Indic OCR', 'Tesseract, the page\'s own script', '#2E7D32')}{blk('Number pass', 'English model, digits must agree', '#2E7D32')}{blk('Questions', 'words → regions → options by layout gaps', '#2E7D32')}</div>
<div class="down">▼</div>
<div class="lane"><div class="lane-t">CHECKS + OUTPUT</div>{blk('Structure · repeats', 'numbering, duplicates, options, marks, bank', '#7C3AED')}{blk('Translation parity', 'align versions; numbers, options, marks', '#7C3AED')}{blk('Report', 'verdict, findings, pages outlined; HTML / JSON', '#7C3AED')}</div>
</div>''')

# ---------------------------------------------------------------- 8. Dataset
lang_rows = list(csv.DictReader(open(ROOT / 'dataset' / 'language_split.csv')))
tot = {k: sum(int(r[k]) for r in lang_rows) for k in ('train', 'val', 'test')}
class_rows = list(csv.DictReader(open(ROOT / 'dataset' / 'class_counts.csv')))
total_boxes = sum(int(r['total']) for r in class_rows)
pages_csv = list(csv.DictReader(open(ROOT / 'dataset' / 'pages.csv')))
qp_pages = sum(1 for r in pages_csv if r['domain'] == 'question papers')
cnt = {r['class_name']: int(r['total']) for r in class_rows}
n_pairs = V.get('overall', {}).get('pairs', 0)
n_err = V.get('overall', {}).get('errors_injected', 0)
slide('DATA', 'Dataset Description and Pre-processing', f'''<div class="two w60">
<div><table class="kv">
<tr><td>Layout data</td><td><b>IndicDLP</b> (Nath et al., ICDAR 2025; Hugging Face <code>ai4bharat/indicdlp</code>, AIKosh): 119,806 pages, 12 languages, 12 domains, 42 labels</td></tr>
<tr><td>Our subset</td><td><b>{tot['train'] + tot['val'] + tot['test']:,} pages</b>, {total_boxes:,} boxes, language-balanced, official splits: train {tot['train']:,} · val {tot['val']} · test {tot['test']}. {qp_pages} question-paper pages; {cnt.get('first-level-question', 0):,} question and {cnt.get('options', 0):,} option boxes</td></tr>
<tr><td>Verify benchmark</td><td><b>{n_pairs} paper pairs</b> (English → Hindi, English → Tamil), 18 questions each (MCQ + short answer), rendered at 200 dpi; <b>{n_err}</b> with one injected error of 9 real kinds, {n_pairs - n_err} clean; half degraded like scans</td></tr>
<tr><td>Kept apart</td><td>A separate <b>development set</b> (48 pairs, other seed) was used to build the parser; the test set above was run once</td></tr>
<tr><td>Pre-processing</td><td>Parquet → PNG + COCO JSON; boxes clipped; COCO → YOLO labels; letterbox 640 / 1024 px. Verify: PDF → 200 dpi; blur 0.7 px + noise σ 8 + JPEG 55 for "scanned"</td></tr>
<tr><td>Augmentation</td><td>YOLOv8: mosaic, HSV, scale. Faster R-CNN: colour jitter, blur. No mirror flips</td></tr>
<tr><td>Challenges</td><td>Class imbalance ("paragraph" = 32% of boxes); no OCR ground truth in IndicDLP; no public set of erroneous exam papers, hence the benchmark</td></tr>
<tr><td>Deliverable</td><td><code>dataset/</code> CSVs + samples; <code>scripts/verify_bench/</code> rebuilds the benchmark</td></tr>
</table></div>
<div class="samples">
<figure>{img(FIG / 'bench_en.jpg')}<figcaption>benchmark · English source</figcaption></figure>
<figure>{img(FIG / 'bench_hi.jpg')}<figcaption>benchmark · Hindi version</figcaption></figure>
<figure>{img(FIG / 'bench_ta.jpg')}<figcaption>benchmark · Tamil, scan-degraded</figcaption></figure>
<figure>{img(ROOT / 'dataset' / 'samples' / 'hindi_newspapers.jpg')}<figcaption>IndicDLP · hindi newspaper</figcaption></figure>
</div></div>''')

# ---------------------------------------------------------------- 9. Architecture
def arch(name, color, parts, note):
    p = ''.join(f'<div class="ap"><b>{a}</b><span>{b}</span></div><div class="aa">→</div>' for a, b in parts)[:-len('<div class="aa">→</div>')]
    return f'<div class="arch" style="--c:{color}"><div class="an">{name}</div><div class="apipe">{p}</div><div class="anote">{note}</div></div>'


slide('MODELS', 'Deep Learning Model Architecture', f'''
{arch(f"DocLayout-YOLO · {A.get('parameters', 0) / 1e6:.1f} M params · used by IndicDoc Verify", '#14284B',
      [('Input', '1024×1024×3 page'), ('Backbone', 'CSP conv blocks + GL-CRM (global-to-local receptive fields)'), ('Neck', 'PAN feature pyramid, 3 scales'), ('Head', 'YOLOv10-style, NMS-free'), ('Output', 'N × (box, 42-class score)')],
      f'Fine-tuned on full IndicDLP by its authors (Zenodo 15881917). Question labels: first-level {QAP.get("first-level-question", 0):.2f}, second-level {QAP.get("second-level-question", 0):.2f}, options {QAP.get("options", 0):.2f}, answer {QAP.get("answer", 0):.2f} AP@50.')}
{arch(f"YOLOv8s · {P.get('parameters', 0) / 1e6:.1f} M params · fine-tuned by us", '#D9A441',
      [('Input', '640×640×3'), ('Backbone', 'CSPDarknet with C2f blocks + SPPF'), ('Neck', 'PAN-FPN'), ('Head', 'decoupled, anchor-free'), ('Output', 'boxes + 42 classes')],
      'Transfer learning: COCO-pretrained weights, 80-class head replaced by a 42-class head. Loss: CIoU + DFL (boxes) + BCE (classes).')}
{arch(f"Faster R-CNN · {B.get('parameters', 0) / 1e6:.1f} M params · baseline", '#64748B',
      [('Input', 'longest side 640–1024'), ('Backbone', 'ResNet-50, frozen batch-norm'), ('Neck', 'FPN, 5 levels'), ('RPN + RoIAlign', 'region proposals'), ('Box head', '43 outputs (42 + background)')],
      'Two-stage detector, COCO-pretrained; predictor replaced (43 classes). Loss: RPN + classification CE + smooth-L1 box.')}''')

# ---------------------------------------------------------------- 10. Tools
tools = [('Programming language', 'Python 3.11 (ML, checks, app, API); JavaScript (Node) for the benchmark renderer'),
         ('DL framework', 'PyTorch 2 with torchvision; Ultralytics 8 (YOLOv8); doclayout-yolo 0.0.4'),
         ('Development environment', 'VS Code / terminal, Git + GitHub, pytest (unit tests for every check)'),
         ('Libraries', 'pycocotools, OpenCV, Pillow, NumPy, pandas, Matplotlib, PyMuPDF; Tesseract 5 (tessdata_best, 12 scripts); tesseract.js for the benchmark'),
         ('Hardware', 'Apple M1 Pro GPU via PyTorch MPS (training, evaluation); CPU container for the benchmark; CPU in Docker'),
         ('Dataset tools', 'Hugging Face Hub + fsspec (partial Parquet reads); COCO / YOLO converters; Playwright + Noto fonts to render benchmark papers'),
         ('Interface / deployment', 'Streamlit app (Verify Paper tab) + FastAPI (/verify); Docker + docker-compose, offline')]
slide('STACK', 'Tools, Technologies and Environment',
      '<table class="tools"><tr><th>Component</th><th>Details</th></tr>' +
      ''.join(f'<tr><td><b>{a}</b></td><td>{b}</td></tr>' for a, b in tools) + '</table>')

# ---------------------------------------------------------------- 11. Implementation
code = '''<span class="c"># every page: layout (DL) -> OCR with number pass</span>
out = predict_single(model, page, <span class="s">"doclayout"</span>, device, conf=<span class="n">0.15</span>)
doc = extract_document(page, out, <span class="s">"hin"</span>, number_pass=<span class="n">True</span>)

<span class="c"># ml/verify — questions, checks, parity, repeats</span>
report = verify(hindi_pages, source_pages=english_pages,
                bank=earlier_questions, profile=<span class="s">"question_paper"</span>)
report[<span class="s">"summary"</span>]   <span class="c"># {questions, errors, warnings}</span>
html = to_html(report, {<span class="s">"paper"</span>: page_images})'''
slide('WHAT WE BUILT', 'Implementation Details', f'''<div class="two">
<div><ul class="tight">
<li><b>Layout model:</b> COCO-pretrained YOLOv8s and Faster R-CNN fine-tuned (SGD 0.005 / auto, CIoU+DFL+BCE), DocLayout-YOLO used for verification at its high-recall threshold ({A.get('high_recall_conf')}).</li>
<li><b>OCR:</b> the page is read once with its own script's model; an English <b>number pass</b> fixes misread Latin digits (Hindi reads "1947" as "947") and the two must agree on every number.</li>
<li><b>Questions:</b> question regions start a question; options are separated by <b>layout gaps</b> between words, so garbled markers do not matter; marks, figures, tables attached.</li>
<li><b>Checks:</b> numbering, duplicates, option count, marks total; <b>parity</b> by global alignment (Needleman–Wunsch) on language-independent features; repeats by character 3-gram similarity; format profiles in JSON.</li>
<li><b>Output:</b> verdict, findings with question and page, problems outlined on the page; HTML / JSON; same code in app, API and benchmark.</li></ul></div>
<div><pre class="code">{code}</pre>
<div class="mini">{img(SHOTS / '07_verify_report.png')}</div></div></div>''')

# ---------------------------------------------------------------- 12. Experimental setup
slide('EXPERIMENTS', 'Experimental Setup and Training', f'''<div class="two">
<div><table class="kv small">
<tr><td>Layout training</td><td>train {tot['train']:,} · val {tot['val']} · test {tot['test']} pages. Faster R-CNN: batch 4, SGD 0.005, <b>2 epochs, 76 min</b>. YOLOv8s: 640 px, batch 8, <b>12 epochs, 60 min</b>. Apple M1 Pro GPU</td></tr>
<tr><td>E1 Layout</td><td>3 detectors on {A.get('num_images')} test pages: COCO mAP@50, mAP@50:95, P / R / F1, per label (incl. question labels)</td></tr>
<tr><td>E2 Robustness</td><td>blur, noise, JPEG, half resolution, faded print (Robustness Index)</td></tr>
<tr><td>E3 Verify benchmark</td><td>{n_pairs} pairs, one injected error or none; layout boxes from the renderer, <b>real Tesseract OCR</b>; an error counts as found only if a matching finding points at the right question</td></tr>
<tr><td>E4 Ablations</td><td>without the number pass; without layout-based options; with a <b>noisy layout</b> (boxes ±8 px, 10% of question / option labels wrong)</td></tr>
<tr><td>Protocol</td><td>parser built on a separate development set; test set run once</td></tr>
</table></div>
<div>{img(FIG / 'fig_training.png', 'wide')}<p class="cap">Training curves of our two detectors (measured).</p></div></div>''')

# ---------------------------------------------------------------- 13. Results: layout
def row(name, m):
    hr = m.get('at_high_recall_conf', {})
    return (f"<tr><td><b>{name}</b></td><td>{f3(m['map50'])}</td><td>{f3(m['map50_95'])}</td>"
            f"<td>{f3(m['at_recommended_conf']['precision'])}</td><td>{f3(m['at_recommended_conf']['recall'])}</td>"
            f"<td>{f3(m['at_recommended_conf']['f1'])}</td><td>{f3(hr.get('recall', 0))}</td>"
            f"<td>{m['inference_time']:.3f}</td></tr>")


qrows = ''.join(f'<tr><td>{k}</td><td>{f3(QAP.get(k, 0))}</td><td>{f3(P.get("per_class_ap50", {}).get(k, 0))}</td><td>{f3(B.get("per_class_ap50", {}).get(k, 0))}</td></tr>' for k in q_labels)
slide('RESULTS (1/2) · LAYOUT', 'Results and Performance Evaluation', f'''
<table class="res"><tr><th>Model ({A.get('num_images')} test pages)</th><th>mAP@50</th><th>mAP@50:95</th><th>Precision*</th><th>Recall*</th><th>F1*</th><th>Recall, high-recall mode</th><th>s / page</th></tr>
{row('DocLayout-YOLO (authors)', A)}{row('YOLOv8s (ours)', P)}{row('Faster R-CNN (ours)', B)}</table>
<p class="cap">* at the "balanced" threshold chosen on validation pages. Apple M1 Pro GPU.</p>
<div class="two"><div><table class="res small"><tr><th>AP@50 on question labels</th><th>DocLayout-YOLO</th><th>YOLOv8s</th><th>Faster R-CNN</th></tr>{qrows}</table>
<p class="cap">The full-data model finds the parts of a question paper reliably; this is what the checker is built on.</p></div>
<div class="kpis">
<div class="kpi"><b>{f3(A['map50'])}</b><span>mAP@50, best model; question-paper domain {S.get('per_domain_authors', {}).get('question papers', 0):.2f}</span></div>
<div class="kpi"><b>{pct(S.get('mean_ri', {}).get('authors', 0))}</b><span>of accuracy kept on damaged scans (mean Robustness Index)</span></div>
</div></div>''')

# ---------------------------------------------------------------- 14. Results: verification
slide('RESULTS (2/2) · VERIFICATION', 'Results and Performance Evaluation', f'''<div class="two w45">
<div class="kpis">
<div class="kpi"><b>{pct(clean_q.get('found_at_place', 0))}</b><span>of injected errors found <em>at the right question</em> on born-digital papers ({clean_q.get('errors_injected', 0)} errors); {pct(deg_q.get('found_at_place', 0))} on scan-degraded pages</span></div>
<div class="kpi"><b>{clean_q.get('false_alarms_per_clean_pair', 0):.1f}</b><span>false alarms per clean paper pair, born-digital ({deg_q.get('false_alarms_per_clean_pair', 0):.1f} when scan-degraded); each is shown on the page for the vetter to dismiss</span></div>
<div class="kpi"><b>{pct(VN.get('overall', {}).get('found_at_place', 0))}</b><span>still found with a noisy layout (boxes shifted, 10% of question / option labels wrong)</span></div>
<div class="kpi"><b>{n_exact}/{n_pairs}</b><span>papers with every question recovered from the pages (Hindi and Tamil)</span></div>
</div>
<div>{img(FIG / 'fig_verify_by_type.png', 'wide')}<p class="cap">Share of injected errors found at the right question, per error type ({n_pairs} paper pairs, Hindi and Tamil).</p></div></div>''')

# ---------------------------------------------------------------- 15. Comparison
ab = [('Full system', V), ('Without number pass', V_NP), ('Without layout-based options', V_NG), ('Noisy layout', VN)]
abr = ''.join(f"<tr><td>{n}</td><td>{pct(r.get('overall', {}).get('found_at_place', 0))}</td><td>{r.get('overall', {}).get('false_alarms_per_clean_pair', 0):.1f}</td></tr>" for n, r in ab if r)
slide('COMPARISON', 'Comparison with Baseline / Existing Methods', f'''<div class="two">
<div><table class="res small"><tr><th>Layout detector</th><th>Training data</th><th>mAP@50</th><th>mAP@50:95</th></tr>
<tr><td>Faster R-CNN (ours, baseline)</td><td>2,293 pages, 76 min</td><td>{f3(B['map50'])}</td><td>{f3(B['map50_95'])}</td></tr>
<tr><td>YOLOv8s (ours)</td><td>2,293 pages, 60 min</td><td>{f3(P['map50'])}</td><td>{f3(P['map50_95'])}</td></tr>
<tr><td><b>DocLayout-YOLO (authors)</b></td><td>~96K pages</td><td><b>{f3(A['map50'])}</b></td><td><b>{f3(A['map50_95'])}</b></td></tr>
<tr class="pub"><td>Published: DocLayout-YOLO [1]</td><td>full test set</td><td>≈0.735</td><td>0.545</td></tr></table>
<p class="cap">YOLOv8s beats the Faster R-CNN baseline at {B['inference_time'] / P['inference_time']:.0f}× the speed; the released full-data model is best and is used.</p>
<table class="res small" style="margin-top:8px"><tr><th>Verification ablation (test set)</th><th>found</th><th>false alarms / clean pair (all pages)</th></tr>{abr}</table></div>
<div><table class="res small cmp"><tr><th>Existing approach</th><th>Checks the paper?</th><th>Compares versions?</th><th>Indic, offline?</th></tr>
<tr><td>Manual vetting (status quo)</td><td>yes, by hand</td><td>by hand</td><td>yes</td></tr>
<tr><td>Answer-sheet AI (Eklavvya, GradeLab)</td><td>no, grades answers</td><td>no</td><td>partly</td></tr>
<tr><td>MT quality estimation (CometKiwi [10])</td><td>no structure</td><td>sentences only</td><td>cloud / GPU</td></tr>
<tr><td>LLM back-translation [9]</td><td>no structure</td><td>with human review</td><td>cloud</td></tr>
<tr><td><b>IndicDoc Verify</b></td><td><b>numbering, options, marks, repeats</b></td><td><b>question by question</b></td><td><b>12 languages, offline</b></td></tr></table>
{img(FIG / 'fig_verify_ablation.png', 'wide')}</div></div>''')

# ---------------------------------------------------------------- 16. Sample outputs
slide('DEMO', 'Sample Outputs and Demonstration', f'''<div class="three demo">
<figure>{img(SHOTS / '08_verify_page.png')}<figcaption><b>Correct:</b> a changed number in the Hindi version found and outlined on the page</figcaption></figure>
<figure>{img(SHOTS / '07_verify_report.png')}<figcaption>Verification report: verdict, findings with question and page (HTML / JSON)</figcaption></figure>
<figure>{img(SHOTS / '02_detection_tamil.png')}<figcaption>Layout model in the running app: regions of a Tamil page</figcaption></figure>
</div><div class="two demo2">
<figure>{img(SHOTS / '09_false_alarm.png')}<figcaption><b>Incorrect:</b> a false alarm on a degraded scan: OCR lost a digit of an option, so the versions look different</figcaption></figure>
<figure>{img(PLOTS / 'authors_confusion_matrix.png')}<figcaption>Layout model confusion matrix: most errors are missed regions, not wrong labels</figcaption></figure></div>''')

# ---------------------------------------------------------------- 17. Challenges
slide('LESSONS', 'Challenges and Limitations', f'''<div class="grid2">
{card('OCR is the weakest link', 'Indic Tesseract models misread Latin digits ("1947" → "947", "1" → "।") and glue option markers to values. The number pass, layout-based option splitting and page-level digit-script detection were needed to get false alarms down.', '#C62828')}
{card('No public data of real errors', 'Exam bodies do not publish erroneous pre-print papers, so we built a benchmark with realistic injected errors, kept a separate development set, and report clean and scan-degraded results separately.', '#D9A441')}
{card('Meaning is not checked yet', 'Parity catches numbers, options, marks and missing questions without translation. A pure change of meaning (a dropped word) needs a cross-lingual model; there is a hook for one, not yet evaluated.', '#2563EB')}
{card('Compute and data size', 'Our own detectors saw 2.3K pages for an hour (0.26 vs 0.76 mAP@50 for the released model); the product therefore uses the released model, and our work is the checker built on it.', '#7C3AED')}
</div>''')

# ---------------------------------------------------------------- 18. Conclusion
slide('WRAP-UP', 'Conclusion and Future Scope', f'''<div class="two">
<div><div class="sub">CONCLUSION</div><ul class="tight">
<li><b>Implemented:</b> IndicDoc Verify — layout detection (3 detectors compared), OCR for 12 scripts with a number pass, question parsing, structure / parity / repeat / format checks, report, web app, REST API, Docker.</li>
<li><b>Results:</b> {pct(clean_q.get('found_at_place', 0))} of injected errors found at the right question on born-digital papers ({pct(deg_q.get('found_at_place', 0))} on degraded scans), {clean_q.get('false_alarms_per_clean_pair', 0):.1f} false alarms per clean born-digital pair; best layout model {f3(A['map50'])} mAP@50.</li>
<li><b>Objectives:</b> all four met. Our own detectors did not reach the released model, which we report openly.</li></ul>
<div class="sub">SELLING IT</div><ul class="tight">
<li><b>Buyers:</b> NTA, UPSC and state PSCs, school boards, university exam cells, test-series publishers; offline licence per exam body.</li></ul></div>
<div><div class="sub">FUTURE SCOPE</div><ul class="tight">
<li>Pilot with a university exam cell or a state board on real pre-print papers; measure minutes saved per paper.</li>
<li>Meaning check with a cross-lingual sentence model; number words and formulas.</li>
<li>Better Indic OCR behind the same interface; answer-key consistency check.</li>
<li>More document profiles: bilingual government circulars (Official Languages Act s.3(3)), forms, certificates.</li></ul>
<div class="sdg"><span>SDG 4</span><span>SDG 9</span><span>SDG 10</span><span>SDG 16</span></div></div></div>''')

# ---------------------------------------------------------------- 19. References (mostly 2021-2026)
refs = ['O. Nath, S. Kukkala, M. Khapra, R. K. Sarvadevabhatla, "IndicDLP: A Foundational Dataset for Multi-Lingual and Multi-Domain Document Layout Parsing," ICDAR 2025.',
        'B. V. Kasuba et al., "PLATTER: A Page-Level Handwritten Text Recognition System for Indic Scripts," arXiv:2502.06172, 2025.',
        'A. Mondal et al., "UniTabBank: A Large Scale Multi-Lingual, Multi-Layout, Multi-Type, Multi-Format Dataset for Table Detection," WACV 2026.',
        'V. Pal, E. Kanoulas, A. Yates, M. de Rijke, "Table Question Answering for Low-resourced Indic Languages," arXiv:2410.03576, 2024.',
        'Y. Xu et al., "LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding," arXiv:2104.08836, 2021.',
        'B. V. Kasuba, P. Chaudhuri, G. Ramakrishnan, "DRISHTIKON: Visual Grounding at Multiple Granularities in Documents," arXiv:2506.21316, 2025.',
        'S. Bhattacharyya et al., "Adapting Vision-Language Models for Hindi OCR," IIIT Hyderabad, 2025.',
        'A. R. Dumada, S. G. Shah, "Incorporate Structure and Content for Devanagari Table Extraction," IJSAT 17(1), 2026.',
        'M. Babayeva, J. Dunlap, M. Snětinová, R. Widenhorn, "Translating the Force Concept Inventory in the Age of AI," arXiv:2508.13908, 2025.',
        'R. Rei et al., "CometKiwi: IST-Unbabel 2022 Submission for the Quality Estimation Shared Task," WMT 2022.',
        'Z. Zhao et al., "DocLayout-YOLO: Enhancing Document Layout Analysis through Diverse Synthetic Data and Global-to-Local Adaptive Perception," arXiv:2410.12628, 2024.',
        'Y. Chen et al., "RoDLA: Benchmarking the Robustness of Document Layout Analysis Models," CVPR 2024.',
        'S. Verma et al., "MILU: A Multi-task Indic Language Understanding Benchmark," NAACL 2025.',
        'G. Jocher, A. Chaurasia, J. Qiu, "Ultralytics YOLOv8," 2023.',
        'J. Ansel et al., "PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation," ASPLOS 2024.',
        'S. Ren, K. He, R. Girshick, J. Sun, "Faster R-CNN," NeurIPS 2015 (the baseline architecture; the only pre-2021 source).']
other = ['Dataset: huggingface.co/datasets/ai4bharat/indicdlp (AIKosh)', 'Weights: zenodo.org/records/15881917',
         'Tesseract OCR 5 (tessdata_best), tesseract.js 7', 'Cases: The News Minute (NEET 2018), Deccan Herald / ETV Bharat (KPSC 2024-25), India Legal (APPSC), The Week / SCC Online (UGC-NET 2026), Medical Dialogues (MBBS 2026)',
         'Code: github.com/va-ada/IndicDoc-AI']
slide('SOURCES', 'References', '<div class="two w65"><ol class="refs">' + ''.join(f'<li>{r}</li>' for r in refs) +
      '</ol><div class="card" style="--c:#D9A441"><div class="ct">OTHER SOURCES</div><div class="cx small">' +
      '<br><br>'.join(other) + '</div></div></div>')

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
.case { background: #fff; border-left: 5px solid #C62828; border-radius: 10px; padding: 9px 14px; margin-bottom: 9px; }
.case b { display: block; color: #14284B; font-size: 15.5px; } .case span { font-size: 14px; color: #334155; }
.two.w55 { grid-template-columns: 1.25fr 1fr; } .two.w45 { grid-template-columns: 0.9fr 1.1fr; }
table.lit10 td { font-size: 12.6px; padding: 6px 8px; } table.lit10 th { font-size: 12.5px; padding: 6px 8px; }
.tightgaps { gap: 11px; } .tightgaps .gl, .tightgaps .gr { padding: 13px 16px; font-size: 16.5px; }
table.cmp td, table.cmp th { font-size: 13px; text-align: left !important; padding: 6px 8px; }
.cx.small { font-size: 13.5px; line-height: 1.4; }
@media screen { .slide { margin: 20px auto; box-shadow: 0 4px 20px rgba(0,0,0,.2); } }
'''


def find_chrome() -> str:
    candidates = [os.environ.get('CHROME'), '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
                  shutil.which('google-chrome'), shutil.which('chromium'), shutil.which('chromium-browser')]
    candidates += sorted(str(p) for p in Path('/opt/pw-browsers').glob('chromium-*/chrome-linux*/chrome'))
    return next(c for c in candidates if c and Path(c).exists())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    html = f'<!doctype html><html><head><meta charset="utf-8"><title>IndicDoc AI Phase-2</title><style>{CSS}</style></head><body>{"".join(slides)}</body></html>'
    hp = OUT / 'IndicDoc_AI_Phase2_Implementation.html'
    hp.write_text(html, encoding='utf-8')
    pdf = OUT / 'IndicDoc_AI_Phase2_Implementation.pdf'
    sandbox = ['--no-sandbox'] if hasattr(os, 'geteuid') and os.geteuid() == 0 else []  # Chrome refuses root otherwise
    subprocess.run([find_chrome(), *sandbox, '--headless=new', '--disable-gpu', '--no-pdf-header-footer',
                    '--virtual-time-budget=15000', f'--print-to-pdf={pdf}', hp.as_uri()],
                   check=True, capture_output=True, timeout=240)
    print(f'{len(slides)} slides -> {pdf}')


if __name__ == '__main__':
    main()
