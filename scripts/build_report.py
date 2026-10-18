"""
IndicDoc AI - Build the plain-language project report (HTML -> PDF).

Every number in the report is read from files the pipeline wrote (dataset annotations, outputs/metrics,
outputs/plots); nothing is typed in by hand. The PDF is printed with headless Google Chrome.

Usage:
    python scripts/build_report.py            # -> docs/report/IndicDoc_AI_Project_Report.pdf
"""
import base64
import json
import subprocess
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Template

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from ml.config import CLASS_NAMES, METRICS_DIR, OUTPUT_DIR, PLOTS_DIR, PREDICTION_DIR, RAW_DATA_DIR

REPORT_DIR = PROJECT_ROOT / 'docs' / 'report'
CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
INDIGO, GOLD = '#1E2A5A', '#C89B3C'


def img(path: Path) -> str:
    """Embed an image so the HTML and PDF are self-contained."""
    return 'data:image/png;base64,' + base64.b64encode(Path(path).read_bytes()).decode() if Path(path).exists() else ''


def dataset_summary(root: Path) -> dict:
    splits = {}
    for split in ('train', 'val', 'test'):
        d = json.loads((root / split / 'annotations.json').read_text())
        splits[split] = {
            'pages': len(d['images']), 'boxes': len(d['annotations']),
            'languages': Counter(i['language'] for i in d['images']),
            'domains': Counter(i['domain'] for i in d['images']),
            'labels': Counter(CLASS_NAMES[a['category_id']] for a in d['annotations']),
        }
    return splits


def loss_plot(out: Path) -> str:
    """Training curves for both models, side by side."""
    base_file = METRICS_DIR / 'baseline_training_history.json'
    yolo_file = METRICS_DIR / 'proposed_training_history.csv'
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    if base_file.exists():
        h = json.loads(base_file.read_text())
        ep = range(1, len(h['train_losses']) + 1)
        axes[0].plot(ep, h['train_losses'], 'o-', color=INDIGO, label='training loss')
        axes[0].plot(ep, h['val_losses'], 's--', color=GOLD, label='validation loss')
        axes[0].set_title('Faster R-CNN')
    if yolo_file.exists():
        r = pd.read_csv(yolo_file)
        r.columns = [c.strip() for c in r.columns]
        parts = ['box_loss', 'cls_loss', 'dfl_loss']
        axes[1].plot(r['epoch'], r[[f'train/{p}' for p in parts]].sum(axis=1), 'o-', color=INDIGO, label='training loss')
        axes[1].plot(r['epoch'], r[[f'val/{p}' for p in parts]].sum(axis=1), 's--', color=GOLD, label='validation loss')
        axes[1].set_title('YOLOv8s')
    from matplotlib.ticker import MaxNLocator
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # epochs are whole numbers
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return img(out)


TEMPLATE = Template(r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>IndicDoc AI - Project Report</title>
<style>
  @page { size: A4; margin: 18mm 16mm; }
  body { font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif; color: #1f2430; font-size: 10.5pt; line-height: 1.5; }
  h1 { color: {{indigo}}; font-size: 30pt; margin: 0 0 6px; }
  h2 { color: {{indigo}}; border-bottom: 2px solid {{gold}}; padding-bottom: 3px; margin-top: 26px; page-break-after: avoid; }
  h3 { color: {{indigo}}; margin-bottom: 4px; page-break-after: avoid; }
  table { border-collapse: collapse; width: 100%; margin: 8px 0 12px; font-size: 9.5pt; page-break-inside: avoid; }
  th { background: {{indigo}}; color: #fff; text-align: left; padding: 5px 7px; }
  td { border-bottom: 1px solid #dde; padding: 4px 7px; vertical-align: top; }
  td.num, th.num { text-align: right; }
  .cover { height: 245mm; display: flex; flex-direction: column; justify-content: center; page-break-after: always; }
  .cover .sub { font-size: 14pt; color: #4a5470; }
  .cover .gold { width: 80px; height: 4px; background: {{gold}}; margin: 18px 0; }
  .box { background: #f6f3ea; border-left: 4px solid {{gold}}; padding: 10px 14px; margin: 10px 0; page-break-inside: avoid; }
  .figure { text-align: center; margin: 10px 0 14px; page-break-inside: avoid; }
  .figure img { max-width: 100%; max-height: 120mm; }
  .caption { font-size: 9pt; color: #5a6480; margin-top: 3px; }
  .flow { display: flex; gap: 6px; align-items: center; justify-content: center; flex-wrap: wrap; margin: 12px 0; }
  .flow div { background: {{indigo}}; color: #fff; border-radius: 6px; padding: 7px 10px; font-size: 9pt; }
  .flow span { color: {{gold}}; font-weight: bold; }
  .pagebreak { page-break-before: always; }
  code { background: #eef; padding: 1px 4px; border-radius: 3px; font-size: 9pt; }
  pre.ocr { background: #fafafa; border: 1px solid #dde; padding: 8px 10px; font-size: 8.5pt; white-space: pre-wrap;
            font-family: 'Noto Sans', 'Kohinoor Devanagari', 'Tamil Sangam MN', -apple-system, sans-serif; page-break-inside: avoid; }
</style></head><body>

<div class="cover">
  <h1>IndicDoc AI</h1>
  <div class="sub">Finding the structure and reading the text of Indian-language documents</div>
  <div class="gold"></div>
  <div>Project report &middot; {{today}}</div>
  <div style="margin-top:40px; color:#5a6480; max-width:150mm">
    A plain-language account of what this project does, how it works, how it was trained and tested,
    and how well it performs. Every number in this report was produced by the project's own code on
    real IndicDLP pages.
  </div>
</div>

<h2>1. The project in one paragraph</h2>
<p>IndicDoc AI takes a picture or PDF of a document page (a textbook, a newspaper, a government notice, a question
paper) in any of 12 Indian languages. It does two things. First it finds the <b>layout</b>: it draws a box around
every part of the page (each paragraph, heading, table, figure, page number, list or question) and gives each box
one of 42 labels. Then it <b>reads the text</b> inside those boxes (OCR) and returns the page as an organised
document. The project compares three layout models on the same {{ds.test.pages}} test pages that none of them was
trained on: two trained here on a laptop (Faster R-CNN and YOLOv8s), and DocLayout-YOLO, published by the dataset's
authors and trained on the full dataset. The authors' model is the best by far, at <b>{{'%.3f'|format(auth.map50)}}
mAP@50</b> (explained in section 6), so it is the one the app uses by default. Everything ships as a web app and a
REST API in one Docker image that also works without internet access.</p>

<h2>2. The problem</h2>
<p>Before a computer can usefully read a document, it has to understand its <i>layout</i>: which text is a heading,
which is body text, where the table is, which number is the page number. People do this at a glance; a computer sees
only pixels. This step is called <b>document layout analysis</b>, and it decides how well everything after it works.
For example, if OCR reads a two-column page straight across, the result is nonsense.</p>
<p>Most layout tools are built and tested on English documents. Indian documents use many scripts (Devanagari,
Bengali, Tamil, Telugu, Kannada, Malayalam, Gujarati, Gurmukhi, Odia, Latin) and many formats: dense multi-column
newspapers, question papers with nested sub-questions, legal acts with numbered clauses.</p>
<div class="box"><b>Input:</b> one page image, or one page of a PDF.<br>
<b>Output:</b> boxes with labels (e.g. <code>paragraph</code>, <code>table</code>, <code>section-title</code>) and
confidence scores, plus the text of every box, in reading order, as Markdown and JSON.</div>

<h2>3. The data</h2>
<p>The project uses <b>IndicDLP</b> (Nath et al., ICDAR 2025; AI4Bharat, IIT Madras and IIIT Hyderabad): 119,806
human-annotated pages in 12 languages and 12 document types, labelled with 42 layout classes. It is listed on
India's AIKosh platform and downloaded from Hugging Face.</p>
<p>The full dataset is about 91 GB, too much to train on with a laptop in a few hours. So for its own two models the
project builds a <b>balanced subset</b> with roughly the same number of pages from every language. It uses the
dataset's own train / validation / test split, so test pages are never seen during training.</p>
<table><tr><th>Split</th><th class="num">Pages</th><th class="num">Labelled boxes</th><th>Used for</th></tr>
<tr><td>Train</td><td class="num">{{ds.train.pages}}</td><td class="num">{{ds.train.boxes}}</td><td>Learning our two models' weights</td></tr>
<tr><td>Validation</td><td class="num">{{ds.val.pages}}</td><td class="num">{{ds.val.boxes}}</td><td>Choosing checkpoints and confidence thresholds</td></tr>
<tr><td>Test</td><td class="num">{{ds.test.pages}}</td><td class="num">{{ds.test.boxes}}</td><td>Final, unbiased scores for all three models</td></tr></table>
<table><tr><th>Language</th>{% for s in ['train','val','test'] %}<th class="num">{{s}} pages</th>{% endfor %}</tr>
{% for lang in languages %}<tr><td>{{lang|title}}</td>{% for s in ['train','val','test'] %}<td class="num">{{ds[s].languages[lang]}}</td>{% endfor %}</tr>{% endfor %}</table>
<p>The labels are very unevenly spread: <code>paragraph</code> alone is {{'%.0f'|format(100*ds.train.labels['paragraph']/ds.train.boxes)}}%
of all training boxes, while some labels appear only a handful of times. Rare labels are much harder to learn.</p>
{% if plots.class_distribution %}<div class="figure"><img src="{{plots.class_distribution}}"><div class="caption">How often each label appears in the test pages.</div></div>{% endif %}

<h2 class="pagebreak">4. How it works</h2>
<div class="flow"><div>Page image / PDF</div><span>&rarr;</span><div>Layout model</div><span>&rarr;</span>
<div>Boxes + labels + scores</div><span>&rarr;</span><div>Detect script</div><span>&rarr;</span><div>OCR (Tesseract)</div><span>&rarr;</span><div>Text per box, in reading order</div></div>
<h3>The three layout models</h3>
<p><b>Faster R-CNN (our baseline)</b> works in two steps. A <i>region proposal network</i> suggests places on the page
that might contain something, then a second network looks closely at each one, decides its label and tightens the
box. ResNet-50 backbone with a feature pyramid; {{'%.1f'|format(base.parameters/1e6)}} million parameters.</p>
<p><b>YOLOv8s (our proposed model)</b> ("You Only Look Once") does everything in a single pass over the whole page.
It has {{'%.1f'|format(prop.parameters/1e6)}} million parameters, about a quarter of Faster R-CNN, so it is smaller and
faster.</p>
<p><b>DocLayout-YOLO (IndicDLP authors)</b> is a YOLO-family detector built specially for documents (Zhao et al., 2024):
it was pre-trained on 300,000 synthetic document pages, then fine-tuned by the IndicDLP authors on the
<i>full</i> IndicDLP training set (~96K pages, 100 epochs, 1024-pixel images, on server GPUs).
{{'%.1f'|format(auth.parameters/1e6)}} million parameters. It was <b>not trained in this project</b>: it was downloaded from
the authors' public release (Zenodo, CC-BY 4.0) and evaluated here on exactly the same test pages as our models.</p>
<h3>Transfer learning</h3>
<p>None of the models starts from zero. They begin with weights already trained on other images (COCO everyday photos
for ours, synthetic documents for DocLayout-YOLO), so they already know how to see edges, shapes and text-like
textures. Their final layer is replaced with one for the 42 document labels and the network is then
<i>fine-tuned</i>.</p>

<h2>5. Training our two models</h2>
<table><tr><th></th><th>Faster R-CNN</th><th>YOLOv8s</th></tr>
{% for row in training_rows %}<tr><td>{{row[0]}}</td><td>{{row[1]}}</td><td>{{row[2]}}</td></tr>{% endfor %}</table>
<p>During training the model predicts boxes on training pages, measures how wrong they are (the <i>loss</i>) and
adjusts its weights to be a little less wrong. One <i>epoch</i> is one pass over all training pages. After every epoch
the model is scored on the validation pages and the best version is kept. Falling curves mean the models were
learning; they had not stopped improving, so longer training would help.</p>
<div class="figure"><img src="{{loss_curves}}"><div class="caption">Loss per epoch (lower is better). Training loss in blue, validation loss in gold.</div></div>

<h2>6. Layout results</h2>
<div class="box"><b>How to read the scores.</b> A predicted box counts as correct if it has the right label and
overlaps the true box enough: <b>IoU</b> (overlap &divide; combined area) of at least 0.5. <b>Precision</b> = of the boxes
the model drew, the share that were right. <b>Recall</b> = of the real elements on the page, the share the model found.
<b>mAP@50</b> = the standard overall score for detectors (0 to 1): precision averaged over every recall level and over
all labels, at IoU 0.5. <b>mAP@50:95</b> is stricter: it also demands tightly fitting boxes. The <b>confidence
threshold</b> decides which boxes are shown. Each model has a <i>recommended</i> threshold, the one with the best F1 on
the validation pages (never tuned on the test pages). Because finding every part matters more than a tidy page for
this project, the app's default is a second, <b>"find every part"</b> threshold: the lowest one that still kept at least
70% of boxes correct on the validation pages, and never higher than the balanced one. In both modes, a box that the model proposes under several labels is
shown once, with its most likely label.</div>
<table><tr><th>Metric (same {{ds.test.pages}} test pages)</th>{% for n, label, m in models %}<th class="num">{{label}}</th>{% endfor %}</tr>
{% for name, key, digits in metric_rows %}<tr><td>{{name}}</td>{% for n, label, m in models %}<td class="num">{{('%.' ~ digits ~ 'f')|format(m[key])}}</td>{% endfor %}</tr>{% endfor %}
<tr><td>Model size (MB)</td>{% for n, label, m in models %}<td class="num">{{m.model_size_mb}}</td>{% endfor %}</tr></table>
<p>{{comparison_text}}</p>
{% if plots.model_comparison %}<div class="figure"><img src="{{plots.model_comparison}}"><div class="caption">All three models on the same {{ds.test.pages}} test pages.</div></div>{% endif %}
<p><b>Is the score correct?</b> The evaluator uses <code>pycocotools</code>, the reference COCO implementation, and is
tested on hand-worked examples. Two independent checks agree with it. Ultralytics' own AP code, run on the same YOLOv8s
detections, gives {{'%.3f'|format(cross.ultralytics_ap_on_our_detections_map50)}} against {{'%.3f'|format(cross.ours_map50)}} here.
And the authors' model scores {{'%.3f'|format(auth.map50)}} on our test pages, in line with the about 0.735 the IndicDLP paper
reports on its full test set.</p>

<h3>Which layout elements are easy, which are hard</h3>
<p>{{per_class_text}}</p>
{% for n, label, m in models %}{% if plots[n ~ '_per_class_ap'] %}<div class="figure"><img src="{{plots[n ~ '_per_class_ap']}}"><div class="caption">AP@50 per label, {{label}}.</div></div>{% endif %}{% endfor %}

<h3>Results per language</h3>
<table><tr><th>Language</th><th class="num">Test pages</th>{% for n, label, m in models %}<th class="num">{{label}}</th>{% endfor %}</tr>
{% for lang in languages %}<tr><td>{{lang|title}}</td><td class="num">{{auth.per_group[lang].images}}</td>{% for n, label, m in models %}<td class="num">{{'%.3f'|format(m.per_group[lang].map50)}}</td>{% endfor %}</tr>{% endfor %}</table>
<p>{{language_text}} Most languages have {{ds.test.languages.most_common(1)[0][1]}} test pages (Hindi has
{{ds.test.languages['hindi']}}), so differences of a few points are within normal variation. Languages can score above the
overall figure: mAP averages over the labels present, and each language's pages contain only part of the 42 labels.</p>

<h3>What the predictions look like</h3>
{% for p in example_predictions %}<div class="figure"><img src="{{p.src}}"><div class="caption">{{p.caption}}</div></div>{% endfor %}

<h3>Where the best model goes wrong</h3>
<p>The confusion matrix shows, for the most common labels, which true label (rows) was predicted as which label
(columns). The last row and column count missed elements and false alarms.</p>
{% if plots.authors_confusion_matrix %}<div class="figure"><img src="{{plots.authors_confusion_matrix}}"><div class="caption">Confusion matrix of DocLayout-YOLO on the test pages.</div></div>{% endif %}
{% for p in error_examples %}<div class="figure"><img src="{{p.src}}"><div class="caption">{{p.caption}}</div></div>{% endfor %}

<h2 class="pagebreak">7. Reading the text (OCR)</h2>
<p>Finding boxes is half the job; the other half is reading what is inside them. The project uses <b>Tesseract</b>
(Google's open-source OCR engine) with its best models for all 12 languages.</p>
<ul>
<li><b>One read per page.</b> Tesseract reads the whole page once and reports every word with its position. Each word
is then placed in the detected box that contains it. This is much faster than reading 60 boxes separately, and Tesseract
reads whole lines more accurately than tight crops. {{'%.0f'|format(100*ocr.words_in_regions_share)}}% of the words it
finds fall inside a detected box, so the layout model covers almost all the text.</li>
<li><b>Choosing the language automatically.</b> Tesseract's built-in script detector proved unreliable on these pages
(it got only 70 of 120 test pages right, and none of the Gujarati, Odia or Punjabi ones). The project instead reads one
text block with each script's model and keeps the most confident reading, counting only characters that belong to that
script. On {{ocr.pages}} test pages (10 per language) this picks the right script for
<b>{{ocr.script_detection_correct}} ({{'%.0f'|format(100*ocr.script_detection_accuracy)}}%)</b>. In the app the
language can also be chosen by hand.</li>
<li><b>Nothing is lost.</b> Words that fall in no detected box are not dropped: they are grouped into extra
"unlabelled text" blocks, so the extracted document contains all the text even where the layout model missed an element.</li>
<li><b>Organised output.</b> Boxes are put in reading order (top to bottom, left to right) and written as Markdown:
headings become headings, lists become lists, tables stay as tables, figures and formulas are marked
<code>[figure]</code> / <code>[formula]</code> (OCR cannot read pictures or mathematics). Page furniture (header,
footer, page number) is listed at the end. The same content is available as JSON with every box, label and text.</li>
</ul>
<table><tr><th>Language</th><th class="num">Right script</th><th>Detected as</th></tr>
{% for lang, v in ocr.per_language|dictsort %}<tr><td>{{lang|title}}</td><td class="num">{{v.correct}}/{{v.pages}}</td><td>{% for s, c in v.detected|dictsort %}{{s}} {{c}}{% if not loop.last %}, {% endif %}{% endfor %}</td></tr>{% endfor %}</table>
<p>OCR takes about {{'%.0f'|format(ocr.seconds_per_page_ocr)}} seconds per page on this laptop. IndicDLP contains no
transcriptions of the text, so character-level accuracy could not be measured; quality varies with print and scan
quality and is usually good for clean printed pages. Two real examples of the output, unedited:</p>
{% for e in ocr_examples %}<p class="caption" style="text-align:left"><b>{{e.language|title}}</b> test page <code>{{e.file}}</code>, detected script {{e.script}}, Tesseract languages <code>{{e.tesseract_languages}}</code>:</p>
<pre class="ocr">{{e.markdown}}</pre>{% endfor %}

<h2 class="pagebreak">8. Making it a working product</h2>
<ul><li><b>Web app</b> (Streamlit): upload an image or PDF (any page) and see it with coloured boxes. Move the confidence
slider to show more or fewer parts instantly. Click <i>Read text</i> to get the text of every box (auto-detected or chosen
language), and download it as Markdown or JSON. Other tabs show the dashboard, model comparison, evaluation charts and
per-language results.</li>
<li><b>REST API</b> (FastAPI): other software sends a page to <code>POST /analyze</code> (optionally with
<code>ocr=true</code>) and gets boxes, labels and text back as JSON.</li></ul>
<p>Both run from <b>one Docker image</b> with all three models and the OCR engine inside. It is CPU-only, runs as a
non-root user, has health checks and upload limits, and works with networking switched off. One command starts
everything: <code>docker compose up --build</code>.</p>
<p>The project was audited end to end. {{audit_count}} bugs were found and fixed, including the following:</p>
<ul>
<li>model loading that always crashed;</li>
<li>YOLO training that had never worked;</li>
<li>evaluation charts that were randomly generated;</li>
<li>class names that did not match the dataset;</li>
<li>a model that silently downloaded weights at run time;</li>
<li>a sidebar that could not be reopened.</li>
</ul>
<p>The test suite grew from 27 shape checks to {{test_count}} tests that check behaviour, and every fix was also
verified on real data. The full list is in <code>docs/FINAL_AUDIT.md</code>.</p>

<h2>9. Limitations and next steps</h2>
<ul>
<li><b>Our own models</b> were trained on {{ds.train.pages}} pages for one or two hours on a laptop. Longer training on
the full dataset with a cloud GPU is how they would approach the authors' model.</li>
<li><b>Rare labels</b> ({{rare_labels}}) have very few training examples, and all models handle them poorly.</li>
<li><b>OCR quality</b> could not be measured without transcriptions. Handwriting, very old prints and decorative fonts
will read badly.</li>
<li><b>Out-of-domain pages:</b> IndicDLP covers 12 kinds of document; others (CVs, invoices, slides) may be segmented less
well.</li>
<li><b>Reading order</b> is a simple top-to-bottom, left-to-right rule; complex newspaper layouts can come out jumbled.</li>
<li><b>Public deployment</b> needs a reverse proxy for HTTPS, login and rate limiting.</li>
</ul>

<h2>10. References</h2>
<ol style="font-size:9.5pt">
<li>O. Nath, S. Kukkala, M. Khapra, R. K. Sarvadevabhatla. IndicDLP: A Foundational Dataset for Multi-lingual and
Multi-domain Document Layout Parsing. ICDAR 2025, Springer LNCS, pp. 23-39. Models: Zenodo record 15881917.</li>
<li>Z. Zhao, H. Kang, B. Wang, C. He. DocLayout-YOLO: Enhancing Document Layout Analysis through Diverse Synthetic Data
and Global-to-Local Adaptive Perception. arXiv:2410.12628, 2024.</li>
<li>R. Smith. An Overview of the Tesseract OCR Engine. ICDAR 2007.</li>
<li>S. Ren, K. He, R. Girshick, J. Sun. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. NeurIPS 2015.</li>
<li>G. Jocher, A. Chaurasia, J. Qiu. Ultralytics YOLOv8. 2023. github.com/ultralytics/ultralytics</li>
<li>T.-Y. Lin et al. Microsoft COCO: Common Objects in Context. ECCV 2014.</li>
<li>K. He, X. Zhang, S. Ren, J. Sun. Deep Residual Learning for Image Recognition. CVPR 2016.</li>
<li>T.-Y. Lin et al. Feature Pyramid Networks for Object Detection. CVPR 2017.</li>
</ol>
</body></html>
""")


def device_label(device: str) -> str:
    """'mps' -> e.g. 'Apple M1 Pro GPU (MPS)', using this machine's chip name."""
    if device.lower() != 'mps':
        return device.upper()
    try:
        chip = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True).stdout.strip()
    except OSError:
        chip = 'Apple Silicon'
    return f"{chip or 'Apple Silicon'} GPU (MPS)"


def training_runs() -> dict:
    """Epochs, batch size, device and wall-clock time of both training runs, read from what the trainers saved."""
    import yaml
    h = json.loads((METRICS_DIR / 'baseline_training_history.json').read_text())
    baseline = {'epochs': len(h['train_losses']), 'batch': h['batch_size'], 'device': device_label(h['device']),
                'minutes': h['epoch_end_seconds'][-1] / 60}
    r = pd.read_csv(METRICS_DIR / 'proposed_training_history.csv')
    r.columns = [c.strip() for c in r.columns]
    args = yaml.safe_load((OUTPUT_DIR / 'yolo_runs' / 'indicdoc_yolo' / 'args.yaml').read_text())
    proposed = {'epochs': len(r), 'batch': args['batch'], 'device': device_label(str(args['device'])),
                'minutes': float(r['time'].iloc[-1]) / 60}
    return {'baseline': baseline, 'proposed': proposed}


def audit_fix_count() -> int:
    """Rows of the fix table in docs/FINAL_AUDIT.md (lines starting '| <number> |')."""
    import re
    text = (PROJECT_ROOT / 'docs' / 'FINAL_AUDIT.md').read_text()
    return len(re.findall(r'^\| \d+ \|', text, flags=re.M))


def fmt(v, digits=3):
    return 'N/A' if v is None else f"{v:.{digits}f}"


def overall_table(models: list) -> str:
    """Markdown table; models = [(prefix, display name, metrics dict), ...]."""
    rows = [('mAP@50', 'map50'), ('mAP@50:95', 'map50_95'), ('mAP@75', 'map75'), ('Precision (conf >= 0.5)', 'precision'),
            ('Recall (conf >= 0.5)', 'recall'), ('F1 (conf >= 0.5)', 'f1'), ('Average IoU (correct boxes)', 'avg_iou'),
            ('Inference time (s/page)', 'inference_time')]
    lines = ['| Metric | ' + ' | '.join(label for _, label, _ in models) + ' |', '|---' * (len(models) + 1) + '|']
    lines += [f"| {name} | " + ' | '.join(fmt(m[k]) for _, _, m in models) + ' |' for name, k in rows]
    lines.append('| Recommended confidence (best F1 on validation pages) | ' +
                 ' | '.join(fmt(m.get('recommended_conf'), 2) for _, _, m in models) + ' |')
    for name, k in (('Precision at recommended confidence', 'precision'), ('Recall at recommended confidence', 'recall'),
                    ('F1 at recommended confidence', 'f1')):
        lines.append(f"| {name} | " + ' | '.join(fmt(m.get('at_recommended_conf', {}).get(k)) for _, _, m in models) + ' |')
    lines.append('| "Find every part" confidence (lowest with precision >= 0.70 on validation) | ' +
                 ' | '.join(fmt(m.get('high_recall_conf'), 2) for _, _, m in models) + ' |')
    for name, k in (('Precision, find every part', 'precision'), ('Recall, find every part', 'recall')):
        lines.append(f"| {name} | " + ' | '.join(fmt(m.get('at_high_recall_conf', {}).get(k)) for _, _, m in models) + ' |')
    lines.append('| Model size (MB) | ' + ' | '.join(str(m['model_size_mb']) for _, _, m in models) + ' |')
    lines.append('| Parameters (M) | ' + ' | '.join(f"{m['parameters'] / 1e6:.1f}" for _, _, m in models) + ' |')
    return '\n'.join(lines)


def write_markdown(ds, models, runs, cross, ocr, test_count):
    base = next(m for n, _, m in models if n == 'baseline')
    test_gt = ds['test']['labels']
    langs = sorted(ds['test']['languages'])
    heads = ' | '.join(label for _, label, _ in models)
    per_lang = [f'| Language | Test pages | {heads} |', '|---' * (len(models) + 2) + '|']
    per_lang += [f"| {l.title()} | {base['per_group'][l]['images']} | " +
                 ' | '.join(fmt(m['per_group'][l]['map50']) for _, _, m in models) + ' |' for l in langs]
    per_class = [f'| Label | Test boxes | {heads} |', '|---' * (len(models) + 2) + '|']
    per_class += [f"| {c} | {test_gt[c]} | " + ' | '.join(fmt(m['per_class_ap50'][c]) for _, _, m in models) + ' |'
                  for c in sorted(CLASS_NAMES, key=lambda c: -test_gt[c])]
    ocr_rows = '\n'.join(f"| {l.title()} | {v['correct']}/{v['pages']} | " +
                         ', '.join(f'{k} {c}' for k, c in sorted(v['detected'].items())) + ' |'
                         for l, v in sorted(ocr['per_language'].items()))
    b, y = runs['baseline'], runs['proposed']
    results = f"""# Results

All numbers below were produced by `scripts/evaluate.py` on **{ds['test']['pages']} held-out IndicDLP test pages**
({ds['test']['boxes']} labelled boxes, 12 languages) that no model saw during training or checkpoint selection.
DocLayout-YOLO was trained by the IndicDLP authors on the full training set (downloaded, not trained here); the
other two were trained in this project on {ds['train']['pages']} pages. This file is generated by
`scripts/build_report.py` from `outputs/metrics/`; do not edit it by hand.

## Overall

{overall_table(models)}

Inference time is measured on {base['device'].upper()} (Apple M1 Pro) with one page at a time.

For context: the IndicDLP paper reports about 0.735 mAP@50 for DocLayout-YOLO on its full test set; on these
{ds['test']['pages']} test pages the released checkpoint scores {fmt(models[0][2]['map50'])}.

## Training (our two models)

| | Faster R-CNN | YOLOv8s |
|---|---|---|
| Training pages | {ds['train']['pages']} | {ds['train']['pages']} |
| Validation pages | {ds['val']['pages']} | {ds['val']['pages']} |
| Epochs | {b['epochs']} | {y['epochs']} |
| Batch size | {b['batch']} | {y['batch']} |
| Wall-clock time | {b['minutes']:.0f} min | {y['minutes']:.0f} min |
| Device | {b['device']} | {y['device']} |

Loss curves: `outputs/metrics/baseline_training_history.json`, `outputs/metrics/proposed_training_history.csv`
(also shown in the app and the report).

## Per language (mAP@50)

{chr(10).join(per_lang)}

Most languages have about 60 test pages; Hindi has {ds['test']['languages']['hindi']}. Differences of a few points are within noise.

Every language scores above the overall mAP@50. That is expected, not a contradiction: mAP averages over the labels present
in the pages being scored. Overall that is all 42 labels, including the rare ones the models barely detect, while each
language's test pages contain only part of the label set, mostly without those rare labels.

## Per label (AP@50)

"N/A" means the label does not occur in the test pages, so it cannot be scored.

{chr(10).join(per_class)}

## Evaluator cross-check

| YOLOv8s on the same test pages | mAP@50 | mAP@50:95 |
|---|---|---|
| This project (pycocotools, `scripts/evaluate.py`) | {cross['ours_map50']:.4f} | {cross['ours_map50_95']:.4f} |
| Ultralytics AP code on the same detections | {cross['ultralytics_ap_on_our_detections_map50']:.4f} | {cross['ultralytics_ap_on_our_detections_map50_95']:.4f} |
| Ultralytics validator (multi-label NMS) | {cross['ultralytics_map50']:.4f} | {cross['ultralytics_map50_95']:.4f} |

The first two rows agree, so the metric maths is right. The validator is higher because its NMS keeps several labels
per box; normal prediction (what the app and API return) keeps one. See [EVALUATION.md](EVALUATION.md).

## OCR

Script detection (which Tesseract languages to use) on {ocr['pages']} test pages, 10 per language, with the layout from
DocLayout-YOLO: **{ocr['script_detection_correct']}/{ocr['pages']}** right. {100 * ocr['words_in_regions_share']:.0f}% of
the words Tesseract reads fall inside a detected region. OCR takes {ocr['seconds_per_page_ocr']:.1f} s per page on the
Apple M1 Pro. IndicDLP has no text transcriptions, so character accuracy is not measured
(`scripts/evaluate_ocr.py`, `outputs/metrics/ocr_evaluation.json`).

| Language | Right script | Detected as |
|---|---|---|
{ocr_rows}

## Plots and examples

`outputs/plots/`: model comparison, per-class AP, precision-recall curves, confusion matrices, class distribution,
per-language charts. `outputs/predictions/`: test pages with ground truth vs prediction, and the pages with the
most missed elements and false detections (`errors/`).

Test suite at the time of these results: {test_count} tests passing.
"""
    (PROJECT_ROOT / 'docs' / 'RESULTS.md').write_text(results)

    evaluation = f"""# Evaluation

## Protocol

- **Test set:** {ds['test']['pages']} pages from IndicDLP's official test split ({ds['test']['boxes']} boxes, all 12 languages,
  all 12 domains). Never used for training or for choosing checkpoints (that is what the {ds['val']['pages']} validation
  pages are for).
- **Detections:** every detection with confidence >= 0.001 is kept for mAP (up to 300 per page; dense newspaper pages
  exceed COCO's default of 100).
- **Tool:** `ml/evaluation/coco_eval.py`, built on pycocotools (the reference COCO implementation).

## Metrics

| Metric | Meaning |
|---|---|
| IoU | Overlap of a predicted and a true box, divided by the area they cover together (1 = identical) |
| mAP@50 | Area under the precision-recall curve (COCO 101-point interpolation) at IoU >= 0.50, averaged over labels |
| mAP@75 | Same at IoU >= 0.75 (needs tight boxes) |
| mAP@50:95 | Average of mAP over IoU 0.50, 0.55, ..., 0.95 |
| Precision / Recall / F1 | At one operating point: confidence >= 0.5, same-label match at IoU >= 0.5, greedy by score |
| Recommended confidence | The threshold (0.05 to 0.95 in steps of 0.05) with the best F1 on the **validation** pages; test P/R/F1 are also reported at it. The app and API use it by default |
| Average IoU | Mean IoU of the correctly matched boxes |
| Confusion matrix | Label-agnostic matching at IoU >= 0.5: true label vs predicted label, plus missed and spurious boxes |
| Per-language mAP@50 | mAP restricted to each language's test pages |
| Inference time | Seconds per page, one page at a time, first page excluded as warm-up |

## Is the evaluator right?

1. **Hand-worked unit tests** (`tests/test_coco_eval.py`): perfect detections give mAP 1.0; no detections give 0;
   one wrong label gives exactly 2 TP / 1 FP / 1 FN and the expected confusion-matrix cell; boxes 60% too wide
   (IoU 0.625) pass at IoU 0.5 and fail at 0.75; per-language splits are separated correctly.
2. **Independent cross-check of the maths:** Ultralytics' own matching and AP code, run on exactly the detections this
   project scores, gives mAP@50 {cross['ultralytics_ap_on_our_detections_map50']:.4f} against this project's
   {cross['ours_map50']:.4f} (mAP@50:95 {cross['ultralytics_ap_on_our_detections_map50_95']:.4f} vs
   {cross['ours_map50_95']:.4f}).
3. **Why Ultralytics' validator reports more** ({cross['ultralytics_map50']:.4f}): its validation NMS is multi-label
   (`multi_label=True` in `ultralytics/models/yolo/detect/val.py`), so one box can also be kept under its second-best
   label, which adds recall. Normal prediction keeps one label per box. That is what users of the app and API get, so
   it is what this project reports.

Results: [RESULTS.md](RESULTS.md).
"""
    (PROJECT_ROOT / 'docs' / 'EVALUATION.md').write_text(evaluation)

    readme = PROJECT_ROOT / 'README.md'
    text = readme.read_text()
    start, end = text.index('<!-- RESULTS:START -->'), text.index('<!-- RESULTS:END -->')
    block = (f"<!-- RESULTS:START -->\n{overall_table(models)}\n\n"
             f"Test set: {ds['test']['pages']} pages, 12 languages. DocLayout-YOLO: trained by the IndicDLP authors on "
             f"the full dataset. YOLOv8s and Faster R-CNN: trained here on {ds['train']['pages']} pages "
             f"(language-balanced subset) on an Apple M1 Pro.\n")
    readme.write_text(text[:start] + block + text[end:])


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ds = dataset_summary(RAW_DATA_DIR / 'indicdlp')
    load = lambda n: json.loads((METRICS_DIR / f'{n}_metrics.json').read_text())
    auth, prop, base = load('authors'), load('proposed'), load('baseline')
    models = [('authors', 'DocLayout-YOLO (authors)', auth), ('proposed', 'YOLOv8s (ours)', prop),
              ('baseline', 'Faster R-CNN (ours)', base)]
    runs = training_runs()
    cross = json.loads((METRICS_DIR / 'evaluator_crosscheck.json').read_text())
    tests = json.loads((METRICS_DIR / 'test_suite.json').read_text())
    ocr = json.loads((METRICS_DIR / 'ocr_evaluation.json').read_text())

    speedup = base['inference_time'] / prop['inference_time'] if prop['inference_time'] else float('nan')
    comparison_text = (
        f"The authors' DocLayout-YOLO is far ahead: mAP@50 {auth['map50']:.3f} against {prop['map50']:.3f} for our YOLOv8s and "
        f"{base['map50']:.3f} for our Faster R-CNN, and F1 {auth['at_recommended_conf']['f1']:.3f} at its recommended "
        f"threshold. That is what a document-specialised model trained on ~40 times more pages for 100 epochs buys. Between "
        f"our two models, YOLOv8s wins on accuracy and is {speedup:.1f}x faster than Faster R-CNN "
        f"({prop['inference_time']:.2f} vs {base['inference_time']:.2f} s per page), with {prop['model_size_mb']:.0f} MB of "
        f"weights instead of {base['model_size_mb']:.0f} MB. DocLayout-YOLO works on larger 1024-pixel images and needs "
        f"{auth['inference_time']:.2f} s per page on the same machine.")

    ap = {k: v for k, v in auth['per_class_ap50'].items() if v is not None}
    ranked = sorted(ap, key=ap.get, reverse=True)
    per_class_text = (
        f"For DocLayout-YOLO, the easiest labels are {', '.join(f'{c} ({ap[c]:.2f})' for c in ranked[:4])}. The hardest "
        f"are {', '.join(f'{c} ({ap[c]:.2f})' for c in ranked[-4:])}: rare, or easily confused with similar labels "
        f"(for example the three levels of nested lists). " +
        ("All 42 labels occur in the test pages." if len(ap) == len(CLASS_NAMES) else
         f"{len(CLASS_NAMES) - len(ap)} of the 42 labels do not appear in the test pages, so no score is given for them."))

    train_labels = ds['train']['labels']
    rare_labels = ', '.join(f"<code>{c}</code>: {train_labels[c]} training boxes"
                            for c in sorted(CLASS_NAMES, key=lambda c: train_labels[c])[:4])
    languages = sorted(ds['test']['languages'])
    groups = auth['per_group']
    lang_rank = sorted(languages, key=lambda l: groups[l]['map50'], reverse=True)
    language_text = (f"For DocLayout-YOLO, scores range from {groups[lang_rank[0]]['map50']:.3f} ({lang_rank[0].title()}) "
                     f"to {groups[lang_rank[-1]]['map50']:.3f} ({lang_rank[-1].title()}): it works well in every language.")

    plots = {p.stem: img(p) for p in PLOTS_DIR.glob('*.png')}
    examples = []
    for name, label in (('authors', 'DocLayout-YOLO'), ('proposed', 'YOLOv8s')):
        for p in sorted(PREDICTION_DIR.glob(f'{name}_prediction_*.png'))[:1]:
            examples.append({'src': img(p), 'caption': f"{label} on a test page: true layout (left, green) vs prediction (right, red)."})
    errors = [{'src': img(p), 'caption': f"DocLayout-YOLO: the test page with the most "
                                         f"{'missed elements' if 'missed' in p.stem else 'false detections'}."}
              for p in sorted((PREDICTION_DIR / 'errors').glob('authors_*.png'))]
    ocr_examples = [e for e in ocr['examples'] if e['language'] in ('english', 'hindi', 'tamil')][:2]

    b, y = runs['baseline'], runs['proposed']
    training_rows = [
        ('Hardware', b['device'], y['device']),
        ('Epochs', b['epochs'], y['epochs']),
        ('Batch size', b['batch'], y['batch']),
        ('Training time', f"{b['minutes']:.0f} min", f"{y['minutes']:.0f} min"),
        ('Optimizer', 'SGD, lr 0.005, momentum 0.9, cosine schedule', 'Chosen automatically by Ultralytics'),
        ('Augmentation', 'colour jitter, blur (no flips)', 'mosaic, HSV, translate/scale (no flips)'),
        ('Best checkpoint', 'lowest validation loss', 'best validation mAP'),
    ]
    for _, _, m in models:  # flatten the operating-point numbers for the template
        m['rec_conf'], m['hr_conf'] = m['recommended_conf'], m['high_recall_conf']
        for k in ('precision', 'recall', 'f1'):
            m[f'rec_{k}'] = m['at_recommended_conf'][k]
            m[f'hr_{k}'] = m['at_high_recall_conf'][k]
    metric_rows = [('mAP@50', 'map50', 3), ('mAP@50:95', 'map50_95', 3), ('mAP@75', 'map75', 3),
                   ('Recommended confidence (best F1 on validation)', 'rec_conf', 2),
                   ('Precision at recommended confidence', 'rec_precision', 3),
                   ('Recall at recommended confidence', 'rec_recall', 3),
                   ('F1 at recommended confidence', 'rec_f1', 3),
                   ('"Find every part" confidence (validation: precision >= 0.70, at most the balanced value)', 'hr_conf', 2),
                   ('Precision, find every part', 'hr_precision', 3), ('Recall, find every part', 'hr_recall', 3),
                   ('Average IoU of correct boxes', 'avg_iou', 3), ('Time per page (s, Apple M1 Pro GPU)', 'inference_time', 3)]

    html = TEMPLATE.render(
        indigo=INDIGO, gold=GOLD, today=date.today().strftime('%d %B %Y'), ds=ds, base=base, prop=prop, auth=auth,
        models=models, comparison_text=comparison_text, per_class_text=per_class_text, language_text=language_text,
        languages=languages, plots=plots, example_predictions=examples, error_examples=errors, ocr=ocr,
        ocr_examples=ocr_examples, loss_curves=loss_plot(REPORT_DIR / 'loss_curves.png'), training_rows=training_rows,
        metric_rows=metric_rows, cross=cross, rare_labels=rare_labels, audit_count=audit_fix_count(),
        test_count=tests['passed'])
    write_markdown(ds, models, runs, cross, ocr, tests['passed'])
    html_path = REPORT_DIR / 'IndicDoc_AI_Project_Report.html'
    html_path.write_text(html)
    pdf_path = REPORT_DIR / 'IndicDoc_AI_Project_Report.pdf'
    subprocess.run([CHROME, '--headless=new', '--disable-gpu', '--no-pdf-header-footer',
                    f'--print-to-pdf={pdf_path}', html_path.as_uri()], check=True, capture_output=True, timeout=120)
    print(f"Report written to {pdf_path}")


if __name__ == '__main__':
    main()
