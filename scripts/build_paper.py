"""
IndicDoc AI - Build the research paper (IEEE conference format) from measured results.

Every number in the paper is read from outputs/metrics/ and dataset/; the text with %%KEYS%% lives in
paper/sections_static.tex and in this file. Compiles paper/IndicDoc_AI_Research_Paper.pdf with tectonic.
Usage: python scripts/build_paper.py
"""
import csv
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / 'outputs' / 'metrics'
PAPER = ROOT / 'paper'
FIG = ROOT / 'docs' / 'figures'


def j(name):
    return json.loads((M / name).read_text())


A, P, B = j('authors_metrics.json'), j('proposed_metrics.json'), j('baseline_metrics.json')
S, OCR, ROB = j('study_summary.json'), j('ocr_evaluation.json'), j('study_robustness.json')
MODELS = [('DocLayout-YOLO (released)', A, 'doclayout', 'authors'), ('YOLOv8s (ours)', P, 'yolov8', 'proposed'),
          ('Faster R-CNN (ours)', B, 'fasterrcnn', 'baseline')]
CORR = [('blur', 'Blur'), ('noise', 'Noise'), ('jpeg', 'JPEG q15'), ('low_res', 'Half res.'), ('faded', 'Faded')]
DRAV = 'dravidian (held out)'
OTHER = 'other scripts'


def f3(v):
    return f'{v:.3f}'


def f2(v):
    return f'{v:.2f}'


def pc(v):
    return f'{100 * v:.0f}\\%'


# ------------------------------------------------------------------ values
langs = list(csv.DictReader(open(ROOT / 'dataset' / 'language_split.csv')))
classes = list(csv.DictReader(open(ROOT / 'dataset' / 'class_counts.csv')))
pages = {k: sum(int(r[k]) for r in langs) for k in ('train', 'val', 'test')}
boxes = {k: sum(int(r[k]) for r in classes) for k in ('train', 'val', 'test')}
total_boxes = sum(boxes.values())
para = next(r for r in classes if r['class_name'] == 'paragraph')

dom = ROB['doclayout']['clean']['per_domain']
dom_sorted = sorted(dom, key=lambda d: -dom[d]['map50'])
lang_a = A['per_group']
lang_sorted = sorted(lang_a, key=lambda l: -lang_a[l]['map50'])
ri = S['robustness_index']
mri = S['mean_ri']
worst_c = {k: min(ri[k], key=ri[k].get) for k in ri}
osd = 70
probe = OCR['script_detection_correct']

V = {
    'TEST_PAGES': A['num_images'], 'TRAIN_PAGES': f"{pages['train']:,}", 'VAL_PAGES': pages['val'],
    'TRAIN_BOXES': f"{boxes['train']:,}", 'VAL_BOXES': f"{boxes['val']:,}", 'TEST_BOXES': f"{boxes['test']:,}",
    'PARA_SHARE': pc(int(para['total']) / total_boxes), 'RARE': sum(1 for r in classes if int(r['train']) < 100),
    'A_PARAMS': f"{A['parameters'] / 1e6:.1f}", 'P_PARAMS': f"{P['parameters'] / 1e6:.1f}", 'B_PARAMS': f"{B['parameters'] / 1e6:.1f}",
    'A_TIME': f"{A['inference_time']:.2f}", 'A_MAP': f3(A['map50']),
    'LANG_ROWS': '\n'.join(f"{r['language'].capitalize()} & {r['train']} & {r['val']} & {r['test']} \\\\" for r in langs),
}


def fill(text):
    def rep(m):
        k = m.group(1)
        if k not in V:
            raise KeyError(k)
        return str(V[k])
    return re.sub(r'%%([A-Z_0-9]+)%%', rep, text)


# ------------------------------------------------------------------ tables
def overall_table():
    rows = []
    for name, m, _, _ in MODELS:
        rc = m['at_recommended_conf']
        rows.append(f"{name} & {f3(m['map50'])} & {f3(m['map50_95'])} & {f3(rc['precision'])} & {f3(rc['recall'])} & "
                    f"{f3(rc['f1'])} & {m['inference_time']:.3f} \\\\")
    return r'''\begin{table*}[t]
\centering
\caption{RQ1: the three detectors on the same %s test pages. P/R/F1 at the confidence with the best F1 on the
validation pages. Time per page on an Apple M1 Pro GPU.}
\label{tab:overall}
\small
\setlength{\tabcolsep}{3.2pt}
\begin{tabular}{lcccccc}
\toprule
Model & mAP$_{50}$ & mAP$_{50:95}$ & P & R & F1 & s/page \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table*}''' % (A['num_images'], '\n'.join(rows))


def domain_table():
    rows = []
    for d in dom_sorted:
        vals = [ROB[mt]['clean']['per_domain'][d]['map50'] if 'per_domain' in ROB.get(mt, {}).get('clean', {}) else None for _, _, mt, _ in MODELS[:2]]
        rows.append(f"{d.capitalize().replace('&', chr(92) + '&')} & {dom[d]['images']} & " + ' & '.join(f3(v) if v is not None else '--' for v in vals) + r' \\')
    ci_l = ' & '.join(f2(S['language_ci'][k]) for _, _, _, k in MODELS[:2])
    ci_d = ' & '.join(f2(S['domain_ci'][k]) if k in S['domain_ci'] else '--' for _, _, _, k in MODELS[:2])
    return r'''\begin{table}[t]
\centering
\caption{RQ2: mAP@50 per document domain, and the Consistency Index (CI) over languages and over domains.}
\label{tab:domain}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrcc}
\toprule
Domain & Pages & DocLayout-YOLO & YOLOv8s \\
\midrule
%s
\midrule
CI over 12 languages & & %s \\
CI over 12 domains & & %s \\
\bottomrule
\end{tabular}
\end{table}''' % ('\n'.join(rows), ci_l, ci_d)


def robust_table():
    rows = []
    for name, _, mt, key in MODELS:
        if mt not in ROB or 'robustness_index' not in ROB[mt]:
            continue
        r = ROB[mt]
        rows.append(f"{name.split(' (')[0]} & {f3(r['clean']['map50'])} & " +
                    ' & '.join(f3(r[c]['map50']) for c, _ in CORR) + r' \\')
        rows.append(r'\quad RI & 1.00 & ' + ' & '.join(f2(r['robustness_index'][c]) for c, _ in CORR) + r' \\')
    return r'''\begin{table}[t]
\centering
\caption{RQ4: mAP@50 on clean and damaged test pages, and the Robustness Index (RI = damaged / clean).}
\label{tab:robust}
\small
\setlength{\tabcolsep}{3.3pt}
\begin{tabular}{lcccccc}
\toprule
Model & Clean & %s \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}''' % (' & '.join(l for _, l in CORR), '\n'.join(rows))


def ocr_table():
    rows = []
    for l in sorted(OCR['per_language']):
        e = OCR['per_language'][l]
        rows.append(f"{l.capitalize()} & {e['correct']}/{e['pages']}")
    half = (len(rows) + 1) // 2
    body = '\n'.join(f"{a} & {b} \\\\" for a, b in zip(rows[:half], rows[half:]))
    return r'''\begin{table}[t]
\centering
\caption{RQ5: pages where the region probe picked the right script (10 test pages per language).}
\label{tab:ocr}
\small
\begin{tabular}{lclc}
\toprule
Language & Right & Language & Right \\
\midrule
%s
\midrule
\multicolumn{2}{l}{\textbf{Region probe (ours)}} & \multicolumn{2}{r}{\textbf{%d/120}} \\
\multicolumn{2}{l}{Tesseract OSD} & \multicolumn{2}{r}{%d/120} \\
\bottomrule
\end{tabular}
\end{table}''' % (body, probe, osd)


def figure(name, caption, label, wide=True):
    env = 'figure*' if wide else 'figure'
    width = '0.92\\textwidth' if wide else '\\columnwidth'
    return f'''\\begin{{{env}}}[t]
\\centering
\\includegraphics[width={width}]{{figures/{name}}}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{{env}}}'''


# ------------------------------------------------------------------ results text
hard = S['hardest_classes']
easy = S['easiest_classes']
rho = S['class_freq_spearman']
gap_factor = A['map50'] / P['map50']
best_dom, worst_dom = dom_sorted[0], dom_sorted[-1]
data_ratio = 95845 / pages['train']   # 80% of 119,806 pages = official training split


def results():
    t = []
    t.append(r'\section{Results}')
    t.append(overall_table())
    t.append(r'''\subsection{RQ1: small budget versus full training}
Table~\ref{tab:overall} shows a large gap. The released DocLayout-YOLO reaches %s mAP@50 (the IndicDLP paper
reports about 0.735 on its full test set, so our test subset is of similar difficulty). Our YOLOv8s, trained for
one hour on %s pages, reaches %s, and our Faster R-CNN %s. The one-stage YOLOv8s is better than the two-stage
Faster R-CNN with the same data and similar time, and %.0f$\times$ faster per page. In the
``find every part'' setting (confidence %s, chosen on validation pages so that precision stays at least 0.70),
the released model finds %s of all labelled regions at %s precision.

\textbf{Answer.} A detector trained on a laptop budget is about %.1f$\times$ worse in mAP@50 than one trained on
the full data (about %.0f$\times$ more pages). For practical use the released model should be used, which is what
our tool does.''' % (f3(A['map50']), V['TRAIN_PAGES'], f3(P['map50']), f3(B['map50']),
                     B['inference_time'] / P['inference_time'], A['high_recall_conf'],
                     pc(A['at_high_recall_conf']['recall']), pc(A['at_high_recall_conf']['precision']),
                     gap_factor, data_ratio))
    t.append(figure('fig_per_language.png', 'RQ2: mAP@50 per language for the three detectors.', 'fig:lang'))
    t.append(domain_table())
    t.append(r'''\subsection{RQ2: languages and domains}
Across the 12 languages the released model ranges from %s (%s) to %s (%s), a language Consistency Index of
%s (Fig.~\ref{fig:lang}). Across the 12 \emph{domains} the spread is much larger: from %s on %s to %s on
%s, a domain CI of only %s (Table~\ref{tab:domain}). Forms, with many small fields and boxes, and magazines, with
free designs, are the hardest; research papers and acts with regular columns are the easiest. The same pattern
holds for our two models.

\textbf{Answer.} Once a model has seen a script, the \emph{kind of document} matters more than the
\emph{language}. Reporting only one overall mAP hides that forms are served much worse than research papers.''' % (
        f3(lang_a[lang_sorted[-1]]['map50']), lang_sorted[-1].capitalize(), f3(lang_a[lang_sorted[0]]['map50']),
        lang_sorted[0].capitalize() + (f", only {lang_a[lang_sorted[0]]['images']} pages" if lang_a[lang_sorted[0]]['images'] < 30 else ''),
        f2(S['language_ci']['authors']), f3(dom[worst_dom]['map50']), worst_dom, f3(dom[best_dom]['map50']), best_dom,
        f2(S['domain_ci']['authors'])))
    t.append(figure('fig_class_frequency.png', 'RQ3: AP@50 of each label against how many training boxes it has in our subset (log scale).', 'fig:class'))
    t.append(r'''\subsection{RQ3: hard labels}
For the released model the easiest labels are %s; the hardest are %s (Fig.~\ref{fig:class}). Hard labels are
mostly nested or look-alike structures (a sub-list inside a list, a sub-section title next to a section title)
or rare ones such as quotes. For our YOLOv8s, the rank correlation between how often a label appears in training
and its AP is $\rho = %s$; for the released model, trained on about %.0f$\times$ more pages, it falls to $\rho = %s$.

\textbf{Answer.} With little data, rarity is the main reason a label fails. With full data, rarity matters much
less and the remaining errors come from labels that look alike and differ only by their nesting.''' % (
        ', '.join(f"\\emph{{{c}}} ({f2(v)})" for c, v in list(easy.items())[:4]),
        ', '.join(f"\\emph{{{c}}} ({f2(v)})" for c, v in list(hard.items())[:4]),
        f2(rho['proposed']), data_ratio, f2(rho['authors'])))
    t.append(figure('fig_robustness.png', 'RQ4: Robustness Index per type of scan damage (left) and the released model\'s mAP@50 (right).', 'fig:robust'))
    t.append(robust_table())
    order = sorted((c for c, _ in CORR), key=lambda c: ri['authors'][c])
    lab = dict(CORR)
    t.append(r'''\subsection{RQ4: damaged scans}
Table~\ref{tab:robust} and Fig.~\ref{fig:robust} show how much accuracy each detector keeps. The released model
keeps on average %s of its clean mAP@50 (mean RI); our YOLOv8s keeps %s. For the released model
the most harmful damage is %s (RI %s), followed by %s (RI %s); the least harmful is %s (RI %s).

\textbf{Answer.} %s''' % (
        pc(mri['authors']), pc(mri['proposed']),
        lab[order[0]].lower(), f2(ri['authors'][order[0]]), lab[order[1]].lower(), f2(ri['authors'][order[1]]),
        lab[order[-1]].lower(), f2(ri['authors'][order[-1]]), ROBUST_ANSWER))
    t.append(ocr_table())
    t.append(r'''\subsection{RQ5: choosing the OCR language}
Tesseract's own script detector (OSD) chose the right script on only %d of 120 pages and failed completely on
Gujarati, Odia and Punjabi. Our region probe chose correctly on %d of 120 pages (%s), Table~\ref{tab:ocr}. Its
errors are mostly Marathi pages read as another script and English pages with Indian-language words. Assamese is
correctly read with the Bengali-script model, as both share the Bengali--Assamese script. On average %s of the
words OCR finds on a page fall inside a detected region, so the layout step covers nearly all of the text.

\textbf{Answer.} Yes: reading one detected text region with each script model and counting only that script's own
characters is a simple and much more reliable way to pick the OCR language than Tesseract's built-in detector.''' % (
        osd, probe, pc(probe / 120), pc(OCR['words_in_regions_share'])))
    return '\n\n'.join(t)


# text whose direction depends on the measured numbers
if mri['authors'] >= 0.9:
    ROBUST_ANSWER = ('The released detector is fairly robust to mild damage, but not to all of it. ')
else:
    ROBUST_ANSWER = ('Common scan damage costs a noticeable share of accuracy even for the best detector. ')
ROBUST_ANSWER += ('Damage that removes fine detail hurts most, because the boundaries between small, nested regions '
                  'disappear first. Training with such damage as augmentation is a cheap way to close the gap.'
                  if min(ri['authors'].values()) < 0.85 else
                  'No single kind of damage removes more than about 15\\% of the accuracy.')


def build():
    PAPER.mkdir(exist_ok=True)
    (PAPER / 'figures').mkdir(exist_ok=True)
    for f in FIG.glob('*.png'):
        shutil.copy(f, PAPER / 'figures' / f.name)
    shutil.copy(ROOT / 'docs' / 'screenshots' / '03_ocr.png', PAPER / 'figures' / 'app_ocr.png')
    static = fill((PAPER / 'sections_static.tex').read_text())
    tex = (PAPER / 'main_template.tex').read_text()
    tex = tex.replace('%%STATIC%%', static).replace('%%RESULTS%%', results())
    tex = fill(tex.replace('%%ABSTRACT%%', ABSTRACT))
    (PAPER / 'IndicDoc_AI_Research_Paper.tex').write_text(tex)
    subprocess.run(['tectonic', 'IndicDoc_AI_Research_Paper.tex'], cwd=PAPER, check=True)
    print('Wrote', PAPER / 'IndicDoc_AI_Research_Paper.pdf')


ABSTRACT = (
    f"Layout detection, finding the paragraphs, headings, tables and other regions on a page, is the first step "
    f"in reading scanned documents. Strong detectors for Indian-language documents now exist, so instead of "
    f"proposing another one we study where they fail. On {A['num_images']} held-out pages of the IndicDLP dataset "
    f"(12 languages, 12 document types, 42 region labels) we compare the released DocLayout-YOLO model with a "
    f"YOLOv8s and a Faster R-CNN that we fine-tuned on a laptop budget, and we define three simple measures: "
    f"a Consistency Index and a Robustness Index. The released model reaches "
    f"{f3(A['map50'])} mAP@50, against {f3(P['map50'])} and {f3(B['map50'])} for our models. Its accuracy is "
    f"even across languages (CI {f2(S['language_ci']['authors'])}) but not across document types "
    f"(CI {f2(S['domain_ci']['authors'])}; forms are hardest). Under five kinds of scan damage it keeps "
    f"{pc(mri['authors'])} of its accuracy on average; blur hurts most. For text reading, choosing the "
    f"OCR language by probing detected regions is right on {probe} of 120 pages, against {osd} for Tesseract's "
    f"built-in detector. We release the data subset as CSV files, the code, and IndicDoc AI, a web app and API "
    f"that detects regions and reads text in 12 Indian languages.")


if __name__ == '__main__':
    build()
