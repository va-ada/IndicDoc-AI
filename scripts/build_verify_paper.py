"""
IndicDoc AI - Build the research paper on IndicDoc Verify in IEEE conference format (two columns, Times-style
font, IEEE tables, figures and references), printed to PDF with headless Chrome / Chromium.

Every number is read from outputs/metrics (layout evaluation and the verification benchmark); nothing is typed in by
hand. The earlier layout-study paper (paper/IndicDoc_AI_Research_Paper.tex, scripts/build_paper.py) is kept as a
companion study.

Usage: python scripts/build_verify_paper.py      -> paper/IndicDoc_Verify_Research_Paper.pdf
"""
import base64
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / 'outputs' / 'metrics'
FIG = ROOT / 'docs' / 'figures'
SHOTS = ROOT / 'docs' / 'screenshots'
OUT = ROOT / 'paper' / 'IndicDoc_Verify_Research_Paper.pdf'
Q_LABELS = ['first-level-question', 'second-level-question', 'third-level-question', 'options', 'answer']


def j(name):
    p = M / name
    return json.loads(p.read_text()) if p.exists() else {}


def img(path, width='100%'):
    data = base64.b64encode(Path(path).read_bytes()).decode()
    mime = 'jpeg' if str(path).endswith('.jpg') else 'png'
    return f'<img src="data:image/{mime};base64,{data}" style="width:{width}">'


def pct(x):
    return f'{100 * x:.0f}%'


def roman(n):
    out = ''
    for v, s in [(10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]:
        while n >= v:
            out, n = out + s, n - v
    return out


class Num:
    t = 0
    f = 0

    def table(self, cap, head, rows, align):
        self.t += 1
        css = {'l': 'left', 'r': 'right', 'c': 'center'}
        th = ''.join(f'<th style="text-align:{css[a]}">{h}</th>' for h, a in zip(head, align))
        body = ''.join('<tr class="mid"><td colspan="%d"></td></tr>' % len(head) if r == '---' else
                       '<tr>' + ''.join(f'<td style="text-align:{css[a]}">{c}</td>' for c, a in zip(r, align)) + '</tr>'
                       for r in rows)
        return (f'<table class="tb"><caption>TABLE {roman(self.t)}<br><span>{cap}</span></caption><thead><tr>{th}'
                f'</tr></thead><tbody>{body}</tbody></table>')

    def figure(self, content, cap, wide=False):
        self.f += 1
        return f'<figure class="{"wide" if wide else ""}">{content}<figcaption>Fig. {self.f}. {cap}</figcaption></figure>'


def build():
    A, P, B = j('authors_metrics.json'), j('proposed_metrics.json'), j('baseline_metrics.json')
    S = j('study_summary.json')
    V, VN = j('verify_benchmark_oracle.json'), j('verify_benchmark_noisy.json')
    V_NP, V_NG = j('verify_benchmark_oracle_no_number_pass.json'), j('verify_benchmark_oracle_no_geometry.json')
    VP = j('verify_benchmark_oracle_posthoc.json')
    vo, cl, dg, vl = V['overall'], V['by_quality']['clean_render'], V['by_quality']['scan_degraded'], V['by_language']
    qap = {k: A['per_class_ap50'].get(k, 0) for k in Q_LABELS}
    lo, hi = min(qap.values()), max(qap.values())
    n = Num()
    labels = V['labels']

    t_layout = n.table('Layout detection on 682 IndicDLP test pages',
                       ['Detector', 'mAP@50', 'mAP@50:95', 'F1', 's/page'],
                       [[name, f"{m['map50']:.3f}", f"{m['map50_95']:.3f}", f"{m['at_recommended_conf']['f1']:.3f}",
                         f"{m['inference_time']:.3f}"] for name, m in
                        (('Faster R-CNN (ours)', B), ('YOLOv8s (ours)', P), ('DocLayout-YOLO [11]', A))], 'lrrrr')
    t_q = n.table('AP@50 on question-paper labels', ['Label', 'DocLayout', 'YOLOv8s', 'F. R-CNN'],
                  [[k.replace('-level-question', '-level q.'), f'{qap[k]:.2f}', f"{P['per_class_ap50'].get(k, 0):.2f}",
                    f"{B['per_class_ap50'].get(k, 0):.2f}"] for k in Q_LABELS], 'lrrr')
    t_err = n.table('Error types injected in the benchmark', ['Error', 'Pairs', 'Found'],
                    [[labels[t], V['by_type'][t]['errors_injected'], pct(V['by_type'][t]['found_at_place'])]
                     for t in labels], 'lrr')
    t_res = n.table('Verification results (test set)', ['Condition', 'Found', 'FA / clean pair'], [
        ['Born-digital', pct(cl['found_at_place']), f"{cl['false_alarms_per_clean_pair']:.2f}"],
        ['Scan-degraded', pct(dg['found_at_place']), f"{dg['false_alarms_per_clean_pair']:.2f}"],
        '---',
        ['English → Hindi', pct(vl['hi']['found_at_place']), f"{vl['hi']['false_alarms_per_clean_pair']:.2f}"],
        ['English → Tamil', pct(vl['ta']['found_at_place']), f"{vl['ta']['false_alarms_per_clean_pair']:.2f}"],
        '---',
        ['All', pct(vo['found_at_place']), f"{vo['false_alarms_per_clean_pair']:.2f}"]], 'lrr')
    abl = [('Full system', V), ('No number pass', V_NP), ('No layout-based options', V_NG), ('Noisy layout', VN)]
    t_abl = n.table('Ablations (test set)', ['System', 'Found', 'FA / clean pair'],
                    [[a, pct(r['overall']['found_at_place']), f"{r['overall']['false_alarms_per_clean_pair']:.2f}"]
                     for a, r in abl if r], 'lrr')

    f_page = n.figure(img(SHOTS / '08_verify_page.png'), 'A changed number in the Hindi version of a benchmark paper, '
                      'found and outlined on the page by IndicDoc Verify.')
    f_type = n.figure(img(FIG / 'fig_verify_by_type.png'), 'Injected errors found at the right question, per error '
                      'type, born-digital vs. scan-degraded pages.')
    f_abl = n.figure(img(FIG / 'fig_verify_ablation.png'), 'Ablations: errors found (left) and false alarms per clean '
                     'pair (right).')

    refs = [
        'O. Nath, S. Kukkala, M. Khapra, and R. K. Sarvadevabhatla, "IndicDLP: A foundational dataset for multi-lingual and multi-domain document layout parsing," in <i>Proc. ICDAR</i>, 2025, pp. 23–39.',
        'B. V. Kasuba <i>et al.</i>, "PLATTER: A page-level handwritten text recognition system for Indic scripts," arXiv:2502.06172, 2025.',
        'A. Mondal <i>et al.</i>, "UniTabBank: A large scale multi-lingual, multi-layout, multi-type, multi-format dataset for table detection," in <i>Proc. WACV</i>, 2026.',
        'Y. Xu <i>et al.</i>, "LayoutXLM: Multimodal pre-training for multilingual visually-rich document understanding," arXiv:2104.08836, 2021.',
        'B. V. Kasuba, P. Chaudhuri, and G. Ramakrishnan, "DRISHTIKON: Visual grounding at multiple granularities in documents," arXiv:2506.21316, 2025.',
        'S. Bhattacharyya <i>et al.</i>, "Adapting vision-language models for Hindi OCR," IIIT Hyderabad, 2025.',
        'M. Babayeva, J. Dunlap, M. Snětinová, and R. Widenhorn, "Translating the Force Concept Inventory in the age of AI," arXiv:2508.13908, 2025.',
        'R. Rei <i>et al.</i>, "CometKiwi: IST-Unbabel 2022 submission for the quality estimation shared task," in <i>Proc. WMT</i>, 2022, pp. 634–645.',
        'S. Verma <i>et al.</i>, "MILU: A multi-task Indic language understanding benchmark," in <i>Proc. NAACL</i>, 2025.',
        'Y. Chen <i>et al.</i>, "RoDLA: Benchmarking the robustness of document layout analysis models," in <i>Proc. CVPR</i>, 2024, pp. 15556–15566.',
        'Z. Zhao, H. Kang, B. Wang, and C. He, "DocLayout-YOLO: Enhancing document layout analysis through diverse synthetic data and global-to-local adaptive perception," arXiv:2410.12628, 2024.',
        'G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLOv8," 2023. [Online]. Available: github.com/ultralytics/ultralytics',
        'S. Ren, K. He, R. Girshick, and J. Sun, "Faster R-CNN: Towards real-time object detection with region proposal networks," in <i>Proc. NeurIPS</i>, 2015.',
        'Tesseract OCR 5 and tessdata_best models, 2021–2025. [Online]. Available: github.com/tesseract-ocr/tesseract',
        '"Madras HC orders CBSE to give 196 marks to Tamil medium students, for errors in 49 questions," <i>The News Minute</i>, Jul. 2018.',
        '"When Kannada was lost in translation: KPSC forces a do-over," <i>The South First</i>, 2024; "Conduct KAS prelims examination again," <i>ETV Bharat</i>, Mar. 2025.',
        '"NTA admits defects in 3 UGC-NET papers, orders re-tests in September," <i>The Week</i>, Aug. 2026.',
        '"HC grants relief to MBBS student, directs NTRUHS to give 1 compensatory mark for question paper error," <i>Medical Dialogues</i>, Jul. 2026.',
        'Rajya Sabha, Unstarred Question 2356 on AI tools in the Supreme Court e-filing system, Mar. 2025.',
    ]

    body = f'''
<div class="head">
<h1>IndicDoc Verify: Layout-Aware Pre-Print Verification of Multilingual Indian Question Papers</h1>
<div class="authors">Prashant Jha, Aditya Soni, Riddhi Patil, Mariam Badure and Joanne Gomes<br>
<i>Department of Artificial Intelligence and Machine Learning, St. Francis Institute of Technology (Autonomous), Mumbai, India</i></div>
</div>
<div class="cols">
<p class="abs"><b><i>Abstract</i>—Indian examinations are set in one language and translated into Hindi and regional
languages, and every version is checked by hand; errors keep reaching candidates, from 49 wrong questions in the Tamil
version of NEET 2018 to re-tests of three UGC-NET papers in 2026. We present IndicDoc Verify, an offline system that
reads a printed question paper in any of 12 Indian languages and flags structural and numerical errors, and every
difference from its source version, before printing. A deep-learning layout detector recovers questions, options and
answers (AP@50 {lo:.2f}–{hi:.2f}); Tesseract reads the text with an English number pass that must agree on every
digit; questions are built using the layout, checked for numbering, duplicates, options, marks and repeats, and two
language versions are aligned and compared without translation. On {vo['pairs']} English–Hindi and English–Tamil
paper pairs with injected errors, read with real OCR, the system finds {pct(cl['found_at_place'])} of errors at the
right question on born-digital papers ({pct(dg['found_at_place'])} on degraded scans) with
{cl['false_alarms_per_clean_pair']:.1f} false alarms per clean pair; the number pass alone cuts false alarms from
{V_NP['overall']['false_alarms_per_clean_pair']:.1f} to {vo['false_alarms_per_clean_pair']:.1f}.</b></p>
<p class="abs"><b><i>Index Terms</i>—document layout analysis, Indian languages, OCR, examinations, translation
quality, verification.</b></p>

<h2>I. Introduction</h2>
<p>National and state examinations in India are multilingual: NEET-UG is offered in 13 languages, and the Union and
state Public Service Commissions, school boards and universities print bilingual papers. A paper is set in one
language, translated, and every version is vetted by hand. When that step fails, candidates who chose a regional
language sit a different paper. In NEET-UG 2018, 49 questions of the Tamil version were wrong and the Madras High Court
ordered 196 grace marks, later set aside by the Supreme Court [15]; the Karnataka civil-services preliminary paper of
2024 had 59 errors in Kannada and was re-held, with errors again [16]; three UGC-NET papers of June 2026 were re-tested
after translation errors and questions repeated from earlier papers [17]; an MBBS paper of 2026 printed the same MCQ
twice [18].</p>
<p>We found no tool that checks a paper and its translations automatically. Translation-quality estimation [8] and
LLM back-translation [7] work on plain sentences and cannot see question numbers, options or marks; the IndicDLP layout
dataset [1] labels questions and options in 12 languages, but detected structure has not been used to verify a
document; and a pre-print paper is too confidential for a cloud service. (Court-filing scrutiny, by contrast, already
uses AI defect detection [19].)</p>
<p>Our contributions: (1) IndicDoc Verify, an offline pipeline that turns a printed paper into structured questions
with a deep-learning layout detector and checks it; (2) <i>translation parity without translation</i>: language
versions are aligned and compared on numbers, options, marks, sub-questions and figures, which are the same in every
language; (3) a <i>number pass</i> in which two OCR models must agree on every digit; (4) a benchmark of
{vo['pairs']} rendered paper pairs with nine kinds of real error, a separate development set, and ablations.</p>

<h2>II. Related Work</h2>
<p><b>Indian-language documents.</b> IndicDLP [1] provides 119,806 pages with 42 labels and benchmarks modern
detectors; DocLayout-YOLO [11] adapts YOLOv10 to documents; RoDLA [10] studies layout robustness. Recognition work
covers handwritten Indic text [2], Hindi OCR with vision-language models [6] and tables [3]; multimodal models [4],
[5] need reliable OCR. MILU [9] collects Indian exam questions in 11 languages for evaluating LLMs.</p>
<p><b>Checking translations.</b> Quality estimation such as CometKiwi [8] scores a translation without a reference,
sentence by sentence. Babayeva <i>et al.</i> [7] translate a physics test with an LLM and still review each item by
hand. Neither uses the page structure of an exam paper, which is where many costly errors lie (a missing option, a
changed number, wrong marks).</p>

<h2>III. IndicDoc Verify</h2>
<p><b>A. Layout.</b> Each page (PDF rendered at 200 dpi) goes through DocLayout-YOLO [11], fine-tuned on IndicDLP by
its authors, at a high-recall threshold chosen on validation pages. We also fine-tuned YOLOv8s [12] and Faster
R-CNN [13] on a 2,293-page language-balanced subset to choose the detector (Table I). The released model recovers the
question-paper labels well (Table II).</p>
{t_layout}
{t_q}
<p><b>B. OCR with a number pass.</b> Tesseract [14] reads the page with the model of its own script. Indic models
misread Latin digits in mixed text ("1947" as "947", "1" as the danda "&#2404;", "A" as "&#2414;"). An English pass
re-reads the page; each numeric token replaces the overlapping word of the Indic reading when that word has no Indic
letters or the English reading is more confident, is inserted where the Indic reading has nothing, and a Latin number
of the Indic reading that the English pass does not confirm is dropped. Pages printed with Indian digits are detected
and keep them.</p>
<p><b>C. Questions.</b> A first-level-question region starts a question (or a text region beginning with a question
number, if the detector missed it); sub-questions, options, figures, tables and formulas follow it. Options are split
by the horizontal gaps between words inside an option region, so a garbled marker does not change the count; marks
("[2 marks]", "[2 अंक]", "(2 மதிப்பெண்கள்)") and the declared maximum are read in all 12 languages.</p>
<p><b>D. Checks.</b> On one paper: numbering gaps and repeats, duplicated questions (character 3-gram cosine
&ge; 0.9 with identical numbers), option count, repeated or empty options, marks against the declared maximum, page
numbering. Against a question bank: repeats (cosine &ge; 0.8, identical numbers). Between versions <i>s</i> and
<i>t</i>, questions are aligned by Needleman–Wunsch with the score</p>
<p class="eq">σ(i, j) = 2 J(N<sub>i</sub>, N<sub>j</sub>) + ν(i, j) + 0.5·[k<sub>i</sub> = k<sub>j</sub>],
&nbsp; gap = −0.6</p>
<p>where <i>N</i> is the multiset of numbers in a question and its options, <i>J</i> their Jaccard similarity, ν = 1
(−0.5) if the printed numbers agree (differ), and <i>k</i> the skeleton (option count, sub-questions, marks, visuals).
Aligned pairs are compared on every number in the stem and options, option count, marks, sub-questions and visuals;
unaligned questions are reported missing or extra. A cross-lingual similarity model can be plugged in as an extra
meaning check. Every finding carries its question, page and box.</p>
<p><b>E. Format profiles and deployment.</b> The same engine checks any document against a JSON profile of required
elements, order and bilingual text (e.g. an Office Memorandum). The system runs offline in one Docker image with a web
app and a REST API; the report outlines each finding on its page (Fig. 1).</p>
{f_page}

<h2>IV. Benchmark</h2>
<p>Erroneous pre-print papers are not public, so we built a benchmark. Each pair is an English source and its Hindi
or Tamil version: 15 MCQs (numeric templates and fixed questions) and 3 short-answer questions with marks, rendered at
200 dpi with Noto fonts by a browser, which also records each region's box. {vo['errors_injected']} of the
{vo['pairs']} test pairs carry one injected error (Table III); {vo['clean_pairs']} are clean. Half are degraded like
scans (blur 0.7 px, Gaussian noise σ = 8, JPEG 55). Pages are read with Tesseract 5 (tesseract.js, best-integer models).
Layout boxes come from the renderer (an oracle layout); a <i>noisy-layout</i> run shifts boxes by up to 8 px and
relabels 10% of question and option regions as paragraphs. An error counts as found only if a finding of the matching
kind points at the right question (±1) or, for paper-level errors, at the paper. The parser was developed on a
separate 48-pair development set; the test set was run once.</p>
{t_err}

<h2>V. Results</h2>
<p><b>Errors found and false alarms.</b> Table IV: on born-digital papers, the normal pre-print case, IndicDoc Verify
finds {pct(cl['found_at_place'])} of injected errors at the right question with {cl['false_alarms_per_clean_pair']:.2f}
false alarms per clean pair; scan degradation lowers this to {pct(dg['found_at_place'])} with
{dg['false_alarms_per_clean_pair']:.2f} false alarms. Fig. 2 shows the result per error type.</p>
{t_res}
{f_type}
<p><b>Ablations.</b> Table V and Fig. 3: removing the number pass raises false alarms to
{V_NP['overall']['false_alarms_per_clean_pair']:.2f} per clean pair; splitting options by their markers instead of the
layout {"raises them to " + f"{V_NG['overall']['false_alarms_per_clean_pair']:.2f}" if V_NG else "was not run"}. A noisy
layout still finds {pct(VN['overall']['found_at_place'])} of errors, because questions are recovered from their text
when a label is wrong.</p>
{t_abl}
{f_abl}
<p><b>Error analysis.</b> Most remaining false alarms are option values that OCR dropped or merged on one side
("216" read as "26"), more often on degraded Hindi pages. The only two misses were Hindi papers whose maximum-marks line
OCR read as "पूर्णाक" (anusvara dropped). After the test run we widened that pattern; re-running the unchanged test set
then finds {pct(VP['overall']['found_at_place'])} of errors with {VP['overall']['false_alarms_per_clean_pair']:.2f}
false alarms per clean pair (post-hoc; the numbers above are from the frozen system).</p>

<h2>VI. Discussion and Limitations</h2>
<p>The checks need no translation, so they work for any pair of the 12 languages and cannot be fooled by fluent but
wrong text in the way a quality-estimation score can. Their blind spot is a change of meaning that keeps numbers and
structure (a dropped word, as in NEET 2021); the meaning hook needs a cross-lingual model that is evaluated on exam
language. The benchmark uses rendered papers and oracle layout boxes; the layout model's own errors are covered by the
noisy-layout run and by its measured AP, and a pilot on real pre-print papers of an exam body is the next step. Number
words and formulas are not compared yet.</p>

<h2>VII. Conclusion</h2>
<p>IndicDoc Verify brings deep-learning layout analysis to a problem that is costly, recurring and so far unautomated:
checking an Indian question paper and its translations before printing. It finds most injected errors at the right
question with few false alarms on born-digital papers, runs offline, and covers 12 languages. Code, benchmark generator
and results are open.</p>

<h2>References</h2>
<ol class="refs">{''.join(f'<li>{r}</li>' for r in refs)}</ol>
</div>'''
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>IndicDoc Verify</title>
<link href="https://fonts.googleapis.com/css2?family=STIX+Two+Text:ital,wght@0,400;0,600;0,700;1,400;1,600&family=Noto+Sans+Devanagari&family=Noto+Sans+Tamil&display=swap" rel="stylesheet">
<style>
@page {{ size: letter; margin: 0.75in 0.625in 1in 0.625in; }}
html {{ font-family: 'STIX Two Text', 'Noto Sans Devanagari', 'Noto Sans Tamil', serif; font-size: 10pt; line-height: 1.18; color: #000; }}
body {{ margin: 0; }}
.head {{ text-align: center; margin-bottom: 14pt; }}
h1 {{ font-size: 22pt; font-weight: 400; line-height: 1.15; margin: 0 0 10pt; }}
.authors {{ font-size: 11pt; line-height: 1.35; }}
.cols {{ column-count: 2; column-gap: 0.25in; text-align: justify; }}
p {{ margin: 0 0 5pt; text-indent: 1em; hyphens: auto; }}
p.abs {{ text-indent: 0; font-size: 9pt; }} p.abs b {{ font-weight: 600; }}
h2 {{ font-size: 10pt; font-weight: 400; font-variant: small-caps; text-align: center; margin: 9pt 0 5pt; break-after: avoid; }}
.eq {{ text-align: center; text-indent: 0; margin: 4pt 0; }}
table.tb {{ border-collapse: collapse; margin: 6pt auto 8pt; font-size: 8pt; break-inside: avoid; width: 100%; }}
table.tb caption {{ font-size: 8pt; font-variant: small-caps; margin-bottom: 3pt; line-height: 1.25; }}
table.tb caption span {{ font-variant: small-caps; }}
table.tb thead th {{ border-top: 1pt solid #000; border-bottom: 0.6pt solid #000; padding: 2pt 3pt; font-weight: 600; }}
table.tb td {{ padding: 1.6pt 3pt; }} table.tb tbody tr:last-child td {{ border-bottom: 1pt solid #000; }}
table.tb tr.mid td {{ border-top: 0.5pt solid #000; padding: 0; }}
figure {{ margin: 6pt 0 8pt; break-inside: avoid; }} figure img {{ display: block; margin: 0 auto; }}
figcaption {{ font-size: 8pt; margin-top: 3pt; text-align: justify; }}
ol.refs {{ font-size: 8pt; padding-left: 0; list-style: none; counter-reset: r; text-align: left; }}
ol.refs li {{ counter-increment: r; position: relative; padding-left: 20pt; margin-bottom: 2pt; }}
ol.refs li::before {{ content: '[' counter(r) ']'; position: absolute; left: 0; }}
</style></head><body>{body}</body></html>'''


def find_chrome():
    cands = [os.environ.get('CHROME'), '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
             shutil.which('google-chrome'), shutil.which('chromium')]
    cands += sorted(str(p) for p in Path('/opt/pw-browsers').glob('chromium-*/chrome-linux*/chrome'))
    return next(c for c in cands if c and Path(c).exists())


def main():
    with tempfile.TemporaryDirectory() as d:
        page = Path(d) / 'paper.html'
        page.write_text(build(), encoding='utf-8')
        sandbox = ['--no-sandbox'] if hasattr(os, 'geteuid') and os.geteuid() == 0 else []
        subprocess.run([find_chrome(), *sandbox, '--headless=new', '--disable-gpu', '--no-pdf-header-footer',
                        '--virtual-time-budget=15000', f'--print-to-pdf={OUT}', page.as_uri()],
                       check=True, capture_output=True, timeout=180)
    print('Paper written to', OUT)


if __name__ == '__main__':
    main()
