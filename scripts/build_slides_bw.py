"""
IndicDoc AI - Phase-2 deck in the department's template format: black and white, 16:9 widescreen slides,
centred serif titles, bullet lists and plain bordered tables, the same 18 sections in the same order.

Content and numbers are the same as the colour deck (imported from build_slides.py, which reads outputs/metrics);
figures are converted to greyscale. Writes an editable PPTX and a PDF (LibreOffice).

Usage: python scripts/build_slides_bw.py
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_slides as C  # noqa: E402  (content and measured numbers of the Phase-2 deck)

OUT = C.OUT / 'IndicDoc_AI_Phase2_Implementation_BW.pptx'
FONT, MONO = 'Times New Roman', 'Courier New'
BLACK, WHITE = RGBColor(0, 0, 0), RGBColor(255, 255, 255)
W, H = 13.333, 7.5
L, R = 0.5, 0.5
CW = W - L - R
TMP = Path(tempfile.mkdtemp())


def f3(v):
    return f'{v:.3f}'


def pct(v):
    return f'{100 * v:.0f}%'


def strip(s: str) -> str:
    return re.sub(r'<[^>]+>', '', s).replace('&amp;', '&')


def runs(par, text, size, bold=False, font=FONT):
    """Text with **bold** spans."""
    for k, part in enumerate(re.split(r'\*\*', text)):
        if not part:
            continue
        r = par.add_run()
        r.text = part
        r.font.size, r.font.name, r.font.color.rgb = Pt(size), font, BLACK
        r.font.bold = bold or (k % 2 == 1)


def title(s, text):
    tb = s.shapes.add_textbox(Inches(L), Inches(0.25), Inches(CW), Inches(0.7))
    p = tb.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    runs(p, text, 30)
    return tb


def bullets(s, items, top, height, size=13, left=L, width=CW, gap=3):
    """items: str, (str, level), '' for an empty line, or ('#', str) for a numbered list item."""
    tb = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    first = True
    for it in items:
        level, text, numbered = 0, it, None
        if isinstance(it, tuple):
            if it[0] == '#':
                numbered, text = it[1], it[2]
            else:
                text, level = it
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(gap)
        if text == '':
            runs(p, ' ', size)
            continue
        sz = size - 1.5 * level
        indent = 0.25 + 0.3 * level
        pPr = p._p.get_or_add_pPr()
        pPr.set('marL', str(int(Inches(indent))))
        pPr.set('indent', str(int(-Inches(0.22 if numbered is None else 0.32))))
        if numbered is None:
            bu = pPr.makeelement(qn('a:buChar'), {'char': '•'})
        else:
            bu = pPr.makeelement(qn('a:buAutoNum'), {'type': 'arabicPeriod', 'startAt': str(numbered)})
        pPr.append(bu)
        runs(p, text, sz)
    return tb


def _border(cell, w=Pt(0.75)):
    tcPr = cell._tc.get_or_add_tcPr()
    for tag in ('a:lnL', 'a:lnR', 'a:lnT', 'a:lnB'):
        ln = tcPr.makeelement(qn(tag), {'w': str(int(w)), 'cmpd': 'sng'})
        fill = ln.makeelement(qn('a:solidFill'), {})
        clr = fill.makeelement(qn('a:srgbClr'), {'val': '000000'})
        fill.append(clr)
        ln.append(fill)
        tcPr.append(ln)


def table(s, head, rows, top, widths, size=10, left=L, row_h=0.28, bold_first_col=False):
    n_r, n_c = len(rows) + 1, len(head)
    shape = s.shapes.add_table(n_r, n_c, Inches(left), Inches(top), Inches(sum(widths)), Inches(row_h * n_r))
    tbl = shape.table
    tbl.first_row = False
    tbl.horz_banding = False
    tblPr = tbl._tbl.tblPr
    for el in tblPr.findall(qn('a:tableStyleId')):
        tblPr.remove(el)
    for j, w in enumerate(widths):
        tbl.columns[j].width = Inches(w)
    for i in range(n_r):
        tbl.rows[i].height = Inches(row_h)
        for j in range(n_c):
            cell = tbl.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE
            _border(cell)
            cell.margin_left = cell.margin_right = Inches(0.04)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            text = head[j] if i == 0 else str(rows[i - 1][j])
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if i == 0 else PP_ALIGN.LEFT
            runs(p, text, size, bold=(i == 0) or (bold_first_col and j == 0))
    return shape


def grey(path: Path) -> Path:
    out = TMP / (path.stem + '_bw.png')
    if not out.exists():
        im = Image.open(path).convert('L')
        im.save(out)
    return out


def image(s, path, left, top, width=None, height=None, frame=True):
    p = s.shapes.add_picture(str(grey(Path(path))), Inches(left), Inches(top),
                             Inches(width) if width else None, Inches(height) if height else None)
    if frame:
        p.line.color.rgb = BLACK
        p.line.width = Pt(0.5)
    return p


def figure(s, path, left, top, width=None, height=None, cap='', cap_size=9.5):
    """Picture with its caption directly below; returns the bottom edge in inches."""
    pic = image(s, path, left, top, width, height)
    bottom = (pic.top + pic.height) / 914400
    if cap:
        caption(s, cap, pic.left / 914400 - 0.1, bottom + 0.03, pic.width / 914400 + 0.2, size=cap_size)
        bottom += 0.35
    return bottom


def caption(s, text, left, top, width, size=9.5, italic=True):
    tb = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(0.3))
    tb.text_frame.word_wrap = True
    p = tb.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    runs(p, text, size)
    for r in p.runs:
        r.font.italic = italic
    return tb


def box(s, x, y, w, h, head, sub='', size=10.5, fill=WHITE):
    sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = BLACK
    sh.line.width = Pt(1)
    sh.shadow.inherit = False
    tf = sh.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.04)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    runs(p, head, size, bold=True)
    if sub:
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.CENTER
        runs(p2, sub, size - 2)
    return sh


def arrow(s, x1, y1, x2, y2):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = BLACK
    c.line.width = Pt(1.25)
    ln = c.line._get_or_add_ln()
    ln.append(ln.makeelement(qn('a:tailEnd'), {'type': 'triangle', 'w': 'med', 'len': 'med'}))
    return c


def code_box(s, text, x, y, w, h, size=9):
    sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = WHITE
    sh.line.color.rgb = BLACK
    sh.line.width = Pt(0.75)
    sh.shadow.inherit = False
    tf = sh.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    for k, line in enumerate(text.split('\n')):
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        runs(p, line or ' ', size, font=MONO)


def main():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    blank = prs.slide_layouts[6]
    new = lambda: prs.slides.add_slide(blank)  # noqa: E731
    A, P, B, V, VN = C.A, C.P, C.B, C.V, C.VN
    QAP, cq, dq, q_ap, tot = C.QAP, C.clean_q, C.deg_q, C.q_ap, C.tot
    T = 1.15            # top of the body
    HALF = (CW - 0.4) / 2
    X2 = L + HALF + 0.4  # left edge of the right column

    # 1. Title slide
    s = new()
    title(s, 'IndicDoc Verify')
    bullets(s, [
        '**Project title:** IndicDoc Verify: Deep-Learning Layout Analysis for Pre-Print Checking of Indian Question '
        'Papers and Their Translations',
        '**Team:** ' + ', '.join(f'{n} ({r})' for _, n, r in sorted(C.TEAM, key=lambda t: t[2])),
        '**Course:** DL Mini Project (Expt. 11, SDG-based)', '',
        'Department of Artificial Intelligence and Machine Learning',
        'St. Francis Institute of Technology (SFIT)', '',
        f'**Project guide:** {C.GUIDE}',
        '**Academic year / semester:** 2026–27, Semester V (TE AIML)',
    ], T + 0.3, 5.5, size=20, gap=6)

    # 2. Content
    s = new()
    title(s, 'Content')
    bullets(s, [('#', k, t) for k, t in enumerate(C.items[:8], 1)], T + 0.2, 5.5, size=20, left=L + 0.8, width=HALF, gap=6)
    bullets(s, [('#', k, t) for k, t in enumerate(C.items[8:], 9)], T + 0.2, 5.5, size=20, left=X2, width=HALF, gap=6)

    # 3. Literature survey summary table
    s = new()
    title(s, 'Literature Survey Summary Table')
    table(s, ['Ref', 'Author', 'Problem', 'Dataset', 'Method', 'Limitation'], C.lit, T,
          [0.5, 1.6, 2.3, 2.6, 2.6, 2.73], size=9.5, row_h=0.56)

    # 4. Research gaps
    s = new()
    title(s, 'Research Gaps')
    table(s, ['#', 'Gap in existing work', 'How this project addresses it'],
          [(c, strip(a), strip(b)) for a, b, c in C.gaps], T + 0.1, [0.6, 5.85, 5.88], size=13, row_h=0.9)
    bullets(s, ['Court e-filing checks already exist (Supreme Court + IIT Madras), so we chose what **nobody automates '
                'yet**: checking a question paper and its translations.'], 6.45, 0.8, size=14)

    # 5. Problem statement and objectives
    s = new()
    title(s, 'Problem Statement and Objectives')
    bullets(s, [
        '**Problem:** exam papers are translated into Hindi and regional languages and checked only by hand; errors '
        'reach lakhs of candidates.',
        '**Main objective:** an offline DL system that flags errors in a paper and every difference from its source '
        'version, before printing.',
        '**Specific objectives:**',
        ('Detect questions, options, answers with a layout detector (compare 3).', 1),
        ('Read text with reliable numbers in 12 languages.', 1),
        ('Check numbering, duplicates, options, marks, parity, repeats.', 1),
        ('Measure on a benchmark; ship app + API in Docker.', 1),
        '**Expected output:** verdict + each finding with question and page (HTML / JSON).',
    ], T, 6, size=15, width=HALF + 0.6, gap=5)
    caption(s, 'It has already happened', X2 + 0.6, T, HALF - 0.6, size=14, italic=False)
    table(s, ['Exam', 'What went wrong'], C.cases, T + 0.45, [1.6, HALF - 2.2], size=11, left=X2 + 0.6, row_h=0.75,
          bold_first_col=True)

    # 6. Proposed system, methodology and novelty
    s = new()
    title(s, 'Proposed System, Methodology and Novelty')
    bullets(s, ['**Workflow (input to output):**', *[('#', k, f'**{b}:** {c}') for k, (_, b, c) in enumerate(C.steps, 1)]],
            T, 6, size=16, width=HALF, gap=6)
    bullets(s, [
        f'**Deep learning:** DocLayout-YOLO finds every question, option and answer (AP@50 {min(q_ap):.2f}–{max(q_ap):.2f}).',
        '**Novelty 1 – parity without translation:** numbers, options, marks and figures must match in every language.',
        '**Novelty 2 – number pass:** two OCR models must agree on every digit.',
        '**USP:** the only tool we found that checks a paper against its translations, offline, in 12 languages.',
    ], T, 6, size=16, left=X2, width=HALF, gap=8)

    # 7. System block diagram
    s = new()
    title(s, 'System Block Diagram')
    lanes = [('User', [('Streamlit app', 'Verify Paper tab'), ('REST API', 'POST /verify'), ('Docker', 'offline')]),
             ('Input', [('Render pages', 'PDF at 200 dpi'), ('Source version', 'optional'), ('Bank, profile', 'old papers, rules')]),
             ('Deep learning', [('Layout detector', 'DocLayout-YOLO'), ('Post-process', 'threshold, de-duplicate'),
                                ('Question regions', 'questions, options')]),
             ('Text', [('Indic OCR', 'Tesseract'), ('Number pass', 'digits must agree'), ('Build questions', 'stem, options, marks')]),
             ('Checks, output', [('Structure, repeats', 'numbering, marks, bank'), ('Translation parity', 'align versions'),
                                 ('Report', 'verdict, HTML / JSON')])]
    y, bw, bh, gap, x0 = T, 3.1, 0.78, 0.38, 2.6
    for k, (lane, blocks) in enumerate(lanes):
        lb = s.shapes.add_textbox(Inches(L + 0.3), Inches(y), Inches(1.8), Inches(bh))
        lb.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        runs(lb.text_frame.paragraphs[0], lane.upper(), 12, bold=True)
        for i, (hd, sub) in enumerate(blocks):
            x = x0 + i * (bw + 0.45)
            box(s, x, y, bw, bh, hd, sub, size=13, fill=RGBColor(0xE6, 0xE6, 0xE6) if lane == 'Deep learning' else WHITE)
            if i:
                arrow(s, x - 0.45, y + bh / 2, x, y + bh / 2)
        if k < len(lanes) - 1:
            xm = x0 + (bw + 0.45) + bw / 2
            arrow(s, xm, y + bh, xm, y + bh + gap)
        y += bh + gap
    caption(s, 'Grey: deep-learning stage. The source version follows the same path.', L, y - 0.2, CW, size=11)

    # 8. Dataset description and pre-processing
    s = new()
    title(s, 'Dataset Description and Pre-processing')
    bullets(s, [
        '**Dataset and source:** IndicDLP (ICDAR 2025), Hugging Face / AIKosh; 12 languages, 42 labels.',
        f'**Samples:** {tot["train"] + tot["val"] + tot["test"]:,} pages, {C.total_boxes:,} boxes, 42 classes.',
        f'**Split:** train {tot["train"]:,} / val {tot["val"]} / test {tot["test"]} pages.',
        f'**Benchmark:** {C.n_pairs} English–Hindi/Tamil paper pairs, {C.n_err} with one injected error.',
        '**Pre-processing:** COCO → YOLO labels, 640 / 1024 px; PDFs at 200 dpi.',
        '**Augmentation:** mosaic, HSV, scale (YOLO); jitter, blur (R-CNN).',
        '**Challenges:** class imbalance (paragraph ≈ 32%); no public data of real errors.',
    ], T, 6, size=15, width=HALF - 0.1, gap=6)
    for i, (pth, cap) in enumerate([(C.FIG / 'bench_en.jpg', 'English source'), (C.FIG / 'bench_hi.jpg', 'Hindi version'),
                                    (C.ROOT / 'dataset' / 'samples' / 'hindi_newspapers.jpg', 'IndicDLP page')]):
        figure(s, pth, X2 + 0.1 + i * 2.02, T + 0.3, height=2.8, cap=cap, cap_size=10)
    caption(s, 'Actual data used', X2, T + 3.75, 6.2, size=12, italic=False)

    # 9. Deep learning model architecture
    s = new()
    title(s, 'Deep Learning Model Architecture')
    bullets(s, [f'**Selected:** DocLayout-YOLO, {A.get("parameters", 0) / 1e6:.1f} M params, pretrained on IndicDLP by its '
                'authors. **Input** page image → **output** boxes with 42 labels. **Modified:** 42-class heads.'],
            T - 0.1, 0.8, size=14)
    archs = [(f'DocLayout-YOLO, {A.get("parameters", 0) / 1e6:.1f} M (used)',
              [('Input', '1024² page'), ('Backbone', 'CSP + GL-CRM'), ('Neck', 'PAN'), ('Head', 'NMS-free'), ('Output', '42 classes')]),
             (f'YOLOv8s, {P.get("parameters", 0) / 1e6:.1f} M (ours)',
              [('Input', '640² page'), ('Backbone', 'CSPDarknet C2f'), ('Neck', 'PAN-FPN'), ('Head', 'anchor-free'), ('Output', '42 classes')]),
             (f'Faster R-CNN, {B.get("parameters", 0) / 1e6:.1f} M (baseline)',
              [('Input', '640–1024 px'), ('Backbone', 'ResNet-50'), ('Neck', 'FPN'), ('RPN + RoIAlign', 'proposals'), ('Box head', '42 + bg')])]
    y, bw, gap = T + 0.85, 2.1, 0.45
    for name, parts in archs:
        tb = s.shapes.add_textbox(Inches(L), Inches(y), Inches(CW), Inches(0.3))
        runs(tb.text_frame.paragraphs[0], name, 13, bold=True)
        for i, (hd, sub) in enumerate(parts):
            x = L + 0.2 + i * (bw + gap)
            box(s, x, y + 0.38, bw, 0.72, hd, sub, size=13, fill=RGBColor(0xE6, 0xE6, 0xE6) if 0 < i < 4 else WHITE)
            if i:
                arrow(s, x - gap, y + 0.74, x, y + 0.74)
        y += 1.6

    # 10. Tools, technologies and environment
    s = new()
    title(s, 'Tools, Technologies and Environment')
    table(s, ['Component', 'Details to report'], C.tools, T, [3.0, 9.33], size=13, row_h=0.72)

    # 11. Implementation details
    s = new()
    title(s, 'Implementation Details')
    bullets(s, [
        '**Built:** checker (ml/verify), OCR + number pass, 2 trained detectors, benchmark, app, API.',
        '**Data loading:** IndicDLP Parquet → COCO; PDFs at 200 dpi.',
        '**Model:** pretrained DocLayout-YOLO; YOLOv8s, Faster R-CNN fine-tuned.',
        '**Loss / optimiser:** CIoU + DFL + BCE; SGD lr 0.005, batch 4.',
        '**Inference:** layout → OCR → questions → checks → report.',
        '**Interface:** Streamlit tab + POST /verify.',
    ], T, 6, size=15, width=HALF, gap=7)
    code = ('out = predict_single(model, page, "doclayout")\n'
            'doc = extract_document(page, out, "hin",\n'
            '                       number_pass=True)\n'
            'report = verify(hindi_pages,\n'
            '                source_pages=english_pages)')
    code_box(s, code, X2, T, HALF, 1.45, size=11)
    figure(s, C.SHOTS / '07_verify_report.png', X2 + 0.6, T + 1.65, width=HALF - 1.2, cap='Report produced by the system')

    # 12. Experimental setup and training
    s = new()
    title(s, 'Experimental Setup and Training')
    bullets(s, [
        f'**Data:** train {tot["train"]:,} / val {tot["val"]} / test {tot["test"]}; benchmark {C.n_pairs} + 48 dev pairs.',
        '**Faster R-CNN:** batch 4, SGD 0.005, 2 epochs, 76 min. **YOLOv8s:** 640 px, batch 8, 12 epochs, 60 min (M1 Pro GPU).',
        '**Experiments:** layout mAP; robustness to scan damage; verification benchmark; ablations.',
    ], T, 2, size=15, gap=5)
    figure(s, C.FIG / 'fig_training.png', L + 1.4, T + 1.75, width=CW - 2.8, cap='Training / validation curves (measured)')

    # 13. Results and performance evaluation
    s = new()
    title(s, 'Results and Performance Evaluation')

    def lrow(n, m):
        rc = m['at_recommended_conf']
        return (n, f3(m['map50']), f3(m['map50_95']), f3(rc['precision']), f3(rc['recall']), f3(rc['f1']))
    table(s, ['Model', 'mAP@50', 'mAP@50:95', 'Precision', 'Recall', 'F1'],
          [lrow('DocLayout-YOLO', A), lrow('YOLOv8s (ours)', P), lrow('Faster R-CNN (ours)', B)],
          T, [1.95, 0.85, 1.0, 0.9, 0.75, 0.52], size=12, row_h=0.4)
    bullets(s, [
        f'**{pct(cq.get("found_at_place", 0))}** of injected errors found at the right question (born-digital); '
        f'{pct(dq.get("found_at_place", 0))} on scans.',
        f'**{cq.get("false_alarms_per_clean_pair", 0):.1f}** false alarms per clean pair (born-digital).',
        f'**{C.n_exact}/{C.n_pairs}** papers: every question recovered.',
        f'Question labels AP@50: {min(q_ap):.2f}–{max(q_ap):.2f}.',
    ], T + 1.85, 4, size=15, width=HALF + 0.2, gap=6)
    figure(s, C.FIG / 'fig_verify_by_type.png', X2 + 0.2, T, width=HALF - 0.2, cap='Errors found per error type')

    # 14. Comparison with baseline / existing methods
    s = new()
    title(s, 'Comparison with Baseline / Existing Methods')
    table(s, ['Detector', 'mAP@50', 'mAP@50:95'],
          [('Faster R-CNN (baseline)', f3(B['map50']), f3(B['map50_95'])), ('YOLOv8s (ours)', f3(P['map50']), f3(P['map50_95'])),
           ('DocLayout-YOLO (used)', f3(A['map50']), f3(A['map50_95']))], T, [3.2, 1.4, 1.37], size=13, row_h=0.42)
    table(s, ['Verification', 'Found', 'False alarms'],
          [(n, pct(r.get('overall', {}).get('found_at_place', 0)), f"{r.get('overall', {}).get('false_alarms_per_clean_pair', 0):.1f}")
           for n, r in (('Full system', V), ('No number pass', C.V_NP), ('No layout options', C.V_NG), ('Noisy layout', VN)) if r],
          T + 2.0, [3.2, 1.4, 1.37], size=13, row_h=0.42)
    table(s, ['Existing approach', 'Checks paper?', 'Compares versions?'],
          [('Manual vetting', 'by hand', 'by hand'), ('Answer-sheet AI', 'no', 'no'), ('MT quality estimation', 'no structure', 'sentences'),
           ('**IndicDoc Verify**', '**yes**', '**question by question**')],
          T, [2.4, 1.6, 1.97], size=13, left=X2 + 0.15, row_h=0.5)
    bullets(s, ['Metric: COCO mAP (layout); errors found and false alarms (verification).',
                'Proposed approach improves: YOLOv8s > baseline; number pass is the largest gain.'],
            T + 2.9, 2, size=14, left=X2 + 0.15, width=HALF - 0.15)

    # 15. Sample outputs and demonstration
    s = new()
    title(s, 'Sample Outputs and Demonstration')
    y = figure(s, C.SHOTS / '08_verify_page.png', L + 0.4, T, width=HALF - 0.8, cap='**Correct:** changed number found and outlined')
    figure(s, C.SHOTS / '09_false_alarm.png', L + 0.4, y + 0.1, width=HALF - 0.8, cap='**Incorrect:** a false alarm on a degraded scan')
    figure(s, C.SHOTS / '02_detection_tamil.png', X2, T, width=3.3, cap='Running app (Tamil page)', cap_size=10)
    figure(s, C.PLOTS / 'authors_confusion_matrix.png', X2 + 3.5, T, width=2.4, cap='Confusion matrix', cap_size=10)

    # 16. Challenges and limitations
    s = new()
    title(s, 'Challenges and Limitations')
    bullets(s, [
        '**OCR misreads digits** in Indic text ("1947" → "947"): solved with the number pass.',
        '**No public data of real errors:** we built a benchmark with injected errors.',
        '**Meaning is not checked yet:** a dropped word needs a cross-lingual model.',
        '**Limited compute:** our detectors trained ~1 hour on 2.3K pages, so the product uses the released model.',
    ], T + 0.3, 5, size=20, gap=14)

    # 17. Conclusion and future scope
    s = new()
    title(s, 'Conclusion and Future Scope')
    bullets(s, ['**Conclusion**',
                ('Implemented: layout detection, OCR with number pass, checks, report, app, API, Docker.', 1),
                (f'Results: {pct(cq.get("found_at_place", 0))} of errors found; best layout model {f3(A["map50"])} mAP@50.', 1),
                ('All four objectives met.', 1)], T + 0.1, 5.5, size=17, width=HALF, gap=8)
    bullets(s, ['**Future scope**',
                ('Pilot with an exam body on real papers.', 1), ('Meaning check (cross-lingual model).', 1),
                ('Better Indic OCR; answer-key check.', 1), ('More document profiles (circulars, forms).', 1)],
            T + 0.1, 5.5, size=17, left=X2, width=HALF, gap=8)

    # 18. References
    s = new()
    title(s, 'References')
    refs = [strip(r) for r in C.refs]
    bullets(s, [('#', k, r) for k, r in enumerate(refs[:8], 1)], T - 0.05, 5.6, size=10.5, width=HALF, gap=3)
    bullets(s, [('#', k, r) for k, r in enumerate(refs[8:], 9)], T - 0.05, 4.4, size=10.5, left=X2, width=HALF, gap=3)
    bullets(s, ['**Other:** ' + '; '.join(C.other)], 5.65, 1.6, size=10, left=X2, width=HALF)

    prs.save(OUT)
    subprocess.run(['soffice', '--headless', '--convert-to', 'pdf', '--outdir', str(OUT.parent), str(OUT)],
                   check=True, capture_output=True, timeout=300)
    print(f'{OUT}\n{OUT.with_suffix(".pdf")}  ({len(prs.slides)} slides)')


if __name__ == '__main__':
    main()
