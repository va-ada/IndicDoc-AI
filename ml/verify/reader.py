"""
Reader B: a second, layout-free reading of a paper, independent of the deep-learning layout detector (reader A).

Reader A finds questions, options and answers with DocLayout-YOLO and reads them with OCR. Reader B never sees the
detector: it groups the page's words into lines and segments the page with the typographic grammar of exam papers
alone (a question starts with its number in the left column, options are cells that open with a marker, section
headings, instructions, 'OR' / 'अथवा' between internal-choice alternatives, running headers and footers). Its words
come from the PDF's own text layer when the file is born-digital (no OCR at all), otherwise from a separate OCR pass.

Because the two readers fail in different ways (a missed or merged detection in A; a garbled line start in B), the
consensus step (ml/verify/consensus.py) only confirms what both read, and re-reads the rest.

read_page() returns a page in the same form as ml.ocr.extract_document(), so the same parser and checks run on it.
"""
import re
from collections import Counter
from statistics import median
from typing import Dict, List, Optional

from ml.verify.text import (is_choice_line, is_instruction, normalize_digits, page_of, question_number, script_of,
                            section_key, strip_marks)

TOP_BAND, BOTTOM_BAND = 0.06, 0.94    # always furniture outside these (fractions of the page height)
EDGE = 0.12                            # repeated lines this close to the top or bottom edge are furniture
# An option marker at the start of a cell, as printed ('(A)', 'A.', 'a)') or as OCR garbles it ('(8)', '(·)', '(()',
# '())', '(?)', '(8)4' with the value run in)
_MARKER = re.compile(r'^[(\[{]\s*[^\s\[\]{}]{0,3}\s*[)\]}]|^[A-Da-d]\s*[.)]$|^\(?[^\s()]{1,2}\)\S*$|^\(\(|^\(\)')


def _box(w: Dict) -> List[float]:
    if 'box' in w:
        return [float(v) for v in w['box']]
    x, y = float(w['left']), float(w['top'])
    return [x, y, x + float(w['width']), y + float(w['height'])]


def _norm_word(w: Dict) -> Dict:
    """A word from Tesseract ({'left', 'top', 'width', 'height', 'conf'}) or a PDF text layer ({'box'})."""
    x1, y1, x2, y2 = _box(w)
    conf = w.get('conf')
    return {'text': str(w['text']), 'left': x1, 'top': y1, 'width': x2 - x1, 'height': y2 - y1,
            'conf': float(conf) if conf is not None else 99.0}


def lines_of(words: List[Dict]) -> List[List[Dict]]:
    """Words grouped into printed lines (top to bottom, left to right), by their vertical overlap."""
    ws = sorted((_norm_word(w) for w in words if str(w.get('text', '')).strip()), key=lambda w: w['top'])
    if not ws:
        return []
    h = median(w['height'] for w in ws) or 1.0
    lines: List[List[Dict]] = []
    for w in ws:
        cy = w['top'] + w['height'] / 2
        best = None
        for line in lines[-4:]:
            ly = median(x['top'] + x['height'] / 2 for x in line)
            if abs(cy - ly) < 0.55 * max(h, min(w['height'], 2 * h)):
                best = line
        if best is None:
            lines.append([w])
        else:
            best.append(w)
    for line in lines:
        line.sort(key=lambda w: w['left'])
    lines.sort(key=lambda line: min(w['top'] for w in line))
    return lines


def _cells(line: List[Dict], h: float) -> List[List[Dict]]:
    cells = [[line[0]]]
    for prev, w in zip(line, line[1:]):
        if w['left'] - (prev['left'] + prev['width']) > 2.0 * h:
            cells.append([w])
        else:
            cells[-1].append(w)
    return cells


def _text(words: List[Dict]) -> str:
    return ' '.join(w['text'] for w in words)


def _is_option_line(line: List[Dict], h: float, strict: bool = False) -> bool:
    """
    Every cell of the line opens with an option marker ('(A) 50   (B) 100'), or the line is one marked option.
    strict: at least two cells each opening with a marker (a single marked line may be a sub-part instead).
    """
    cells = [c for c in _cells(line, h) if not _MARKS_CELL.match(_text(c))]
    if not cells:
        return False
    marked = all(_MARKER.match(normalize_digits(c[0]['text'])) for c in cells)
    if strict:
        return marked and len(cells) >= 2
    return marked and (len(cells) >= 2 or len(line) >= 2)


# A sub-part marker: '(a)', '(b)', '(i)', '(ii)', '(iii)', '(iv)' (and 'a.' / 'i.'); MCQ options use '(A)'-'(D)'
_SUB_MARKER = re.compile(r'^\s*\(?(?:[a-f]|i{1,3}|iv|v|vi)\)\s+\S|^\s*(?:[a-f]|i{1,3}|iv|v)\.\s+\S')
_MARKS_CELL = re.compile(r'^\s*[\[({]?\s*\d{0,2}(?:\.5)?\s*(?:marks?|अंक)?\s*[\])}]\s*$|^\s*[\[{]\s*\d{1,2}\s*$',
                        re.IGNORECASE)


def _region(label: str, lines: List[List[Dict]], **extra) -> Dict:
    words = [w for line in lines for w in line]
    box = [min(w['left'] for w in words), min(w['top'] for w in words),
           max(w['left'] + w['width'] for w in words), max(w['top'] + w['height'] for w in words)]
    return {'label': label, 'box': [round(v, 1) for v in box], 'score': None, 'detected': False,
            'text': '\n'.join(_text(line) for line in lines), 'words': words, **extra}


_LOOSE_NUMBER = re.compile(r'^\s*[(\[]?(\d{1,3})\s*[.,)\]।:]\s*(?=\S)')


def _skeleton(text: str) -> str:
    """A furniture line with its digits removed: 'Page 8 of 24' and 'Page 9 of 24' are the same footer."""
    return re.sub(r'\s+', ' ', re.sub(r'[0-9\u0966-\u096F]+', '#', normalize_digits(text))).strip().lower()


def _wordy(skeleton: str) -> bool:
    """A repeated edge line is furniture only if it has words: '(क) # (ख) #' (options with their values removed)
    repeats on every page too."""
    letters = re.sub(r'\([^)\s]{1,4}\)|[^\w]|[#_\d]', '', skeleton)
    return len(letters) >= 3


def split_columns(words: List[Dict], width: float) -> List[List[Dict]]:
    """
    The words of a page in reading order of its columns: [left, right] for a two-column paper (NEET, JEE and many
    state papers), [all] otherwise. Two columns are recognised by a vertical gutter in the middle third of the page
    that no printed line crosses, with text on both sides of it.
    """
    ws = [_norm_word(w) for w in words if str(w.get('text', '')).strip()]
    if len(ws) < 40:
        return [ws]
    lines = lines_of(ws)
    bins = 60
    lo, hi = int(bins / 3), int(2 * bins / 3)
    crossing = [0] * bins
    for line in lines:
        for w in line:
            a, b = int(w['left'] / width * bins), int((w['left'] + w['width']) / width * bins)
            for k in range(max(0, a), min(bins, b + 1)):
                crossing[k] += 1
    centre = min(range(lo, hi), key=lambda k: crossing[k])
    gutter = centre * width / bins
    left = [w for w in ws if w['left'] + w['width'] / 2 < gutter]
    right = [w for w in ws if w['left'] + w['width'] / 2 >= gutter]
    if crossing[centre] > max(2, 0.03 * len(lines)) or min(len(left), len(right)) < 0.25 * len(ws):
        return [ws]
    # A line of one column runs across the gutter with ordinary word spacing; two columns leave a wide gap there
    h = median(w['height'] for w in ws) or 1.0
    spanning = 0
    for line in lines:
        before = [w for w in line if w['left'] + w['width'] <= gutter]
        after = [w for w in line if w['left'] >= gutter]
        if before and after and after[0]['left'] - (before[-1]['left'] + before[-1]['width']) < 2.0 * h:
            spanning += 1
    if spanning > max(1, 0.05 * len(lines)):
        return [ws]

    def numbered(col):
        return sum(question_number(strip_marks(_text(ln)))[0] is not None for ln in lines_of(col))
    # Option grids leave a gap in the middle of a one-column page too; only a paper whose questions start in both
    # halves has two columns
    two = numbered(left) >= 1 and numbered(right) >= 1
    return [left, right] if two else [ws]


def read_pages(pages: List[Dict], source: str = 'ocr') -> List[Dict]:
    """
    Every page of a paper read without the layout model. pages: [{'words': [...], 'size': (w, h)}, ...] with the
    words in pixel coordinates (Tesseract rows from ml.ocr.read_words, or PDF text-layer words with a 'box').
    Running headers and footers are lines near the top or bottom edge whose text (digits aside) repeats on other
    pages; question numbers are read with the number expected next in mind ('14,' is question 14 after 13).
    """
    all_lines = []
    for p in pages:
        cols = split_columns(p['words'], p['size'][0])
        # left column, then the right one; each line remembers its column (question numbers sit at its left edge)
        all_lines.append([[{**w, 'col': c} for w in ln] for c, col in enumerate(cols) for ln in lines_of(col)])
    edge = Counter()
    for p, lines in zip(pages, all_lines):
        height = p['size'][1]
        seen = {_skeleton(_text(ln)) for ln in lines
                if ln[0]['top'] < EDGE * height or ln[0]['top'] > (1 - EDGE) * height}
        edge.update(seen)
    repeated = {k for k, c in edge.items() if c >= max(2, 0.3 * len(pages)) and _wordy(k)}
    out, expected = [], None
    for p, lines in zip(pages, all_lines):
        page, expected = _read_lines(lines, p['size'], repeated, expected, source)
        out.append(page)
    return out


def read_page(words: List[Dict], size, source: str = 'ocr') -> Dict:
    """One page (or a crop) read without the layout model; see read_pages()."""
    return _read_lines(lines_of(words), size, set(), None, source)[0]


def _read_lines(lines, size, repeated, expected, source):
    width, height = size
    if not lines:
        return {'script': None, 'languages': None, 'regions': [], 'reader': 'B', 'source': source}, expected
    h = median(w['height'] for line in lines for w in line) or 1.0

    def furniture(line):
        top, text = line[0]['top'], _text(line)
        if page_of(text):
            return 'page-number'
        near_edge = top < EDGE * height or top > (1 - EDGE) * height
        # a numbered question or an option line is content, however close to the edge a tight page prints it
        content = question_number(text)[0] is not None or bool(_MARKER.match(text))
        if near_edge and not content and (_skeleton(text) in repeated or top < TOP_BAND * height
                                          or top > BOTTOM_BAND * height):
            return 'header' if top < height / 2 else 'footer'
        return None

    body = [ln for ln in lines if not furniture(ln)]
    margins = {}
    for c in {ln[0].get('col', 0) for ln in lines}:
        col = [ln for ln in body if ln[0].get('col', 0) == c]
        margins[c] = sorted(ln[0]['left'] for ln in col)[len(col) // 10] if col else 0.0
    regions: List[Dict] = []
    block: List[List[Dict]] = []
    kind = None                 # kind of the open text block: 'question', 'instruction' or 'text'
    open_q = False              # a question is open (its options and continuation lines follow)

    def flush():
        nonlocal block, kind
        if block:
            regions.append(_region('first-level-question' if kind == 'question' else 'paragraph', block))
        block, kind = [], None

    for line in lines:
        text = _text(line)
        f = furniture(line)
        if f:
            flush()
            regions.append(_region(f, [line]))
            continue
        if section_key(text) and len(text.split()) <= 4:
            flush()
            regions.append(_region('section-title', [line]))
            open_q = False
            continue
        if is_instruction(text):
            flush()
            block, kind, open_q = [line], 'instruction', False
            continue
        number, _ = question_number(strip_marks(text))
        if number is None and expected is not None:
            m = _LOOSE_NUMBER.match(normalize_digits(text))
            # '(3) Oxygen (4) Hydrogen' under a question whose options are numbered: options, not question 3
            bracketed_option = open_q and normalize_digits(text).lstrip().startswith('(') and _is_option_line(line, h)
            if m and int(m.group(1)) == expected and not bracketed_option:
                number = expected
                # '14,' (a comma read for the full stop): written back as '14.' so the parser reads the same number
                if re.fullmatch(r'[(\[]?\d{1,3}\s*[.,)\]।:]', normalize_digits(line[0]['text'])):
                    line = [{**line[0], 'text': f'{number}.'}] + line[1:]
        if number is not None and line[0]['left'] <= margins.get(line[0].get('col', 0), 0.0) + 2.5 * h and number <= 200 \
                and (expected is None or 0 < number <= expected + 3 or number == 1):
            flush()
            block, kind, open_q = [line], 'question', True
            expected = number + 1
            continue
        if is_choice_line(text):
            flush()
            regions.append(_region('paragraph', [line]))
            continue
        marks_cell = len(line) >= 2 and _MARKS_CELL.match(_cells(line, h)[-1][-1]['text'] if _cells(line, h) else '')
        if open_q and (marks_cell or _SUB_MARKER.match(normalize_digits(text))) \
                and not _is_option_line(line, h, strict=True):
            # '(b) ...' after 'OR', '(i) ... [1]' in a case study: a part of the question, not an option
            flush()
            regions.append(_region('second-level-question', [line]))
            continue
        if open_q and _is_option_line(line, h):
            flush()
            if regions and regions[-1]['label'] == 'options':
                prev = regions.pop()
                regions.append(_region('options', [prev['words'], line]))
            else:
                regions.append(_region('options', [line]))
            continue
        if not block:
            kind = 'text'
        block.append(line)
    flush()
    script = script_of(' '.join(r['text'] for r in regions if r['label'] not in ('header', 'footer', 'page-number')))
    return ({'script': script, 'languages': None, 'regions': regions, 'reader': 'B', 'source': source,
             'words_on_page': sum(len(r['words']) for r in regions)}, expected)


def text_layer_usable(pdf_words: List[Dict], ocr_script: Optional[str]) -> bool:
    """
    A PDF's text layer replaces OCR only when it really holds the page's text: enough words, and in the script the
    page image shows. Hindi PDFs set in legacy (non-Unicode) fonts carry Latin garbage instead of Devanagari.
    """
    if len(pdf_words) < 20:
        return False
    script = script_of(' '.join(str(w['text']) for w in pdf_words))
    return ocr_script is None or script == ocr_script
