"""
OCR for detected layout regions, with Tesseract (Smith, "An Overview of the Tesseract OCR Engine", ICDAR 2007).

The page is read ONCE with Tesseract (word boxes in TSV form); each word is then assigned to the detected layout
region that contains its centre. One call per page instead of one per region keeps a 60-region newspaper page fast,
and Tesseract sees whole lines, which it reads better than tight crops.

Models: tessdata_best for the 12 IndicDLP languages, kept in outputs/models/tessdata (download with
scripts/download_models.py). The page's script is detected by reading one text region with each script's model
and keeping the most confident reading (detect_script_by_probe).
"""
import io
import os
import re
import shutil
import subprocess
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from ml.config import CLASS_NAMES, MODEL_DIR

TESSDATA_DIR = MODEL_DIR / 'tessdata'

# Display name -> Tesseract code, for the 12 IndicDLP languages
OCR_LANGUAGES = {'English': 'eng', 'Hindi': 'hin', 'Marathi': 'mar', 'Bengali': 'ben', 'Assamese': 'asm',
                 'Gujarati': 'guj', 'Punjabi': 'pan', 'Kannada': 'kan', 'Malayalam': 'mal', 'Tamil': 'tam',
                 'Telugu': 'tel', 'Odia': 'ori'}

# Script reported by Tesseract OSD -> languages written in it (English is added: Indian documents mix it in)
_SCRIPT_LANGUAGES = {'Devanagari': 'hin+mar', 'Bengali': 'ben+asm', 'Gujarati': 'guj', 'Gurmukhi': 'pan',
                     'Kannada': 'kan', 'Malayalam': 'mal', 'Oriya': 'ori', 'Tamil': 'tam', 'Telugu': 'tel'}

# Regions that are not plain text: OCR of pictures and mathematics only produces junk
_NO_TEXT = {'figure', 'flag', 'formula'}
# Page furniture: kept, but listed after the main content in Markdown
_FURNITURE = {'header', 'footer', 'page-number', 'folio', 'footnote', 'jumpline', 'website-link', 'contact-info'}
_HEADINGS = {'headline': '#', 'chapter-title': '#', 'section-title': '##', 'sub-headline': '##',
             'sub-section-title': '###', 'subsub-headline': '###', 'subsub-section-title': '####'}
# Bullet glyphs Tesseract reads in front of list items ('«' is its usual misreading of a round bullet)
_BULLET = re.compile(r'^[\s«»•·●○◦▪■□►▸‣⁃\-\*–—+]+')
# Start of a numbered item: '1.', '2)', '(a)', 'iv.' ...
_NUMBERED = re.compile(r'^\s*(\(?\d{1,3}[.)]|\([a-zA-Z]{1,4}\)|[a-zA-Z][.)]\s|[ivxIVX]{1,4}[.)])')


def _list_items(text: str) -> List[str]:
    """
    Split an OCR'd list into items: a line starting with a bullet glyph or a number opens a new item, other lines
    continue the previous one (wrapped text). If no line has a marker, each line is its own item.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not any(_BULLET.match(l) or _NUMBERED.match(l) for l in lines):
        return lines
    items: List[str] = []
    for line in lines:
        if _BULLET.match(line) or _NUMBERED.match(line) or not items:
            items.append(_BULLET.sub('', line).strip())
        else:
            items[-1] += ' ' + line
    return [i for i in items if i]
_LISTS = {'ordered-list': '', 'unordered-list': '', 'options': '', 'sub-ordered-list': '  ',
          'sub-unordered-list': '  ', 'subsub-ordered-list': '    ', 'subsub-unordered-list': '    '}


def tesseract_available() -> bool:
    return shutil.which('tesseract') is not None and (TESSDATA_DIR / 'eng.traineddata').exists()


def languages_for_script(script: str) -> str:
    """Tesseract language string for a script name from OSD (unknown scripts fall back to English)."""
    langs = _SCRIPT_LANGUAGES.get(script)
    return f'{langs}+eng' if langs else 'eng'


def _run_tesseract(image: np.ndarray, args: List[str], timeout: int = 180, single_thread: bool = False) -> str:
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format='PNG')
    env = {**os.environ, 'TESSDATA_PREFIX': str(TESSDATA_DIR)}
    if single_thread:
        # Tesseract builds with OpenMP (e.g. Debian's) oversubscribe the CPU when several run in parallel
        env['OMP_THREAD_LIMIT'] = '1'
    proc = subprocess.run(['tesseract', 'stdin', 'stdout', *args], input=buf.getvalue(),
                          capture_output=True, env=env, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"tesseract failed: {proc.stderr.decode(errors='replace')[:300]}")
    return proc.stdout.decode('utf-8', errors='replace')


def read_words(image: np.ndarray, languages: str, psm: int = 3, binarize: bool = False,
               single_thread: bool = False, scale: Optional[float] = None) -> List[Dict]:
    """
    Every word Tesseract finds on the page, with its box (page pixels) and line identifiers.
    psm / binarize / scale: page segmentation mode, Otsu binarisation, enlargement factor (default: x2 below 1600 px);
    the verification's second reader reads a scanned page with another configuration (psm 4, enlarged 1.5 times) so
    that its OCR errors are not the first reader's.
    """
    if binarize:
        import cv2
        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        _, bw = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
    h, w = image.shape[:2]
    # Tesseract reads best around 300 dpi; small renders are upscaled and the boxes scaled back
    if scale is None:
        scale = 2 if max(h, w) < 1600 else 1
    if scale != 1:
        image = np.array(Image.fromarray(image).resize((int(w * scale), int(h * scale)), Image.LANCZOS))
    # -c tessedit_create_tsv=1 rather than the 'tsv' config file, which lives outside our tessdata folder
    tsv = _run_tesseract(image, ['-l', languages, '--psm', str(psm), '-c', 'tessedit_create_tsv=1'],
                         single_thread=single_thread)
    rows = tsv.splitlines()
    header = rows[0].split('\t')
    words = []
    for row in rows[1:]:
        cols = row.split('\t')
        if len(cols) != len(header):
            continue
        r = dict(zip(header, cols))
        if r['level'] != '5' or not r['text'].strip():
            continue
        words.append({'text': r['text'].strip(), 'conf': float(r['conf']),
                      'left': int(r['left']) / scale, 'top': int(r['top']) / scale,
                      'width': int(r['width']) / scale, 'height': int(r['height']) / scale,
                      'block_num': int(r['block_num']), 'par_num': int(r['par_num']), 'line_num': int(r['line_num'])})
    return words


# A token that is a number, possibly with brackets, punctuation and a short unit: '1947', '(1)', '24.', '8A', '120°C'
_NUMERIC_TOKEN = re.compile(r'^[(\[]?[0-9][0-9.,:/%°\-]*[)\].,:;]?[A-Za-z°/]{0,4}[)\].,;]?$')
NUMBER_PASS_MIN_CONF = 55
_GLUED_MARKER = re.compile(r'^(?:\(?[^\s()0-9]{1,3}\)|\([0-9]\)|[A-Za-z])(?=[0-9])')
_INDIC_DIGIT = re.compile(r'[\u0966-\u096F\u09E6-\u09EF\u0A66-\u0A6F\u0AE6-\u0AEF\u0B66-\u0B6F\u0BE6-\u0BEF'
                          r'\u0C66-\u0C6F\u0CE6-\u0CEF\u0D66-\u0D6F]')
# Indic letters and vowel signs, without the Indian digits and the danda ('।' is how the Hindi model reads '1')
_INDIC_LETTER = re.compile(r'[\u0900-\u0963\u0970-\u09E5\u09F0-\u0A65\u0A70-\u0AE5\u0AF0-\u0B65\u0B70-\u0BE5'
                           r'\u0BF0-\u0C65\u0C70-\u0CE5\u0CF0-\u0D65\u0D70-\u0D7F]')
_LATIN_UNIT = re.compile(r'^[A-Za-z°/%.]{1,4}$')
_DIGITS_THEN_UNIT = re.compile(r'^(\d{1,6})[.,]?[A-Z]$')   # '24.V', '5A': a number and a one-letter unit
_BARE_NUMBER = re.compile(r'^[0-9][0-9.,]*$')  # Latin digits only ('\d' also matches Indian digits)
_HAS_DIGIT = re.compile(r'[0-9\u0966-\u096F\u09E6-\u09EF\u0A66-\u0A6F\u0AE6-\u0AEF\u0B66-\u0B6F\u0BE6-\u0BEF'
                        r'\u0C66-\u0C6F\u0CE6-\u0CEF\u0D66-\u0D6F]')


def _overlap(a: Dict, b: Dict) -> float:
    """Intersection over the smaller of two word boxes."""
    ix = min(a['left'] + a['width'], b['left'] + b['width']) - max(a['left'], b['left'])
    iy = min(a['top'] + a['height'], b['top'] + b['height']) - max(a['top'], b['top'])
    if ix <= 0 or iy <= 0:
        return 0.0
    return ix * iy / max(1.0, min(a['width'] * a['height'], b['width'] * b['height']))


def _digits(text: str) -> str:
    return ''.join(ch for ch in text if ch.isascii() and ch.isdigit())


KEEP_DIGIT_CONF = 90   # an Indic-model digit this confident stands unless the English model reads something sure there


def _sure_digit(w: Dict, latin_words: List[Dict]) -> bool:
    """A confidently read short number that the English reading does not contradict with a confident token."""
    if w.get('conf', 0) < KEEP_DIGIT_CONF or len(_digits(w['text'])) > 2:
        return False
    return not any(_overlap(w, lw) > 0.3 and lw.get('conf', 0) >= 80 for lw in latin_words)


def merge_number_reading(words: List[Dict], latin_words: List[Dict]) -> List[Dict]:
    """
    Fix the numbers of an Indic-language reading with a second reading by the English model.
    Indic Tesseract models misread Latin digits in mixed text (the Hindi model reads '1947' as '947' and '10' as
    '70', and sometimes reads a letter as a digit); the English model reads Latin digits reliably but not Indic
    letters. So for Latin digits the English reading decides:
      - a numeric token of the English reading replaces the overlapping word of the main reading when that word is
        a number with different digits (a bracket the main reading had, as in an option marker, is kept);
      - it is inserted when the main reading has no word there;
      - a bare Latin number of the main reading that no English numeric token confirms is dropped.
    Indic words and Indian-script digits ('१२०') are never touched.
    """
    out = [dict(w) for w in words]
    # An option marker run into the value is not part of the number: '(3)6A' -> '6A', 'M2A' ('(1) 2A') -> '2A'
    latin_words = [{**lw, 'text': _GLUED_MARKER.sub('', lw['text'])} for lw in latin_words]
    numeric_latin = [lw for lw in latin_words if _NUMERIC_TOKEN.match(lw['text'])]
    # Pages printed with Indian digits ('१२०') keep them: the English model cannot read them. On pages printed with
    # Latin digits, an Indian digit in the Indic reading is a misread ('1A' -> '८') and the English reading decides.
    indic_digits = sum(bool(_INDIC_DIGIT.search(w['text'])) for w in out)
    latin_digits = sum(bool(re.search('[0-9]', w['text'])) for w in out)
    indic_page = indic_digits > latin_digits
    # Any English token with the same digits confirms a number, even when it ran into a marker ('(m30' for '(ग) 30')
    with_digits = [lw for lw in latin_words if _digits(lw['text'])]
    for w in out:
        if _BARE_NUMBER.match(w['text']) and not any(_overlap(w, lw) > 0.3 and _digits(w['text']) in _digits(lw['text'])
                                                     for lw in with_digits) and not _sure_digit(w, latin_words):
            w['text'] = '·'
        # On a page printed with Latin digits, an Indian digit in a word without Indic letters is the Indic model's
        # reading of Latin text ('C(x)' -> '(७)'); unless the English reading has digits there, it is not a number
        elif not indic_page and _INDIC_DIGIT.search(w['text']) and not _INDIC_LETTER.search(w['text']) \
                and not any(_overlap(w, lw) > 0.3 for lw in with_digits):
            w['text'] = _INDIC_DIGIT.sub('·', w['text'])
        # Latin units read by the Indic model as digits ('A' -> '८', '°C' -> '९६' or '“6'): a confident short Latin
        # reading without digits at the same place wins
        elif _HAS_DIGIT.search(w['text']) and not _BARE_NUMBER.match(w['text']) \
                and not (indic_page and _INDIC_DIGIT.search(w['text'])):
            near = [lw for lw in latin_words if _overlap(w, lw) > 0.5]
            if len(near) == 1 and near[0].get('conf', 0) >= 80 and _LATIN_UNIT.match(near[0]['text']):
                w['text'] = near[0]['text']
    # Only confident English readings may change the text: where the English model "reads" digits into Indic
    # letters its confidence is low (median 32 on the benchmark's development pages, against 93 for real digits)
    for lw in numeric_latin:
        confident = lw.get('conf', 100) >= NUMBER_PASS_MIN_CONF
        hits = [w for w in out if _overlap(w, lw) > 0.3]
        if not hits:
            if not confident:
                continue
            same_line = [w for w in out if abs((w['top'] + w['height'] / 2) - (lw['top'] + lw['height'] / 2))
                         < 0.6 * max(w['height'], lw['height'])]
            ids = min(same_line, key=lambda w: abs(w['left'] - lw['left'])) if same_line else None
            if ids is not None:
                out.append({**lw, 'block_num': ids['block_num'], 'par_num': ids['par_num'], 'line_num': ids['line_num']})
            continue
        if len(hits) > 1:
            # The English token spans two Indic words ('में ।20' read as 'F120'): the one with digits (or without
            # Indic letters) is the number
            numeric = [h for h in hits if _HAS_DIGIT.search(h['text']) or not _INDIC_LETTER.search(h['text'])]
            if len(numeric) == 1:
                hits = numeric
        w = hits[0]
        # A word of the Indic reading without Indic letters (a number, '।' for '1', a stray symbol) takes the
        # English number; a word with Indic letters only when the English reading is confident or more confident
        letters = bool(_INDIC_LETTER.search(w['text']))
        unit = _DIGITS_THEN_UNIT.match(lw['text'])
        if len(hits) == 1 and letters and len(w['text']) == 1 and unit and not _HAS_DIGIT.search(w['text']) \
                and not any(_digits(o['text']) == unit.group(1) and abs(o['top'] - w['top']) < w['height']
                            and abs(w['left'] - (o['left'] + o['width'])) < 6 * w['height'] for o in out):
            # '24 V' read by the Indic model as the single letter 'ए' (its reading of 'V'), by the English model as
            # '24.V': the digits the Indic reading lost are put back before the unit
            out.append({**w, 'text': unit.group(1), 'left': lw['left'],
                        'width': max(1.0, w['left'] - lw['left'] - 2)})
            continue
        if indic_page and _INDIC_DIGIT.search(w['text']):
            continue
        if len(hits) == 1 and (_HAS_DIGIT.search(w['text']) or not letters or len(w['text']) <= 2) \
                and _digits(w['text']) != _digits(lw['text']) \
                and (not letters or confident or lw.get('conf', 0) > w.get('conf', 0)):
            # '(m2' (marker and number run together) + English '12' -> '(?) 12': the option marker survives
            w['text'] = ('(?) ' + lw['text']) if w['text'].startswith('(') and not lw['text'].startswith('(') \
                else lw['text']
    _fix_option_markers(out, latin_words)
    return out


_INDIC_MARKER = re.compile(r'^[(\[{][^\s()\[\]{}]{0,3}[)\]}]')     # '(8)', '(0)', '(·)', '(()' at a word's start
_LATIN_MARKER = re.compile(r'^\(([A-Da-d])\)')


def _fix_option_markers(words: List[Dict], latin_words: List[Dict]) -> None:
    """
    Option markers printed in Latin letters on an Indic page ('(A)' - '(D)', as CBSE prints them) are read by the
    Indic model as digits or symbols ('(8)', '(0)', '(·)'). Where the English reading has a clean marker at the same
    place, the marker is taken from it (and the rest of the word, the option's value, is kept).
    """
    for w in words:
        m = _INDIC_MARKER.match(w['text'])
        if not m or _INDIC_LETTER.search(m.group(0)) or _LATIN_MARKER.match(w['text']):
            continue
        near = [lw for lw in latin_words if _overlap(w, lw) > 0.2 and _LATIN_MARKER.match(lw['text'])]
        if near:
            best = max(near, key=lambda lw: lw.get('conf', 0))
            w['text'] = _LATIN_MARKER.match(best['text']).group(0) + w['text'][m.end():]


_STARTS_NUMBERED = re.compile(r'^\s*(?:Q\.?\s*|प्रश्न\s*)?[(\[]?[0-9\u0966-\u096F]{1,3}\s*[.):।]')
_MARGIN_NUMBER = re.compile(r'^\(?Q?\.?(\d{1,3})[.):]?$')


def read_margin_numbers(image: np.ndarray, regions: List[Dict], labels=('first-level-question',)) -> None:
    """
    The number of a question the page OCR skipped. Exam papers print question numbers in a margin column ('12.'
    hanging left of the text); Tesseract's page segmentation often drops such short isolated tokens. For a question
    region whose text does not start with a number, the strip left of its first word is read alone (English model,
    digits only, three times enlarged) and the number found is put in front of the region's text.
    """
    hgt, wid = image.shape[:2]
    for r in regions:
        if r['label'] not in labels or _STARTS_NUMBERED.match(r.get('text', '') or ''):
            continue
        words = r.get('words') or []
        x1, y1 = r['box'][0], r['box'][1]
        if words:
            first_line = printed_lines(words)[0]
            first = first_line[0]
            lh, top, right = first['height'], first['top'], first['left']
        else:
            lh, top, right = 0.02 * hgt, y1, x1 + 0.06 * (r['box'][2] - x1)
        cx1, cx2 = int(max(0, x1 - 0.03 * wid)), int(right - 2)
        cy1, cy2 = int(max(0, top - 0.35 * lh)), int(min(hgt, top + 1.35 * lh))
        if cx2 - cx1 < 8 or cy2 - cy1 < 8:
            continue
        crop = Image.fromarray(image[cy1:cy2, cx1:cx2]).convert('L')
        crop = np.array(crop.resize((crop.width * 3, crop.height * 3), Image.LANCZOS).convert('RGB'))
        try:
            text = _run_tesseract(crop, ['-l', 'eng', '--psm', '7', '-c', 'tessedit_char_whitelist=0123456789.()Q'])
        except (RuntimeError, subprocess.TimeoutExpired):
            continue
        m = _MARGIN_NUMBER.match(text.strip().replace(' ', ''))
        if m:
            text = r.get('text') or ''
            first = text.split(maxsplit=1)
            if first and re.fullmatch(r'[(\[]?[0-9\u0966-\u096F]{1,3}[^\w\s]?', first[0]):
                text = first[1] if len(first) > 1 else ''      # the page OCR's garbled reading of the same number
            r['text'] = f'{int(m.group(1))}. ' + text
            if 'words' in r:
                r['words'] = [{'text': f'{int(m.group(1))}.', 'left': float(cx1), 'top': float(top),
                               'width': float(right - cx1 - 4), 'height': float(lh), 'conf': 80.0}] + list(words)


def printed_lines(words: List[Dict]) -> List[List[Dict]]:
    """
    Words grouped into printed lines by their vertical overlap (top to bottom, each line left to right).
    Tesseract's own line ids are not used: a hanging question number ('12.' in a margin column) is often a block of
    its own, and ordering by block would put it after the first line of its question.
    """
    lines: List[List[Dict]] = []
    for w in sorted(words, key=lambda w: w['top']):
        cy = w['top'] + w['height'] / 2
        for line in lines[-3:]:
            ref = line[0]
            if abs(cy - (ref['top'] + ref['height'] / 2)) < 0.5 * max(1.0, min(w['height'], ref['height'])):
                line.append(w)
                break
        else:
            lines.append([w])
    for line in lines:
        line.sort(key=lambda w: w['left'])
    lines.sort(key=lambda ln: min(w['top'] for w in ln))
    return lines


def assign_words(regions: List[Dict], words: List[Dict], min_conf: float = 0.0,
                 include_words: bool = False) -> List[Dict]:
    """
    Put each word into the smallest region containing its centre and build that region's text, line by line.
    Returns copies of the regions with a 'text' field, in the same order (and their 'words' with boxes when
    include_words is set: verification uses the word positions to separate the options of a question).
    """
    per_region = defaultdict(list)
    for word in words:
        if not word['text'].strip() or word['conf'] < min_conf:
            continue
        cx, cy = word['left'] + word['width'] / 2, word['top'] + word['height'] / 2
        containing = [i for i, r in enumerate(regions)
                      if r['box'][0] <= cx <= r['box'][2] and r['box'][1] <= cy <= r['box'][3]]
        if containing:
            area = lambda i: (regions[i]['box'][2] - regions[i]['box'][0]) * (regions[i]['box'][3] - regions[i]['box'][1])
            per_region[min(containing, key=area)].append(word)

    out = []
    for i, region in enumerate(regions):
        ordered = printed_lines(per_region[i])
        text = '\n'.join(' '.join(w['text'] for w in ws) for ws in ordered)
        out.append({**region, 'text': text})
        if include_words:
            out[-1]['words'] = [{k: w[k] for k in ('text', 'left', 'top', 'width', 'height')} for w in per_region[i]]
    return out


def _contains(region: Dict, word: Dict) -> bool:
    cx, cy = word['left'] + word['width'] / 2, word['top'] + word['height'] / 2
    return region['box'][0] <= cx <= region['box'][2] and region['box'][1] <= cy <= region['box'][3]


def leftover_regions(regions: List[Dict], words: List[Dict], min_conf: float = 0.0) -> List[Dict]:
    """
    Text the layout model put in no box at all, grouped by Tesseract paragraph, as 'unlabelled-text' regions.
    Keeps the extracted document complete when the detector misses an element (e.g. a heading).
    """
    groups = defaultdict(list)
    for word in words:
        if word['text'].strip() and word['conf'] >= min_conf and not any(_contains(r, word) for r in regions):
            groups[(word['block_num'], word['par_num'])].append(word)
    out = []
    for ws in groups.values():
        lines = defaultdict(list)
        for w in ws:
            lines[w['line_num']].append(w)
        ordered = sorted(lines.values(), key=lambda line: np.mean([w['top'] for w in line]))
        text = '\n'.join(' '.join(w['text'] for w in sorted(line, key=lambda w: w['left'])) for line in ordered)
        box = [min(w['left'] for w in ws), min(w['top'] for w in ws),
               max(w['left'] + w['width'] for w in ws), max(w['top'] + w['height'] for w in ws)]
        out.append({'label': 'unlabelled-text', 'box': [round(float(v), 1) for v in box], 'score': None,
                    'text': text, 'detected': False})
    return sorted(out, key=lambda r: r['box'][1])


def reading_order(regions: List[Dict]) -> List[Dict]:
    """Top to bottom; regions that overlap vertically by at least half form a row, read left to right."""
    rows: List[List[Dict]] = []
    for region in sorted(regions, key=lambda r: r['box'][1]):
        y1, y2 = region['box'][1], region['box'][3]
        if rows:
            row_top = min(r['box'][1] for r in rows[-1])
            row_bottom = min(r['box'][3] for r in rows[-1])
            overlap = min(y2, row_bottom) - max(y1, row_top)
            if overlap >= 0.5 * min(y2 - y1, row_bottom - row_top):
                rows[-1].append(region)
                continue
        rows.append([region])
    return [r for row in rows for r in sorted(row, key=lambda r: r['box'][0])]


def to_markdown(regions: List[Dict]) -> str:
    """Regions (already in reading order) as Markdown: headings, lists, tables, captions; page furniture last."""
    body, furniture = [], []
    for r in regions:
        label, text = r['label'], r.get('text', '').strip()
        if label in _NO_TEXT:
            body.append(f'[{label}]')
            continue
        if not text:
            continue
        if label in _FURNITURE:
            furniture.append(f"- {label}: {' '.join(text.split())}")
        elif label in _HEADINGS:
            body.append(f"{_HEADINGS[label]} {' '.join(text.split())}")
        elif label in _LISTS:
            body.append('\n'.join(f'{_LISTS[label]}- {item}' for item in _list_items(text)))
        elif label == 'table':
            body.append(f'```\n{text}\n```')
        elif label in ('figure-caption', 'table-caption'):
            body.append(f'*{" ".join(text.split())}*')
        elif label == 'quote':
            body.append('\n'.join(f'> {line}' for line in text.splitlines()))
        else:
            body.append(text)
    md = '\n\n'.join(body)
    if furniture:
        md += '\n\n---\n\n' + '\n'.join(furniture)
    return md + '\n'


def extract_document(image: np.ndarray, detections: Dict, languages: Optional[str] = 'auto',
                     number_pass: bool = False) -> Dict:
    """
    OCR a page given its layout detections (0-indexed labels, boxes as x1, y1, x2, y2 in page pixels).
    languages: a Tesseract language string such as 'hin+eng', or 'auto' to detect the script first.
    number_pass: read the page a second time with the English model and use its digits (merge_number_reading);
    verification needs every number exact, so it turns this on for Indic-language pages.
    """
    script = None
    if languages in (None, 'auto'):
        script = detect_script_by_probe(image, detections)
        languages = languages_for_script(script)
    if number_pass and languages != 'eng':
        # The Indic model alone reads Indic words best (with '+eng' Tesseract sometimes swaps a Hindi word for an
        # English guess: 'पूर्णांक' -> 'guia'); the English number pass then supplies the Latin digits
        indic = '+'.join(code for code in languages.split('+') if code != 'eng')
        words = merge_number_reading(read_words(image, indic), read_words(image, 'eng'))
    else:
        words = read_words(image, languages)
    regions = [{'label': CLASS_NAMES[int(l)], 'box': [round(float(v), 1) for v in box], 'score': round(float(s), 3)}
               for box, l, s in zip(detections['boxes'], detections['labels'], detections['scores'])]
    regions = [{**r, 'detected': True} if r['label'] not in _NO_TEXT else {**r, 'text': '', 'detected': True}
               for r in assign_words(regions, words, include_words=number_pass)]
    if number_pass:
        read_margin_numbers(image, regions)
    # Words in no detected box are kept as 'unlabelled-text' regions, so no text is silently lost
    regions += leftover_regions(regions, words)
    ordered = reading_order(regions)
    assigned = sum(len(r['text'].split()) for r in ordered if r['detected'])
    return {'script': script, 'languages': languages, 'regions': ordered, 'markdown': to_markdown(ordered),
            'words_on_page': len(words), 'words_in_regions': assigned}


# One representative Tesseract model per script for the language probe (Marathi shares Devanagari with Hindi,
# Assamese shares the Bengali script)
_PROBE_LANGUAGES = {'Latin': 'eng', 'Devanagari': 'hin', 'Bengali': 'ben', 'Gujarati': 'guj', 'Gurmukhi': 'pan',
                    'Kannada': 'kan', 'Malayalam': 'mal', 'Oriya': 'ori', 'Tamil': 'tam', 'Telugu': 'tel'}
# Unicode ranges of each script, to count only the characters a model reads in ITS OWN script
# (the Indic models also contain Latin letters and read English text confidently)
_SCRIPT_RANGES = {'Latin': [(0x41, 0x5A), (0x61, 0x7A)], 'Devanagari': [(0x0900, 0x097F)],
                  'Bengali': [(0x0980, 0x09FF)], 'Gurmukhi': [(0x0A00, 0x0A7F)], 'Gujarati': [(0x0A80, 0x0AFF)],
                  'Oriya': [(0x0B00, 0x0B7F)], 'Tamil': [(0x0B80, 0x0BFF)], 'Telugu': [(0x0C00, 0x0C7F)],
                  'Kannada': [(0x0C80, 0x0CFF)], 'Malayalam': [(0x0D00, 0x0D7F)]}


def _in_script(ch: str, script: str) -> bool:
    return any(lo <= ord(ch) <= hi for lo, hi in _SCRIPT_RANGES[script])


_TEXT_LABELS_FOR_PROBE = {'paragraph', 'ordered-list', 'unordered-list', 'answer', 'first-level-question',
                          'second-level-question', 'options', 'sidebar', 'quote'}


def _probe_crop(image: np.ndarray, detections: Optional[Dict]) -> np.ndarray:
    """The largest detected body-text region (most text, least noise); the page centre if there is none."""
    h, w = image.shape[:2]
    best = None
    if detections is not None:
        for box, label in zip(detections['boxes'], detections['labels']):
            if CLASS_NAMES[int(label)] in _TEXT_LABELS_FOR_PROBE:
                x1, y1, x2, y2 = (int(v) for v in box)
                if best is None or (x2 - x1) * (y2 - y1) > (best[2] - best[0]) * (best[3] - best[1]):
                    best = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
    if best is None or (best[2] - best[0]) < 40 or (best[3] - best[1]) < 20:
        best = (w // 6, h // 4, 5 * w // 6, 3 * h // 4)
    x1, y1, x2, y2 = best
    crop = image[y1:y2, x1:x2]
    if crop.shape[0] > 700:  # enough text to judge; keeps the probe fast
        crop = crop[:700]
    return crop


def detect_script_by_probe(image: np.ndarray, detections: Optional[Dict] = None) -> str:
    """
    Script of the page: read one text region with each script's model and keep the most confident reading.
    A model shown the wrong script produces few, low-confidence words in its own script, so the mean
    confidence of words containing that script's characters, weighted by how many such characters were
    read, separates the scripts well. Accuracy on IndicDLP test pages is measured by scripts/evaluate_ocr.py (during
    development Tesseract's own orientation-and-script detector got only 70 of 120 pages right).
    """
    from concurrent.futures import ThreadPoolExecutor
    crop = _probe_crop(image, detections)
    if crop.shape[0] < 600:
        crop = np.array(Image.fromarray(crop).resize((crop.shape[1] * 2, crop.shape[0] * 2), Image.LANCZOS))

    def score(item):
        script, lang = item
        try:
            tsv = _run_tesseract(crop, ['-l', lang, '--psm', '6', '-c', 'tessedit_create_tsv=1'], timeout=120,
                                 single_thread=True)
        except (RuntimeError, subprocess.TimeoutExpired):
            return script, 0.0
        confs, chars = [], 0
        for row in tsv.splitlines()[1:]:
            cols = row.split('\t')
            if len(cols) == 12 and cols[0] == '5' and cols[11].strip() and float(cols[10]) >= 0:
                own = sum(_in_script(ch, script) for ch in cols[11])
                if own:
                    confs.append(float(cols[10]))
                    chars += own
        return script, (np.mean(confs) if confs else 0.0) * np.log1p(chars)

    with ThreadPoolExecutor(max_workers=len(_PROBE_LANGUAGES)) as pool:
        scores = dict(pool.map(score, _PROBE_LANGUAGES.items()))
    return max(scores, key=scores.get)
