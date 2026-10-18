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


def read_words(image: np.ndarray, languages: str) -> List[Dict]:
    """Every word Tesseract finds on the page, with its box (page pixels) and line identifiers."""
    h, w = image.shape[:2]
    # Tesseract reads best around 300 dpi; small renders are upscaled and the boxes scaled back
    scale = 2 if max(h, w) < 1600 else 1
    if scale > 1:
        image = np.array(Image.fromarray(image).resize((w * scale, h * scale), Image.LANCZOS))
    # -c tessedit_create_tsv=1 rather than the 'tsv' config file, which lives outside our tessdata folder
    tsv = _run_tesseract(image, ['-l', languages, '--psm', '3', '-c', 'tessedit_create_tsv=1'])
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


def assign_words(regions: List[Dict], words: List[Dict], min_conf: float = 0.0) -> List[Dict]:
    """
    Put each word into the smallest region containing its centre and build that region's text, line by line.
    Returns copies of the regions with a 'text' field, in the same order.
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
        lines = defaultdict(list)
        for word in per_region[i]:
            lines[(word['block_num'], word['par_num'], word['line_num'])].append(word)
        ordered = sorted(lines.values(), key=lambda ws: np.mean([w['top'] for w in ws]))
        text = '\n'.join(' '.join(w['text'] for w in sorted(ws, key=lambda w: w['left'])) for ws in ordered)
        out.append({**region, 'text': text})
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


def extract_document(image: np.ndarray, detections: Dict, languages: Optional[str] = 'auto') -> Dict:
    """
    OCR a page given its layout detections (0-indexed labels, boxes as x1, y1, x2, y2 in page pixels).
    languages: a Tesseract language string such as 'hin+eng', or 'auto' to detect the script first.
    """
    script = None
    if languages in (None, 'auto'):
        script = detect_script_by_probe(image, detections)
        languages = languages_for_script(script)
    words = read_words(image, languages)
    regions = [{'label': CLASS_NAMES[int(l)], 'box': [round(float(v), 1) for v in box], 'score': round(float(s), 3)}
               for box, l, s in zip(detections['boxes'], detections['labels'], detections['scores'])]
    regions = [{**r, 'detected': True} if r['label'] not in _NO_TEXT else {**r, 'text': '', 'detected': True}
               for r in assign_words(regions, words)]
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
