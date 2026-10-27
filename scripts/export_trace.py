"""
Record what IndicDoc Verify reads from a paper, so a run on one machine can be replayed and debugged on another.

For every page (rendered at the verification resolution) the trace holds:
  - detections: every layout box the model proposes (score >= 0.05), before the app's threshold
  - page: the exact extract_document() result the app verifies (threshold, de-duplication, OCR, number pass)
  - words: full-page Tesseract words with the page's Indic model and with the English model
  - pdf_words: the PDF's own text layer, if it has one, in the same pixel coordinates
The PDF is needed to replay a trace (pages are re-rendered at the same resolution).

Usage:
    python scripts/export_trace.py paper.pdf                 # -> paper.trace.json.gz
    python scripts/export_trace.py paper.pdf --lang hin      # force the OCR language of the paper
"""
import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import backend.main as api  # noqa: E402  (same model loading, threshold and OCR as the app)
from ml.ocr import detect_script_by_probe, extract_document, languages_for_script, read_words  # noqa: E402
from ml.inference.predict import dedupe_boxes, predict_single  # noqa: E402
from ml.config import CLASS_NAMES  # noqa: E402
from ml.preprocessing.preprocess import pdf_page_count, render_pdf_page  # noqa: E402
from PIL import Image  # noqa: E402


def pdf_words(pdf: bytes, index: int, dpi: int):
    import pymupdf
    doc = pymupdf.open(stream=pdf, filetype='pdf')
    s = dpi / 72.0
    return [{'text': w[4], 'box': [round(w[0] * s, 1), round(w[1] * s, 1), round(w[2] * s, 1), round(w[3] * s, 1)],
             'block': w[5], 'line': w[6], 'word': w[7]} for w in doc[index].get_text('words')]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('pdf', type=Path)
    ap.add_argument('--lang', default='auto', help="Tesseract language of the paper, or 'auto' per page")
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    if not api.CHECKPOINTS['doclayout'].exists() or not api.tesseract_available():
        sys.exit('Needs the layout model and Tesseract: run python scripts/download_models.py first')
    data = args.pdf.read_bytes()
    images = [cv2.cvtColor(render_pdf_page(data, i, dpi=api.VERIFY_DPI, max_pixels=api.MAX_PIXELS), cv2.COLOR_BGR2RGB)
              for i in range(pdf_page_count(data))]
    conf = api.recommended_conf('doclayout', 'all')
    model, device = api.get_model('doclayout')
    pages = []
    t0 = time.time()
    for i, img in enumerate(images):
        raw = predict_single(model, Image.fromarray(img), 'doclayout', device, conf=0.05)
        keep = raw['scores'] >= conf
        out = dedupe_boxes({**raw, 'boxes': raw['boxes'][keep], 'labels': raw['labels'][keep], 'scores': raw['scores'][keep]})
        script = detect_script_by_probe(img, out)
        lang = args.lang if args.lang != 'auto' else languages_for_script(script)
        page = extract_document(img, out, lang, number_pass=True)
        indic = '+'.join(c for c in lang.split('+') if c != 'eng') or 'eng'
        pages.append({
            'index': i, 'size': [int(img.shape[1]), int(img.shape[0])],
            'sha1': hashlib.sha1(np.ascontiguousarray(img).tobytes()).hexdigest(),
            'script': script, 'lang': lang,
            'detections': [{'label': CLASS_NAMES[int(l)], 'box': [round(float(v), 1) for v in b], 'score': round(float(s), 3)}
                           for b, l, s in zip(raw['boxes'], raw['labels'], raw['scores'])],
            'page': page,
            'words': {indic: read_words(img, indic), **({'eng': read_words(img, 'eng')} if indic != 'eng' else {})},
            'pdf_words': pdf_words(data, i, api.VERIFY_DPI) if args.pdf.suffix.lower() == '.pdf' else [],
        })
        print(f'page {i + 1}/{len(images)}  {script}  {len(page["regions"])} regions  {time.time() - t0:.0f}s', flush=True)
    out_path = args.out or args.pdf.with_suffix('.trace.json.gz')
    trace = {'file': args.pdf.name, 'dpi': api.VERIFY_DPI, 'threshold': conf, 'pages': pages}
    with gzip.open(out_path, 'wt', encoding='utf-8') as fh:
        json.dump(trace, fh, ensure_ascii=False)
    print(f'trace written to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
