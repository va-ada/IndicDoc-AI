"""
Reader B's inputs for the original verification benchmark (English source + Hindi/Tamil translation pairs, made by
generate.py, render.js, degrade.py and ocr.js), so the dual-reader consensus can be measured on it:

    pdf/<paper>.pdf     each paper printed to PDF from its HTML (born-digital, with a text layer)
    textlayer.json      that text layer per page, in the page images' pixel coordinates (200 dpi)
    ocr_b.json          for the scan-degraded papers (no usable text layer): Tesseract words with reader B's own
                        configuration (psm 6 on a binarised image), the page's Indic model and English

Reader A keeps the benchmark's existing OCR (tesseract.js, psm 3), so the two readers do not share an OCR engine build.

Usage: python scripts/verify_bench/pairs_prepare.py data/verify_bench [--workers 4]
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cbse_prepare import make_pdf, text_layer  # noqa: E402

INDIC = {'hi': 'hin', 'ta': 'tam'}


def ocr_task(task):
    from ml.ocr import read_words
    path, lang, key = task
    img = np.asarray(Image.open(path).convert('RGB'))
    return path.name, key, read_words(img, lang, psm=6, binarize=True, single_thread=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bench', type=Path)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--skip_ocr', action='store_true')
    args = ap.parse_args()
    root = args.bench
    cases = json.loads((root / 'cases.json').read_text())
    layout = json.loads((root / 'layout.json').read_text())
    (root / 'pdf').mkdir(exist_ok=True)
    tl_path = root / 'textlayer.json'
    tl = json.loads(tl_path.read_text()) if tl_path.exists() else {}
    for case in cases:
        for side in ('src', 'tgt'):
            name = f"{case['id']}_{side}"
            if name in tl:
                continue
            make_pdf(root / 'html' / f'{name}.html', root / 'pdf' / f'{name}.pdf')
            tl[name] = text_layer(root / 'pdf' / f'{name}.pdf')
            assert len(tl[name]) == len(layout[name]), (name, len(tl[name]), len(layout[name]))
        tl_path.write_text(json.dumps(tl, ensure_ascii=False))
    print('text layers:', len(tl), flush=True)
    if args.skip_ocr:
        return
    ocr_path = root / 'ocr_b.json'
    ocr = json.loads(ocr_path.read_text()) if ocr_path.exists() else {}
    tasks = []
    for case in cases:
        if not case['degraded']:
            continue
        for side in ('src', 'tgt'):
            for pg in layout[f"{case['id']}_{side}"]:
                path = root / 'pages' / pg['image']
                langs = ['eng'] if side == 'src' else [INDIC[case['lang']], 'eng']
                tasks += [(path, lang, lang) for lang in langs if lang not in ocr.get(path.name, {})]
    print('OCR tasks:', len(tasks), flush=True)
    with Pool(args.workers) as pool:
        for k, (name, key, words) in enumerate(pool.imap_unordered(ocr_task, tasks, chunksize=2), 1):
            ocr.setdefault(name, {})[key] = words
            if k % 100 == 0:
                print(f'  {k}/{len(tasks)}', flush=True)
                ocr_path.write_text(json.dumps(ocr, ensure_ascii=False))
    ocr_path.write_text(json.dumps(ocr, ensure_ascii=False))
    print('OCR done:', len(ocr), 'pages')


if __name__ == '__main__':
    main()
