"""
Inputs of both readers for the CBSE-format benchmark (after cbse.py and render.js):

    pdf/<case>.pdf      the paper printed to PDF from the same HTML (born-digital, with a text layer)
    textlayer.json      that text layer, per page, in the page images' pixel coordinates (200 dpi)
    pages/*.png         half of the papers degraded like scans (blur, noise, JPEG); those are "scanned PDFs":
                        reader B has no text layer for them and reads them with its own OCR configuration
    ocr.json            Tesseract words per page: 'hin' / 'eng' (psm 3, reader A and the number pass) and, for
                        scanned papers, 'hin_b' / 'eng_b' (reader B's OCR configuration, ml/verify/pipeline.py SECOND_OCR)

Usage: python scripts/verify_bench/cbse_prepare.py data/cbse_bench [--workers 4]
"""
import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

PRINT_CSS = ('<style>@page{size:A4;margin:0} html,body{margin:0;background:#fff!important}'
             '.page{zoom:0.6399;margin:0!important;break-after:page} .page:last-child{break-after:auto}</style>')
SCALE = 200 / 72  # PDF points -> pixels of the 200 dpi page images


def page_langs(k: int):
    """Scripts on page k of the CBSE layout: cover both, then Hindi / English pages alternating."""
    if k == 0:
        return ['hin', 'eng']
    if k == 1:
        return ['hin', 'eng']
    if k == 2:
        return ['eng']
    return ['hin', 'eng'] if (k - 3) % 2 == 0 else ['eng']


def make_pdf(html: Path, out: Path):
    from build_lab_report import find_chrome
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / html.name
        src.write_text(html.read_text(encoding='utf-8').replace('</head>', PRINT_CSS + '</head>', 1), encoding='utf-8')
        sandbox = ['--no-sandbox'] if os.geteuid() == 0 else []
        subprocess.run([find_chrome(), *sandbox, '--headless=new', '--disable-gpu', '--no-pdf-header-footer',
                        '--allow-file-access-from-files', '--virtual-time-budget=10000', f'--print-to-pdf={out}',
                        src.as_uri()], check=True, capture_output=True, timeout=180)


def text_layer(pdf: Path):
    import pymupdf
    doc = pymupdf.open(pdf)
    return [[{'text': w[4], 'box': [round(v * SCALE, 1) for v in w[:4]]} for w in page.get_text('words')] for page in doc]


def degrade(path: Path, rng):
    img = Image.open(path).convert('L').filter(ImageFilter.GaussianBlur(0.7))
    arr = np.asarray(img, dtype=np.float32) + rng.normal(0, 8, (img.height, img.width))
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=55)
    Image.open(buf).convert('RGB').save(path)


def ocr_task(task):
    from ml.ocr import read_words
    path, lang, psm, binarize, key = task
    img = np.asarray(Image.open(path).convert('RGB'))
    if key.endswith('_b'):   # reader B: the product's second OCR configuration (ml/verify/pipeline.py)
        from ml.verify.pipeline import SECOND_OCR
        return path.name, key, read_words(img, lang, single_thread=True, **SECOND_OCR)
    return path.name, key, read_words(img, lang, psm=psm, binarize=binarize, single_thread=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bench', type=Path)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    root = args.bench
    cases = json.loads((root / 'cases.json').read_text())
    layout = json.loads((root / 'layout.json').read_text())
    (root / 'pdf').mkdir(exist_ok=True)
    tl_path = root / 'textlayer.json'
    tl = json.loads(tl_path.read_text()) if tl_path.exists() else {}
    for case in cases:
        cid = case['id']
        if cid not in tl:
            make_pdf(root / 'html' / f'{cid}.html', root / 'pdf' / f'{cid}.pdf')
            tl[cid] = text_layer(root / 'pdf' / f'{cid}.pdf')
            assert len(tl[cid]) == len(layout[cid]), (cid, len(tl[cid]), len(layout[cid]))
    tl_path.write_text(json.dumps(tl, ensure_ascii=False))
    print('text layers:', len(tl))

    marker = root / '.degraded'
    if not marker.exists():
        rng = np.random.default_rng(7)
        for case in cases:
            if case['degraded']:
                for pg in layout[case['id']]:
                    degrade(root / 'pages' / pg['image'], rng)
        marker.write_text('done')
    ocr_path = root / 'ocr.json'
    ocr = json.loads(ocr_path.read_text()) if ocr_path.exists() else {}
    tasks = []
    for case in cases:
        for k, pg in enumerate(layout[case['id']]):
            path = root / 'pages' / pg['image']
            for lang in page_langs(k):
                tasks.append((path, lang, 3, False, lang))
                if case['degraded']:
                    tasks.append((path, lang, 6, True, lang + '_b'))
    tasks = [t for t in tasks if t[4] not in ocr.get(t[0].name, {})]
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
