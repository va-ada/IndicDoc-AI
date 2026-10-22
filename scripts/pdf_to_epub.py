"""
IndicDoc AI - Convert a whole scanned PDF (e.g. a textbook chapter) into one accessible EPUB3 book, offline.

Each page: layout detection (DocLayout-YOLO) -> OCR with automatic script choice -> semantic chapter. With --lang auto the script is detected on every page
(books can mix languages).
The book gets a table of contents and a page list that matches the printed pages.
Usage: python scripts/pdf_to_epub.py chapter.pdf --out chapter.epub [--lang auto|tam+eng|...] [--conf 0.15]
"""
import argparse
import sys
from pathlib import Path

import cv2
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.accessible import to_epub
from ml.config import AUTHORS_CHECKPOINT, get_device
from ml.inference.predict import dedupe_boxes, load_model, predict_single
from ml.ocr import extract_document
from ml.preprocessing.preprocess import pdf_page_count, render_pdf_page


def convert(pdf_bytes: bytes, title: str, languages: str = 'auto', conf: float = 0.15, log=print) -> bytes:
    device = get_device()
    model = load_model(str(AUTHORS_CHECKPOINT), 'doclayout', device)
    pages = []
    n = pdf_page_count(pdf_bytes)
    for i in range(n):
        rgb = cv2.cvtColor(render_pdf_page(pdf_bytes, page_index=i), cv2.COLOR_BGR2RGB)
        det = dedupe_boxes(predict_single(model, Image.fromarray(rgb), 'doclayout', device, conf=conf))
        doc = extract_document(rgb, det, languages=languages)
        pages.append(doc)
        log(f'page {i + 1}/{n}: {len(doc["regions"])} regions, {doc["languages"]}')
    return to_epub(pages, title)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('pdf')
    p.add_argument('--out', default=None)
    p.add_argument('--lang', default='auto')
    p.add_argument('--conf', type=float, default=0.15)
    a = p.parse_args()
    src = Path(a.pdf)
    out = Path(a.out) if a.out else src.with_suffix('.epub')
    out.write_bytes(convert(src.read_bytes(), src.stem, a.lang, a.conf))
    print('Wrote', out)
