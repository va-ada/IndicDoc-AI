"""
IndicDoc AI - Download the pretrained files the app uses but git does not hold.

1. DocLayout-YOLO trained on the full IndicDLP dataset by the dataset's authors
   (Nath et al., ICDAR 2025; Zenodo record 15881917, licence CC-BY 4.0).
2. Tesseract OCR models ("tessdata_best", Apache-2.0) for the 12 IndicDLP languages.

Usage:
    python scripts/download_models.py
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import AUTHORS_CHECKPOINT
from ml.ocr import OCR_LANGUAGES, TESSDATA_DIR

AUTHORS_URL = 'https://zenodo.org/api/records/15881917/files/doclayout_yolo_indicdlp.pt/content'
AUTHORS_SHA256_PREFIX = '473b988715cd78428835'  # first 20 hex digits, checked after download
TESSDATA_URL = 'https://github.com/tesseract-ocr/tessdata_best/raw/main/{}.traineddata'


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [ok] {dest.relative_to(PROJECT_ROOT)} already present")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + '.part')
    print(f"  downloading {dest.name} ...")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def main():
    print("DocLayout-YOLO (IndicDLP authors):")
    download(AUTHORS_URL, AUTHORS_CHECKPOINT)
    digest = hashlib.sha256(AUTHORS_CHECKPOINT.read_bytes()).hexdigest()
    if not digest.startswith(AUTHORS_SHA256_PREFIX):
        raise SystemExit(f"Checksum mismatch for {AUTHORS_CHECKPOINT}: {digest[:20]}...")

    print("Tesseract OCR models:")
    for code in sorted(set(OCR_LANGUAGES.values())):
        download(TESSDATA_URL.format(code), TESSDATA_DIR / f'{code}.traineddata')
    print("Done.")


if __name__ == '__main__':
    main()
