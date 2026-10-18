"""
IndicDoc AI - Measure the OCR stage on IndicDLP test pages.

IndicDLP has no text transcriptions, so character accuracy cannot be scored. What is measured:
  - script detection: does detect_script_by_probe pick the script of the page's language?
  - coverage and speed: words read per page, the share falling inside detected regions, seconds per page.
The layout comes from the deployed model (the IndicDLP authors' DocLayout-YOLO), as in the app.

Usage:
    python scripts/evaluate_ocr.py --pages_per_language 10
"""
import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import AUTHORS_CHECKPOINT, METRICS_DIR, RAW_DATA_DIR, SEED, get_device
from ml.inference.predict import load_model, predict_single
from ml.ocr import detect_script_by_probe, extract_document, languages_for_script

EXPECTED_SCRIPT = {'english': 'Latin', 'hindi': 'Devanagari', 'marathi': 'Devanagari', 'bengali': 'Bengali',
                   'assamese': 'Bengali', 'gujarati': 'Gujarati', 'punjabi': 'Gurmukhi', 'kannada': 'Kannada',
                   'malayalam': 'Malayalam', 'odia': 'Oriya', 'tamil': 'Tamil', 'telugu': 'Telugu'}


def main(args):
    root = RAW_DATA_DIR / 'indicdlp' / 'test'
    gt = json.loads((root / 'annotations.json').read_text())
    by_lang = defaultdict(list)
    for img in gt['images']:
        by_lang[img['language']].append(img)
    device = get_device()
    model = load_model(str(AUTHORS_CHECKPOINT), 'doclayout', device)
    rng = random.Random(SEED)
    per_language, times, coverage, examples = {}, [], [], []
    for lang in sorted(by_lang):
        pages = rng.sample(by_lang[lang], min(args.pages_per_language, len(by_lang[lang])))
        got = Counter()
        for p in pages:
            image = np.array(Image.open(root / 'images' / p['file_name']).convert('RGB'))
            det = predict_single(model, Image.fromarray(image), 'doclayout', device, conf=args.conf)
            start = time.time()
            script = detect_script_by_probe(image, det)
            doc = extract_document(image, det, languages=languages_for_script(script))
            times.append(time.time() - start)
            got[script] += 1
            if len(examples) < len(per_language) + 1 and not any(e['language'] == lang for e in examples):
                examples.append({'language': lang, 'file': p['file_name'], 'script': script,
                                 'tesseract_languages': doc['languages'], 'markdown': doc['markdown'][:1500]})
            if doc['words_on_page']:
                coverage.append(doc['words_in_regions'] / doc['words_on_page'])
        per_language[lang] = {'pages': len(pages), 'correct': got[EXPECTED_SCRIPT[lang]], 'detected': dict(got)}
        print(f"{lang:10} script correct {got[EXPECTED_SCRIPT[lang]]}/{len(pages)}  {dict(got)}")
    correct = sum(v['correct'] for v in per_language.values())
    total = sum(v['pages'] for v in per_language.values())
    report = {'script_detection_correct': correct, 'pages': total, 'script_detection_accuracy': correct / total,
              'per_language': per_language, 'seconds_per_page_ocr': float(np.mean(times)),
              'words_in_regions_share': float(np.mean(coverage)), 'examples': examples,
              'note': 'IndicDLP has no text ground truth, so OCR character accuracy is not measured here.'}
    (METRICS_DIR / 'ocr_evaluation.json').write_text(json.dumps(report, indent=2))
    print(f"script detection {correct}/{total} | OCR {report['seconds_per_page_ocr']:.1f}s/page | "
          f"{100 * report['words_in_regions_share']:.0f}% of words inside detected regions")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pages_per_language', type=int, default=10)
    parser.add_argument('--conf', type=float, default=0.35)
    main(parser.parse_args())
