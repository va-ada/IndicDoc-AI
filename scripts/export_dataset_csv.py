"""
IndicDoc AI - Export the IndicDLP subset used in this project as plain CSV files (the dataset deliverable).

Writes to dataset/:
  pages.csv          one row per page: split, file name, language, domain, size, number of boxes
  boxes.csv          one row per labelled box: split, file name, class id, class name, x, y, width, height (pixels)
  class_counts.csv   boxes per class in each split
  language_split.csv pages per language in each split
  samples/           one real page per language (with its boxes drawn), for a quick look
Usage: python scripts/export_dataset_csv.py
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from ml.config import CLASS_NAMES

DATA = PROJECT_ROOT / 'data' / 'raw' / 'indicdlp'
OUT = PROJECT_ROOT / 'dataset'


def main():
    (OUT / 'samples').mkdir(parents=True, exist_ok=True)
    class_counts = defaultdict(Counter)
    lang_counts = defaultdict(Counter)
    samples = {}
    with open(OUT / 'pages.csv', 'w', newline='', encoding='utf-8') as fp, \
            open(OUT / 'boxes.csv', 'w', newline='', encoding='utf-8') as fb:
        pw, bw = csv.writer(fp), csv.writer(fb)
        pw.writerow(['split', 'file_name', 'language', 'domain', 'width', 'height', 'num_boxes'])
        bw.writerow(['split', 'file_name', 'class_id', 'class_name', 'x', 'y', 'width', 'height'])
        for split in ('train', 'val', 'test'):
            gt = json.loads((DATA / split / 'annotations.json').read_text())
            by_img = defaultdict(list)
            for a in gt['annotations']:
                by_img[a['image_id']].append(a)
            for im in gt['images']:
                anns = by_img[im['id']]
                pw.writerow([split, im['file_name'], im['language'], im['domain'], im['width'], im['height'], len(anns)])
                lang_counts[im['language']][split] += 1
                for a in anns:
                    x, y, w, h = (round(v, 1) for v in a['bbox'])
                    bw.writerow([split, im['file_name'], a['category_id'], CLASS_NAMES[a['category_id']], x, y, w, h])
                    class_counts[CLASS_NAMES[a['category_id']]][split] += 1
                # a sample per language: the test page with the most different labels
                if split == 'test':
                    best = samples.get(im['language'])
                    if best is None or len({a['category_id'] for a in anns}) > best[2]:
                        samples[im['language']] = (im, anns, len({a['category_id'] for a in anns}))

    with open(OUT / 'class_counts.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['class_id', 'class_name', 'train', 'val', 'test', 'total'])
        for k, name in enumerate(CLASS_NAMES):
            c = class_counts[name]
            w.writerow([k, name, c['train'], c['val'], c['test'], sum(c.values())])
    with open(OUT / 'language_split.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['language', 'train', 'val', 'test', 'total'])
        for lang in sorted(lang_counts):
            c = lang_counts[lang]
            w.writerow([lang, c['train'], c['val'], c['test'], sum(c.values())])

    for lang, (im, anns, _) in sorted(samples.items()):
        page = Image.open(DATA / 'test' / 'images' / im['file_name']).convert('RGB')
        draw = ImageDraw.Draw(page)
        for a in anns:
            x, y, w, h = a['bbox']
            draw.rectangle([x, y, x + w, y + h], outline=(200, 30, 30), width=3)
            draw.text((x + 3, y + 2), CLASS_NAMES[a['category_id']], fill=(200, 30, 30))
        page.thumbnail((900, 1300))
        page.save(OUT / 'samples' / f"{lang}_{im['domain']}.jpg", quality=85)
    print(f"Wrote {OUT}")


if __name__ == '__main__':
    main()
