"""
Evaluate dual-reader consensus verification on the CBSE-format benchmark (bilingual papers, Hindi and English
interleaved page by page; scripts/verify_bench/cbse.py, cbse_prepare.py).

For every paper the two readers read the whole file independently:
    reader A  layout boxes + Tesseract OCR of the page images (psm 3, number pass). --layout oracle uses the
              renderer's boxes; --layout noisy shifts them, flips 10% of question / option labels to 'paragraph',
              drops 5% of the question boxes and merges 5% of neighbouring ones, as a detector does
    reader B  layout-free: the PDF text layer for born-digital papers; its own OCR configuration (psm 6, binarised)
              for the scanned half
and the third reading (a crop of one question, OCR at twice the resolution) is used where they disagree.

Reported per system - reader A alone, reader B alone, and the consensus:
    found       share of planted errors found at the right question (a matching finding within one question)
    alarms      errors raised per clean paper (and warnings), i.e. false alarms
    review      for the consensus, how many of its findings it marks for a human look

Usage: python scripts/evaluate_consensus.py --bench data/cbse_bench [--layout noisy] [--no_third]
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts' / 'verify_bench'))

from ml.ocr import assign_words, leftover_regions, merge_number_reading, read_margin_numbers, reading_order  # noqa: E402
from ml.verify import verify  # noqa: E402
from ml.verify.consensus import make_rereader  # noqa: E402
from ml.verify.reader import read_pages  # noqa: E402

EXPECTED = {
    'number_changed': {'NUMBER_MISMATCH'},
    'option_number_changed': {'OPTION_NUMBER_MISMATCH'},
    'option_dropped': {'OPTION_COUNT_MISMATCH', 'OPTION_COUNT'},
    'question_dropped': {'MISSING_IN_TRANSLATION', 'NUMBER_GAP', 'QUESTION_COUNT', 'QUESTION_COUNT_DECLARED'},
    'marks_changed': {'MARKS_MISMATCH', 'MARKS_TOTAL'},
    'total_wrong': {'TOTAL_MISMATCH', 'MARKS_TOTAL'},
    'numbering_error': {'NUMBER_REPEATED', 'NUMBER_GAP', 'NUMBER_ORDER'},
    'question_duplicated': {'DUPLICATE_QUESTION'},
    'subpart_dropped': {'SUBQUESTION_MISMATCH'},
    'choice_dropped': {'SUBQUESTION_MISMATCH', 'CHOICE_MISMATCH'},
}
PAPER_LEVEL = {'QUESTION_COUNT', 'MARKS_TOTAL', 'TOTAL_MISMATCH', 'QUESTION_COUNT_DECLARED'}


def page_words(entry, b=False):
    """Words of one page image for a reader: the Indic reading fixed by the English number pass where present."""
    suffix = '_b' if b else ''
    eng = entry.get('eng' + suffix, [])
    if 'hin' + suffix in entry:
        return merge_number_reading(entry['hin' + suffix], eng)
    return eng


def reader_a_pages(cid, layout, ocr, mode, rng, images=None):
    pages = []
    for k, pg in enumerate(layout[cid]):
        regions = []
        qboxes = []
        for r in pg['regions']:
            box, label = list(r['box']), r['label']
            if mode == 'noisy':
                box = [v + rng.uniform(-8, 8) for v in box]
                if label in ('first-level-question', 'options', 'second-level-question') and rng.random() < 0.10:
                    label = 'paragraph'
                if label == 'first-level-question' and rng.random() < 0.05:
                    continue                                   # missed detection: its words become leftover text
            region = {'label': label, 'box': [round(v, 1) for v in box], 'score': round(rng.uniform(0.55, 0.98), 3)
                      if mode == 'noisy' else 0.95}
            if mode == 'noisy' and label == 'first-level-question' and qboxes and rng.random() < 0.05:
                prev = qboxes[-1]                              # two neighbouring questions in one box
                prev['box'] = [min(prev['box'][0], box[0]), min(prev['box'][1], box[1]),
                               max(prev['box'][2], box[2]), max(prev['box'][3], box[3])]
                continue
            if label == 'first-level-question':
                qboxes.append(region)
            regions.append(region)
        words = page_words(ocr[pg['image']])
        regions = [{**r, 'detected': True} for r in assign_words(regions, words, include_words=True)]
        if images is not None:
            read_margin_numbers(images[k], regions)       # as ml.ocr.extract_document(number_pass=True) does
        regions += leftover_regions(regions, words)
        pages.append({'regions': reading_order(regions), 'script': None, 'sheet': k})
    return pages


def reader_b_pages(cid, case, layout, ocr, textlayer):
    imgs = layout[cid]
    if not case['degraded']:
        inp = [{'words': words, 'size': (1654, 2339)} for words in textlayer[cid]]
        pages = read_pages(inp, source='text-layer')
    else:
        inp = [{'words': page_words(ocr[pg['image']], b=True), 'size': (1654, 2339)} for pg in imgs]
        pages = read_pages(inp, source='ocr')
    for k, p in enumerate(pages):
        p['sheet'] = k
    return pages


def judge(issues, gt):
    if gt['type'] == 'clean':
        return None
    exp = EXPECTED[gt['type']]
    q = gt.get('question')
    hits = [i for i in issues if i['code'] in exp and i['severity'] in ('error', 'warning')
            and (q is None or i['code'] in PAPER_LEVEL or i.get('question') in (q - 1, q, q + 1))]
    return bool(hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', default='data/cbse_bench')
    ap.add_argument('--layout', choices=['oracle', 'noisy'], default='oracle')
    ap.add_argument('--no_third', action='store_true', help='consensus without the third reading')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    root = Path(args.bench)
    cases = json.loads((root / 'cases.json').read_text())
    layout = json.loads((root / 'layout.json').read_text())
    ocr = json.loads((root / 'ocr.json').read_text())
    textlayer = json.loads((root / 'textlayer.json').read_text())
    rng = random.Random(0)
    rows = []
    for case in cases[:args.limit or None]:
        cid = case['id']
        images = {k: np.asarray(Image.open(root / 'pages' / pg['image']).convert('RGB'))
                  for k, pg in enumerate(layout[cid])}
        pa = reader_a_pages(cid, layout, ocr, args.layout, rng, images)
        pb = reader_b_pages(cid, case, layout, ocr, textlayer)
        rereader = None if args.no_third else make_rereader(images)
        rep = verify(pa, pages_b=pb, rereader=rereader, bilingual='auto', return_readers=True)
        ra, rb = rep['reader_issues']['a'], rep['reader_issues']['b']
        row = {'id': cid, 'type': case['gt']['type'], 'degraded': case['degraded'],
               'found': {'a': judge(ra, case['gt']), 'b': judge(rb, case['gt']), 'consensus': judge(rep['issues'], case['gt'])},
               'errors': {'a': sum(i['severity'] == 'error' for i in ra), 'b': sum(i['severity'] == 'error' for i in rb),
                          'consensus': sum(i['severity'] == 'error' for i in rep['issues'])},
               'warnings': {'a': sum(i['severity'] == 'warning' for i in ra), 'b': sum(i['severity'] == 'warning' for i in rb),
                            'consensus': sum(i['severity'] == 'warning' for i in rep['issues'])},
               'review': sum(i.get('consensus', {}).get('status') == 'review' for i in rep['issues']),
               'stats': rep['consensus']['stats'], 'versions': rep.get('versions', {}).get('mode'),
               'questions': {v: len(rep[k]['questions']) if rep.get(k) else None for v, k in (('target', 'paper'), ('source', 'source'))},
               'codes': sorted({(i['code'], i['severity'], i.get('consensus', {}).get('status')) for i in rep['issues']})}
        rows.append(row)
        print(cid, 'found', row['found'], 'err', row['errors'], 'warn', row['warnings'], 'review', row['review'],
              'q', row['questions'], flush=True)

    def agg(sel):
        out = {}
        for sysname in ('a', 'b', 'consensus'):
            err = [r for r in sel if r['type'] != 'clean']
            clean = [r for r in sel if r['type'] == 'clean']
            out[sysname] = {'found': round(sum(bool(r['found'][sysname]) for r in err) / len(err), 3) if err else None,
                            'errors_per_clean_paper': round(sum(r['errors'][sysname] for r in clean) / len(clean), 2) if clean else None,
                            'warnings_per_clean_paper': round(sum(r['warnings'][sysname] for r in clean) / len(clean), 2) if clean else None}
        return out
    summary = {'layout': args.layout, 'third_reading': not args.no_third, 'papers': len(rows),
               'overall': agg(rows), 'born_digital': agg([r for r in rows if not r['degraded']]),
               'scanned': agg([r for r in rows if r['degraded']]),
               'by_type': {t: agg([r for r in rows if r['type'] == t]) for t in sorted({r['type'] for r in rows})},
               'rows': rows}
    print(json.dumps({k: v for k, v in summary.items() if k not in ('rows', 'by_type')}, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
