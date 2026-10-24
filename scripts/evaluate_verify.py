"""
Evaluate IndicDoc Verify on the benchmark built by scripts/verify_bench (rendered English/Hindi and English/Tamil
paper pairs, one injected error per pair or none, half of them degraded like scans, read with Tesseract).

For every pair the target pages go through the product's own OCR assembly (ml.ocr.assign_words, leftover_regions,
reading_order) and ml.verify.verify(); an injected error counts as found when an issue of the matching kind points at
the right question (or at the paper, for paper-level errors). Clean pairs measure false alarms.

    --layout oracle   region boxes from the renderer (what a perfect layout model would return)
    --layout noisy    boxes jittered by up to 8 px and 10% of question / option labels changed to 'paragraph',
                      to test the parser's text fallbacks against layout-model mistakes

Usage:
    python scripts/evaluate_verify.py --bench data/verify_bench --layout oracle
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.ocr import assign_words, leftover_regions, merge_number_reading, reading_order  # noqa: E402
from ml.verify import verify  # noqa: E402
from ml.verify.checks import ERROR, WARNING  # noqa: E402

EXPECTED = {
    'number_changed': {'NUMBER_MISMATCH'},
    'option_number_changed': {'OPTION_NUMBER_MISMATCH'},
    'option_dropped': {'OPTION_COUNT_MISMATCH', 'OPTION_COUNT'},
    'question_dropped': {'MISSING_IN_TRANSLATION', 'NUMBER_GAP', 'QUESTION_COUNT'},
    'question_duplicated': {'DUPLICATE_QUESTION'},
    'marks_changed': {'MARKS_MISMATCH', 'MARKS_TOTAL'},
    'total_wrong': {'TOTAL_MISMATCH', 'MARKS_TOTAL'},
    'numbering_error': {'NUMBER_REPEATED', 'NUMBER_GAP', 'NUMBER_ORDER'},
    'bank_repeat': {'REPEATED_FROM_BANK'},
}
PAPER_LEVEL = {'QUESTION_COUNT', 'MARKS_TOTAL', 'TOTAL_MISMATCH'}
LABELS = {'number_changed': 'Number changed in stem', 'option_number_changed': 'Number changed in an option',
          'option_dropped': 'Option missing', 'question_dropped': 'Question missing',
          'question_duplicated': 'Question printed twice', 'marks_changed': 'Marks differ',
          'total_wrong': 'Wrong maximum marks', 'numbering_error': 'Question numbering error',
          'bank_repeat': 'Repeated from a past paper'}


def build_pages(name, layout, ocr, mode, rng):
    """extract_document()-style page dicts from rendered boxes + Tesseract words."""
    pages = []
    for pg in layout[name]:
        regions = []
        for r in pg['regions']:
            box, label = list(r['box']), r['label']
            if mode == 'noisy':
                box = [v + rng.uniform(-8, 8) for v in box]
                if label in ('first-level-question', 'options') and rng.random() < 0.10:
                    label = 'paragraph'
            regions.append({'label': label, 'box': [round(v, 1) for v in box], 'score': 1.0})
        words = ocr[pg['image']]
        if pg['image'] + '#eng' in ocr:  # number pass (ml.ocr.extract_document(number_pass=True))
            words = merge_number_reading(words, ocr[pg['image'] + '#eng'])
        regions = [{**r, 'detected': True} for r in assign_words(regions, words, include_words=True)]
        regions += leftover_regions(regions, words)
        pages.append({'regions': reading_order(regions), 'script': None})
    return pages


def found(issues, gt):
    """(found anywhere, found at the right place) for one injected error."""
    exp = EXPECTED[gt['type']]
    hits = [i for i in issues if i['code'] in exp and i['severity'] in (ERROR, WARNING)]
    q = gt.get('question')
    placed = [i for i in hits if q is None or i['code'] in PAPER_LEVEL or i.get('question') in (q - 1, q, q + 1)]
    return bool(hits), bool(placed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', default='data/verify_bench')
    ap.add_argument('--layout', choices=['oracle', 'noisy'], default='oracle')
    ap.add_argument('--out', default=None)
    ap.add_argument('--report_example', default=None, help='write the HTML report of this case id')
    args = ap.parse_args()
    root = Path(args.bench)
    cases = json.loads((root / 'cases.json').read_text())
    layout = json.loads((root / 'layout.json').read_text())
    ocr = json.loads((root / 'ocr.json').read_text())
    bank = json.loads((root / 'bank.json').read_text())
    rng = random.Random(0)

    per_case, spurious = [], Counter()
    for case in cases:
        src = build_pages(f"{case['id']}_src", layout, ocr, args.layout, rng)
        tgt = build_pages(f"{case['id']}_tgt", layout, ocr, args.layout, rng)
        report = verify(tgt, source_pages=src, bank=bank[case['lang']])
        issues = [i for i in report['issues'] if i['severity'] in (ERROR, WARNING)]
        gt = case['gt']
        row = {'id': case['id'], 'lang': case['lang'], 'type': gt['type'], 'degraded': case['degraded'],
               'questions_read': report['summary']['questions'], 'issues': len(issues),
               'errors': report['summary']['errors']}
        if gt['type'] == 'clean':
            row['false_alarms'] = len(issues)
            spurious.update(i['code'] for i in issues)
        else:
            row['found'], row['found_at_place'] = found(issues, gt)
            other = [i for i in issues if i['code'] not in EXPECTED[gt['type']]]
            row['other_issues'] = len(other)
            spurious.update(i['code'] for i in other)
        per_case.append(row)
        if args.report_example == case['id']:
            from ml.verify.report import to_html
            import numpy as np
            from PIL import Image
            imgs = [np.asarray(Image.open(root / 'pages' / pg['image']).convert('RGB')) for pg in layout[f"{case['id']}_tgt"]]
            report['meta'] = {'Paper': 'Model paper (Hindi version)', 'Checked against': 'English source',
                              'Languages': 'Hindi, English'}
            (root / f"{case['id']}_report.html").write_text(to_html(report, {'paper': imgs},
                                                                     title='IndicDoc Verify: pre-print report'))

    def summarize(rows):
        err = [r for r in rows if r['type'] != 'clean']
        clean = [r for r in rows if r['type'] == 'clean']
        out = {'pairs': len(rows)}
        if err:
            out['errors_injected'] = len(err)
            out['found'] = sum(r['found'] for r in err) / len(err)
            out['found_at_place'] = sum(r['found_at_place'] for r in err) / len(err)
            out['other_issues_per_paper'] = sum(r['other_issues'] for r in err) / len(err)
        if clean:
            out['clean_pairs'] = len(clean)
            out['clean_pairs_with_any_issue'] = sum(r['false_alarms'] > 0 for r in clean) / len(clean)
            out['false_alarms_per_clean_pair'] = sum(r['false_alarms'] for r in clean) / len(clean)
        out['questions_read_ratio'] = sum(r['questions_read'] for r in rows) / max(1, sum(
            18 - (r['type'] == 'question_dropped') for r in rows))
        return out

    by_type = {t: summarize([r for r in per_case if r['type'] == t]) for t in EXPECTED}
    result = {
        'layout': args.layout, 'ocr': 'Tesseract 5 (tesseract.js, tessdata_best int models eng/hin/tam)',
        'overall': summarize(per_case),
        'by_type': by_type,
        'by_language': {lg: summarize([r for r in per_case if r['lang'] == lg]) for lg in ('hi', 'ta')},
        'by_quality': {'clean_render': summarize([r for r in per_case if not r['degraded']]),
                       'scan_degraded': summarize([r for r in per_case if r['degraded']])},
        'by_type_language': {t: {lg: summarize([r for r in per_case if r['type'] == t and r['lang'] == lg])
                                 for lg in ('hi', 'ta')} for t in EXPECTED},
        'spurious_issue_codes': dict(spurious.most_common()),
        'labels': LABELS,
        'cases': per_case,
    }
    out = Path(args.out or PROJECT_ROOT / 'outputs' / 'metrics' / f'verify_benchmark_{args.layout}.json')
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    o = result['overall']
    print(f"{args.layout}: {o['errors_injected']} injected errors, found {o['found']:.1%} "
          f"(at the right question {o['found_at_place']:.1%}); clean pairs with any issue "
          f"{o['clean_pairs_with_any_issue']:.1%}, false alarms per clean pair {o['false_alarms_per_clean_pair']:.2f}")
    for t, s in by_type.items():
        print(f"  {t:22s} found {s['found']:.0%} placed {s['found_at_place']:.0%} other {s['other_issues_per_paper']:.2f}")
    print('  spurious codes:', dict(spurious.most_common(8)))


if __name__ == '__main__':
    main()
