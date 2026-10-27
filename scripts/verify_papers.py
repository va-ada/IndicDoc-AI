"""
IndicDoc Verify from the command line: one paper, a translation against its source, or a whole folder of papers.

Every file is read by both readers (layout model + OCR, and the layout-free reading of its text layer or a second
OCR pass) and checked; bilingual files are split into their versions by themselves. For each paper an HTML and a
JSON report are written; for a folder, summary.csv lists every paper's verdict.

Usage:
    python scripts/verify_papers.py paper.pdf                          # one paper (bilingual files recognised)
    python scripts/verify_papers.py hindi.pdf --source english.pdf     # a translation against its source
    python scripts/verify_papers.py papers/ --out reports/             # every PDF / image in a folder
Options: --lang hin+eng (OCR language, default auto), --single (reader A only), --bank earlier_questions.json
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SUFFIXES = {'.pdf', '.png', '.jpg', '.jpeg', '.tif', '.tiff'}


def verdict(summary) -> str:
    if summary['errors']:
        return 'do not print'
    return 'check before printing' if summary['warnings'] or summary.get('review') else 'ready for printing'


def check_one(api, path: Path, source: Path, args, bank):
    from ml.verify import verify
    second_reading, make_rereader, version_langs = api.second_reading, api.make_rereader, api.version_langs
    data = path.read_bytes()
    images = api._pages_rgb(data)
    pages = api._read_paper(images, args.lang)
    kwargs = {}
    src_images = src_pages = None
    if source is not None:
        src_data = source.read_bytes()
        src_images = api._pages_rgb(src_data)
        src_pages = api._read_paper(src_images, args.source_lang)
    if not args.single:
        kwargs = {'pages_b': second_reading(images, pages, data if data[:5] == b'%PDF-' else None, api.VERIFY_DPI),
                  'rereader': make_rereader(images), 'langs': {'target': version_langs(pages)}}
        if source is not None:
            kwargs['source_pages_b'] = second_reading(src_images, src_pages,
                                                      src_data if src_data[:5] == b'%PDF-' else None, api.VERIFY_DPI)
            kwargs['rereader'] = {'target': make_rereader(images), 'source': make_rereader(src_images)}
            kwargs['langs']['source'] = version_langs(src_pages)
    meta = {'File': path.name, 'Pages': len(images), **({'Checked against': source.name} if source else {})}
    report = verify(pages, source_pages=src_pages, bank=bank, meta=meta,
                    bilingual='auto' if source is None else 'off', **kwargs)
    imgs = {'paper': images, **({'source': src_images} if src_images else {})}
    return report, imgs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('path', type=Path, help='a paper (PDF or image) or a folder of papers')
    ap.add_argument('--source', type=Path, help='the source-language version of the paper')
    ap.add_argument('--out', type=Path, help='folder for the reports (default: next to the paper)')
    ap.add_argument('--lang', default='auto')
    ap.add_argument('--source_lang', default='auto')
    ap.add_argument('--bank', type=Path, help="earlier papers' questions: JSON [{'text', 'source'}]")
    ap.add_argument('--single', action='store_true', help='reader A only, no consensus')
    args = ap.parse_args(argv)

    import backend.main as api
    from ml.verify.report import to_html
    if not api.CHECKPOINTS['doclayout'].exists() or not api.tesseract_available():
        sys.exit('Needs the layout model and Tesseract: python scripts/download_models.py')
    bank = json.loads(args.bank.read_text()) if args.bank else None
    files = sorted(p for p in args.path.iterdir() if p.suffix.lower() in SUFFIXES) if args.path.is_dir() else [args.path]
    if args.source and len(files) != 1:
        sys.exit('--source goes with one paper')
    out_dir = args.out or (args.path if args.path.is_dir() else args.path.parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in files:
        t0 = time.time()
        try:
            report, imgs = check_one(api, f, args.source, args, bank)
        except Exception as e:      # an unreadable file must not stop a folder run
            detail = getattr(e, 'detail', None) or str(e)
            rows.append({'file': f.name, 'verdict': f'not checked: {detail}'})
            print(f'{f.name}: not checked ({detail})', flush=True)
            continue
        s = report['summary']
        (out_dir / f'{f.stem}_report.html').write_text(to_html(report, imgs, title='IndicDoc Verify: pre-print report'),
                                                       encoding='utf-8')
        (out_dir / f'{f.stem}_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
        row = {'file': f.name, 'verdict': verdict(s), 'questions': s['questions'], 'errors': s['errors'],
               'warnings': s['warnings'], 'confirmed': s.get('confirmed', ''), 'review': s.get('review', ''),
               'dismissed': s.get('dismissed', ''), 'versions': report['meta'].get('Versions', ''),
               'seconds': round(time.time() - t0)}
        rows.append(row)
        print(f"{f.name}: {row['verdict']} ({s['errors']} errors, {s['warnings']} warnings, {s['questions']} questions)",
              flush=True)
    if len(rows) > 1 or args.path.is_dir():
        with open(out_dir / 'summary.csv', 'w', newline='', encoding='utf-8') as fh:
            fields = list(max(rows, key=len))
            writer = csv.DictWriter(fh, fieldnames=fields, restval='')
            writer.writeheader()
            writer.writerows(rows)
        print('summary:', out_dir / 'summary.csv')
    return rows


if __name__ == '__main__':
    main()
