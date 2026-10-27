"""
Replay a verification from a trace (scripts/export_trace.py) on a machine without the layout model.

Reader A's pages are taken from the trace exactly as the recording machine produced them (layout model + OCR);
reader B and the third reading run here, from the PDF itself. Prints the summary and the consensus statistics, and
writes the HTML report next to the trace.

Usage:
    python scripts/replay_trace.py paper.pdf paper.trace.json.gz [--single] [--mode auto|off]
"""
import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.preprocessing.preprocess import pdf_page_count, render_pdf_page  # noqa: E402
from ml.verify import verify  # noqa: E402
from ml.verify.consensus import make_rereader  # noqa: E402
from ml.verify.pipeline import second_reading, version_langs  # noqa: E402
from ml.verify.report import to_html  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('pdf', type=Path)
    ap.add_argument('trace', type=Path)
    ap.add_argument('--single', action='store_true', help='reader A only (the result the recording machine gave)')
    ap.add_argument('--mode', choices=['auto', 'off'], default='auto', help="'auto' finds bilingual versions")
    args = ap.parse_args()
    trace = json.load(gzip.open(args.trace, 'rt', encoding='utf-8'))
    data = args.pdf.read_bytes()
    dpi = trace.get('dpi', 200)
    images = [cv2.cvtColor(render_pdf_page(data, i, dpi=dpi), cv2.COLOR_BGR2RGB) for i in range(pdf_page_count(data))]
    pages_a = []
    for k, tp in enumerate(trace['pages']):
        page = dict(tp['page'])
        page['sheet'] = k
        pages_a.append(page)
        if list(images[k].shape[1::-1]) != tp['size']:
            print(f'warning: page {k + 1} renders at {images[k].shape[1::-1]}, the trace has {tp["size"]}')
    kwargs = {}
    if not args.single:
        pages_b = second_reading(images, pages_a, data, dpi,
                                 progress=lambda k, n: print(f'  reader B: page {k}/{n}', end='\r', flush=True))
        print()
        kwargs = {'pages_b': pages_b, 'rereader': make_rereader(images), 'langs': {'target': version_langs(pages_a)}}
    rep = verify(pages_a, bilingual=args.mode, meta={'File': args.pdf.name, 'Pages': len(images)},
                 return_readers=not args.single, **kwargs)
    print('versions:', rep.get('meta', {}).get('Versions', 'one paper'))
    print('summary :', rep['summary'])
    print('findings:', dict(Counter((i['code'], i.get('consensus', {}).get('status')) for i in rep['issues'])))
    if not args.single:
        ra = rep['reader_issues']['a']
        print(f"reader A alone: {sum(i['severity'] == 'error' for i in ra)} errors, "
              f"{sum(i['severity'] == 'warning' for i in ra)} warnings")
        for v, st in rep['consensus']['stats'].items():
            print(f'  {v}: {st}')
    out = args.trace.with_name(args.trace.name.replace('.trace.json.gz', '') + ('_single' if args.single else '')
                               + '_report.html')
    out.write_text(to_html(rep, {'paper': images}, title='IndicDoc Verify: pre-print report'), encoding='utf-8')
    print('report  :', out)


if __name__ == '__main__':
    main()
