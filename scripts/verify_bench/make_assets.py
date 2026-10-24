"""
Pictures of real IndicDoc Verify output on the benchmark's test set, for the slides and the report:

    docs/figures/bench_{en,hi,ta}.jpg       sample benchmark pages (English source, Hindi, Tamil scan-degraded)
    docs/screenshots/07_verify_report.png   top of an HTML verification report
    docs/screenshots/08_verify_page.png     a page with a correctly found error outlined
    docs/screenshots/09_false_alarm.png     a page with a false alarm outlined (an honest failure case)

Usage: python scripts/verify_bench/make_assets.py data/verify_bench
"""
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from evaluate_verify import EXPECTED, build_pages  # noqa: E402
from ml.verify import verify  # noqa: E402
from ml.verify.report import annotate, to_html  # noqa: E402

FIG, SHOTS = ROOT / 'docs' / 'figures', ROOT / 'docs' / 'screenshots'


def chrome():
    cands = [os.environ.get('CHROME'), shutil.which('chromium'), shutil.which('google-chrome'),
             '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome']
    cands += sorted(str(p) for p in Path('/opt/pw-browsers').glob('chromium-*/chrome-linux*/chrome'))
    return next(c for c in cands if c and Path(c).exists())


def crop_around(img: Image.Image, box, pad=260, height=900):
    x1, y1, x2, y2 = box
    top = max(0, int(y1) - pad)
    return img.crop((0, top, img.width, min(img.height, top + height)))


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else 'data/verify_bench')
    cases = json.loads((root / 'cases.json').read_text())
    layout = json.loads((root / 'layout.json').read_text())
    ocr = json.loads((root / 'ocr.json').read_text())
    bank = json.loads((root / 'bank.json').read_text())
    rng = random.Random(0)
    page_img = lambda name, p: Image.open(root / 'pages' / layout[name][p]['image']).convert('RGB')

    # sample pages
    clean_hi = next(c for c in cases if c['lang'] == 'hi' and not c['degraded'])
    deg_ta = next(c for c in cases if c['lang'] == 'ta' and c['degraded'])
    for name, case, side in (('bench_en', clean_hi, 'src'), ('bench_hi', clean_hi, 'tgt'), ('bench_ta', deg_ta, 'tgt')):
        im = page_img(f"{case['id']}_{side}", 0)
        im.thumbnail((900, 1300))
        im.save(FIG / f'{name}.jpg', quality=88)

    def run(case):
        src = build_pages(f"{case['id']}_src", layout, ocr, 'oracle', rng)
        tgt = build_pages(f"{case['id']}_tgt", layout, ocr, 'oracle', rng)
        return verify(tgt, source_pages=src, bank=bank[case['lang']],
                      meta={'Paper': f"Model paper, {'Hindi' if case['lang'] == 'hi' else 'Tamil'} version",
                            'Checked against': 'English source'})

    # a correctly found changed number, Hindi, born-digital: the case whose report has the fewest other findings
    best = None
    for case in cases:
        if case['lang'] == 'hi' and case['gt']['type'] == 'number_changed' and not case['degraded']:
            rep = run(case)
            hits = [(n, i) for n, i in enumerate(rep['issues'], 1)
                    if i['code'] == 'NUMBER_MISMATCH' and i['question'] == case['gt']['question']]
            if hits and (best is None or len(rep['issues']) < len(best[1]['issues'])):
                best = (case, rep, hits)
    case, report, hits = best
    n, issue = hits[0]
    img = page_img(f"{case['id']}_tgt", issue['page'])
    crop_around(annotate(np.asarray(img), [(n, issue)]), issue['box']).save(SHOTS / '08_verify_page.png')
    imgs = [np.asarray(page_img(f"{case['id']}_tgt", p)) for p in range(len(layout[f"{case['id']}_tgt"]))]
    with tempfile.TemporaryDirectory() as tmp:
        html = Path(tmp) / 'report.html'
        html.write_text(to_html(report, {'paper': imgs}, title='IndicDoc Verify: pre-print report'))
        sandbox = ['--no-sandbox'] if os.geteuid() == 0 else []
        subprocess.run([chrome(), *sandbox, '--headless=new', '--disable-gpu', '--hide-scrollbars',
                        f'--screenshot={SHOTS / "07_verify_report.png"}', '--window-size=1280,900', html.as_uri()],
                       check=True, capture_output=True, timeout=120)
    (root / f"{case['id']}_report.html").write_text(to_html(report, {'paper': imgs}, title='IndicDoc Verify: pre-print report'))

    # a false alarm on a scan-degraded clean pair
    for case in cases:
        if case['gt']['type'] == 'clean' and case['degraded']:
            report = run(case)
            bad = [(n, i) for n, i in enumerate(report['issues'], 1)
                   if i['severity'] == 'error' and i.get('box') and i['code'].startswith('OPTION_NUMBER')]
            if bad:
                n, issue = bad[0]
                img = page_img(f"{case['id']}_tgt", issue['page'])
                crop_around(annotate(np.asarray(img), [(n, issue)]), issue['box'], pad=200, height=600).save(
                    SHOTS / '09_false_alarm.png')
                print('false alarm example:', case['id'], issue['message'])
                break
    print('assets written')


if __name__ == '__main__':
    main()
