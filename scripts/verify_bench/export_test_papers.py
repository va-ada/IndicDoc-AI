"""
Export a small set of benchmark papers as PDFs for trying IndicDoc Verify by hand (app or API).

Each test is an English source paper and its Hindi or Tamil version, taken from the benchmark's test set
(data/verify_bench, built by generate.py and render.js). The version has one planted error, or none; ANSWER_KEY.md
says what and where. Born-digital tests are printed from the benchmark HTML with Chrome; the scanned test is made
from the benchmark's scan-degraded page images.

Usage: python scripts/verify_bench/export_test_papers.py [data/verify_bench] [docs/test_papers]
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from build_lab_report import find_chrome  # noqa: E402
from evaluate_verify import EXPECTED  # noqa: E402

# (folder name, benchmark case, how to check it in the app)
TESTS = [
    ('01_hindi_no_error', 'hi_009_clean', 'translation'),
    ('02_hindi_number_changed', 'hi_012_number_changed', 'translation'),
    ('03_hindi_option_number_changed', 'hi_020_option_number_changed', 'translation'),
    ('04_hindi_question_missing', 'hi_031_question_dropped', 'translation'),
    ('05_hindi_marks_changed', 'hi_043_marks_changed', 'translation'),
    ('06_hindi_repeated_from_old_paper', 'hi_064_bank_repeat', 'bank'),
    ('07_tamil_no_error', 'ta_000_clean', 'translation'),
    ('08_tamil_option_missing', 'ta_024_option_dropped', 'translation'),
    ('09_tamil_question_printed_twice', 'ta_038_question_duplicated', 'one'),
    ('10_tamil_wrong_maximum_marks', 'ta_051_total_wrong', 'one'),
    ('11_tamil_numbering_error', 'ta_055_numbering_error', 'one'),
    ('12_tamil_scanned_number_changed', 'ta_012_number_changed', 'translation'),
]
# CBSE-format bilingual papers from the development set (scripts/verify_bench/cbse.py, seed 7)
CBSE_EXAMPLES = [('13_cbse_bilingual/no_error.pdf', 'cbse_000_clean'),
                 ('13_cbse_bilingual/number_changed.pdf', 'cbse_002_number_changed'),
                 ('13_cbse_bilingual/internal_choice_missing.pdf', 'cbse_011_choice_dropped')]
LANG = {'hi': ('Hindi', 'hin'), 'ta': ('Tamil', 'tam')}
HOW = {
    'translation': 'Verify Paper → "Translation against its source": paper = `{v}`, source = `english_source.pdf`',
    'one': 'Verify Paper → "One paper": paper = `{v}` (the error is inside this paper; the English source is there '
           'to compare too)',
    'bank': 'Verify Paper → "Translation against its source" with `{v}` and `english_source.pdf`; under "More '
            'checks" upload `question_bank_{l}.json` as the question bank',
}
WHAT = {
    'clean': 'None: a correct translation. Anything the report lists is a false alarm.',
    'number_changed': 'Q{q}: a number in the question was changed ({d}).',
    'option_number_changed': 'Q{q}: a number in an option was changed ({d}).',
    'option_dropped': 'Q{q}: an option is missing ({d}).',
    'question_dropped': 'Q{q} is missing from the version ({d}).',
    'question_duplicated': 'Q{q} is printed twice ({d}).',
    'marks_changed': 'Q{q}: the marks differ from the source ({d}).',
    'total_wrong': 'The printed maximum marks are wrong ({d}).',
    'numbering_error': 'Question numbering is wrong ({d}).',
    'bank_repeat': 'Q{q} repeats a question of an earlier paper ({d}).',
}
PRINT_CSS = ('<style>@page{size:A4;margin:0} html,body{margin:0;background:#fff!important}'
             '.page{zoom:0.6399;margin:0!important;break-after:page} .page:last-child{break-after:auto}</style>')


def html_to_pdf(html: Path, out: Path, tmp: Path):
    src = tmp / html.name
    src.write_text(html.read_text(encoding='utf-8').replace('</head>', PRINT_CSS + '</head>', 1), encoding='utf-8')
    sandbox = ['--no-sandbox'] if hasattr(os, 'geteuid') and os.geteuid() == 0 else []
    subprocess.run([find_chrome(), *sandbox, '--headless=new', '--disable-gpu', '--no-pdf-header-footer',
                    '--allow-file-access-from-files', '--virtual-time-budget=10000', f'--print-to-pdf={out}',
                    src.as_uri()], check=True, capture_output=True, timeout=120)


def images_to_pdf(images, out: Path):
    doc = pymupdf.open()
    for img in images:
        page = doc.new_page(width=595.28, height=841.89)
        page.insert_image(page.rect, filename=str(img))
    doc.save(out, deflate=True)


def main():
    bench = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / 'data' / 'verify_bench')
    out = Path(sys.argv[2] if len(sys.argv) > 2 else ROOT / 'docs' / 'test_papers')
    cases = {c['id']: c for c in json.loads((bench / 'cases.json').read_text())}
    bank = json.loads((bench / 'bank.json').read_text())
    measured = {c['id']: c for c in json.loads((ROOT / 'outputs' / 'metrics' / 'verify_benchmark_oracle.json').read_text())['cases']}
    out.mkdir(parents=True, exist_ok=True)
    for lang in LANG:
        (out / f'question_bank_{lang}.json').write_text(json.dumps(bank[lang], ensure_ascii=False, indent=1),
                                                        encoding='utf-8')
    rows = []
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        for folder, cid, how in TESTS:
            case = cases[cid]
            name, code = LANG[case['lang']]
            version = f'{name.lower()}_version.pdf'
            dest = out / folder
            dest.mkdir(exist_ok=True)
            for side, fname in (('src', 'english_source.pdf'), ('tgt', version)):
                if case['degraded']:
                    images_to_pdf(sorted((bench / 'pages').glob(f'{cid}_{side}_p*.png')), dest / fname)
                else:
                    html_to_pdf(bench / 'html' / f'{cid}_{side}.html', dest / fname, tmp)
            gt = case['gt']
            rows.append((folder, name, 'scanned' if case['degraded'] else 'born-digital',
                         WHAT[gt['type']].format(q=gt.get('question'), d=gt.get('detail', '')),
                         ' or '.join(sorted(EXPECTED.get(gt['type'], {'none'}))),
                         HOW[how].format(v=version, l=case['lang']), measured[cid].get('other_issues', measured[cid].get('false_alarms', 0))))
            print(folder, '<-', cid)

    md = ['# Test papers for IndicDoc Verify', '',
          'Each folder has an English source paper and its Hindi or Tamil version (18 questions, 3 pages). The version '
          'has one planted error, or none. Upload them in the app\'s **Verify Paper** tab (or `POST /verify`) and check '
          'that the report points at the question below. Set the language of the paper (Hindi / Tamil) and of the '
          'source (English), or leave auto-detect.', '',
          'These papers come from the test set of the verification benchmark (`scripts/verify_bench/`), on which the '
          'system found 98% of the planted errors. The report can also list a few other findings (warnings, '
          'occasional false alarms): the vetter confirms each one.', '',
          '"Other findings" is how many more the report listed for that pair in the benchmark run.', '',
          '| Folder | Language | Kind | Planted error | Expected finding | Other findings | How to check |',
          '|---|---|---|---|---|---|---|']
    md += [f'| `{f}` | {n} | {k} | {w} | {c} | {o} | {h} |' for f, n, k, w, c, h, o in rows]
    md += ['', 'API example (test 02):', '', '```bash',
           'curl -F file=@02_hindi_number_changed/hindi_version.pdf -F source=@02_hindi_number_changed/english_source.pdf \\',
           '     "http://localhost:8000/verify?lang=hin&source_lang=eng&output=html" > report.html', '```', '',
           'Rebuild: `python scripts/verify_bench/export_test_papers.py` (needs `data/verify_bench`, made by the benchmark '
           'scripts).', '']
    cbse_root = ROOT / 'data' / 'cbse_dev'
    if (cbse_root / 'cases.json').exists():
        cbse_cases = {c['id']: c for c in json.loads((cbse_root / 'cases.json').read_text())}
        md += ['## CBSE-format bilingual papers (one PDF, Hindi and English page by page)', '',
               'Each file is one paper in the Central Board\'s layout: bilingual cover, instructions in both languages, '
               'sections A-E (38 questions, 80 marks), a Hindi page followed by the same questions in English. Upload '
               'the one PDF in **One paper** mode: the two versions are found and checked against each other.', '',
               '| File | Planted error (in the Hindi version) | Expected finding |', '|---|---|---|']
        for name, cid in CBSE_EXAMPLES:
            case = cbse_cases.get(cid)
            if case is None:
                continue
            (out / name).parent.mkdir(exist_ok=True)
            shutil.copy(cbse_root / 'pdf' / f'{cid}.pdf', out / name)
            gt = case['gt']
            what = 'None: a correct paper.' if gt['type'] == 'clean' else f"Q{gt.get('question')}: {gt.get('detail', '')}."
            exp = {'clean': 'none', 'number_changed': 'NUMBER_MISMATCH', 'choice_dropped': 'CHOICE_MISMATCH',
                   'subpart_dropped': 'SUBQUESTION_MISMATCH'}.get(gt['type'], gt['type'])
            md.append(f'| `{name}` | {what} | {exp} |')
            print(name, '<-', cid)
        md.append('')
    (out / 'ANSWER_KEY.md').write_text('\n'.join(md), encoding='utf-8')
    print('written to', out)


if __name__ == '__main__':
    main()
