"""
Build the IndicDoc Verify benchmark: pairs of question papers (English source + Hindi or Tamil translation), each
with at most one injected error of a kind that has reached candidates in real Indian exams, written as printable
HTML pages for render.js.

Usage:
    python scripts/verify_bench/generate.py --out data/verify_bench --per_type 6 --clean 12
Then:
    node scripts/verify_bench/render.js data/verify_bench      # pages -> PNG + region boxes
    node scripts/verify_bench/ocr.js data/verify_bench         # Tesseract (tesseract.js) word boxes
    python scripts/evaluate_verify.py --bench data/verify_bench
"""
import argparse
import copy
import html
import json
import random
from pathlib import Path

from questions import FIXED_NUMERIC, FIXED_TEXT, HEADER, MARKERS, SHORT, TEMPLATES, numeric_options

ERROR_TYPES = ['number_changed', 'option_number_changed', 'option_dropped', 'question_dropped',
               'question_duplicated', 'marks_changed', 'total_wrong', 'numbering_error', 'bank_repeat']
N_MCQ, PER_PAGE = 15, [5, 7, 7]


# ------------------------------------------------------------------------------------- logical paper
def make_mcq_template(rng, t_idx=None):
    t_idx = rng.randrange(len(TEMPLATES)) if t_idx is None else t_idx
    v = TEMPLATES[t_idx]['values'](rng)
    return {'kind': 'tmpl', 't': t_idx, 'v': v, 'opts': numeric_options(TEMPLATES[t_idx]['answer'](v), rng)}


def make_paper(rng, bank_keys=frozenset()):
    """Language-independent paper: 15 MCQs (templated + fixed) and 3 short-answer questions."""
    qs, used_fixed = [], set()
    while len(qs) < N_MCQ:
        r = rng.random()
        if r < 0.6:
            q = make_mcq_template(rng)
            key = (q['t'], tuple(sorted(q['v'].items())))
            if key in bank_keys or any(x.get('key') == key for x in qs):
                continue
            q['key'] = key
        elif r < 0.8:
            i = rng.randrange(len(FIXED_NUMERIC))
            if ('fn', i) in used_fixed:
                continue
            used_fixed.add(('fn', i))
            q = {'kind': 'fixed_num', 'i': i, 'opts': list(FIXED_NUMERIC[i][1])}
            rng.shuffle(q['opts'])
        else:
            i = rng.randrange(len(FIXED_TEXT))
            if ('ft', i) in used_fixed:
                continue
            used_fixed.add(('ft', i))
            order = list(range(4)); rng.shuffle(order)
            q = {'kind': 'fixed_text', 'i': i, 'order': order}
        qs.append(q)
    for i in rng.sample(range(len(SHORT)), 3):
        qs.append({'kind': 'short', 'i': i, 'marks': rng.choice([2, 3, 5])})
    for k, q in enumerate(qs, 1):
        q['number'] = k
    style = rng.choice(['digit', 'letter'])
    total = N_MCQ + sum(q['marks'] for q in qs if q['kind'] == 'short')
    return {'questions': qs, 'total': total, 'style': style}


def render_question(q, lang, style):
    """Text of one question in one language: (stem, [options], marks or None)."""
    if q['kind'] in ('tmpl', 'bank'):
        t = TEMPLATES[q['t']]
        stem = t['stem'][lang].format(**q['v'])
        return stem, [t['unit'][lang].format(x=o) for o in q['opts']], None
    if q['kind'] == 'fixed_num':
        stem, _, unit = FIXED_NUMERIC[q['i']]
        return stem[lang], [unit.format(x=o) for o in q['opts']], None
    if q['kind'] == 'fixed_text':
        stem, opts = FIXED_TEXT[q['i']]
        return stem[lang], [opts[lang][k] for k in q['order']], None
    return SHORT[q['i']][lang], [], q['marks']


# ---------------------------------------------------------------------------------------- injection
def inject(paper, kind, rng, bank_questions):
    """Apply one error to the target paper (and the source too where the error is the paper-setter's).
    Returns (source, target, ground truth)."""
    src, tgt = copy.deepcopy(paper), copy.deepcopy(paper)
    qs = tgt['questions']
    mcq = [q for q in qs if q['kind'] != 'short']
    gt = {'type': kind}
    if kind == 'number_changed':
        q = rng.choice([q for q in qs if q['kind'] == 'tmpl'])
        # only a value that is printed in the stem (some templates keep a hidden value, e.g. the marked price)
        key = rng.choice(sorted(k for k in q['v'] if '{' + k + '}' in TEMPLATES[q['t']]['stem']['en']))
        old = q['v'][key]
        used = set(q['v'].values()) | set(q['opts'])
        new = next(x for x in rng.sample(range(2, 3 * old + 20), 3 * old + 18) if x not in used and x != old)
        q['v'][key] = new
        gt.update(question=q['number'], detail=f'{key}: {old} -> {new}')
    elif kind == 'option_number_changed':
        q = rng.choice([q for q in mcq if q['kind'] in ('tmpl', 'fixed_num')])
        k = rng.randrange(4)
        old = q['opts'][k]
        q['opts'][k] = next(x for x in (old + d for d in (rng.choice([1, 2, 10, 100]), 11, 13)) if x not in q['opts'])
        gt.update(question=q['number'], detail=f'option {k + 1}: {old} -> {q["opts"][k]}')
    elif kind == 'option_dropped':
        q = rng.choice(mcq)
        k = rng.randrange(4)
        if q['kind'] == 'fixed_text':
            q['order'].pop(k)
        else:
            q['opts'].pop(k)
        gt.update(question=q['number'], detail=f'option {k + 1} removed')
    elif kind == 'question_dropped':
        k = rng.randrange(1, len(qs) - 1)
        dropped = qs.pop(k)
        renumber = rng.random() < 0.5
        if renumber:
            for j, q in enumerate(qs, 1):
                q['number'] = j
        gt.update(question=dropped['number'], detail='renumbered' if renumber else 'number gap left')
    elif kind == 'question_duplicated':
        j, k = sorted(rng.sample(range(len(mcq)), 2))
        for paper_ in (src, tgt):
            m = [q for q in paper_['questions'] if q['kind'] != 'short']
            number = m[k]['number']
            m[k].clear(); m[k].update(copy.deepcopy(m[j])); m[k]['number'] = number
        gt.update(question=mcq[k]['number'], detail=f'repeats question {mcq[j]["number"]}', first=mcq[j]['number'])
    elif kind == 'marks_changed':
        q = rng.choice([q for q in qs if q['kind'] == 'short'])
        old = q['marks']
        q['marks'] = rng.choice([m for m in (1, 2, 3, 4, 5, 6) if m != old])
        gt.update(question=q['number'], detail=f'marks {old} -> {q["marks"]}')
    elif kind == 'total_wrong':
        old = tgt['total']
        tgt['total'] = old + rng.choice([-10, -5, 5, 10])
        gt.update(question=None, detail=f'maximum marks {old} -> {tgt["total"]}')
    elif kind == 'numbering_error':
        k = rng.randrange(1, len(qs) - 1)
        qs[k]['number'] = qs[k - 1]['number']
        gt.update(question=k + 1, detail=f'question {k + 1} printed as {qs[k]["number"]}')
    elif kind == 'bank_repeat':
        b = rng.choice(bank_questions)
        k = rng.choice([i for i, q in enumerate(qs) if q['kind'] != 'short'])
        for paper_ in (src, tgt):
            number = paper_['questions'][k]['number']
            paper_['questions'][k] = {**copy.deepcopy(b), 'number': number}
        gt.update(question=qs[k]['number'], detail=f'repeats {b["source"]}')
    return src, tgt, gt


# ------------------------------------------------------------------------------------------------ HTML
FONTS = {'en': "'Noto Serif', 'Noto Sans', serif", 'hi': "'Noto Sans Devanagari', 'Noto Sans', sans-serif",
         'ta': "'Noto Sans Tamil', 'Noto Sans', sans-serif"}


def paper_html(paper, lang, fonts):
    """Fixed-size pages (1240 x 1754 px = A4 at 150 dpi); every element the layout model would detect carries its
    IndicDLP label in data-label, so render.js can record the boxes."""
    style = paper['style']
    marker = MARKERS[style][lang]
    blocks, section_b_started = [], False
    for q in paper['questions']:
        stem, opts, marks = render_question(q, lang, style)
        if q['kind'] == 'short' and not section_b_started:
            blocks.append(f'<div class="sec" data-label="section-title">{HEADER["section_b"][lang]}</div>')
            section_b_started = True
        m = f'<span class="mk">{HEADER["marks"][lang].format(m=marks)}</span>' if marks else ''
        qhtml = f'<div class="q" data-label="first-level-question"><b>{q["number"]}.</b> {html.escape(stem)}{m}</div>'
        if opts:
            cells = ''.join(f'<span>{marker[i]} {html.escape(o)}</span>' for i, o in enumerate(opts))
            qhtml += f'<div class="opts" data-label="options">{cells}</div>'
        blocks.append(f'<div class="blk">{qhtml}</div>')
    header = (f'<div class="title" data-label="headline">{HEADER["title"][lang]}</div>'
              f'<div class="meta" data-label="paragraph">{HEADER["meta"][lang].format(total=paper["total"])}</div>'
              f'<div class="ins" data-label="paragraph">{HEADER["instructions"][lang]}</div>'
              f'<div class="sec" data-label="section-title">{HEADER["section_a"][lang]}</div>')
    pages, i, p = [], 0, 0
    while i < len(blocks):
        n = PER_PAGE[min(p, len(PER_PAGE) - 1)]
        chunk = blocks[i:i + n]
        # keep a section title with the question after it
        if chunk and chunk[-1].startswith('<div class="sec"'):
            chunk = chunk[:-1]
        i += len(chunk)
        body = (header if p == 0 else '') + ''.join(chunk)
        pages.append(f'<div class="page" data-page="{p}">{body}<div class="pn" data-label="page-number">{p + 1}</div></div>')
        p += 1
    return f'''<!doctype html><html><head><meta charset="utf-8"><style>
@font-face {{ font-family: 'Noto Serif'; src: url('{fonts}/noto-serif/files/noto-serif-latin-400-normal.woff2'); }}
@font-face {{ font-family: 'Noto Serif'; font-weight: 700; src: url('{fonts}/noto-serif/files/noto-serif-latin-700-normal.woff2'); }}
@font-face {{ font-family: 'Noto Sans'; src: url('{fonts}/noto-sans/files/noto-sans-latin-400-normal.woff2'); }}
@font-face {{ font-family: 'Noto Sans Devanagari'; src: url('{fonts}/noto-sans-devanagari/files/noto-sans-devanagari-devanagari-400-normal.woff2'); }}
@font-face {{ font-family: 'Noto Sans Devanagari'; font-weight: 700; src: url('{fonts}/noto-sans-devanagari/files/noto-sans-devanagari-devanagari-700-normal.woff2'); }}
@font-face {{ font-family: 'Noto Sans Tamil'; src: url('{fonts}/noto-sans-tamil/files/noto-sans-tamil-tamil-400-normal.woff2'); }}
@font-face {{ font-family: 'Noto Sans Tamil'; font-weight: 700; src: url('{fonts}/noto-sans-tamil/files/noto-sans-tamil-tamil-700-normal.woff2'); }}
body {{ margin: 0; background: #888; }}
.page {{ width: 1240px; height: 1754px; background: #fff; position: relative; box-sizing: border-box; padding: 90px 100px;
  font-family: {FONTS[lang]}; font-size: 25px; line-height: 1.5; color: #111; margin-bottom: 20px; overflow: hidden; }}
.title {{ text-align: center; font-size: 34px; font-weight: 700; margin-bottom: 18px; }}
.meta {{ text-align: center; margin-bottom: 14px; white-space: pre; }}
.ins {{ margin-bottom: 26px; }}
.sec {{ font-weight: 700; font-size: 28px; margin: 22px 0 14px; display: inline-block; }}
.blk {{ margin-bottom: 30px; }}
.q {{ display: block; }} .mk {{ float: right; margin-left: 20px; }}
.opts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px 40px; margin: 10px 0 0 34px; }}
.pn {{ position: absolute; bottom: 60px; left: 0; right: 0; text-align: center; width: 40px; margin: 0 auto; }}
</style></head><body>{''.join(pages)}</body></html>'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/verify_bench')
    ap.add_argument('--per_type', type=int, default=6)
    ap.add_argument('--clean', type=int, default=12)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    fonts = (Path(__file__).resolve().parent / 'node_modules' / '@fontsource').as_uri()
    (out / 'html').mkdir(parents=True, exist_ok=True)

    # "Last year's" papers: the question bank the repeat check searches
    bank_logical = []
    for b in range(3):
        for q in make_paper(rng)['questions']:
            if q['kind'] == 'tmpl':
                bank_logical.append({**q, 'kind': 'bank', 'source': f'Model paper 2025 set {b + 1}, Q{q["number"]}'})
    bank_keys = frozenset(q['key'] for q in bank_logical)
    bank = {lang: [{'text': render_question(q, lang, 'digit')[0], 'source': q['source']} for q in bank_logical]
            for lang in ('hi', 'ta')}
    (out / 'bank.json').write_text(json.dumps(bank, ensure_ascii=False, indent=1))

    cases = []
    for lang in ('hi', 'ta'):
        kinds = ['clean'] * args.clean + [k for k in ERROR_TYPES for _ in range(args.per_type)]
        for n, kind in enumerate(kinds):
            paper = make_paper(rng, bank_keys)
            if kind == 'clean':
                src, tgt, gt = paper, paper, {'type': 'clean', 'question': None}
            else:
                src, tgt, gt = inject(paper, kind, rng, bank_logical)
            cid = f'{lang}_{n:03d}_{kind}'
            for side, p, lg in (('src', src, 'en'), ('tgt', tgt, lang)):
                (out / 'html' / f'{cid}_{side}.html').write_text(paper_html(p, lg, fonts))
            degraded = rng.random() < 0.5
            cases.append({'id': cid, 'lang': lang, 'gt': gt, 'degraded': degraded})
    (out / 'cases.json').write_text(json.dumps(cases, ensure_ascii=False, indent=1))
    print(f'{len(cases)} paper pairs written to {out}')


if __name__ == '__main__':
    main()
