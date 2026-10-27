"""
CBSE-format bilingual papers for the IndicDoc Verify benchmark: one file per paper, Hindi and English interleaved page
by page, as the Central Board prints them.

Every paper has the board's structure: a bilingual cover (maximum marks, "contains N printed pages", "38
questions"), general instructions in Hindi and in English that give the marks of each block of questions, and five
sections: A - 18 MCQs and 2 assertion-reason questions (1 mark), B - 5 very short answer questions (2 marks), C - 6
short answer questions (3 marks), D - 4 long answer questions (5 marks), E - 3 case studies with sub-parts (4 marks);
internal choice ('OR' / 'अथवा') in two questions of B, C and D. 38 questions, 80 marks. Each Hindi page is followed by
the English page with the same questions; running header and 'Page X of N' footer on every page.

One error of a kind that reaches candidates is planted in the Hindi version (or in both, where it is the setter's),
or none. Each paper is written as one HTML file for render.js (page images + layout boxes); make_cbse_pdfs() prints the
same HTML to PDF, whose text layer is the born-digital input of reader B.

Usage:
    python scripts/verify_bench/cbse.py --out data/cbse_bench --per_type 3 --clean 6 --seed 42
    node scripts/verify_bench/render.js data/cbse_bench
    python scripts/verify_bench/cbse_prepare.py data/cbse_bench      # PDF text layer, degradation, OCR
    python scripts/evaluate_consensus.py --bench data/cbse_bench
"""
import argparse
import copy
import html
import json
import random
from pathlib import Path

from generate import make_mcq_template
from questions import FIXED_NUMERIC, FIXED_TEXT, SHORT, TEMPLATES

ERROR_TYPES = ['number_changed', 'option_number_changed', 'option_dropped', 'question_dropped', 'marks_changed',
               'total_wrong', 'numbering_error', 'question_duplicated', 'subpart_dropped', 'choice_dropped']
SECTIONS = [('A', 1, 20, 1), ('B', 21, 25, 2), ('C', 26, 31, 3), ('D', 32, 35, 5), ('E', 36, 38, 4)]
SEC_HI = dict(zip('ABCDE', 'कखगघङ'))
QP_CODE = '465'

AR = [  # assertion, reason (both languages)
    ({'en': 'The sum of the angles of a triangle is 180 degrees.',
      'hi': 'किसी त्रिभुज के कोणों का योग 180 अंश होता है।'},
     {'en': 'An exterior angle of a triangle is equal to the sum of the two opposite interior angles.',
      'hi': 'त्रिभुज का बाह्य कोण दो सम्मुख अंतः कोणों के योग के बराबर होता है।'}),
    ({'en': 'Pure water boils at 100 °C at sea level.',
      'hi': 'समुद्र तल पर शुद्ध जल 100 °C पर उबलता है।'},
     {'en': 'The boiling point of a liquid decreases when the pressure on it increases.',
      'hi': 'किसी द्रव पर दाब बढ़ने से उसका क्वथनांक घटता है।'}),
    ({'en': 'A leap year has 366 days.', 'hi': 'एक अधिवर्ष में 366 दिन होते हैं।'},
     {'en': 'In a leap year, February has 29 days.', 'hi': 'अधिवर्ष में फरवरी में 29 दिन होते हैं।'}),
    ({'en': 'The number 91 is a prime number.', 'hi': 'संख्या 91 एक अभाज्य संख्या है।'},
     {'en': 'A prime number has exactly two factors.', 'hi': 'एक अभाज्य संख्या के ठीक दो गुणनखंड होते हैं।'}),
]
AR_PREAMBLE = {
    'en': ('Questions number 19 and 20 are Assertion - Reason based questions of 1 mark each. Two statements are '
           'given - one labelled Assertion (A) and the other labelled Reason (R). Select the correct answer to these '
           'questions from the codes (A), (B), (C) and (D) as given below :',
           ['Both Assertion (A) and Reason (R) are true and Reason (R) is the correct explanation of Assertion (A).',
            'Both Assertion (A) and Reason (R) are true, but Reason (R) is not the correct explanation of Assertion (A).',
            'Assertion (A) is true, but Reason (R) is false.', 'Assertion (A) is false, but Reason (R) is true.']),
    'hi': ('प्रश्न संख्या 19 और 20 अभिकथन एवं तर्क आधारित प्रश्न हैं और प्रत्येक प्रश्न का 1 अंक है। दो कथन दिए गए हैं जिनमें '
           'एक को अभिकथन (A) तथा दूसरे को तर्क (R) द्वारा अंकित किया गया है। इन प्रश्नों के सही उत्तर नीचे दिए गए कोडों (A), '
           '(B), (C) और (D) में से चुनकर दीजिए :',
           ['अभिकथन (A) और तर्क (R) दोनों सही हैं और तर्क (R), अभिकथन (A) की सही व्याख्या करता है।',
            'अभिकथन (A) और तर्क (R) दोनों सही हैं, परन्तु तर्क (R), अभिकथन (A) की सही व्याख्या नहीं करता है।',
            'अभिकथन (A) सही है, परन्तु तर्क (R) ग़लत है।', 'अभिकथन (A) ग़लत है, परन्तु तर्क (R) सही है।']),
}
CASES = [
    {'values': lambda r: {'b': r.choice([400, 500, 600]), 'n': r.choice([30, 40, 50]), 'r': r.choice([5, 10]),
                          't': r.choice([800, 900, 1000])},
     'stem': {'en': 'A school library has {b} books. Every month {n} new books are added and {r} old books are removed.',
              'hi': 'एक विद्यालय के पुस्तकालय में {b} पुस्तकें हैं। प्रत्येक महीने {n} नई पुस्तकें जोड़ी जाती हैं और {r} पुरानी पुस्तकें हटाई जाती हैं।'},
     'parts': [({'en': 'How many books will the library have after 1 month?', 'hi': '1 महीने बाद पुस्तकालय में कितनी पुस्तकें होंगी?'}, 1),
               ({'en': 'How many books will the library have after 3 months?', 'hi': '3 महीने बाद पुस्तकालय में कितनी पुस्तकें होंगी?'}, 1),
               ({'en': 'After how many months will the library have {t} books?', 'hi': 'कितने महीनों बाद पुस्तकालय में {t} पुस्तकें होंगी?'}, 2)]},
    {'values': lambda r: {'p': r.choice([10, 12, 15]), 'q': r.choice([40, 45, 60]), 'a': r.choice([3, 4, 5]),
                          'c': r.choice([2, 6, 8]), 'd': r.choice([5, 10, 20])},
     'stem': {'en': 'A shop sells a pen for Rs {p} and a notebook for Rs {q}. A student buys {a} pens and {c} notebooks.',
              'hi': 'एक दुकान में एक पेन {p} रुपये का और एक कॉपी {q} रुपये की मिलती है। एक विद्यार्थी {a} पेन और {c} कॉपियाँ खरीदता है।'},
     'parts': [({'en': 'What is the cost of the pens?', 'hi': 'पेनों का मूल्य कितना है?'}, 1),
               ({'en': 'What is the cost of the notebooks?', 'hi': 'कॉपियों का मूल्य कितना है?'}, 1),
               ({'en': 'What is the total cost after a discount of {d}%?', 'hi': '{d}% छूट के बाद कुल मूल्य कितना होगा?'}, 2)]},
    {'values': lambda r: {'v': r.choice([600, 900, 1200]), 'f': r.choice([30, 60]), 'e': r.choice([10, 20]),
                          'm': r.choice([5, 15, 25])},
     'stem': {'en': 'A tank holds {v} litres of water. One pipe fills it at {f} litres per minute and another pipe empties it at {e} litres per minute.',
              'hi': 'एक टंकी में {v} लीटर पानी आता है। एक पाइप इसे {f} लीटर प्रति मिनट की दर से भरता है और दूसरा पाइप इसे {e} लीटर प्रति मिनट की दर से खाली करता है।'},
     'parts': [({'en': 'How long does the first pipe take to fill the empty tank?', 'hi': 'पहला पाइप खाली टंकी को कितने समय में भरेगा?'}, 1),
               ({'en': 'How much water flows out through the second pipe in {m} minutes?', 'hi': 'दूसरे पाइप से {m} मिनट में कितना पानी निकलेगा?'}, 1),
               ({'en': 'If both pipes are open, how long will the empty tank take to fill?', 'hi': 'दोनों पाइप खुले हों तो खाली टंकी कितने समय में भरेगी?'}, 2)]},
]
DESC_SUFFIX = {'en': ' Show your working.', 'hi': ' अपना हल लिखिए।'}
T = {  # fixed texts
    'title': {'en': 'SCIENCE AND MATHEMATICS', 'hi': 'विज्ञान एवं गणित'},
    'time': {'en': 'Time allowed : 3 hours', 'hi': 'निर्धारित समय : 3 घण्टे'},
    'max': {'en': 'Maximum Marks : {m}', 'hi': 'अधिकतम अंक : {m}'},
    'pages': {'en': 'Please check that this question paper contains {p} printed pages.',
              'hi': 'कृपया जाँच कर लें कि इस प्रश्न-पत्र में मुद्रित पृष्ठ {p} हैं।'},
    'nq': {'en': 'Please check that this question paper contains {n} questions.',
           'hi': 'कृपया जाँच कर लें कि इस प्रश्न-पत्र में {n} प्रश्न हैं।'},
    'gi': {'en': 'General Instructions :', 'hi': 'सामान्य निर्देश :'},
    'gi_lead': {'en': 'Read the following instructions very carefully and strictly follow them :',
                'hi': 'निम्नलिखित निर्देशों को बहुत सावधानी से पढ़िए और उनका सख़्ती से पालन कीजिए :'},
    'or': {'en': 'OR', 'hi': 'अथवा'},
    'section': {'en': 'SECTION - {s}', 'hi': 'खण्ड - {s}'},
}
SEC_INTRO = {
    'A': {'en': 'This section comprises 20 Multiple Choice Questions (MCQs) of 1 mark each.',
          'hi': 'इस खण्ड में 20 बहुविकल्पीय प्रश्न (MCQ) हैं, जिनमें प्रत्येक प्रश्न 1 अंक का है।'},
    'B': {'en': 'This section comprises 5 Very Short Answer (VSA) type questions of {m} marks each.',
          'hi': 'इस खण्ड में 5 अति लघु-उत्तरीय (VSA) प्रकार के प्रश्न हैं, जिनमें प्रत्येक के {m} अंक हैं।'},
    'C': {'en': 'This section comprises 6 Short Answer (SA) type questions of {m} marks each.',
          'hi': 'इस खण्ड में 6 लघु-उत्तरीय (SA) प्रकार के प्रश्न हैं, जिनमें प्रत्येक के {m} अंक हैं।'},
    'D': {'en': 'This section comprises 4 Long Answer (LA) type questions of {m} marks each.',
          'hi': 'इस खण्ड में 4 दीर्घ-उत्तरीय (LA) प्रकार के प्रश्न हैं, जिनमें प्रत्येक के {m} अंक हैं।'},
    'E': {'en': 'This section comprises 3 case study based questions of {m} marks each.',
          'hi': 'इस खण्ड में 3 प्रकरण अध्ययन आधारित प्रश्न हैं, जिनमें प्रत्येक के {m} अंक हैं।'},
}


def instructions(marks, lang):
    m = marks
    if lang == 'en':
        return ['This question paper contains 38 questions. All questions are compulsory.',
                'This question paper is divided into five Sections - A, B, C, D and E.',
                f'In Section - A, Questions Number 1 to 18 are Multiple Choice Questions (MCQs) and Questions Number 19 and 20 are Assertion-Reason based questions of {m["A"]} mark each.',
                f'In Section - B, Questions Number 21 to 25 are Very Short Answer (VSA) type questions, carrying {m["B"]} marks each.',
                f'In Section - C, Questions Number 26 to 31 are Short Answer (SA) type questions, carrying {m["C"]} marks each.',
                f'In Section - D, Questions Number 32 to 35 are Long Answer (LA) type questions, carrying {m["D"]} marks each.',
                f'In Section - E, Questions Number 36 to 38 are case study based questions, carrying {m["E"]} marks each.',
                'There is no overall choice. However, an internal choice has been provided in 2 questions in Section - B, 2 questions in Section - C and 2 questions in Section - D.',
                'Use of calculators is NOT allowed.']
    return ['इस प्रश्न-पत्र में 38 प्रश्न हैं। सभी प्रश्न अनिवार्य हैं।',
            'यह प्रश्न-पत्र पाँच खण्डों में विभाजित है - क, ख, ग, घ एवं ङ।',
            f'खण्ड - क में प्रश्न संख्या 1 से 18 तक बहुविकल्पीय तथा प्रश्न संख्या 19 एवं 20 अभिकथन एवं तर्क आधारित {m["A"]} अंक के प्रश्न हैं।',
            f'खण्ड - ख में प्रश्न संख्या 21 से 25 तक अति लघु-उत्तरीय (VSA) प्रकार के {m["B"]} अंकों के प्रश्न हैं।',
            f'खण्ड - ग में प्रश्न संख्या 26 से 31 तक लघु-उत्तरीय (SA) प्रकार के {m["C"]} अंकों के प्रश्न हैं।',
            f'खण्ड - घ में प्रश्न संख्या 32 से 35 तक दीर्घ-उत्तरीय (LA) प्रकार के {m["D"]} अंकों के प्रश्न हैं।',
            f'खण्ड - ङ में प्रश्न संख्या 36 से 38 तक प्रकरण अध्ययन आधारित {m["E"]} अंकों के प्रश्न हैं।',
            'प्रश्न-पत्र में समग्र विकल्प नहीं दिया गया है। यद्यपि, खण्ड - ख के 2 प्रश्नों में, खण्ड - ग के 2 प्रश्नों में तथा खण्ड - घ के 2 प्रश्नों में आंतरिक विकल्प का प्रावधान दिया गया है।',
            'कैल्कुलेटर का उपयोग वर्जित है।']


# ------------------------------------------------------------------------------------- logical paper
def _desc_item(rng, used):
    """A descriptive question: a numeric template stem (asked to be worked out) or a short-answer prompt."""
    while True:
        if rng.random() < 0.6:
            q = make_mcq_template(rng)
            key = ('m', q['t'], tuple(sorted(q['v'].items())))
            item = {'kind': 'tmpl', 't': q['t'], 'v': q['v']}
        else:
            i = rng.randrange(len(SHORT))
            key = ('s', i)
            item = {'kind': 'short', 'i': i}
        if key not in used:
            used.add(key)
            return item


def make_paper(rng):
    qs, used, fixed = [], set(), set()
    while len(qs) < 18:
        r = rng.random()
        if r < 0.65:
            q = make_mcq_template(rng)
            key = ('m', q['t'], tuple(sorted(q['v'].items())))
            if key in used:
                continue
            used.add(key)
            qs.append({'type': 'mcq', **q})
        elif r < 0.85:
            i = rng.randrange(len(FIXED_NUMERIC))
            if ('fn', i) in fixed:
                continue
            fixed.add(('fn', i))
            opts = list(FIXED_NUMERIC[i][1]); rng.shuffle(opts)
            qs.append({'type': 'mcq', 'kind': 'fixed_num', 'i': i, 'opts': opts})
        else:
            i = rng.randrange(len(FIXED_TEXT))
            if ('ft', i) in fixed:
                continue
            fixed.add(('ft', i))
            order = list(range(4)); rng.shuffle(order)
            qs.append({'type': 'mcq', 'kind': 'fixed_text', 'i': i, 'order': order})
    for i in rng.sample(range(len(AR)), 2):
        qs.append({'type': 'ar', 'i': i})
    marks = {s: m for s, _, _, m in SECTIONS}
    for sec, lo, hi, m in SECTIONS[1:4]:
        choice_at = set(rng.sample(range(lo, hi + 1), 2))
        for n in range(lo, hi + 1):
            items = [_desc_item(rng, used)] + ([_desc_item(rng, used)] if n in choice_at else [])
            qs.append({'type': 'desc', 'items': items})
    for i in range(3):
        c = CASES[i]
        qs.append({'type': 'case', 'i': i, 'v': c['values'](rng), 'parts': [0, 1, 2]})
    for k, q in enumerate(qs, 1):
        q['number'] = k
        q['section'] = next(s for s, lo, hi, _ in SECTIONS if lo <= k <= hi)
    return {'questions': qs, 'marks': marks, 'total': 80, 'qp': QP_CODE}


# ------------------------------------------------------------------------------------------- rendering
def _mcq_text(q, lang):
    if q['kind'] in ('tmpl', 'bank'):
        t = TEMPLATES[q['t']]
        return t['stem'][lang].format(**q['v']), [t['unit'][lang].format(x=o) for o in q['opts']]
    if q['kind'] == 'fixed_num':
        stem, _, unit = FIXED_NUMERIC[q['i']]
        return stem[lang], [unit.format(x=o) for o in q['opts']]
    stem, opts = FIXED_TEXT[q['i']]
    return stem[lang], [opts[lang][k] for k in q['order']]


def _desc_text(item, lang):
    if item['kind'] == 'tmpl':
        return TEMPLATES[item['t']]['stem'][lang].format(**item['v']) + DESC_SUFFIX[lang]
    return SHORT[item['i']][lang]


def _num(q):
    return q.get('printed', q['number'])


def question_html(q, lang):
    """HTML of one question; data-label marks what the layout model would detect (IndicDLP labels)."""
    n = _num(q)
    e = html.escape
    if q['type'] == 'mcq':
        stem, opts = _mcq_text(q, lang)
        cells = ''.join(f'<span>({"ABCD"[k]}) {e(o)}</span>' for k, o in enumerate(opts))
        return (f'<div class="blk"><div class="q" data-label="first-level-question"><b>{n}.</b> {e(stem)}</div>'
                f'<div class="opts" data-label="options">{cells}</div></div>')
    if q['type'] == 'ar':
        a, r = AR[q['i']]
        la, lr = ('Assertion (A) :', 'Reason (R) :') if lang == 'en' else ('अभिकथन (A) :', 'तर्क (R) :')
        return (f'<div class="blk"><div class="q" data-label="first-level-question"><b>{n}.</b> <b>{la}</b> {e(a[lang])}'
                f'<br><b>{lr}</b> {e(r[lang])}</div></div>')
    if q['type'] == 'desc':
        items = q['items']
        if len(items) == 1:
            return f'<div class="blk"><div class="q" data-label="first-level-question"><b>{n}.</b> {e(_desc_text(items[0], lang))}</div></div>'
        out = (f'<div class="blk"><div class="q" data-label="first-level-question"><b>{n}.</b> (a) {e(_desc_text(items[0], lang))}</div>')
        if not q.get('choice_dropped'):
            out += (f'<div class="or" data-label="paragraph">{T["or"][lang]}</div>'
                    f'<div class="sub" data-label="second-level-question">(b) {e(_desc_text(items[1], lang))}</div>')
        return out + '</div>'
    c = CASES[q['i']]
    out = f'<div class="blk"><div class="q" data-label="first-level-question"><b>{n}.</b> {e(c["stem"][lang].format(**q["v"]))}</div>'
    for k, part in enumerate(q['parts']):
        text, mk = c['parts'][part]
        out += (f'<div class="sub" data-label="second-level-question">({"i" * (k + 1) if k < 3 else "iv"}) '
                f'{e(text[lang].format(**q["v"]))}<span class="mk">[{mk}]</span></div>')
    return out + '</div>'


# questions per page pair, in order: MCQs 1-18, AR 19-20, B, C, D, E
CHUNKS = [(1, 5), (6, 10), (11, 14), (15, 18), (19, 20), (21, 23), (24, 25), (26, 28), (29, 31), (32, 33),
          (34, 35), (36, 36), (37, 37), (38, 38)]


def pages_of(paper):
    """Page plan: cover, Hindi instructions, English instructions, then a Hindi and an English page per chunk."""
    plan = [('cover', None, None), ('instr', 'hi', None), ('instr', 'en', None)]
    for chunk in CHUNKS:
        plan += [('q', 'hi', chunk), ('q', 'en', chunk)]
    return plan


def paper_html(versions, fonts, total_pages_note=None):
    """versions: {'en': paper, 'hi': paper}. One HTML document with every page of the bilingual paper."""
    plan = pages_of(versions['en'])
    n_pages = len(plan)
    note = total_pages_note or n_pages
    pages = []
    for p, (kind, lang, chunk) in enumerate(plan, 1):
        body = ''
        if kind == 'cover':
            en, hi = versions['en'], versions['hi']
            body = (f'<div class="cover"><div class="code" data-label="paragraph">Q.P. Code {QP_CODE}</div>'
                    f'<div class="title" data-label="headline">{T["title"]["hi"]}</div>'
                    f'<div class="title" data-label="headline">{T["title"]["en"]}</div>'
                    f'<div class="row"><span data-label="paragraph">{T["time"]["hi"]}</span>'
                    f'<span data-label="paragraph">{T["max"]["hi"].format(m=hi["total"])}</span></div>'
                    f'<div class="row"><span data-label="paragraph">{T["time"]["en"]}</span>'
                    f'<span data-label="paragraph">{T["max"]["en"].format(m=en["total"])}</span></div>'
                    f'<div class="notes"><div data-label="paragraph">(I) {T["pages"]["hi"].format(p=note)}<br>'
                    f'(II) {T["nq"]["hi"].format(n=38)}</div>'
                    f'<div data-label="paragraph">(I) {T["pages"]["en"].format(p=note)}<br>'
                    f'(II) {T["nq"]["en"].format(n=38)}</div></div></div>')
        elif kind == 'instr':
            paper = versions[lang]
            items = ''.join(f'<li>{html.escape(t)}</li>' for t in instructions(paper['marks'], lang))
            body = (f'<div class="gi" data-label="section-title">{T["gi"][lang]}</div>'
                    f'<div data-label="paragraph">{T["gi_lead"][lang]}</div>'
                    f'<ol class="ins" type="i" data-label="ordered-list">{items}</ol>')
        else:
            paper = versions[lang]
            lo, hi_ = chunk
            qs = [q for q in paper['questions'] if lo <= q['number'] <= hi_]
            for q in qs:
                first_of_section = q['number'] in (s[1] for s in SECTIONS)
                if first_of_section:
                    sec = q['section']
                    label = T['section'][lang].format(s=sec if lang == 'en' else SEC_HI[sec])
                    body += (f'<div class="sec" data-label="section-title">{label}</div>'
                             f'<div class="intro" data-label="paragraph">{html.escape(SEC_INTRO[sec][lang].format(m=paper["marks"][sec]))}</div>')
                if q['type'] == 'ar' and q['number'] == 19:
                    lead, opts = AR_PREAMBLE[lang]
                    body += (f'<div class="pre" data-label="paragraph">{html.escape(lead)}'
                             + ''.join(f'<div>({"ABCD"[k]}) {html.escape(o)}</div>' for k, o in enumerate(opts)) + '</div>')
                body += question_html(q, lang)
        lang_cls = lang or 'hi'
        pages.append(f'<div class="page {lang_cls}" data-page="{p - 1}">'
                     f'<div class="hdr" data-label="header">* &#9632; *</div>{body}'
                     f'<div class="ftr" data-label="footer"><span>{QP_CODE}</span><span data-label="page-number">Page {p} of {n_pages}</span>'
                     f'<span>{"P.T.O." if p < n_pages else ""}</span></div></div>')
    return f'''<!doctype html><html><head><meta charset="utf-8"><style>
@font-face {{ font-family: 'Noto Serif'; src: url('{fonts}/noto-serif/files/noto-serif-latin-400-normal.woff2'); }}
@font-face {{ font-family: 'Noto Serif'; font-weight: 700; src: url('{fonts}/noto-serif/files/noto-serif-latin-700-normal.woff2'); }}
@font-face {{ font-family: 'Noto Sans Devanagari'; src: url('{fonts}/noto-sans-devanagari/files/noto-sans-devanagari-devanagari-400-normal.woff2'); }}
@font-face {{ font-family: 'Noto Sans Devanagari'; font-weight: 700; src: url('{fonts}/noto-sans-devanagari/files/noto-sans-devanagari-devanagari-700-normal.woff2'); }}
body {{ margin: 0; background: #888; }}
.page {{ width: 1240px; height: 1754px; background: #fff; position: relative; box-sizing: border-box; padding: 130px 110px 120px;
  font-family: 'Noto Serif', 'Noto Sans Devanagari', serif; font-size: 24px; line-height: 1.5; color: #111; margin-bottom: 20px; overflow: hidden; }}
.page.hi {{ font-family: 'Noto Sans Devanagari', 'Noto Serif', sans-serif; }}
.hdr {{ position: absolute; top: 45px; left: 0; right: 0; text-align: center; font-size: 26px; }}
.ftr {{ position: absolute; bottom: 55px; left: 110px; right: 110px; display: flex; justify-content: space-between; font-size: 22px; }}
.title {{ text-align: center; font-size: 34px; font-weight: 700; margin: 10px 0; }}
.code {{ text-align: right; font-weight: 700; margin-bottom: 30px; }}
.row {{ display: flex; justify-content: space-between; margin: 6px 0; }}
.notes {{ display: grid; grid-template-columns: 1fr 1fr; gap: 50px; margin-top: 60px; font-size: 22px; }}
.gi {{ font-weight: 700; font-size: 26px; }}
.ins li {{ margin: 8px 0; }}
.sec {{ text-align: center; font-weight: 700; font-size: 26px; margin: 10px 0 6px; }}
.intro {{ margin-bottom: 22px; }}
.pre {{ margin-bottom: 22px; }} .pre div {{ margin-left: 36px; }}
.blk {{ margin-bottom: 26px; }}
.q {{ display: block; padding-left: 46px; text-indent: -46px; }} .q b:first-child {{ display: inline-block; width: 46px; text-indent: 0; }}
.opts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4px 40px; margin: 8px 0 0 46px; }}
.or {{ text-align: center; font-weight: 700; margin: 8px 0; }}
.sub {{ margin: 6px 0 0 46px; }} .mk {{ float: right; margin-left: 20px; }}
</style></head><body>{''.join(pages)}</body></html>'''


# ---------------------------------------------------------------------------------------- injection
def inject(paper, kind, rng):
    """(english, hindi, ground truth): the error is planted in the Hindi version, or in both for setter errors."""
    en, hi = copy.deepcopy(paper), copy.deepcopy(paper)
    gt = {'type': kind}
    mcq = [q for q in hi['questions'] if q['type'] == 'mcq']
    if kind == 'number_changed':
        q = rng.choice([q for q in mcq if q['kind'] == 'tmpl'])
        key = rng.choice(sorted(k for k in q['v'] if '{' + k + '}' in TEMPLATES[q['t']]['stem']['en']))
        old = q['v'][key]
        used = set(q['v'].values()) | set(q['opts'])
        same_template = [x['v'] for x in hi['questions'] if x is not q and x.get('t') == q['t'] and x['type'] == 'mcq']
        q['v'][key] = next(x for x in rng.sample(range(2, 3 * old + 20), 3 * old + 18)
                           if x not in used and x != old and {**q['v'], key: x} not in same_template)
        gt.update(question=q['number'], detail=f'{key}: {old} -> {q["v"][key]}')
    elif kind == 'option_number_changed':
        q = rng.choice([q for q in mcq if q['kind'] in ('tmpl', 'fixed_num')])
        k = rng.randrange(4)
        old = q['opts'][k]
        q['opts'][k] = next(x for x in (old + d for d in (rng.choice([1, 2, 10, 100]), 11, 13)) if x not in q['opts'])
        gt.update(question=q['number'], detail=f'option {"ABCD"[k]}: {old} -> {q["opts"][k]}')
    elif kind == 'option_dropped':
        q = rng.choice(mcq)
        k = rng.randrange(4)
        (q['order'] if q['kind'] == 'fixed_text' else q['opts']).pop(k)
        gt.update(question=q['number'], detail=f'option {"ABCD"[k]} removed')
    elif kind == 'question_dropped':
        q = rng.choice([q for q in hi['questions'] if q['type'] in ('mcq', 'desc') and q['number'] not in (1, 38)])
        hi['questions'].remove(q)
        gt.update(question=q['number'], detail='question missing, number gap left')
    elif kind == 'marks_changed':
        sec = rng.choice(['B', 'C', 'D'])
        old = hi['marks'][sec]
        hi['marks'][sec] = old + rng.choice([-1, 1])
        first = next(lo for s, lo, _, _ in SECTIONS if s == sec)
        gt.update(question=first, detail=f'Section {sec}: {old} -> {hi["marks"][sec]} marks each (Hindi instructions)')
    elif kind == 'total_wrong':
        hi['total'] = rng.choice([70, 75, 85, 90])
        gt.update(question=None, detail=f'maximum marks 80 -> {hi["total"]} (Hindi cover)')
    elif kind == 'numbering_error':
        q = rng.choice([q for q in hi['questions'] if 2 < q['number'] < 37])
        q['printed'] = q['number'] - 1
        gt.update(question=q['number'], detail=f'question {q["number"]} printed as {q["printed"]} (Hindi)')
    elif kind == 'question_duplicated':
        a, b = sorted(rng.sample([q['number'] for q in mcq], 2))
        for version in (en, hi):
            qs = version['questions']
            src = copy.deepcopy(qs[a - 1])
            src['number'], src['section'] = b, qs[b - 1]['section']
            qs[b - 1] = src
        gt.update(question=b, detail=f'question {b} repeats question {a} (both versions)', first=a)
    elif kind == 'subpart_dropped':
        q = rng.choice([q for q in hi['questions'] if q['type'] == 'case'])
        k = rng.randrange(3)
        q['parts'].pop(k)
        gt.update(question=q['number'], detail=f'case-study part ({"i" * (k + 1)}) missing (Hindi)')
    elif kind == 'choice_dropped':
        q = rng.choice([q for q in hi['questions'] if q['type'] == 'desc' and len(q['items']) == 2])
        q['choice_dropped'] = True
        gt.update(question=q['number'], detail='internal choice (b) missing (Hindi)')
    return en, hi, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/cbse_bench')
    ap.add_argument('--per_type', type=int, default=3)
    ap.add_argument('--clean', type=int, default=6)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    fonts = (Path(__file__).resolve().parent / 'node_modules' / '@fontsource').as_uri()
    (out / 'html').mkdir(parents=True, exist_ok=True)
    cases = []
    kinds = ['clean'] * args.clean + [k for k in ERROR_TYPES for _ in range(args.per_type)]
    for n, kind in enumerate(kinds):
        paper = make_paper(rng)
        if kind == 'clean':
            en, hi, gt = paper, paper, {'type': 'clean', 'question': None}
        else:
            en, hi, gt = inject(paper, kind, rng)
        cid = f'cbse_{n:03d}_{kind}'
        (out / 'html' / f'{cid}.html').write_text(paper_html({'en': en, 'hi': hi}, fonts))
        cases.append({'id': cid, 'gt': gt, 'degraded': rng.random() < 0.5})
    (out / 'cases.json').write_text(json.dumps(cases, ensure_ascii=False, indent=1))
    print(f'{len(cases)} bilingual papers written to {out}')


if __name__ == '__main__':
    main()
