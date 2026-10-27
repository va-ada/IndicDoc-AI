"""
Bilingual papers: find the language versions inside one file.

Indian board papers are printed in two languages in one of two ways:
    page by page   every page in one language, the versions interleaved (CBSE: a Hindi page, then the same
                   questions on an English page) or one after the other
    line by line   both languages on the same page, each question in both (many state boards and PSCs)
detect_versions() tells which, from the script of each page's text; split_versions() returns one page list per
version, each page tagged with its sheet in the file, so findings still point at the right sheet.
"""
from collections import Counter
from typing import Dict, List, Optional

from ml.verify.text import question_number, script_of, split_lines_by_script, strip_marks

_FURNITURE = {'header', 'footer', 'page-number', 'folio', 'website-link', 'jumpline'}
# Name of each script's version, and the order in which versions are preferred as the source
VERSION_NAMES = {'Latin': 'english', 'Devanagari': 'hindi', 'Tamil': 'tamil', 'Bengali': 'bengali',
                 'Gujarati': 'gujarati', 'Gurmukhi': 'punjabi', 'Oriya': 'odia', 'Telugu': 'telugu',
                 'Kannada': 'kannada', 'Malayalam': 'malayalam'}


def _line_scripts(page: Dict) -> Counter:
    """Letters per script on the page, outside the running header and footer (so short Latin units and option
    markers on a Hindi page do not outweigh its Hindi sentences)."""
    c = Counter()
    for r in page['regions']:
        if r['label'] in _FURNITURE:
            continue
        for ch in r.get('text') or '':
            s = script_of(ch)
            if s:
                c[s] += 1
    return c


def _has_question(page: Dict) -> bool:
    return any(r['label'] == 'first-level-question' or question_number(strip_marks(r.get('text', '') or ''))[0]
               for r in page['regions'] if r['label'] not in _FURNITURE)


def page_script(page: Dict) -> Optional[str]:
    c = _line_scripts(page)
    return c.most_common(1)[0][0] if c else page.get('script')


def detect_versions(pages: List[Dict]) -> Dict:
    """
    {'mode': 'single'|'pages'|'lines', 'scripts': [...], 'source': script}. 'pages': question pages are in two
    scripts (each version on its own pages); 'lines': question pages carry both scripts in similar amounts.
    """
    qpages = [p for p in pages if _has_question(p)]
    if not qpages:
        return {'mode': 'single', 'scripts': []}
    per_page = [page_script(p) for p in qpages]
    counts = Counter(s for s in per_page if s)
    if len(counts) >= 2:
        (s1, n1), (s2, n2) = counts.most_common(2)
        if n2 >= 2 and n2 >= 0.25 * len(qpages):
            return {'mode': 'pages', 'scripts': [s1, s2], 'source': 'Latin' if 'Latin' in (s1, s2) else s1}
    mixed = 0
    for p in qpages:
        c = _line_scripts(p)
        top = c.most_common(2)
        if len(top) == 2 and top[1][1] >= 0.25 * top[0][1]:
            mixed += 1
    if mixed >= 0.6 * len(qpages):
        total = Counter()
        for p in qpages:
            total.update(_line_scripts(p))
        (s1, _), (s2, _) = total.most_common(2)
        return {'mode': 'lines', 'scripts': [s1, s2], 'source': 'Latin' if 'Latin' in (s1, s2) else s1}
    return {'mode': 'single', 'scripts': list(counts)}


def split_versions(pages: List[Dict], plan: Dict) -> Dict[str, List[Dict]]:
    """
    One page list per script of the plan. Page by page: each question page goes to its script's version; pages
    without questions (cover, instructions printed in both languages) go to both, line by line. Line by line: every
    page is split by the script of its lines; lines without letters (numbers, formulas) go to both versions.
    """
    scripts = plan['scripts']
    keep = {s: s for s in scripts}
    out = {s: [] for s in scripts}
    for k, page in enumerate(pages):
        sheet = page.get('sheet', k)
        if plan['mode'] == 'pages' and _has_question(page):
            s = page_script(page)
            if s in out:
                out[s].append({**page, 'sheet': sheet})
            continue
        per = {s: [] for s in scripts}
        for region in page['regions']:
            if region['label'] in _FURNITURE or region['label'] in ('figure', 'table', 'formula'):
                for s in scripts:
                    per[s].append(region)
                continue
            words = region.get('words') or []
            if words:
                # Word by word: a cover in two columns (Hindi left, English right) shares its lines between languages
                from ml.verify.reader import lines_of
                lines = lines_of(words)
                for s in scripts:
                    mine = [[w for w in ln if script_of(w['text']) in (s, None)] for ln in lines]
                    text = '\n'.join(' '.join(w['text'] for w in ln) for ln in mine if any(script_of(w['text']) == s for w in ln))
                    if text.strip():
                        per[s].append({**region, 'text': text, 'words': [w for ln in mine for w in ln]})
                continue
            texts = split_lines_by_script(region.get('text', '') or '', keep)
            for s, text in texts.items():
                if text.strip():
                    per[s].append({**region, 'text': text})
        for s in scripts:
            out[s].append({**page, 'regions': per[s], 'sheet': sheet})
    return out
