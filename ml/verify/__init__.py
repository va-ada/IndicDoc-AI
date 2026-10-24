"""
IndicDoc Verify: pre-print checking of question papers and official documents in 12 Indian languages.

    verify(pages)                                 structure checks on one paper
    verify(pages, source_pages=src)               + parity with the source-language version
    verify(pages, bank=[...])                     + questions repeated from earlier papers
    verify(pages, profile='office_memorandum')    format check of any document type against a rule profile
    verify(bilingual_pages, split_scripts=...)    one bilingual paper, checked version against version

`pages` are extract_document() results (layout regions with OCR text), one per page, so the checks run on top of
the same detector and OCR as the rest of the product, offline.
"""
from typing import Callable, Dict, List, Optional, Sequence

from ml.verify.checks import ERROR, INFO, WARNING, Issue, align, check_paper, compare_versions, find_repeats
from ml.verify.paper import Paper, Question, parse_paper
from ml.verify.profiles import check_format, list_profiles, load_profile
from ml.verify.text import split_lines_by_script


def split_bilingual(pages: List[Dict], scripts: Dict[str, str]) -> Dict[str, List[Dict]]:
    """
    One bilingual paper (both languages on the same pages) as one page list per language: every region's lines
    go to the stream of their script; lines without letters (numbers, formulas, option values) go to both.
    scripts: e.g. {'Latin': 'english', 'Devanagari': 'hindi'}.
    """
    out = {name: [] for name in scripts.values()}
    for page in pages:
        per = {name: [] for name in scripts.values()}
        for region in page['regions']:
            texts = split_lines_by_script(region.get('text', '') or '', scripts)
            for name, text in texts.items():
                if text.strip() or region['label'] in ('figure', 'table', 'formula'):
                    per[name].append({**region, 'text': text})
        for name in out:
            out[name].append({**page, 'regions': per[name]})
    return out


def verify(pages: List[Dict], source_pages: Optional[List[Dict]] = None, bank: Optional[Sequence[Dict]] = None,
           profile=None, split_scripts: Optional[Dict[str, str]] = None,
           semantic: Optional[Callable[[str, str], float]] = None, meta: Optional[Dict] = None) -> Dict:
    """
    Run every applicable check and return {'summary', 'issues', 'paper', 'source', 'meta'} (JSON-serialisable).
    With split_scripts, `pages` is one bilingual paper: the first script's stream is the source, the second the
    translation.
    """
    if split_scripts:
        streams = split_bilingual(pages, split_scripts)
        names = list(split_scripts.values())
        source_pages, pages = streams[names[0]], streams[names[1]]

    paper = parse_paper(pages)
    issues: List[Issue] = []
    # Question-level checks run for question papers: no profile, or a profile that asks for them
    if profile is None or load_profile(profile).get('question_checks', False):
        issues += check_paper(paper, version='target' if source_pages else 'paper')
    source = None
    if source_pages is not None:
        source = parse_paper(source_pages)
        issues += check_paper(source, version='source')
        issues += compare_versions(source, paper, semantic=semantic)
    if bank:
        issues += find_repeats(paper, bank)
    if profile is not None:
        issues += check_format(pages, profile)

    out = [i.to_dict() for i in issues]
    summary = {'questions': len(paper.questions), 'errors': sum(i['severity'] == ERROR for i in out),
               'warnings': sum(i['severity'] == WARNING for i in out), 'info': sum(i['severity'] == INFO for i in out)}
    return {'summary': summary, 'issues': out, 'paper': paper.to_dict(),
            'source': source.to_dict() if source else None, 'meta': meta or {}}


__all__ = ['verify', 'parse_paper', 'check_paper', 'compare_versions', 'find_repeats', 'check_format',
           'list_profiles', 'load_profile', 'split_bilingual', 'align', 'Paper', 'Question', 'Issue']
