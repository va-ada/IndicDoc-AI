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


def _checks(papers: Dict[str, Paper], profile=None, semantic=None) -> List[Dict]:
    """Every check on one reading of the paper: per version, and between the versions when there are two."""
    issues: List[Issue] = []
    question_checks = profile is None or load_profile(profile).get('question_checks', False)
    for version, paper in papers.items():
        if question_checks:
            issues += check_paper(paper, version=version)
    if 'source' in papers and 'target' in papers:
        issues += compare_versions(papers['source'], papers['target'], semantic=semantic)
    return [i.to_dict() for i in issues]


def _summary(issues: List[Dict], papers: Dict[str, Paper]) -> Dict:
    main = papers.get('target') or papers.get('paper') or next(iter(papers.values()))
    return {'questions': len(main.questions), 'errors': sum(i['severity'] == ERROR for i in issues),
            'warnings': sum(i['severity'] == WARNING for i in issues), 'info': sum(i['severity'] == INFO for i in issues)}


def verify(pages: List[Dict], source_pages: Optional[List[Dict]] = None, bank: Optional[Sequence[Dict]] = None,
           profile=None, split_scripts: Optional[Dict[str, str]] = None,
           semantic: Optional[Callable[[str, str], float]] = None, meta: Optional[Dict] = None,
           pages_b: Optional[List[Dict]] = None, source_pages_b: Optional[List[Dict]] = None,
           rereader=None, bilingual: str = 'off', langs: Optional[Dict[str, str]] = None,
           return_readers: bool = False) -> Dict:
    """
    Run every applicable check and return {'summary', 'issues', 'paper', 'source', 'meta', ...} (JSON-serialisable).

    pages / source_pages: reader A (layout + OCR) pages of the paper and of its source version.
    pages_b / source_pages_b: reader B (layout-free) pages of the same files; with them, every finding is the
        consensus of the two readers (ml/verify/consensus.py) and carries a 'consensus' record.
    rereader(sheet, box, lang) -> words: a third, higher-resolution reading of one question, used where the two
        readers disagree; a dict {'source': ..., 'target': ...} when the versions are separate files.
    bilingual: 'auto' finds the two language versions inside one file (page by page, as CBSE prints them, or line
        by line) and checks them against each other; 'off' treats the file as one paper.
    split_scripts: one bilingual paper with both languages on the same pages, scripts named explicitly
        (the first script's stream is the source, the second the translation).
    langs: OCR language of each version for the third reading, e.g. {'target': 'hin', 'source': 'eng'}.
    """
    from ml.verify.consensus import combine, fuse, verify_numbers
    from ml.verify.versions import VERSION_NAMES, detect_versions, split_versions
    meta = dict(meta or {})
    readers = {'a': {'paper': pages}}
    if pages_b is not None:
        readers['b'] = {'paper': pages_b}
    plan = None
    if split_scripts:
        for r, pp in list(readers.items()):
            streams = split_bilingual(pp['paper'], split_scripts)
            names = list(split_scripts.values())
            readers[r] = {'source': streams[names[0]], 'target': streams[names[1]]}
    elif source_pages is not None:
        readers['a'] = {'source': source_pages, 'target': pages}
        if 'b' in readers:
            readers['b'] = {'source': source_pages_b if source_pages_b is not None else source_pages,
                            'target': pages_b}
    elif bilingual == 'auto':
        plan = detect_versions(pages_b if pages_b is not None else pages)
        if plan['mode'] in ('pages', 'lines'):
            src_script = plan['source']
            tgt_script = next(s for s in plan['scripts'] if s != src_script)
            for r, pp in list(readers.items()):
                parts = split_versions(pp['paper'], plan)
                readers[r] = {'source': parts[src_script], 'target': parts[tgt_script]}
            meta.setdefault('Versions', f'{VERSION_NAMES.get(src_script, src_script)} (source) and '
                                        f'{VERSION_NAMES.get(tgt_script, tgt_script)}, '
                                        f'{"page by page" if plan["mode"] == "pages" else "line by line"} in one file')
            langs = langs or {'source': _lang(src_script), 'target': _lang(tgt_script)}

    papers = {r: {v: parse_paper(pp) for v, pp in versions.items()} for r, versions in readers.items()}
    issues_a = _checks(papers['a'], profile, semantic)
    consensus = None
    if 'b' in papers:
        issues_b = _checks(papers['b'], profile, semantic)
        fused, records = {}, {}
        for v in papers['a']:
            rr = rereader.get(v) if isinstance(rereader, dict) else rereader
            fused[v], records[v] = fuse(papers['a'][v], papers['b'][v], rr, (langs or {}).get(v))
        issues_f = _checks(fused, profile, semantic)
        layer = {v: 'text-layer' for v, pb in papers['b'].items()
                 if pb.questions and all(q.source == 'text-layer' for q in pb.questions)}
        issues, dismissed, stats = combine(issues_a, issues_b, issues_f, records, layer)
        if rereader is not None:
            rr = rereader if isinstance(rereader, dict) else {v: rereader for v in fused}
            issues, rejected = verify_numbers(issues, fused, rr)
            dismissed += rejected
        final_papers = fused
        consensus = {'stats': stats, 'dismissed': dismissed,
                     'readers': {'a': 'deep-learning layout + OCR',
                                 'b': _reader_b_source(pages_b)},
                     'comparison': _comparison(papers, records)}
    else:
        issues, final_papers = issues_a, papers['a']
    main_version = 'target' if 'target' in final_papers else 'paper'
    extra: List[Issue] = []
    if bank:
        extra += find_repeats(final_papers[main_version], bank)
    if profile is not None:
        extra += check_format(readers['a'].get(main_version) or pages, profile)
    issues = issues + [i.to_dict() for i in extra]
    paper = final_papers[main_version]
    source = final_papers.get('source')
    out = {'summary': _summary(issues, final_papers), 'issues': issues, 'paper': paper.to_dict(),
           'source': source.to_dict() if source else None, 'meta': meta}
    if plan is not None:
        out['versions'] = plan
    if return_readers:
        out['reader_issues'] = {'a': issues_a, **({'b': issues_b} if 'b' in papers else {})}
    if consensus is not None:
        out['consensus'] = consensus
        out['summary']['confirmed'] = sum(i.get('consensus', {}).get('status') == 'confirmed' for i in issues)
        out['summary']['review'] = sum(i.get('consensus', {}).get('status') == 'review' for i in issues)
        out['summary']['dismissed'] = len(consensus['dismissed'])
    return out


def _lang(script: str) -> str:
    from ml.ocr import languages_for_script
    return languages_for_script(script)


def _reader_b_source(pages_b) -> str:
    sources = {p.get('source', 'ocr') for p in pages_b or []}
    if sources == {'text-layer'}:
        return 'layout-free, PDF text layer'
    if 'text-layer' in sources:
        return 'layout-free, PDF text layer and second OCR pass'
    return 'layout-free, second OCR pass'


__all__ = ['verify', 'parse_paper', 'check_paper', 'compare_versions', 'find_repeats', 'check_format',
           'list_profiles', 'load_profile', 'split_bilingual', 'align', 'Paper', 'Question', 'Issue']


def _reading(q) -> str:
    """What one reader read for a question, in a few words: 'Q12 · 4 options · 2 marks · 3 parts'."""
    if q is None:
        return 'not found'
    bits = [f'Q{q.number}' if q.number is not None else 'Q?']
    if q.options:
        groups = q.option_groups
        bits.append(f'{len(groups)}×{groups[0]} options' if len(groups) > 1 and len(set(groups)) == 1
                    else f'{len(q.options)} options')
    if q.marks is not None:
        bits.append(f'{q.marks:g} marks')
    if q.sub_questions or q.choices:
        bits.append(f'{q.sub_questions} parts' + (f', {q.choices + 1} choices' if q.choices else ''))
    return ' · '.join(bits)


def _comparison(papers: Dict, records: Dict) -> List[Dict]:
    """Question by question: what reader A and reader B read, the fields they agreed on, and how it was settled.
    Shown in the app so the dual reading can be followed (and explained) question by question."""
    rows = []
    for v, recs in records.items():
        qa_all, qb_all = papers['a'][v].questions, papers['b'][v].questions
        for rec in recs:
            qa = qa_all[rec['a']] if rec.get('a') is not None else None
            qb = qb_all[rec['b']] if rec.get('b') is not None else None
            if rec.get('dropped'):
                outcome = 'dropped: a fragment only one reader saw'
            elif rec.get('uncertain'):
                outcome = 'needs a human look: ' + ', '.join(rec['uncertain'])
            elif rec.get('third'):
                outcome = 'settled by a third reading'
            elif rec.get('disagree') and rec.get('authority') == 'text-layer':
                outcome = "settled by the PDF's own text"
            elif rec.get('disagree'):
                outcome = 'settled by the more confident reader'
            else:
                outcome = 'both agree'
            rows.append({'version': v, 'reader_a': _reading(qa), 'reader_b': _reading(qb),
                         'agree': list(rec.get('agree', [])), 'disagree': list(rec.get('disagree', [])),
                         'outcome': outcome})
    return rows
