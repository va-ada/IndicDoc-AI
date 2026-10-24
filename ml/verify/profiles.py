"""
Format verification with profiles: a JSON rule file says which elements a kind of document must contain, where,
and in what order (e.g. an Office Memorandum per the Central Secretariat Manual of Office Procedure, or the front
matter of a question paper). Elements are found by layout label, by a pattern in the OCR text, or both, so the
same engine works for any document type a customer describes, in any of the 12 languages.
"""
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from ml.verify.checks import ERROR, INFO, WARNING, Issue
from ml.verify.text import SCRIPT_RANGES, normalize_digits

PROFILE_DIR = Path(__file__).parent / 'profiles'


def list_profiles() -> Dict[str, str]:
    """Profile id -> display name, for every JSON file in ml/verify/profiles."""
    return {p.stem: json.loads(p.read_text())['name'] for p in sorted(PROFILE_DIR.glob('*.json'))}


def load_profile(profile) -> Dict:
    """A profile id ('office_memorandum'), a path to a JSON file, or an already-loaded dict."""
    if isinstance(profile, dict):
        return profile
    path = Path(profile)
    if not path.suffix:
        path = PROFILE_DIR / f'{profile}.json'
    return json.loads(path.read_text())


def _find(element: Dict, pages: List[Dict]) -> Optional[Dict]:
    """First region (in reading order) that satisfies the element's label and text rules, with its position."""
    labels = set(element.get('labels', []))
    pattern = re.compile(element['pattern'], re.IGNORECASE | re.MULTILINE) if element.get('pattern') else None
    where = element.get('where')
    page_range = range(len(pages))
    if where == 'first_page':
        page_range = range(min(1, len(pages)))
    elif where == 'last_page':
        page_range = range(max(0, len(pages) - 1), len(pages))
    for p in page_range:
        for k, region in enumerate(pages[p]['regions']):
            if labels and region['label'] not in labels:
                continue
            if pattern and not pattern.search(normalize_digits(region.get('text', '') or '')):
                continue
            return {'page': p, 'order': k, 'box': region['box'], 'label': region['label']}
    return None


def _script_shares(pages: List[Dict]) -> Dict[str, float]:
    counts = Counter()
    for page in pages:
        for region in page['regions']:
            for ch in region.get('text', '') or '':
                for script, ranges in SCRIPT_RANGES.items():
                    if any(lo <= ord(ch) <= hi for lo, hi in ranges):
                        counts[script] += 1
                        break
    total = sum(counts.values())
    return {s: c / total for s, c in counts.items()} if total else {}


def check_format(pages: List[Dict], profile) -> List[Issue]:
    """Issues for missing, misplaced or out-of-order elements, and for the profile's bilingual rule."""
    prof = load_profile(profile)
    issues: List[Issue] = []
    found = {}
    for el in prof['elements']:
        hit = _find(el, pages)
        if hit:
            found[el['id']] = hit
        elif el.get('required', True):
            where = {'first_page': ' on the first page', 'last_page': ' on the last page'}.get(el.get('where'), '')
            issues.append(Issue('FORMAT_MISSING', ERROR, f'{el["name"]} not found{where}.', version='paper',
                                details={'element': el['id']}))
        else:
            issues.append(Issue('FORMAT_OPTIONAL_MISSING', INFO, f'{el["name"]} not found (optional).',
                                version='paper', details={'element': el['id']}))

    names = {el['id']: el['name'] for el in prof['elements']}
    present = [e for e in prof.get('order', []) if e in found]
    for a, b in zip(present, present[1:]):
        pa, pb = found[a], found[b]
        if (pb['page'], pb['order']) < (pa['page'], pa['order']):
            issues.append(Issue('FORMAT_ORDER', WARNING, f'{names[b]} appears before {names[a]}.', page=pb['page'],
                                box=pb['box'], version='paper', details={'before': b, 'after': a}))

    rule = prof.get('bilingual')
    if rule:
        shares = _script_shares(pages)
        low = [s for s in rule['scripts'] if shares.get(s, 0.0) < rule.get('min_share', 0.15)]
        if low:
            issues.append(Issue('BILINGUAL', WARNING, f'Not bilingual: little or no {" / ".join(low)} text. '
                                f'{rule.get("rule", "")}'.strip(), version='paper',
                                details={'shares': {k: round(v, 3) for k, v in shares.items()}}))

    if prof.get('page_numbers') and len(pages) > 1:
        unnumbered = [i + 1 for i, pg in enumerate(pages) if not any(r['label'] == 'page-number'
                                                                     for r in pg['regions'])]
        if unnumbered:
            issues.append(Issue('PAGE_NUMBER_MISSING', WARNING, f'No printed page number on sheet(s) '
                                f'{", ".join(map(str, unnumbered[:10]))}.', version='paper'))
    return issues
