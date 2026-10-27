"""
Dual-reader consensus: the paper is read twice, independently, and a finding is trusted only when both readings agree.

    reader A  deep-learning layout (DocLayout-YOLO) + OCR of the detected regions        (ml.ocr.extract_document)
    reader B  layout-free reading of the PDF text layer, or of a separate OCR pass      (ml.verify.reader)

Each reader parses the paper and runs every check on its own. Their questions are then paired, field by field
(number, options, the numbers in the stem and options, marks). Where the readers agree, the value is confirmed;
where they disagree, the question is re-read a third time (a higher-resolution OCR of exactly that question, parsed
with the layout-free grammar) and the majority decides; without a third reading the more confident reader is used and
the field is marked uncertain. The checks then run once more on the agreed ("fused") paper.

Every finding of the fused paper gets a status:
    confirmed   both readers raised it independently, with similar confidence
    resolved    the readers disagreed; the third reading decided, and it supports the finding
    review      it depends on a value the readers could not agree on: shown for a human, never as a hard error
Findings that only one reader raised and the fused paper does not reproduce are dismissed as reading errors (listed
separately, so nothing is hidden). Confidence is reported for each reader and for the result.
"""
import copy
import subprocess
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ml.verify.paper import Paper, Question, parse_paper
from ml.verify.text import normalized, numerals, similarity

# Two readers' confidences for the same question count as "similar" within this margin (0-1 scale)
CONFIDENCE_MARGIN = 0.25
FIELDS = ('number', 'options', 'numerals', 'marks', 'parts')   # parts: sub-questions and internal-choice alternatives

Rereader = Callable[[int, List[float], Optional[str]], List[Dict]]


def _qnums(q: Question) -> Counter:
    out = Counter(q.stem_numerals)
    for o in q.options:
        out.update(numerals(o))
    return out


def pair_questions(a: Sequence[Question], b: Sequence[Question]) -> List[Tuple[Optional[int], Optional[int]]]:
    """Global alignment of two readings of the same paper: printed number, text, numbers and page must agree."""
    na, nb = [_qnums(q) for q in a], [_qnums(q) for q in b]

    def score(i, j):
        x, y = a[i], b[j]
        s = 0.0
        if x.number is not None and y.number is not None:
            s += 2.0 if x.number == y.number else -1.0
        s += 2.0 * similarity(x.stem, y.stem)
        if na[i] or nb[j]:
            s += sum((na[i] & nb[j]).values()) / max(1, sum((na[i] | nb[j]).values()))
        s += 0.5 if x.page == y.page else -1.0
        return s

    n, m, gap = len(a), len(b), -0.7
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i * gap
    for j in range(1, m + 1):
        dp[0][j] = j * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = max(dp[i - 1][j - 1] + score(i - 1, j - 1), dp[i - 1][j] + gap, dp[i][j - 1] + gap)
    pairs, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and abs(dp[i][j] - (dp[i - 1][j - 1] + score(i - 1, j - 1))) < 1e-9:
            pairs.append((i - 1, j - 1)); i, j = i - 1, j - 1
        elif i > 0 and abs(dp[i][j] - (dp[i - 1][j] + gap)) < 1e-9:
            pairs.append((i - 1, None)); i -= 1
        else:
            pairs.append((None, j - 1)); j -= 1
    return pairs[::-1]


def _field(q: Question, name: str):
    if name == 'number':
        return q.number
    if name == 'options':
        return len(q.options)
    if name == 'numerals':
        return tuple(sorted(_qnums(q).elements()))
    if name == 'marks':
        return q.marks
    if name == 'parts':
        return q.sub_questions, q.choices
    raise KeyError(name)


def _union_box(qs: Sequence[Question], page: int) -> Optional[List[float]]:
    boxes = [r['box'] for q in qs if q is not None for r in q.boxes if r['page'] == page]
    if not boxes:
        return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def third_reading(qa: Optional[Question], qb: Optional[Question], rereader: Optional[Rereader],
                  lang: Optional[str]) -> Optional[Question]:
    """Re-read exactly this question (the union of both readers' boxes, slightly enlarged) and parse it alone."""
    if rereader is None:
        return None
    from ml.verify.reader import read_page
    q0 = qa or qb
    box = _union_box([qa, qb], q0.page)
    if box is None:
        return None
    pad = 0.15 * (box[3] - box[1]) + 12
    box = [box[0] - 4 * pad, box[1] - pad, box[2] + pad, box[3] + pad]
    words = rereader(q0.page, box, lang)
    if not words:
        return None
    h = max(1.0, box[3] - box[1])
    page = read_page(words, (box[2], max(box[3], h)), source='re-read')
    # A crop has no running header or footer: keep every line
    for r in page['regions']:
        if r['label'] in ('header', 'footer'):
            r['label'] = 'paragraph'
    parsed = parse_paper([{**page, 'sheet': q0.page}])
    return parsed.questions[0] if parsed.questions else None


def _vote(values: List, confs: List[Optional[float]]):
    """Majority of the readings; a tie goes to the most confident reading."""
    counts = Counter(v for v in values)
    top = counts.most_common()
    if len(top) == 1 or top[0][1] > top[1][1]:
        return top[0][0], top[0][1] >= 2
    best = max(range(len(values)), key=lambda k: confs[k] if confs[k] is not None else -1)
    return values[best], False


def _reinstate_gaps(fused: List[Question], records: List[Dict]) -> None:
    """
    A question only one reader found, without a readable number, is dropped as a fragment, unless the agreed
    numbering has exactly one number missing where it stood ('2, 3, ...' with the dropped region before 2): then it
    is that question, whose number the readers could not read.
    """
    for rec in reversed([r for r in records if r.get('_candidate')]):
        pos, only = rec.pop('_candidate')
        before = next((q.number for q in reversed(fused[:pos]) if q.number is not None), 0)
        after = next((q.number for q in fused[pos:] if q.number is not None), None)
        if only.number is not None or after is None or after - before != 2 or not (only.stem or only.options):
            continue
        q = copy.deepcopy(only)
        q.number = before + 1
        fused.insert(pos, q)
        rec['dropped'] = False
        rec['uncertain'] = ['number']
        rec['single'] = 'a' if rec['a'] is not None else 'b'
        rec['reinstated'] = True
    for rec in records:
        rec.pop('_candidate', None)


def fuse(pa: Paper, pb: Paper, rereader: Optional[Rereader] = None, lang: Optional[str] = None) -> Tuple[Paper, List[Dict]]:
    """
    The agreed paper and one record per question: which fields both readers read alike, which needed a third
    reading, the readers' confidences. Questions only one reader found are kept when the third reading (or, without
    one, a readable question number) shows they are real, and dropped as fragments otherwise.
    """
    fused, records = [], []
    for i, j in pair_questions(pa.questions, pb.questions):
        qa = pa.questions[i] if i is not None else None
        qb = pb.questions[j] if j is not None else None
        rec = {'a': i, 'b': j, 'agree': [], 'disagree': [], 'third': False, 'uncertain': [],
               'conf_a': qa.confidence if qa else None, 'conf_b': qb.confidence if qb else None}
        if qa is None or qb is None:
            only = qa or qb
            b_layer = any(q.source == 'text-layer' for q in pb.questions if q.page == only.page)
            if b_layer:
                # The page's own text decides: a question in it is real, one only the layout reader saw is a fragment
                third, real = None, qb is not None and qb.number is not None
            else:
                third = third_reading(qa, qb, rereader, lang)
                real = (third is not None and third.number is not None) if rereader else only.number is not None
            if not real:
                rec['dropped'] = True
                rec['_candidate'] = (len(fused), only)      # reinstated below if the numbering needs it
                records.append(rec)
                continue
            q = copy.deepcopy(third if third is not None and third.number is not None else only)
            q.boxes, q.page, q.section = only.boxes, only.page, only.section
            q.marks = only.marks if only.marks is not None else q.marks
            rec['uncertain'] = [] if third is not None or b_layer else list(FIELDS)
            rec['third'] = third is not None
            if b_layer:
                rec['authority'] = 'text-layer'
            rec['single'] = 'a' if qa else 'b'
            fused.append(q)
            records.append(rec)
            continue
        q = copy.deepcopy(qa)
        third = None
        disagree = [f for f in FIELDS if _field(qa, f) != _field(qb, f)]
        if qb.source == 'text-layer':
            # Reader B read the PDF's own text: for what is written (number, options, numbers, marks) it is exact,
            # so it decides; the layout reader still supplies figures and tables
            q = copy.deepcopy(qb)
            q.boxes = qa.boxes + [b for b in qb.boxes if b not in qa.boxes]
            q.figures, q.tables, q.formulas = qa.figures, qa.tables, qa.formulas
            q.sub_questions = max(qa.sub_questions, qb.sub_questions)
            rec['agree'] = [f for f in FIELDS if f not in disagree]
            rec['disagree'] = disagree
            rec['authority'] = 'text-layer'
            fused.append(q)
            records.append(rec)
            continue
        conf_far = (qa.confidence is not None and qb.confidence is not None
                    and abs(qa.confidence - qb.confidence) > CONFIDENCE_MARGIN)
        if disagree or conf_far:
            third = third_reading(qa, qb, rereader, lang)
            rec['third'] = third is not None
        for f in FIELDS:
            va, vb = _field(qa, f), _field(qb, f)
            if va == vb and not conf_far:
                rec['agree'].append(f)
                continue
            if va != vb:
                rec['disagree'].append(f)
            if f == 'number' and (va is None) != (vb is None) and third is None:
                q.number = va if va is not None else vb          # one reader read the number, the other could not
                continue
            readings = [qa, qb] + ([third] if third is not None else [])
            values = [_field(r, f) for r in readings]
            confs = [r.confidence for r in readings]
            value, majority = _vote(values, confs)
            winner = readings[values.index(value)]
            if not majority:
                rec['uncertain'].append(f)
            if f == 'number':
                q.number = value
            elif f == 'options':
                q.options, q.option_markers = list(winner.options), list(winner.option_markers)
            elif f == 'numerals':
                q.stem = winner.stem
                if len(winner.options) == len(q.options):
                    q.options = list(winner.options)
            elif f == 'marks':
                q.marks = value
            elif f == 'parts':
                q.sub_questions, q.choices = value
                q.sub_marks = winner.sub_marks
        # B reads no figures or tables, so those come from the layout reader only
        q.figures, q.tables, q.formulas = qa.figures, qa.tables, qa.formulas
        ca, cb = qa.confidence, qb.confidence
        q.confidence = round(sum(c for c in (ca, cb) if c is not None) / max(1, sum(c is not None for c in (ca, cb))), 3) \
            if (ca is not None or cb is not None) else None
        fused.append(q)
        records.append(rec)
    _reinstate_gaps(fused, records)
    for k, q in enumerate(fused):
        q.index = k
    paper = copy.copy(pa)
    paper.questions = fused
    # Paper-level facts the readers both read from the instructions: the text layer's when B read one, else A's
    # unless only B found the value
    b_layer = bool(pb.questions) and all(q.source == 'text-layer' for q in pb.questions)
    for attr in ('declared_total', 'declared_questions', 'declared_pages', 'marks_each'):
        if (b_layer or getattr(paper, attr) is None) and getattr(pb, attr) is not None:
            setattr(paper, attr, getattr(pb, attr))
    if b_layer:
        paper.page_of, paper.page_numbers = pb.page_of, pb.page_numbers
    if not any(paper.page_of) and any(pb.page_of):
        paper.page_of, paper.page_numbers = pb.page_of, pb.page_numbers
    return paper, records


def combine(issues_a: List[Dict], issues_b: List[Dict], issues_f: List[Dict],
            records: Dict[str, List[Dict]], layer_versions: Optional[Dict[str, str]] = None
            ) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Statuses for the fused paper's findings, the single-reader findings the fused paper does not reproduce
    (dismissed as reading errors), and agreement statistics per version.
    records: per version ('paper', 'source', 'target') the fuse() records.
    layer_versions: versions reader B read entirely from a PDF text layer ({'target': 'text-layer'}); for their
    paper-level findings (marks total, maximum marks, declared counts) the text layer decides as well.
    """
    layer_versions = layer_versions or {}
    maps = {}
    for version, recs in records.items():
        fa, fb, by_f, k = {}, {}, {}, 0
        for r in recs:
            if r.get('dropped'):
                continue
            if r['a'] is not None:
                fa[r['a']] = k
            if r['b'] is not None:
                fb[r['b']] = k
            by_f[k] = r
            k += 1
        maps[version] = {'a': fa, 'b': fb, 'f': by_f}

    def keyed(issues, reader):
        out = {}
        for it in issues:
            version = it.get('version')
            q = it.get('details', {}).get('qindex')
            if q is not None and reader != 'f':
                q = maps.get(version, {}).get(reader, {}).get(q, ('unpaired', reader, q))
            out.setdefault((it['code'], version, q), []).append(it)
        return out

    ka, kb, kf = keyed(issues_a, 'a'), keyed(issues_b, 'b'), keyed(issues_f, 'f')
    final = []
    for key, items in kf.items():
        code, version, fq = key
        rec = maps.get(version, {}).get('f', {}).get(fq) if fq is not None else None
        in_a, in_b = key in ka, key in kb
        conf_a = rec['conf_a'] if rec else None
        conf_b = rec['conf_b'] if rec else None
        similar = conf_a is None or conf_b is None or abs(conf_a - conf_b) <= CONFIDENCE_MARGIN
        third = bool(rec and rec.get('third'))
        authority = (rec.get('authority') if rec else None) or (layer_versions.get(version) if fq is None else None)
        if in_a and in_b and (similar or authority):
            status = 'confirmed'
        elif authority == 'text-layer' and in_b:
            status = 'resolved'     # the readers disagreed and the PDF's own text decided
        elif third and not (rec and rec.get('uncertain')):
            status = 'resolved'
        elif in_a and in_b:
            status = 'confirmed' if not (rec and rec.get('uncertain')) else 'review'
        else:
            status = 'review'
        for it in items:
            it = copy.deepcopy(it)
            it['consensus'] = {'status': status, 'reader_a': in_a, 'reader_b': in_b,
                               'confidence_a': conf_a, 'confidence_b': conf_b,
                               'confidence': _final_conf(status, conf_a, conf_b), 'third_reading': third}
            if status == 'review' and it['severity'] in ('error', 'warning'):
                # Not a finding: the readers could not agree on what is printed here. Listed for a look by eye.
                it['severity'] = 'info'
                it['message'] += ' (uncertain reading: the two readers disagree here, please check by eye)'
            final.append(it)
    dismissed = []
    for reader, keyed_issues in (('a', ka), ('b', kb)):
        for key, items in keyed_issues.items():
            if key not in kf:
                dismissed += [{**it, 'raised_by': reader} for it in items]
    stats = {}
    for version, recs in records.items():
        both = [r for r in recs if r['a'] is not None and r['b'] is not None]
        fields_total = len(FIELDS) * len(both)
        ca = [r['conf_a'] for r in both if r['conf_a'] is not None]
        cb = [r['conf_b'] for r in both if r['conf_b'] is not None]
        stats[version] = {
            'questions_both': len(both), 'only_a': sum(r['b'] is None and r['a'] is not None for r in recs),
            'only_b': sum(r['a'] is None and r['b'] is not None for r in recs),
            'dropped_fragments': sum(bool(r.get('dropped')) for r in recs),
            'field_agreement': round(sum(len(r['agree']) for r in both) / fields_total, 3) if fields_total else None,
            'third_readings': sum(bool(r.get('third')) for r in recs),
            'decided_by_text_layer': sum(r.get('authority') == 'text-layer' and bool(r.get('disagree')) for r in recs),
            'uncertain_questions': sum(bool(r.get('uncertain')) for r in recs if not r.get('dropped')),
            'mean_confidence_a': round(sum(ca) / len(ca), 3) if ca else None,
            'mean_confidence_b': round(sum(cb) / len(cb), 3) if cb else None,
        }
    return final, dismissed, stats


def _final_conf(status: str, ca: Optional[float], cb: Optional[float]) -> Optional[float]:
    vals = [c for c in (ca, cb) if c is not None]
    if not vals:
        return None
    c = sum(vals) / len(vals)
    if status == 'resolved':
        c *= 0.9
    elif status == 'review':
        c *= 0.6
    return round(c, 3)


def make_rereader(images) -> Rereader:
    """
    The third reading: OCR of one question's region only, at twice the resolution, with another page segmentation
    (psm 6, a uniform block) than the full-page readings, and the English number pass on Indic pages.
    images: page images (RGB arrays) indexed by sheet.
    """
    from ml.ocr import merge_number_reading, read_words

    def reread(sheet: int, box: List[float], lang: Optional[str]) -> List[Dict]:
        img = images[sheet]
        hgt, wid = img.shape[:2]
        x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
        x2, y2 = min(wid, int(box[2]) + 1), min(hgt, int(box[3]) + 1)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return []
        crop = img[y1:y2, x1:x2]
        indic = '+'.join(c for c in (lang or 'eng').split('+') if c != 'eng')
        try:
            # One thread: a crop is small, and OpenMP threads busy-waiting next to other OCR jobs made a crop take a
            # minute instead of a second
            if indic:
                words = merge_number_reading(read_words(crop, indic, psm=6, single_thread=True),
                                             read_words(crop, 'eng', psm=6, single_thread=True))
            else:
                words = read_words(crop, 'eng', psm=6, single_thread=True)
        except (RuntimeError, subprocess.TimeoutExpired):
            return []           # no third reading for this question: the disagreement stays for a human to review
        return [{**w, 'left': w['left'] + x1, 'top': w['top'] + y1} for w in words]
    return reread


NUMBER_CODES = ('NUMBER_MISMATCH', 'OPTION_NUMBER_MISMATCH')


def _reread_numerals(q: Question, rereader: Optional[Rereader]) -> Optional[Counter]:
    """The numbers of a question as the English model reads its crop (exact text for a text-layer question)."""
    if q.source == 'text-layer':
        return numerals(' '.join([q.stem] + list(q.options)))
    box = _union_box([q], q.page)
    if rereader is None or box is None:
        return None
    # Two crops, the region as found and with a margin: OCR of one crop can drop a short token ('(D)5A') that the
    # other reads; a number counts as read if either reading has it
    pad = 0.15 * (box[3] - box[1]) + 12
    found = Counter()
    for crop in (box, [box[0] - 4 * pad, box[1] - pad, box[2] + pad, box[3] + pad]):
        words = rereader(q.page, crop, 'eng')
        found |= numerals(' '.join(w['text'] for w in words)) if words else Counter()
    return found or None


def verify_numbers(issues: List[Dict], papers: Dict[str, Paper], rereaders: Dict[str, Optional[Rereader]]
                   ) -> Tuple[List[Dict], List[Dict]]:
    """
    A number that differs between the versions is checked once more, with the model that reads digits best: both
    questions are re-read by the English OCR model (Latin digits and units are printed the same in every version).
    The finding stands unless both re-readings agree on every disputed number; then it was a misreading of the
    Indic model (or of a scan) and goes to the dismissed list. Text-layer questions are their own re-reading.
    """
    kept, dismissed = [], []
    for it in issues:
        d = it.get('details', {})
        if it['code'] not in NUMBER_CODES or it['severity'] != 'error' or 'sindex' not in d \
                or 'source' not in papers or 'target' not in papers:
            kept.append(it)
            continue
        s, t = papers['source'].questions[d['sindex']], papers['target'].questions[d['qindex']]
        if s.source == 'text-layer' and t.source == 'text-layer':
            kept.append(it)
            continue
        if it['code'] == 'NUMBER_MISMATCH':
            disputed = set(d.get('source_only', [])) | set(d.get('target_only', []))
        else:
            disputed = {x for o in d.get('options', []) for x in o['source_only'] + o['target_only']}
        rs, rt = _reread_numerals(s, rereaders.get('source')), _reread_numerals(t, rereaders.get('target'))
        if not disputed or rs is None or rt is None or any((rs[x] > 0) != (rt[x] > 0) for x in disputed):
            it = copy.deepcopy(it)
            it.setdefault('consensus', {})['reread'] = 'confirms' if rs is not None and rt is not None else None
            kept.append(it)
        else:
            dismissed.append({**it, 'raised_by': 'consensus', 'reason': 'the English re-reading of both versions '
                              'agrees on these numbers'})
    return kept, dismissed
