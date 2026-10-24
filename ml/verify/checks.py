"""
Checks run on parsed question papers. Each check reflects an error that has reached candidates in a real Indian exam
(see docs/PRODUCT.md for the cases):

  check_paper      one paper: numbering gaps / repeats, duplicated questions, missing or repeated options,
                   marks that do not add up to the printed total, page numbering
  compare_versions two language versions of the same paper: missing / extra questions, option count, marks,
                   sub-questions, the numbers in every stem and option, figures / tables / formulas
  find_repeats     questions that repeat an earlier paper (a question bank)

Issues carry the question, page and box, so the report can point the human vetter at the exact place.
"""
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ml.verify.paper import Paper, Question
from ml.verify.text import numerals, normalized, similarity

ERROR, WARNING, INFO = 'error', 'warning', 'info'

# Two questions whose text is at least this similar (character 3-gram cosine) are the same question.
# Chosen on the benchmark's clean papers, where no two different questions came close (scripts/evaluate_verify.py).
DUPLICATE_SIMILARITY = 0.9
REPEAT_SIMILARITY = 0.8


@dataclass
class Issue:
    code: str
    severity: str
    message: str
    question: Optional[int] = None       # printed question number (or position + 1 if unreadable)
    page: Optional[int] = None
    box: Optional[List[float]] = None
    version: str = 'paper'               # 'paper', 'source' or 'target'
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


def _qid(q: Question) -> int:
    return q.number if q.number is not None else q.index + 1


def _issue(code, severity, message, q: Optional[Question] = None, version='paper', **details) -> Issue:
    return Issue(code, severity, message, question=_qid(q) if q else None, page=q.page if q else None,
                 box=q.box if q else None, version=version, details=details)


# ------------------------------------------------------------------------------------------ one paper
def check_paper(paper: Paper, version: str = 'paper') -> List[Issue]:
    issues: List[Issue] = []
    qs = paper.questions
    if not qs:
        return [Issue('NO_QUESTIONS', ERROR, 'No questions were found on these pages.', version=version)]

    # Numbering: every number once, no gaps. Sections may restart at 1, so a reset is not a gap.
    numbered = [q for q in qs if q.number is not None]
    for q in qs:
        if q.number is None:
            issues.append(_issue('NUMBER_UNREADABLE', WARNING, f'Question at position {q.index + 1} has no '
                                 'readable number.', q, version))
    prev = None
    seen = {}
    for q in numbered:
        if prev is not None and q.number not in (1, prev.number + 1) and q.section == prev.section:
            if q.number > prev.number + 1:
                missing = list(range(prev.number + 1, q.number))
                issues.append(_issue('NUMBER_GAP', ERROR, f'Question numbers jump from {prev.number} to {q.number} '
                                     f'(missing {", ".join(map(str, missing[:5]))}).', q, version, missing=missing))
            elif q.number <= prev.number:
                issues.append(_issue('NUMBER_ORDER', ERROR, f'Question {q.number} comes after question '
                                     f'{prev.number}.', q, version))
        key = (q.section, q.number)
        if key in seen:
            issues.append(_issue('NUMBER_REPEATED', ERROR, f'Question number {q.number} is used twice '
                                 f'(pages {seen[key].page + 1} and {q.page + 1}).', q, version))
        seen[key] = q
        prev = q

    # The same question printed twice (e.g. an MCQ repeated in an MBBS paper, AP High Court, July 2026)
    for i, a in enumerate(qs):
        for b in qs[i + 1:]:
            if len(normalized(a.stem)) >= 15 and similarity(a.stem, b.stem) >= DUPLICATE_SIMILARITY \
                    and a.stem_numerals == b.stem_numerals:
                issues.append(_issue('DUPLICATE_QUESTION', ERROR, f'Question {_qid(b)} repeats question {_qid(a)}.',
                                     b, version, first=_qid(a), similarity=round(similarity(a.stem, b.stem), 3)))

    # Options: MCQs in one paper normally all have the same number of options
    counts = Counter(len(q.options) for q in qs if q.options)
    if counts:
        usual, n_usual = counts.most_common(1)[0]
        for q in qs:
            if q.options and len(q.options) != usual and n_usual >= 3:
                issues.append(_issue('OPTION_COUNT', ERROR, f'Question {_qid(q)} has {len(q.options)} options; the '
                                     f'other questions have {usual}.', q, version, found=len(q.options),
                                     expected=usual))
        for q in qs:
            texts = [normalized(o) for o in q.options if normalized(o)]
            dup = [t for t, c in Counter(texts).items() if c > 1]
            if dup:
                issues.append(_issue('OPTION_REPEATED', ERROR, f'Question {_qid(q)} prints the same option twice '
                                     f'("{dup[0][:40]}").', q, version))
            if any(not normalized(o) for o in q.options):
                issues.append(_issue('OPTION_EMPTY', WARNING, f'Question {_qid(q)} has an empty option.', q, version))

    # Marks: printed per-question marks should add up to the printed maximum
    with_marks = [q for q in qs if q.marks is not None]
    if paper.declared_total is not None and with_marks:
        if len(with_marks) == len(qs):
            total = sum(q.marks for q in qs)
            if abs(total - paper.declared_total) > 1e-6:
                issues.append(Issue('MARKS_TOTAL', ERROR, f'Question marks add up to {total:g}, but the paper says '
                                    f'maximum marks {paper.declared_total}.', version=version,
                                    details={'sum': total, 'declared': paper.declared_total}))
        else:
            missing = [_qid(q) for q in qs if q.marks is None]
            issues.append(Issue('MARKS_MISSING', WARNING, f'{len(missing)} question(s) show no marks '
                                f'({", ".join(map(str, missing[:8]))}), so the total cannot be checked.',
                                version=version, details={'questions': missing}))

    # Page numbers: printed page numbers should run without gaps
    printed = [(i, n) for i, n in enumerate(paper.page_numbers) if n is not None]
    for (i, a), (j, b) in zip(printed, printed[1:]):
        if b - a != j - i:
            issues.append(Issue('PAGE_NUMBERS', WARNING, f'Printed page number goes from {a} to {b} between '
                                f'sheets {i + 1} and {j + 1}.', page=j, version=version))
    return issues


# ------------------------------------------------------------------------------------- two versions
def _skeleton(q: Question) -> Tuple:
    return (len(q.options), q.sub_questions, q.marks, q.figures + q.tables + q.formulas)


def align(src: Sequence[Question], tgt: Sequence[Question]) -> List[Tuple[Optional[int], Optional[int]]]:
    """
    Pair the questions of two versions with a global alignment (Needleman-Wunsch). The score uses what is the same
    in every language: the printed question number, the numbers in the stem and options, the option count,
    sub-questions, marks and visuals. Content keeps the pairing right even when a dropped question shifted the
    numbering of every later one. Returns (source index, target index) pairs; None = no partner.
    """
    def nums(q: Question) -> Counter:
        out = Counter(q.stem_numerals)
        for o in q.options:
            out.update(numerals(o))
        return out

    src_n, tgt_n = [nums(q) for q in src], [nums(q) for q in tgt]

    def score(i: int, j: int) -> float:
        a, b, na, nb = src[i], tgt[j], src_n[i], tgt_n[j]
        content = (sum((na & nb).values()) / max(1, sum((na | nb).values()))) if (na or nb) else 0.5
        number = 0.0 if a.number is None or b.number is None else (1.0 if a.number == b.number else -0.5)
        return 2 * content + number + 0.5 * (_skeleton(a) == _skeleton(b))

    n, m, gap = len(src), len(tgt), -0.6
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


def _diff(a: Counter, b: Counter) -> Tuple[List[str], List[str]]:
    return sorted((a - b).elements()), sorted((b - a).elements())


def compare_versions(source: Paper, target: Paper,
                     semantic: Optional[Callable[[str, str], float]] = None,
                     semantic_threshold: float = 0.6) -> List[Issue]:
    """
    Differences between a source paper and its translation that a candidate would notice. No translation is
    needed: numbers, option counts, marks and structure must match in every language.
    semantic: optional cross-lingual similarity (e.g. a LaBSE sentence-embedding cosine); low scores are flagged
    for the vetter as possible changes of meaning.
    """
    issues: List[Issue] = []
    if source.declared_total is not None and target.declared_total is not None \
            and source.declared_total != target.declared_total:
        issues.append(Issue('TOTAL_MISMATCH', ERROR, f'Maximum marks: {source.declared_total} in the source, '
                            f'{target.declared_total} in the translation.', version='target'))
    if len(source.questions) != len(target.questions):
        issues.append(Issue('QUESTION_COUNT', ERROR, f'{len(source.questions)} questions in the source, '
                            f'{len(target.questions)} in the translation.', version='target'))

    for i, j in align(source.questions, target.questions):
        if j is None:
            q = source.questions[i]
            issues.append(_issue('MISSING_IN_TRANSLATION', ERROR, f'Question {_qid(q)} of the source is not in the '
                                 'translation.', q, 'source'))
            continue
        if i is None:
            q = target.questions[j]
            issues.append(_issue('EXTRA_IN_TRANSLATION', ERROR, f'Question {_qid(q)} of the translation has no '
                                 'counterpart in the source.', q, 'target'))
            continue
        s, t = source.questions[i], target.questions[j]
        qn = _qid(s)
        if len(s.options) != len(t.options):
            issues.append(_issue('OPTION_COUNT_MISMATCH', ERROR, f'Question {qn}: {len(s.options)} options in the '
                                 f'source, {len(t.options)} in the translation.', t, 'target',
                                 source=len(s.options), target=len(t.options)))
        if s.sub_questions != t.sub_questions:
            issues.append(_issue('SUBQUESTION_MISMATCH', ERROR, f'Question {qn}: {s.sub_questions} sub-questions in '
                                 f'the source, {t.sub_questions} in the translation.', t, 'target'))
        if s.marks is not None and t.marks is not None and s.marks != t.marks:
            issues.append(_issue('MARKS_MISMATCH', ERROR, f'Question {qn}: {s.marks:g} marks in the source, '
                                 f'{t.marks:g} in the translation.', t, 'target'))
        only_s, only_t = _diff(s.stem_numerals, t.stem_numerals)
        if only_s or only_t:
            issues.append(_issue('NUMBER_MISMATCH', ERROR, f'Question {qn}: numbers differ - source has '
                                 f'{", ".join(only_s) or "nothing extra"}, translation has '
                                 f'{", ".join(only_t) or "nothing extra"}.', t, 'target',
                                 source_only=only_s, target_only=only_t))
        # One finding per question (the vetter checks the question once), listing every option that differs
        differing = []
        for k, (so, to) in enumerate(zip(s.options, t.options)):
            os_, ot = _diff(numerals(so), numerals(to))
            if os_ or ot:
                differing.append({'option': k + 1, 'source_only': os_, 'target_only': ot})
        if differing:
            parts = '; '.join(f'option {d["option"]}: source {", ".join(d["source_only"]) or "-"}, translation '
                              f'{", ".join(d["target_only"]) or "-"}' for d in differing)
            issues.append(_issue('OPTION_NUMBER_MISMATCH', ERROR, f'Question {qn}: option numbers differ ({parts}); '
                                 'the answer key may no longer hold.', t, 'target', options=differing))
        for attr in ('figures', 'tables', 'formulas'):
            if getattr(s, attr) != getattr(t, attr):
                issues.append(_issue('VISUAL_MISMATCH', WARNING, f'Question {qn}: {getattr(s, attr)} {attr} in the '
                                     f'source, {getattr(t, attr)} in the translation.', t, 'target', kind=attr))
        if semantic is not None and s.stem and t.stem:
            score = semantic(s.stem, t.stem)
            if score < semantic_threshold:
                issues.append(_issue('MEANING_CHECK', WARNING, f'Question {qn}: the translation may not say the same '
                                     f'thing (cross-lingual similarity {score:.2f}).', t, 'target',
                                     similarity=round(score, 3)))
    return issues


# ------------------------------------------------------------------------------------ question bank
def find_repeats(paper: Paper, bank: Sequence[Dict], threshold: float = REPEAT_SIMILARITY) -> List[Issue]:
    """
    Questions that already appeared in earlier papers (UGC-NET June 2026 papers were re-held partly over questions
    repeated from earlier years). bank: [{'text': ..., 'source': 'UGC-NET Dec 2025, Q14'}, ...] in the same language.
    """
    issues = []
    entries = [(b, None) for b in bank]
    for q in paper.questions:
        if len(normalized(q.stem)) < 15:
            continue
        best, best_score = None, 0.0
        for entry, _ in entries:
            # Same wording AND same numbers: a question with new values is a new question
            if numerals(entry.get('text', '')) != q.stem_numerals:
                continue
            sc = similarity(q.stem, entry.get('text', ''))
            if sc > best_score:
                best, best_score = entry, sc
        if best is not None and best_score >= threshold:
            issues.append(_issue('REPEATED_FROM_BANK', WARNING, f'Question {_qid(q)} closely matches '
                                 f'{best.get("source", "an earlier paper")} (similarity {best_score:.2f}).', q,
                                 similarity=round(best_score, 3), source=best.get('source')))
    return issues
