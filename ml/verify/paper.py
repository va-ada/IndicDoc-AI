"""
Turn the layout + OCR output of a question paper (one extract_document() result per page) into questions.

The layout model already labels question papers: first-, second- and third-level questions, options and answers
(IndicDLP classes; DocLayout-YOLO reaches 0.81-0.90 AP@50 on them, see outputs/metrics/authors_metrics.json). A
question is a first-level-question region plus everything that follows it until the next one: sub-questions,
option regions, figures, tables and formulas. Regions the detector missed are recovered from their text when they
start with a question number.
"""
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ml.verify.text import (declared_pages, declared_questions, declared_total, is_choice_line, is_instruction,
                            marks_each, marks_in, marks_schedule, numerals, option_family, page_of, question_number,
                            section_key, split_options, stem_before_options, strip_marks)

QUESTION = {'first-level-question'}
SUB_QUESTION = {'second-level-question', 'third-level-question'}
OPTIONS = {'options'}
VISUALS = {'figure': 'figures', 'table': 'tables', 'formula': 'formulas'}
TEXT_FALLBACK = {'paragraph', 'unlabelled-text', 'ordered-list', 'answer'}
FURNITURE = {'header', 'footer', 'page-number', 'folio', 'website-link', 'jumpline'}


@dataclass
class Question:
    index: int                      # position in the paper, 0-based
    number: Optional[int]           # printed number, None if it could not be read
    page: int
    box: List[float]
    stem: str = ''
    options: List[str] = field(default_factory=list)
    option_markers: List[str] = field(default_factory=list)
    sub_questions: int = 0
    marks: Optional[float] = None
    sub_marks: float = 0.0           # marks printed on the sub-questions, used when the question has none
    figures: int = 0
    tables: int = 0
    formulas: int = 0
    section: str = ''
    boxes: List[Dict] = field(default_factory=list)  # every region of the question: {'page', 'box', 'label'}
    choices: int = 0                 # internal-choice alternatives ('OR' / 'अथवा') beyond the first
    marks_source: str = ''           # 'printed', 'sub-questions', 'instructions' or 'each'
    confidence: Optional[float] = None  # how sure the reader is of what it read (0-1), from OCR and layout scores
    source: Optional[str] = None     # where the reader got the text: 'text-layer', 'ocr', 're-read' (None: layout + OCR)

    @property
    def text(self) -> str:
        return ' '.join([self.stem] + self.options)

    @property
    def stem_numerals(self) -> Counter:
        return numerals(self.stem)

    def to_dict(self) -> Dict:
        return {'index': self.index, 'number': self.number, 'page': self.page, 'box': self.box, 'stem': self.stem,
                'options': self.options, 'sub_questions': self.sub_questions, 'marks': self.marks,
                'figures': self.figures, 'tables': self.tables, 'formulas': self.formulas, 'section': self.section,
                'choices': self.choices, 'marks_source': self.marks_source, 'confidence': self.confidence,
                'sub_marks': self.sub_marks}


@dataclass
class Paper:
    questions: List[Question]
    declared_total: Optional[int]
    marks_each: Optional[float]
    page_numbers: List[Optional[int]]
    pages: int
    script: Optional[str] = None
    page_of: List[Optional[List[int]]] = field(default_factory=list)  # printed 'Page X of Y' per sheet
    declared_questions: Optional[int] = None   # 'This question paper contains 38 questions'
    declared_pages: Optional[int] = None       # 'contains 23 printed pages'
    page_indices: List[int] = field(default_factory=list)  # sheet of the original file for each page here

    def to_dict(self) -> Dict:
        return {'questions': [q.to_dict() for q in self.questions], 'declared_total': self.declared_total,
                'marks_each': self.marks_each, 'page_numbers': self.page_numbers, 'pages': self.pages,
                'script': self.script, 'page_of': self.page_of, 'declared_questions': self.declared_questions,
                'declared_pages': self.declared_pages}


def _starts_question(region: Dict) -> bool:
    if is_instruction(region.get('text', '') or ''):
        return False
    if region['label'] in QUESTION:
        return True
    # The detector missed it, but the text starts like a question ('12.', 'Q.12', 'प्रश्न 12')
    return region['label'] in TEXT_FALLBACK and question_number(strip_marks(region.get('text', '')))[0] is not None


def parse_paper(pages: List[Dict]) -> Paper:
    """
    pages: extract_document() results in page order (each has 'regions' in reading order, with 'label', 'box' and
    'text'). Returns the questions with their numbers, options, marks and attached figures/tables/formulas.
    """
    questions: List[Question] = []
    header_text, page_numbers, page_ofs, section = [], [], [], ''
    current: Optional[Question] = None
    repeated = _repeated_numbers(pages)

    for k, page in enumerate(pages):
        p = page.get('sheet', k)     # sheet of the original file (a bilingual paper's version holds every other sheet)
        page_no, pof = None, None
        for region in page['regions']:
            label, text = region['label'], region.get('text', '') or ''
            ref = {'page': p, 'box': region['box'], 'label': label}
            found = page_of(text)
            if found and len(text) < 80:
                pof = found
                continue
            if label == 'page-number':
                n = [v for v in numerals(text) if v not in repeated]
                page_no = int(float(n[0])) if len(n) == 1 else page_no
                continue
            if label in FURNITURE:
                header_text.append(text)
                continue
            key = section_key(text)
            if key and (label in ('section-title', 'sub-section-title', 'headline', 'chapter-title')
                        or len(text.split()) <= 4 or is_instruction(text.split('\n', 1)[-1])):
                section = key
                header_text.append(text)
                current = None
                continue
            if label in ('section-title', 'sub-section-title', 'headline', 'chapter-title'):
                header_text.append(text)
                continue
            if is_instruction(text):
                header_text.append(text)  # instructions between sections end the question before them
                current = None
                continue
            if _starts_question(region):
                # Right-aligned marks ('[2 marks]') are often read as the region's first line: drop them first
                number, rest = question_number(strip_marks(text))
                current = Question(index=len(questions), number=number, page=p, box=region['box'], section=section,
                                   boxes=[ref], marks=marks_in(text), source=page.get('source'))
                questions.append(current)
                _absorb_text(current, rest)
                continue
            if current is None:
                header_text.append(text)  # instructions, title, maximum marks ... before question 1
                continue
            current.boxes.append(ref)
            if is_choice_line(text):
                current.choices += 1
                continue
            if label in SUB_QUESTION:
                current.sub_questions += 1
                _absorb_text(current, text, sub=True)
            elif label in OPTIONS:
                _absorb_options(current, text, region.get('words'))
            elif label in VISUALS:
                setattr(current, VISUALS[label], getattr(current, VISUALS[label]) + 1)
            else:
                _absorb_text(current, text)
        page_numbers.append(pof[0] if pof else page_no)
        page_ofs.append(list(pof) if pof else None)

    _split_merged_numbers(questions)
    _unify_option_family(questions)
    head = '\n'.join(header_text)
    each = marks_each(head)
    per_q, per_section = marks_schedule(head)
    for q in questions:
        if q.marks is not None:
            q.marks_source = q.marks_source or 'printed'
        elif q.number is not None and q.number in per_q:
            q.marks, q.marks_source = per_q[q.number], 'instructions'
        elif q.section and q.section in per_section:
            q.marks, q.marks_source = per_section[q.section], 'instructions'
        elif q.sub_marks:
            q.marks, q.marks_source = q.sub_marks, 'sub-questions'
        elif each is not None:
            q.marks, q.marks_source = each, 'each'
        q.confidence = _confidence(q, {pg.get('sheet', k): pg for k, pg in enumerate(pages)})
    script = next((pg.get('script') for pg in pages if pg.get('script')), None)
    return Paper(questions=questions, declared_total=declared_total(head), marks_each=each,
                 page_numbers=page_numbers, pages=len(pages), script=script, page_of=page_ofs,
                 declared_questions=declared_questions(head), declared_pages=declared_pages(head),
                 page_indices=[pg.get('sheet', k) for k, pg in enumerate(pages)])


def _repeated_numbers(pages: List[Dict]) -> set:
    """Numbers printed in the page furniture of most sheets (a paper code such as '465'), which are not page numbers."""
    counts = Counter()
    for page in pages:
        seen = set()
        for region in page['regions']:
            if region['label'] in FURNITURE:
                seen |= set(numerals(region.get('text', '') or ''))
        counts.update(seen)
    return {n for n, c in counts.items() if len(pages) >= 3 and c >= 0.6 * len(pages)}


def _confidence(q: Question, pages: Dict[int, Dict]) -> Optional[float]:
    """Mean OCR confidence of the question's words (0-1), times the layout score of its regions when known."""
    confs, scores = [], []
    for ref in q.boxes:
        for region in pages[ref['page']]['regions']:
            if region['box'] == ref['box'] and region['label'] == ref['label']:
                confs += [float(w.get('conf', 0)) for w in region.get('words', []) or [] if float(w.get('conf', -1)) >= 0]
                if region.get('score') is not None and region.get('detected', True):
                    scores.append(float(region['score']))
                break
    if not confs:
        return None
    c = sum(confs) / len(confs) / 100.0
    if q.source == 'text-layer':
        return 0.99   # the PDF's own text: nothing was recognised, so nothing can be misread
    if scores:
        c *= min(1.0, sum(scores) / len(scores) + 0.25)  # a detector score of 0.75 or more does not lower it
    return round(c, 3)


def _unify_option_family(questions: List[Question]) -> None:
    """
    A paper prints its MCQ options with one marker family ('(A)'-'(D)'). Sub-parts and internal-choice alternatives
    use another ('(a) ... OR (b) ...', '(i) (ii) (iii)' in case studies); those are not options, so they are moved to
    the question's text and counted as sub-questions.
    """
    fams = Counter(option_family(q.option_markers) for q in questions if option_family(q.option_markers))
    if not fams:
        return
    main, _ = fams.most_common(1)[0]
    for q in questions:
        fam = option_family(q.option_markers)
        if fam and fam != main:
            q.stem = ' '.join([q.stem] + [f'({m}) {t}' for m, t in zip(q.option_markers, q.options)])
            q.sub_questions += len(q.options)
            q.options, q.option_markers = [], []


_MERGED = re.compile(r'^(\d{1,3})\.(\d+)\b\s*(.*)$', re.DOTALL)
_BARE_START = re.compile(r'^(\d{1,3})\s+(\D.*)$', re.DOTALL)


def _split_merged_numbers(questions: List[Question]) -> None:
    """
    OCR often drops the space after a question number when the stem starts with a number: '5. 16 V ...' is read as
    '5.16 V ...', which is not a question number. When a question has no number and its text starts with 'N.M'
    where N is exactly the next number in sequence, it is split back into the number N and the stem 'M ...'.
    """
    prev = None
    for q in questions:
        if q.number is None and prev is not None:
            m = _MERGED.match(q.stem)
            bare = _BARE_START.match(q.stem)
            if m and int(m.group(1)) == prev + 1:
                q.number, q.stem = int(m.group(1)), f'{m.group(2)} {m.group(3)}'.strip()
            elif bare and int(bare.group(1)) == prev + 1:
                # '14 यदि ...': the full stop after the number was lost, but it is exactly the next number
                q.number, q.stem = int(bare.group(1)), bare.group(2).strip()
        prev = q.number if q.number is not None else (prev + 1 if prev is not None else None)


_LEADING_MARKER = re.compile(r'^\s*[(\[{][^\s()\[\]{}]{0,4}[)\]}]\s*')


def _absorb_text(q: Question, text: str, sub: bool = False) -> None:
    """Add question text; options printed inside the same region are split off, marks recorded."""
    if sub:
        # '(ii) How many ...': the sub-part's marker is not part of its text (OCR reads '(ii)' as '(7)' or '(॥)')
        text = '\n'.join(_LEADING_MARKER.sub('', ln, count=1) for ln in text.splitlines())
    m = marks_in(text)
    if m is not None:
        if sub:
            q.sub_marks += m
        elif q.marks is None:
            q.marks = m
    clean = strip_marks(text)
    opts = split_options(clean)
    if opts and not q.options:
        q.stem = ' '.join((q.stem + ' ' + stem_before_options(clean)).split())
        q.option_markers = [m for m, _ in opts]
        q.options = [t for _, t in opts]
    else:
        q.stem = ' '.join((q.stem + ' ' + clean).split())


_MARKER_TOKEN = re.compile(r'^[(\[]?[^\s()\[\]]{1,3}[)\].]$|^[(\[][^\s()\[\]]{1,3}$|^[^\s()]{1,2}$|^\(\)$')


def option_cells(words: List[Dict]) -> List[List[Dict]]:
    """
    The options of a question from the positions of its words. Options are typeset as separate cells (a grid or
    one per line), so a horizontal gap much wider than a word space, or a new line, starts a new option. This keeps
    working when OCR garbles or drops the option markers themselves ('(4)' read as '4)' or lost).
    """
    if not words:
        return []
    height = sorted(w['height'] for w in words)[len(words) // 2]
    lines: List[List[Dict]] = []
    for w in sorted(words, key=lambda w: w['top'] + w['height'] / 2):
        cy = w['top'] + w['height'] / 2
        if lines and abs(cy - (lines[-1][0]['top'] + lines[-1][0]['height'] / 2)) < 0.5 * height:
            lines[-1].append(w)
        else:
            lines.append([w])
    cells = []
    for line in lines:
        line.sort(key=lambda w: w['left'])
        cell = [line[0]]
        for prev, w in zip(line, line[1:]):
            if w['left'] - (prev['left'] + prev['width']) > 2.0 * height:
                cells.append(cell)
                cell = [w]
            else:
                cell.append(w)
        cells.append(cell)
    return cells


# An option marker that OCR ran into the option's value, and what is left of the value
_GLUED = [re.compile(r'^\(?[^\s()]{1,3}\)(\S+)$'),   # '(3)12', '2)8', '(I)3A'
          re.compile(r'^\([^\s()](\d\S*)$'),          # '(128' = '(1' + '28', '(M3A'
          re.compile(r'^[^\w\s(\[](\S+)$'),             # '©2A' ('(c)' read as one symbol), '@7A'
          re.compile(r'^[A-Za-z](\d\S*)$')]             # 'M4A' ('(a)' read as 'M')


def strip_marker(tokens: List[str]) -> List[str]:
    """Remove the option marker from the start of an option: a token of its own, or glued to the value."""
    if not tokens:
        return tokens
    if len(tokens) > 1 and _MARKER_TOKEN.match(tokens[0]):
        return tokens[1:]
    for pattern in _GLUED:
        m = pattern.match(tokens[0])
        if m:
            return [m.group(1)] + tokens[1:]
    return tokens


def _cell_options(cell: List[Dict]) -> List[str]:
    """One cell's option text without its marker; a cell holding several marked options ('(a) x (b) y') is split."""
    text = ' '.join(w['text'] for w in cell)
    inline = split_options(text, options_box=True)
    if len(inline) >= 2:
        return [t for _, t in inline]
    return [' '.join(strip_marker(text.split()))]


def _absorb_options(q: Question, text: str, words: Optional[List[Dict]] = None) -> None:
    """
    An options region: several options in one box (separated by layout when word positions are known, else by
    their markers), or one option per box (appended in order).
    """
    if words:
        opts = [o for cell in option_cells(words) for o in _cell_options(cell)]
        q.options += opts
        q.option_markers += [str(len(q.option_markers) + k + 1) for k in range(len(opts))]
        return
    opts = split_options(text, options_box=True)
    if len(opts) >= 2:
        q.option_markers += [m for m, _ in opts]
        q.options += [t for _, t in opts]
    elif text.strip():
        q.options.append(' '.join(text.split()))
        q.option_markers.append(str(len(q.options)))
