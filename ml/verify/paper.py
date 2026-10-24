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

from ml.verify.text import (declared_total, marks_each, marks_in, numerals, question_number, split_options,
                            stem_before_options, strip_marks)

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

    @property
    def text(self) -> str:
        return ' '.join([self.stem] + self.options)

    @property
    def stem_numerals(self) -> Counter:
        return numerals(self.stem)

    def to_dict(self) -> Dict:
        return {'index': self.index, 'number': self.number, 'page': self.page, 'box': self.box, 'stem': self.stem,
                'options': self.options, 'sub_questions': self.sub_questions, 'marks': self.marks,
                'figures': self.figures, 'tables': self.tables, 'formulas': self.formulas, 'section': self.section}


@dataclass
class Paper:
    questions: List[Question]
    declared_total: Optional[int]
    marks_each: Optional[float]
    page_numbers: List[Optional[int]]
    pages: int
    script: Optional[str] = None

    def to_dict(self) -> Dict:
        return {'questions': [q.to_dict() for q in self.questions], 'declared_total': self.declared_total,
                'marks_each': self.marks_each, 'page_numbers': self.page_numbers, 'pages': self.pages,
                'script': self.script}


def _starts_question(region: Dict) -> bool:
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
    header_text, page_numbers, section = [], [], ''
    current: Optional[Question] = None

    for p, page in enumerate(pages):
        page_no = None
        for region in page['regions']:
            label, text = region['label'], region.get('text', '') or ''
            ref = {'page': p, 'box': region['box'], 'label': label}
            if label == 'page-number':
                n = numerals(text)
                page_no = int(float(next(iter(n)))) if len(n) == 1 else page_no
                continue
            if label in FURNITURE:
                header_text.append(text)
                continue
            if label in ('section-title', 'sub-section-title', 'headline', 'chapter-title'):
                section = ' '.join(text.split())
                header_text.append(text)
                continue
            if _starts_question(region):
                # Right-aligned marks ('[2 marks]') are often read as the region's first line: drop them first
                number, rest = question_number(strip_marks(text))
                current = Question(index=len(questions), number=number, page=p, box=region['box'], section=section,
                                   boxes=[ref], marks=marks_in(text))
                questions.append(current)
                _absorb_text(current, rest)
                continue
            if current is None:
                header_text.append(text)  # instructions, title, maximum marks ... before question 1
                continue
            current.boxes.append(ref)
            if label in SUB_QUESTION:
                current.sub_questions += 1
                _absorb_text(current, text, sub=True)
            elif label in OPTIONS:
                _absorb_options(current, text, region.get('words'))
            elif label in VISUALS:
                setattr(current, VISUALS[label], getattr(current, VISUALS[label]) + 1)
            else:
                _absorb_text(current, text)
        page_numbers.append(page_no)

    _split_merged_numbers(questions)
    head = '\n'.join(header_text)
    each = marks_each(head)
    for q in questions:
        if q.marks is None and q.sub_marks:
            q.marks = q.sub_marks
        if q.marks is None and each is not None:
            q.marks = each
    script = next((pg.get('script') for pg in pages if pg.get('script')), None)
    return Paper(questions=questions, declared_total=declared_total(head), marks_each=each,
                 page_numbers=page_numbers, pages=len(pages), script=script)


_MERGED = re.compile(r'^(\d{1,3})\.(\d+)\b\s*(.*)$', re.DOTALL)


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
            if m and int(m.group(1)) == prev + 1:
                q.number, q.stem = int(m.group(1)), f'{m.group(2)} {m.group(3)}'.strip()
        prev = q.number if q.number is not None else (prev + 1 if prev is not None else None)


def _absorb_text(q: Question, text: str, sub: bool = False) -> None:
    """Add question text; options printed inside the same region are split off, marks recorded."""
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
