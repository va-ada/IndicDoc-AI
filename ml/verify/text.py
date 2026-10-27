"""
Text helpers for verification: digits in any Indian script, question numbers, marks, option markers, scripts.

Everything here works on the OCR text of detected regions and needs no model, so a translated paper can be
compared with its source without translating it: numbers, option markers and structure are the same in every
language even when the words are not.
"""
import re
import unicodedata
from collections import Counter
from typing import Dict, List, Optional, Tuple

# Zero digit of each Indian script (and Latin); the next nine code points are 1..9
_ZEROS = [0x0030, 0x0966, 0x09E6, 0x0A66, 0x0AE6, 0x0B66, 0x0BE6, 0x0C66, 0x0CE6, 0x0D66]
_DIGIT_MAP = {z + i: str(i) for z in _ZEROS for i in range(10)}

# Unicode blocks of the scripts the product reads; 'Latin' is ASCII letters only
SCRIPT_RANGES = {'Latin': [(0x41, 0x5A), (0x61, 0x7A)], 'Devanagari': [(0x0900, 0x097F)],
                 'Bengali': [(0x0980, 0x09FF)], 'Gurmukhi': [(0x0A00, 0x0A7F)], 'Gujarati': [(0x0A80, 0x0AFF)],
                 'Oriya': [(0x0B00, 0x0B7F)], 'Tamil': [(0x0B80, 0x0BFF)], 'Telugu': [(0x0C00, 0x0C7F)],
                 'Kannada': [(0x0C80, 0x0CFF)], 'Malayalam': [(0x0D00, 0x0D7F)]}
_DIGIT_CODEPOINTS = set(_DIGIT_MAP)
# Letters, vowel signs and viramas of all Indian scripts (Python's \w misses the combining vowel signs)
INDIC = '\u0900-\u0DFF'


_ZERO_WIDTH = {0x200B: None, 0x200C: None, 0x200D: None, 0xFEFF: None}


def normalize_digits(text: str) -> str:
    """
    'प्रश्न १२' -> 'प्रश्न 12': every Indian-script digit becomes its ASCII digit. Zero-width joiners (Tesseract
    writes U+200C after Tamil and Malayalam viramas) are removed so that patterns match across them.
    """
    return ''.join(_DIGIT_MAP.get(ord(ch), ch) for ch in text.translate(_ZERO_WIDTH))


def script_of(text: str) -> Optional[str]:
    """The script with the most letters in `text` (digits and punctuation do not count); None if no letters."""
    counts = Counter()
    for ch in text:
        cp = ord(ch)
        if cp in _DIGIT_CODEPOINTS:
            continue
        for script, ranges in SCRIPT_RANGES.items():
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[script] += 1
                break
    return counts.most_common(1)[0][0] if counts else None


# ---------------------------------------------------------------------------------------------- numbers
# A letter before the digits makes them part of a name ('CO2', 'H2O'); a unit after them does not ('3A', '5kg')
_NUMBER = re.compile(r'(?<![\w.])(\d{1,3}(?:,\d{2,3})+|\d+)(?:\.(\d+))?(?!\d)')


# OCR runs the article into a quantity: 'A24V battery' for 'A 24 V battery'
_GLUED_ARTICLE = re.compile(r'(?<![\w.])A(\d{1,4}(?:\.\d+)?)(?=\s*(?:V|W|J|N|Hz|kg|km|cm|mm|m|s|g|L)\b)')


# A one-character label in brackets ('Reason (R)' read as '(2)', '(i)' as '(1)') is a label, not a quantity
_BRACKET_LABEL = re.compile(r'(?<![\w)])\(\s*\d\s*\)')


def numerals(text: str) -> Counter:
    """
    Multiset of the numbers written in `text`, script-independent: '१,२०० kg and 2.50 m' -> {'1200': 1, '2.5': 1}.
    Thousands separators are dropped (Indian 1,00,000 and Western 100,000 compare equal) and trailing decimal
    zeros removed, so typesetting differences between versions do not count as mismatches.
    """
    out = Counter()
    t = _BRACKET_LABEL.sub(' ', _GLUED_ARTICLE.sub(r'A \1', normalize_digits(text)))
    for m in _NUMBER.finditer(t):
        whole = m.group(1).replace(',', '').lstrip('0') or '0'
        frac = (m.group(2) or '').rstrip('0')
        out[f'{whole}.{frac}' if frac else whole] += 1
    return out


# ------------------------------------------------------------------------------------- question numbers
# 'Q.12', 'Q 12', 'Question 12', '12.', '12)', 'प्रश्न 12', 'प्र. 12', 'கேள்வி 12', 'প্রশ্ন 12' at the start of a region
_Q_WORDS = r'(?:Q(?:ue(?:stion)?)?\s*(?:No\.?)?|प्रश्न\s*(?:सं(?:ख्या)?\.?)?|प्र\.|கேள்வி|வினா|প্রশ্ন|પ્રશ્ન|ਪ੍ਰਸ਼ਨ|ପ୍ରଶ୍ନ|ప్రశ్న|ಪ್ರಶ್ನೆ|ചോദ്യം)'
_QNUM = re.compile(rf'^\s*(?:{_Q_WORDS}\s*[.:\-]?\s*(\d{{1,3}})\b\s*[.):\-।]?|(\d{{1,3}})\s*[.):।](?!\d))', re.IGNORECASE)


def question_number(text: str) -> Tuple[Optional[int], str]:
    """Number at the start of a question and the text after it: '12. Find x' -> (12, 'Find x').
    An instruction that starts with question numbers ('Questions number 19 and 20 are ...') is not a question."""
    t = normalize_digits(text)
    if is_instruction(t):
        return None, t.strip()
    m = _QNUM.match(t)
    if not m:
        return None, t.strip()
    return int(m.group(1) or m.group(2)), t[m.end():].strip()


# ------------------------------------------------------------------------------------------------ marks
_MARK_WORDS = r'(?:marks?|mks?|अंक|गुण|மதிப்பெண்[\w\u0900-\u0DFF]*|নম্বর|নম্বৰ|গুণ|ગુણ|ਅੰਕ|ନମ୍ବର|మార్కులు|ಅಂಕ[\w\u0900-\u0DFF]*|മാർക്ക്[\w\u0900-\u0DFF]*)'
# Marks are printed in brackets or at the end of a line: '(2 marks)', '[2 अंक]', '(2M)', '... 2 Marks', '... [2]'.
# A number followed by 'marks' in the middle of a sentence ('she scored 45 marks in ...') is part of the question.
_MARKS_BRACKETED = re.compile(rf'[\[(]\s*(\d{{1,2}}(?:\.5)?)\s*(?:{_MARK_WORDS}|M)\s*[\])]', re.IGNORECASE)
_MARKS_EOL = re.compile(rf'(\d{{1,2}}(?:\.5)?)\s*{_MARK_WORDS}\s*[.)]?\s*$', re.IGNORECASE | re.MULTILINE)
# '[2]' at the end of a line; OCR reads the brackets as '{1]' or '[1}' too (never '(1)': that can be maths, f(2))
# '[1]' at a line's end, also as OCR reads it: '{1]', '[1}', '(1]', '[1)'
_MARKS_BRACKET_EOL = re.compile(r'(?:[\[{]\s*(\d{1,2}(?:\.5)?)\s*[\]})]|\(\s*(\d{1,2}(?:\.5)?)\s*[\]}])\s*$', re.MULTILINE)
_TOTAL = re.compile(rf'(?:max(?:imum)?\.?\s*marks?|total\s*marks?|full\s*marks?|पूर्णा[ंँ]?क|अधिकतम\s*अंक|कुल\s*अंक|'
                    rf'மொத்த\s*மதிப்பெண்[\w\u0900-\u0DFF]*|அதிகபட்ச\s*மதிப்பெண்[\w\u0900-\u0DFF]*|পূর্ণমান|মোট\s*নম্বর|કુલ\s*ગુણ|ਕੁੱਲ\s*ਅੰਕ|'
                    rf'ମୋଟ\s*ନମ୍ବର|మొత్తం\s*మార్కులు|ಒಟ್ಟು\s*ಅಂಕ[\w\u0900-\u0DFF]*|ആകെ\s*മാർക്ക്[\w\u0900-\u0DFF]*)\s*[:\-–]?\s*(\d{{1,4}})',
                    re.IGNORECASE)
_EACH = re.compile(rf'(?:each\s+question\s+carries|प्रत्येक\s+प्रश्न|ஒவ்வொரு\s+(?:கேள்விக்கும்|வினாவுக்கும்))\D{{0,20}}?'
                   rf'(\d{{1,2}}(?:\.5)?)[\s।|]*{_MARK_WORDS}', re.IGNORECASE)  # '1 । अंक': OCR adds a danda


def marks_in(text: str) -> Optional[float]:
    """Marks printed with a question ('(2 marks)', '[2 अंक]', '... [2]' at a line end), or None."""
    t = normalize_digits(text)
    m = _MARKS_BRACKETED.search(t) or _MARKS_EOL.search(t) or _MARKS_BRACKET_EOL.search(t)
    return float(next(g for g in m.groups() if g)) if m else None


def strip_marks(text: str) -> str:
    t = normalize_digits(text)
    for pattern in (_MARKS_BRACKETED, _MARKS_EOL, _MARKS_BRACKET_EOL):
        t = pattern.sub(' ', t)
    return t


def declared_total(text: str) -> Optional[int]:
    """'Maximum Marks: 80' / 'पूर्णांक : 80' / 'மொத்த மதிப்பெண்கள்: 80' -> 80."""
    m = _TOTAL.search(normalize_digits(text))
    return int(m.group(1)) if m else None


def marks_each(text: str) -> Optional[float]:
    """'Each question carries 4 marks' / 'प्रत्येक प्रश्न 4 अंक का है' -> 4.0."""
    m = _EACH.search(normalize_digits(text))
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------------------------------------- options
# Families of option markers, each in printed order. A paper uses one family per question.
OPTION_FAMILIES = {
    'upper': list('ABCDEF'), 'lower': list('abcdef'), 'digit': list('123456'),
    'devanagari': ['क', 'ख', 'ग', 'घ', 'ङ'], 'tamil': ['அ', 'ஆ', 'இ', 'ஈ', 'உ'],
    'bengali': ['ক', 'খ', 'গ', 'ঘ', 'ঙ'], 'roman': ['i', 'ii', 'iii', 'iv', 'v'],
}
_OPT_MARKER = re.compile(r'(?:(?<=\s)|^)\(\s*([A-Fa-f1-6]|[कखगघङ]|[அஆஇஈஉ]|[কখগঘঙ]|i{1,3}|iv|v)\s*\)'
                         r'|(?:(?<=\s)|^)([A-Fa-f]|[कखगघङ]|[அஆஇஈஉ]|[কখগঘঙ])\s*[.)](?=\s)')


_BRACKET_TOKEN = re.compile(r'(?:(?<=\s)|^)\((?:[^\s()]{1,3}\)|(?=\S))')


def split_options(text: str, options_box: bool = False) -> List[Tuple[str, str]]:
    """
    Options printed in `text` as (marker, option text): '(1) 40 km (2) 60 km' -> [('1', '40 km'), ('2', '60 km')].
    Only the longest run that starts at a family's first marker and follows its order counts, so a stray '(2)'
    inside a stem is not taken for an option. Digits must be in brackets ('(1)'), because '1.' starts a question.
    options_box: the text is a region the layout model labelled 'options'. There every word that opens with a
    bracket starts an option, which still counts options whose marker OCR garbled ('(ஈ)' read as '(FF)').
    """
    t = normalize_digits(text)
    if options_box:
        starts = [m.start() for m in _BRACKET_TOKEN.finditer(t)]
        family = split_options(text)
        if len(starts) >= 2 and len(starts) > len(family):
            out = []
            for k, start in enumerate(starts):
                chunk = t[start:starts[k + 1] if k + 1 < len(starts) else len(t)]
                close = chunk.find(')')
                marker, body = (chunk[1:close], chunk[close + 1:]) if 0 < close <= 4 else (str(k + 1), chunk[1:])
                out.append((marker.strip() or str(k + 1), ' '.join(body.split())))
            return out
        return family
    hits = [(m.start(), m.end(), m.group(1) or m.group(2)) for m in _OPT_MARKER.finditer(t)]
    best: List[Tuple[int, int, str]] = []
    for family in OPTION_FAMILIES.values():
        run: List[Tuple[int, int, str]] = []
        for hit in hits:
            if len(run) < len(family) and hit[2] == family[len(run)]:
                run.append(hit)
        if len(run) >= 2 and len(run) > len(best):
            best = run
    out = []
    for k, (start, end, marker) in enumerate(best):
        stop = best[k + 1][0] if k + 1 < len(best) else len(t)
        out.append((marker, ' '.join(t[end:stop].split())))
    return out


def stem_before_options(text: str) -> str:
    """The question text before its first option marker (the whole text if it has no options)."""
    opts = split_options(text)
    if not opts:
        return text
    t = normalize_digits(text)
    m = re.search(r'(?:(?<=\s)|^)\(?\s*' + re.escape(opts[0][0]) + r'\s*[.)]', t)
    return t[:m.start()] if m else t


# ------------------------------------------------------------------------------------------- similarity
def normalized(text: str) -> str:
    """Lower-case, NFC, digits normalised, punctuation and spacing removed: for comparing two question texts."""
    t = unicodedata.normalize('NFC', normalize_digits(text)).lower()
    t = re.sub(rf'[^\w\s{INDIC}]', ' ', t)
    return ' '.join(t.split())


def ngrams(text: str, n: int = 3) -> Counter:
    t = normalized(text).replace(' ', '_')
    return Counter(t[i:i + n] for i in range(max(0, len(t) - n + 1)))


def similarity(a: str, b: str) -> float:
    """Cosine similarity of character 3-gram counts (robust to OCR slips; 1.0 = same text)."""
    ga, gb = ngrams(a), ngrams(b)
    if not ga or not gb:
        return 0.0
    dot = sum(v * gb.get(k, 0) for k, v in ga.items())
    na = sum(v * v for v in ga.values()) ** 0.5
    nb = sum(v * v for v in gb.values()) ** 0.5
    return dot / (na * nb)


def split_lines_by_script(text: str, keep: Dict[str, str]) -> Dict[str, str]:
    """
    Lines of a bilingual region sorted into one text per script. Lines with no letters (a question number, a
    formula, '(1) 40') go to every script, because both versions need them.
    keep: script name -> stream name, e.g. {'Latin': 'english', 'Devanagari': 'hindi'}.
    """
    out = {name: [] for name in keep.values()}
    for line in text.splitlines():
        s = script_of(line)
        if s is None:
            for lines in out.values():
                lines.append(line)
        elif s in keep:
            out[keep[s]].append(line)
    return {name: '\n'.join(lines) for name, lines in out.items()}


# -------------------------------------------------------------------------- paper structure and furniture
# 'Page 8 of 24', 'पृष्ठ 8 / 24': the printed page number with the page count
_PAGE_OF = re.compile(r'(?:page|pg\.?|पृष्ठ|पेज|பக்கம்|পৃষ্ঠা|পৃষ্ঠা|પૃષ્ઠ|ਪੰਨਾ|ପୃଷ୍ଠା|పేజీ|ಪುಟ|പേജ്)\s*(?:no\.?)?\s*'
                      r'(\d{1,3})\s*(?:of|/|का|में\s*से|का\s*कुल)\s*(\d{1,3})', re.IGNORECASE)


def page_of(text: str) -> Optional[Tuple[int, int]]:
    """'Page 8 of 24' -> (8, 24); None if the text has no such footer."""
    m = _PAGE_OF.search(normalize_digits(text))
    return (int(m.group(1)), int(m.group(2))) if m else None


# Section headings: 'SECTION – A', 'Section B', 'खण्ड – क', 'भाग 2', 'பகுதி அ'. Letters of every script map to one key,
# so the Hindi and the English version of a paper share section keys.
_SECTION_WORDS = (r'(?:SECTION|Section|PART|Part|खण्ड|खंड|भाग|பகுதி|বিভাগ|খণ্ড|વિભાગ|ਭਾਗ|ਖੰਡ|ବିଭାଗ|విభాగం|ವಿಭಾಗ|'
                  r'വിഭാഗം)')
_SECTION = re.compile(rf'^\s*{_SECTION_WORDS}\s*[–—\-:.]?\s*([A-Za-z]\b|[IVX]{{1,4}}\b|\d{{1,2}}\b|[\u0900-\u0DFF]{{1,2}})')
_SECTION_LETTERS = {**{c: chr(ord('A') + k) for k, c in enumerate('कखगघङचछ')},
                    **{c: chr(ord('A') + k) for k, c in enumerate('அஆஇஈஉஊ')},
                    **{c: chr(ord('A') + k) for k, c in enumerate('কখগঘঙ')},
                    **{c: chr(ord('A') + k) for k, c in enumerate('કખગઘ')}}
_ROMAN = {'I': 'A', 'II': 'B', 'III': 'C', 'IV': 'D', 'V': 'E', 'VI': 'F'}


def section_key(text: str) -> Optional[str]:
    """'SECTION – B' / 'खण्ड – ख' / 'Part II' -> 'B'; None if the text is not a section heading."""
    t = normalize_digits(text)
    first = t.strip().splitlines()[0] if t.strip() else ''
    if len(first) > 40:
        return None
    m = _SECTION.match(first)
    if not m:
        return None
    k = m.group(1).strip()
    if k[0] in _SECTION_LETTERS:
        return _SECTION_LETTERS[k[0]]
    if k.upper() in _ROMAN and not k.isdigit():
        return _ROMAN[k.upper()]
    if k.isdigit():
        return chr(ord('A') + int(k) - 1) if 0 < int(k) < 27 else None
    return k.upper() if k.isascii() else None


# Instructions, not questions: they end the question before them and feed the marks rules
_INSTRUCTION = re.compile(
    r'^\s*(?:\(?[ivx]{1,5}\)\s*)?(?:'
    r'(?:Questions?|प्रश्न)\s*(?:numbers?|nos?\.?|संख्या|सं\.?)?\s*\d{1,3}\s*(?:and|to|&|-|–|से|तथा|और|एवं|व)\s*\d{1,3}'
    r'|This\s+section\s+(?:comprises|consists|contains|has)|This\s+question\s+paper\s+(?:contains|has|is)'
    r'|इस\s+(?:खण्ड|खंड|प्रश्न[-\s]*पत्र)|General\s+Instructions|सामान्य\s+निर्देश|Read\s+the\s+following\s+instructions'
    r'|निम्नलिखित\s+निर्देशों|Please\s+check\s+that|कृपया\s+जाँच'
    r'|In\s+Section\s*[–\-]?\s*[A-E]\b|खण्ड\s*[–\-]?\s*[कखगघङ]\s+में'
    # a section's own note under its heading, also when OCR garbled its first word ('इस' read as '3a')
    r'|\S{0,3}\s*(?:खण्ड|खंड)\s+में\s+\d{1,2}\s.*प्रश्न\s+हैं|\S{0,3}\s*section\s+(?:comprises|consists\s+of|contains)\s+\d)',
    re.IGNORECASE)


def is_instruction(text: str) -> bool:
    return bool(_INSTRUCTION.match(normalize_digits(text)))


# Internal choice between two alternatives of one question
_CHOICE = re.compile(r'^\s*(?:OR|Or|अथवा|या|அல்லது|অথবা|અથવા|ਜਾਂ|ଅଥବା|లేదా|ಅಥವಾ|അല്ലെങ്കിൽ)\s*$')


def is_choice_line(text: str) -> bool:
    return bool(_CHOICE.match(text))


_MARK_VALUE = re.compile(rf'(\d{{1,2}}(?:\.5)?)\s*(?:{_MARK_WORDS}|अंकों|अंको)', re.IGNORECASE)
_RANGE = re.compile(r'(\d{1,3})\s*(?:to|–|-|से)\s*(\d{1,3})')
_PAIR = re.compile(r'(\d{1,3})\s*(?:&|and|एवं|और|तथा|व)\s*(\d{1,3})(?!\s*(?:to|–|-|से))')
_SECTION_IN = re.compile(rf'(?:In\s+)?{_SECTION_WORDS}\s*[–—\-]?\s*([A-Za-z]|[क-ङ])\b', re.IGNORECASE)


def _sentences(text: str) -> List[str]:
    t = ' '.join(normalize_digits(text).split())
    t = re.sub(r'(\d)\s*[।|]\s+(?=(?:से|to)\s*\d)', r'\1 ', t)   # '1 । से 18': a misread digit stroke, not a full stop
    return [p for p in re.split(r'(?<=[।.;])\s+|\s(?=\((?:[ivx]{1,5})\)\s)', t) if p.strip()]


def marks_schedule(text: str) -> Tuple[Dict[int, float], Dict[str, float]]:
    """
    Marks the instructions give to questions and sections, as ({question: marks}, {section key: marks}):
    'Questions Number 21 to 25 are ... carrying 2 marks each' -> {21: 2, ..., 25: 2};
    'This section comprises 5 ... questions of 2 marks each' (under 'SECTION - B') -> {'B': 2}.
    """
    per_q: Dict[int, float] = {}
    per_section: Dict[str, float] = {}
    section, fresh = None, 0
    for sent in _sentences(text):
        key = section_key(sent)
        if key:
            section, fresh = key, 2
        fresh -= 1
        if fresh <= 0:
            section = None    # 'this section' refers to a heading just above it, never to one pages back
        if _TOTAL.search(sent):
            continue
        values = {float(m.group(1)) for m in _MARK_VALUE.finditer(sent)}
        if len(values) != 1:
            continue
        value = values.pop()
        numbers = []
        for m in _RANGE.finditer(sent):
            lo, hi = int(m.group(1)), int(m.group(2))
            if 0 < lo <= hi <= lo + 60:
                numbers += range(lo, hi + 1)
        for m in _PAIR.finditer(sent):
            numbers += [int(m.group(1)), int(m.group(2))]
        numbers = [n for n in numbers if n != value or len(numbers) > 1]
        if numbers:
            for n in numbers:
                per_q.setdefault(n, value)
            continue
        m = _SECTION_IN.search(sent)
        sec = section_key(f'SECTION {m.group(1)}') if m else None
        if sec or (section and re.search(r'this\s+section|इस\s+(?:खण्ड|खंड)', sent, re.IGNORECASE)):
            per_section.setdefault(sec or section, value)
    return per_q, per_section


_DECLARED_QUESTIONS = re.compile(r'(?:contains|has|consists\s+of|comprises)\s+(\d{1,3})\s+questions'
                                 r'|में\s+(\d{1,3})\s+प्रश्न\s+हैं|(\d{1,3})\s+प्रश्न\s+हैं', re.IGNORECASE)
_DECLARED_PAGES = re.compile(r'(?:contains|has)\s+(\d{1,3})\s+printed\s+pages|(\d{1,3})\s+(?:मुद्रित\s+)?पृष्ठ\s+हैं',
                             re.IGNORECASE)


def declared_questions(text: str) -> Optional[int]:
    """'This question paper contains 38 questions' / 'इस प्रश्न-पत्र में 38 प्रश्न हैं' -> 38."""
    for m in _DECLARED_QUESTIONS.finditer(normalize_digits(text)):
        return int(next(g for g in m.groups() if g))
    return None


def declared_pages(text: str) -> Optional[int]:
    """'Please check that this question paper contains 23 printed pages' -> 23."""
    for m in _DECLARED_PAGES.finditer(normalize_digits(text)):
        return int(next(g for g in m.groups() if g))
    return None


def option_family(markers: List[str]) -> Optional[str]:
    """The marker family of a question's options ('A','B','C' -> 'upper'); None if unknown or mixed."""
    if len(markers) < 2:
        return None
    for name, fam in OPTION_FAMILIES.items():
        if all(m in fam for m in markers) and name != 'digit':
            return name
    return None
