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


def numerals(text: str) -> Counter:
    """
    Multiset of the numbers written in `text`, script-independent: '१,२०० kg and 2.50 m' -> {'1200': 1, '2.5': 1}.
    Thousands separators are dropped (Indian 1,00,000 and Western 100,000 compare equal) and trailing decimal
    zeros removed, so typesetting differences between versions do not count as mismatches.
    """
    out = Counter()
    for m in _NUMBER.finditer(normalize_digits(text)):
        whole = m.group(1).replace(',', '').lstrip('0') or '0'
        frac = (m.group(2) or '').rstrip('0')
        out[f'{whole}.{frac}' if frac else whole] += 1
    return out


# ------------------------------------------------------------------------------------- question numbers
# 'Q.12', 'Q 12', 'Question 12', '12.', '12)', 'प्रश्न 12', 'प्र. 12', 'கேள்வி 12', 'প্রশ্ন 12' at the start of a region
_Q_WORDS = r'(?:Q(?:ue(?:stion)?)?\s*(?:No\.?)?|प्रश्न\s*(?:सं(?:ख्या)?\.?)?|प्र\.|கேள்வி|வினா|প্রশ্ন|પ્રશ્ન|ਪ੍ਰਸ਼ਨ|ପ୍ରଶ୍ନ|ప్రశ్న|ಪ್ರಶ್ನೆ|ചോദ്യം)'
_QNUM = re.compile(rf'^\s*(?:{_Q_WORDS}\s*[.:\-]?\s*(\d{{1,3}})\b\s*[.):\-]?|(\d{{1,3}})\s*[.):](?!\d))', re.IGNORECASE)


def question_number(text: str) -> Tuple[Optional[int], str]:
    """Number at the start of a question and the text after it: '12. Find x' -> (12, 'Find x')."""
    t = normalize_digits(text)
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
_MARKS_BRACKET_EOL = re.compile(r'\[\s*(\d{1,2}(?:\.5)?)\s*\]\s*$', re.MULTILINE)
_TOTAL = re.compile(rf'(?:max(?:imum)?\.?\s*marks?|total\s*marks?|full\s*marks?|पूर्णांक|अधिकतम\s*अंक|कुल\s*अंक|'
                    rf'மொத்த\s*மதிப்பெண்[\w\u0900-\u0DFF]*|அதிகபட்ச\s*மதிப்பெண்[\w\u0900-\u0DFF]*|পূর্ণমান|মোট\s*নম্বর|કુલ\s*ગુણ|ਕੁੱਲ\s*ਅੰਕ|'
                    rf'ମୋଟ\s*ନମ୍ବର|మొత్తం\s*మార్కులు|ಒಟ್ಟು\s*ಅಂಕ[\w\u0900-\u0DFF]*|ആകെ\s*മാർക്ക്[\w\u0900-\u0DFF]*)\s*[:\-–]?\s*(\d{{1,4}})',
                    re.IGNORECASE)
_EACH = re.compile(rf'(?:each\s+question\s+carries|प्रत्येक\s+प्रश्न|ஒவ்வொரு\s+(?:கேள்விக்கும்|வினாவுக்கும்))\D{{0,20}}?'
                   rf'(\d{{1,2}}(?:\.5)?)\s*{_MARK_WORDS}', re.IGNORECASE)


def marks_in(text: str) -> Optional[float]:
    """Marks printed with a question ('(2 marks)', '[2 अंक]', '... [2]' at a line end), or None."""
    t = normalize_digits(text)
    m = _MARKS_BRACKETED.search(t) or _MARKS_EOL.search(t) or _MARKS_BRACKET_EOL.search(t)
    return float(m.group(1)) if m else None


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
