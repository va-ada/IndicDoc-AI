"""
Tests for dual-reader verification: the paper-structure rules (sections, instructions, marks schedule, 'Page X of Y'),
reader B (layout-free reading), bilingual version detection, and the consensus of two readings.
"""
from ml.verify import verify
from ml.verify.consensus import fuse, pair_questions
from ml.verify.paper import parse_paper
from ml.verify.reader import read_page, read_pages, split_columns, text_layer_usable
from ml.verify.text import (declared_pages, declared_questions, is_choice_line, is_instruction, marks_schedule,
                            page_of, question_number, section_key)
from ml.verify.versions import detect_versions, split_versions


# ------------------------------------------------------------------------------------------- helpers
def line(text, y, x=100, h=30, conf=95.0, gap_at=None):
    """Words of one printed line; gap_at: word index after which a wide gap starts (an option grid's second cell)."""
    words, cx = [], x
    for k, t in enumerate(text.split()):
        if gap_at is not None and k == gap_at:
            cx += 400
        w = 14 * len(t)
        words.append({'text': t, 'left': cx, 'top': y, 'width': w, 'height': h, 'conf': conf})
        cx += w + 12
    return words


def page(lines, size=(1650, 2330)):
    words = []
    for spec in lines:
        words += line(*spec) if isinstance(spec, tuple) else spec
    return {'words': words, 'size': size}


EN_PAGE = page([
    ('SECTION - A', 300),
    ('This section comprises 3 Multiple Choice Questions of 1 mark each.', 360),
    ('1. A car travels 120 km in 2 hours. What is its speed?', 440),
    ('(A) 60 km/h (B) 50 km/h', 490, 160, 30, 95.0, 3),
    ('(C) 40 km/h (D) 30 km/h', 540, 160, 30, 95.0, 3),
    ('2. In which year did India become independent?', 620),
    ('(A) 1945 (B) 1946', 670, 160, 30, 95.0, 2),
    ('(C) 1947 (D) 1948', 720, 160, 30, 95.0, 2),
    ('3. (a) Explain photosynthesis.', 800),
    ('OR', 850, 700),
    ('(b) State Ohm\'s law.', 900, 160),
    ('465 Page 3 of 8 P.T.O.', 2200),
])


# --------------------------------------------------------------------------------------- text rules
class TestStructureRules:
    def test_page_of_footer(self):
        assert page_of('465 Page 8 of 24 P.T.O.') == (8, 24)
        assert page_of('पृष्ठ 3 / 12') == (3, 12)
        assert page_of('Question 8 of the paper') is None

    def test_section_keys_match_across_languages(self):
        assert section_key('SECTION – B') == section_key('खण्ड – ख') == 'B'
        assert section_key('Part II') == 'B'
        assert section_key('1. A car travels 120 km') is None

    def test_instructions_are_not_questions(self):
        for t in ('Questions number 19 and 20 are Assertion-Reason based questions',
                  'प्रश्न संख्या 19 और 20 अभिकथन एवं तर्क आधारित प्रश्न हैं', 'This section comprises 5 questions'):
            assert is_instruction(t)
            assert question_number(t)[0] is None
        assert question_number('19. Assertion (A): ...')[0] == 19
        assert question_number('14। यदि x वस्तुओं')[0] == 14   # a danda read for the full stop

    def test_marks_schedule_from_instructions(self):
        en = ('(iii) In Section – A, Questions Number 1 to 18 are MCQs and Questions Number 19 & 20 are '
              'Assertion-Reason based questions of 1 mark each. (iv) In Section – B, Questions Number 21 to 25 are '
              'Very Short Answer type questions, carrying 2 marks each. Maximum Marks : 80')
        per_q, _ = marks_schedule(en)
        assert per_q[1] == per_q[20] == 1.0 and per_q[21] == per_q[25] == 2.0 and 26 not in per_q
        hi = 'खण्ड – ख में प्रश्न संख्या 21 से 25 तक अति लघु-उत्तरीय प्रकार के 2 अंकों के प्रश्न हैं।'
        assert marks_schedule(hi)[0][23] == 2.0
        assert marks_schedule('SECTION – C This section comprises 6 questions of 3 marks each.')[1] == {'C': 3.0}

    def test_declared_counts(self):
        assert declared_questions('This question paper contains 38 questions.') == 38
        assert declared_questions('इस प्रश्न-पत्र में 38 प्रश्न हैं।') == 38
        assert declared_pages('Please check that this question paper contains 23 printed pages.') == 23
        assert is_choice_line('OR') and is_choice_line('अथवा') and not is_choice_line('ORDER')


# -------------------------------------------------------------------------------------------- reader B
class TestReaderB:
    def test_reads_structure_without_layout(self):
        pg = read_page(EN_PAGE['words'], EN_PAGE['size'], source='text-layer')
        labels = [r['label'] for r in pg['regions']]
        assert labels.count('first-level-question') == 3
        assert labels.count('options') == 2
        assert 'section-title' in labels and 'page-number' in labels
        paper = parse_paper([pg])
        assert [q.number for q in paper.questions] == [1, 2, 3]
        assert [len(q.options) for q in paper.questions] == [4, 4, 0]
        assert paper.questions[2].choices == 1             # (a) OR (b): one internal choice
        assert paper.questions[0].marks == 1.0             # from 'This section comprises ... 1 mark each'
        assert paper.page_of == [[3, 8]]

    def test_running_footer_and_expected_number(self):
        a = page([('1. First question here?', 400), ('465 Page 1 of 2', 2200)])
        b = page([('2, Second question here?', 400), ('465 Page 2 of 2', 2200)])   # ',' read for '.'
        pages = read_pages([a, b])
        nums = [q.number for q in parse_paper(pages).questions]
        assert nums == [1, 2]

    def test_two_column_paper(self):
        words = []
        for k in range(3):
            words += line(f'{k + 1}. Left column question number {k + 1} with some text here', 300 + k * 200, 100)
            words += line('(1) 10 (2) 20', 350 + k * 200, 140, 30, 95.0, 2)
            words += line(f'{k + 4}. Right column question {k + 4} with words', 300 + k * 200, 880)
            words += line('(1) 30 (2) 40', 350 + k * 200, 920, 30, 95.0, 2)
        paper = parse_paper(read_pages([{'words': words, 'size': (1650, 2330)}]))
        assert [q.number for q in paper.questions] == [1, 2, 3, 4, 5, 6]
        assert [len(q.options) for q in paper.questions] == [2] * 6
        # One column with short stems and option grids is not mistaken for two
        assert split_columns(EN_PAGE['words'], 1650) == [split_columns(EN_PAGE['words'], 1650)[0]]

    def test_text_layer_check(self):
        good = line('यह एक हिंदी प्रश्न है ' * 5, 100)
        assert text_layer_usable(good, 'Devanagari')
        garbage = line("ou ,d fgUnh iz'zu gS " * 6, 100)                    # a legacy (non-Unicode) Hindi font
        assert not text_layer_usable(garbage, 'Devanagari')


# --------------------------------------------------------------------------------------- versions
class TestVersions:
    def test_page_interleaved_bilingual_paper(self):
        hi = page([('1. एक कार 2 घंटे में 120 किमी चलती है। उसकी औसत चाल क्या है?', 400),
                   ('(A) 60 किमी/घंटा (B) 50 किमी/घंटा', 450, 160, 30, 95.0, 2)])
        en = page([('1. A car travels 120 km in 2 hours. What is its average speed?', 400),
                   ('(A) 60 km/h (B) 50 km/h', 450, 160, 30, 95.0, 3)])
        pages = read_pages([hi, en, hi, en])
        for k, p in enumerate(pages):
            p['sheet'] = k
        plan = detect_versions(pages)
        assert plan['mode'] == 'pages' and plan['source'] == 'Latin'
        parts = split_versions(pages, plan)
        assert [p['sheet'] for p in parts['Devanagari']] == [0, 2]
        assert [p['sheet'] for p in parts['Latin']] == [1, 3]


# -------------------------------------------------------------------------------------- consensus
def _ocr_page_misreading_q2():
    """Reader A's view of EN_PAGE: OCR dropped the number of question 2 and misread one option value."""
    words = []
    for w in EN_PAGE['words']:
        w = dict(w)
        if w['text'] == '2.' and w['top'] == 620:
            continue
        if w['text'] == '1946':
            w['text'] = '1940'
        words.append(w)
    return read_pages([{'words': words, 'size': EN_PAGE['size']}], source='ocr')


class TestConsensus:
    def test_pairing_follows_content_and_position(self):
        a = parse_paper(_ocr_page_misreading_q2())
        b = parse_paper(read_pages([EN_PAGE], source='text-layer'))
        pairs = pair_questions(a.questions, b.questions)
        # Without its number, question 2 ran into question 1 in reader A; the alignment keeps the rest in step
        assert len(a.questions) == 2 and pairs == [(0, 0), (None, 1), (1, 2)]

    def test_text_layer_decides_and_single_reader_findings_are_dismissed(self):
        pa = _ocr_page_misreading_q2()
        pb = read_pages([EN_PAGE], source='text-layer')
        rep = verify(pa, pages_b=pb, return_readers=True)
        a_codes = {i['code'] for i in rep['reader_issues']['a']}
        assert 'NUMBER_GAP' in a_codes                        # reader A alone: 'numbers jump from 1 to 3'
        assert not [i for i in rep['issues'] if i['code'] in ('NUMBER_GAP', 'OPTION_COUNT')]
        assert any(d['code'] == 'NUMBER_GAP' and d['raised_by'] == 'a' for d in rep['consensus']['dismissed'])
        fused, _ = fuse(parse_paper(pa), parse_paper(pb))
        assert [q.number for q in fused.questions] == [1, 2, 3] and '1946' in fused.questions[1].options

    def test_both_readers_confirm_a_real_error(self):
        dup = page([('1. A car travels 120 km in 2 hours. What is its speed?', 400),
                    ('2. A car travels 120 km in 2 hours. What is its speed?', 500)])
        pa = read_pages([dup], source='ocr')
        pb = read_pages([dup], source='text-layer')
        rep = verify(pa, pages_b=pb)
        dups = [i for i in rep['issues'] if i['code'] == 'DUPLICATE_QUESTION']
        assert dups and dups[0]['consensus']['status'] == 'confirmed'
        assert dups[0]['consensus']['reader_a'] and dups[0]['consensus']['reader_b']
        assert rep['summary']['confirmed'] >= 1

    def test_unsettled_disagreement_is_review_not_error(self):
        pa = read_pages([page([('1. A train 225 m long passes a pole in 15 seconds?', 400)])], source='ocr')
        pb = read_pages([page([('1. A train 225 m long passes a pole in 18 seconds?', 400)])], source='ocr')
        src = read_pages([page([('1. A train 225 m long passes a pole in 15 seconds?', 400)])], source='ocr')
        rep = verify(pa, source_pages=src, pages_b=pb, source_pages_b=src)
        mism = [i for i in rep['issues'] if i['code'] == 'NUMBER_MISMATCH']
        # The readers read the translation differently and nothing re-read it: a warning for the vetter, not an error
        assert all(i['severity'] == 'warning' and i['consensus']['status'] == 'review' for i in mism)


def test_reader_numbered_options_and_questions_at_the_top_edge():
    """Options numbered '(1)'-'(4)' are not questions, and a tight page's first question is not a header."""
    p1 = page([('1. Which gas do plants take in?', 300), ('(1) Nitrogen (2) Carbon dioxide', 350, 160, 30, 95.0, 2),
               ('(3) Oxygen (4) Hydrogen', 400, 160, 30, 95.0, 2), ('2. Which is the largest organ?', 480),
               ('(1) Lungs (2) Liver', 530, 160, 30, 95.0, 2), ('(3) Skin (4) Heart', 580, 160, 30, 95.0, 2)])
    p2 = page([('3. How many days are there in a leap year?', 100), ('(1) 366 (2) 367', 160, 160, 30, 95.0, 2),
               ('(3) 365 (4) 364', 210, 160, 30, 95.0, 2), ('4. Who wrote the national anthem?', 300),
               ('(1) Naidu (2) Tagore', 350, 160, 30, 95.0, 2), ('(3) Chatterjee (4) Gandhi', 400, 160, 30, 95.0, 2)])
    pages = read_pages([p1, p2], source='text-layer')
    for k, p in enumerate(pages):
        p['sheet'] = k
    paper = parse_paper(pages)
    assert [q.number for q in paper.questions] == [1, 2, 3, 4]
    assert [len(q.options) for q in paper.questions] == [4, 4, 4, 4]


def test_fragment_filling_a_numbering_gap_is_reinstated():
    from ml.verify.consensus import _reinstate_gaps
    from ml.verify.paper import Question
    q = lambda n, i: Question(index=i, number=n, page=0, box=[0, 0, 1, 1], stem='a question')
    fused = [q(2, 0), q(3, 1)]
    records = [{'a': 0, 'b': None, 'dropped': True, '_candidate': (0, q(None, 0))},
               {'a': 1, 'b': 0}, {'a': 2, 'b': 1}]
    _reinstate_gaps(fused, records)
    assert [x.number for x in fused] == [1, 2, 3] and not records[0]['dropped']
    fused = [q(1, 0), q(2, 1)]                       # nothing missing: the fragment stays dropped
    records = [{'a': 0, 'b': 0}, {'a': 1, 'b': None, 'dropped': True, '_candidate': (1, q(None, 0))}, {'a': 2, 'b': 1}]
    _reinstate_gaps(fused, records)
    assert [x.number for x in fused] == [1, 2] and records[1]['dropped']


def test_number_mismatch_is_rechecked_by_the_english_reading():
    from ml.verify.consensus import verify_numbers
    from ml.verify.paper import Paper, Question

    def paper(stem, source):
        q = Question(index=0, number=1, page=0, box=[0, 0, 100, 40], stem=stem, source=source,
                     boxes=[{'page': 0, 'box': [0, 0, 100, 40]}])
        return Paper(questions=[q], declared_total=None, marks_each=None, page_numbers=[1], pages=1)
    papers = {'source': paper('A 12 V battery', 'text-layer'), 'target': paper('12 · · की बैटरी 5', 'ocr')}
    issue = {'code': 'NUMBER_MISMATCH', 'severity': 'error', 'version': 'target', 'message': '',
             'details': {'qindex': 0, 'sindex': 0, 'source_only': [], 'target_only': ['5']}}
    misread = lambda sheet, box, lang: [{'text': '12', 'left': 0, 'top': 0, 'width': 10, 'height': 10}]
    kept, dismissed = verify_numbers([issue], papers, {'target': misread})
    assert not kept and dismissed[0]['raised_by'] == 'consensus'
    really_there = lambda sheet, box, lang: [{'text': t, 'left': 0, 'top': 0, 'width': 10, 'height': 10}
                                             for t in ('12', '5')]
    kept, dismissed = verify_numbers([issue], papers, {'target': really_there})
    assert kept and not dismissed
