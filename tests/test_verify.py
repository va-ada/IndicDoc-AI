"""Tests for ml.verify: parsing question papers from layout + OCR regions, and every check."""
from ml.verify import check_format, list_profiles, parse_paper, split_bilingual, verify
from ml.verify.checks import align, compare_versions, find_repeats
from ml.verify.report import to_html
from ml.verify.text import (declared_total, marks_each, marks_in, numerals, question_number, script_of,
                            split_options)


def region(label, text, y, page_h=40):
    return {'label': label, 'box': [50, y, 550, y + page_h], 'score': 0.9, 'text': text, 'detected': True}


def page(regions, script=None):
    return {'regions': regions, 'script': script}


def english_paper(n=5, options=4, total=None, marks=4, drop=None, dup=None, numbers=None, opts_override=None):
    """A clean MCQ paper: question k asks about k*10 km; options are four speeds."""
    regs = [region('section-title', 'PHYSICS', 10),
            region('paragraph', f'Time: 3 hours    Maximum Marks: {total if total is not None else n * marks}\n'
                                f'Instructions: Each question carries {marks} marks.', 40)]
    y = 100
    nums = numbers or list(range(1, n + 1))
    for k in nums:
        if k == drop:
            continue
        qtext = f'{k}. A car travels {k * 10} km in {k + 1} hours. Find the average speed of the car on day {k}.'
        regs.append(region('first-level-question', qtext, y)); y += 50
        n_opt = (opts_override or {}).get(k, options)
        regs.append(region('options', ' '.join(f'({i + 1}) {k * 5 + i} km/h' for i in range(n_opt)), y)); y += 50
        if k == dup:
            regs.append(region('first-level-question', qtext.replace(f'{k}.', '99.', 1), y)); y += 50
    regs.append(region('page-number', '1', 1000))
    return [page(regs, 'Latin')]


def hindi_paper(n=5, options=4, marks=4, changes=None, drop=None):
    """Hindi version of english_paper (Devanagari digits on purpose): same numbers, same structure."""
    changes = changes or {}
    regs = [region('section-title', 'भौतिकी', 10),
            region('paragraph', f'समय: 3 घंटे    पूर्णांक : {n * marks}\nनिर्देश: प्रत्येक प्रश्न {marks} अंक का है।', 40)]
    y = 100
    to_dev = str.maketrans('0123456789', '०१२३४५६७८९')
    for k in range(1, n + 1):
        if k == drop:
            continue
        dist = changes.get(k, k * 10)
        qtext = f'{k}. एक कार {k + 1} घंटे में {dist} किमी चलती है। दिन {k} पर कार की औसत चाल ज्ञात कीजिए।'
        regs.append(region('first-level-question', qtext.translate(to_dev), y)); y += 50
        regs.append(region('options', ' '.join(f'({i + 1}) {k * 5 + i} किमी/घंटा' for i in range(options)), y)); y += 50
    regs.append(region('page-number', '१', 1000))
    return [page(regs, 'Devanagari')]


def codes(issues):
    return sorted({i['code'] if isinstance(i, dict) else i.code for i in issues})


# --------------------------------------------------------------------------------------------- text
def test_digits_and_numerals_are_script_independent():
    assert numerals('१,२०० kg और 2.50 m') == numerals('1200 kg and 2.5 m')
    assert numerals('1,00,000') == numerals('100000')
    assert numerals('CO2 and H2O') == {}  # subscripts inside formulas are not free-standing numbers


def test_question_numbers_in_several_languages():
    assert question_number('Q.12 Find x')[0] == 12
    assert question_number('प्रश्न १२. x ज्ञात करें')[0] == 12
    assert question_number('கேள்வி 7: விடை')[0] == 7
    assert question_number('3.5 is a decimal')[0] is None


def test_marks_and_totals():
    assert marks_in('Explain photosynthesis. (2 marks)') == 2
    assert marks_in('व्याख्या कीजिए। [3 अंक]') == 3
    assert marks_in('A car went 60 m') is None
    assert marks_in('She scored 45 marks in maths and 50 in science. What is her total?') is None
    assert declared_total('पूर्णांक : ८०') == 80
    assert declared_total('மொத்த மதிப்பெண்கள்: 90') == 90
    assert marks_each('Each question carries 4 marks.') == 4


def test_options_follow_marker_order():
    assert [m for m, _ in split_options('(1) 40 (2) 60 (3) 80 (4) 120')] == ['1', '2', '3', '4']
    assert [m for m, _ in split_options('(क) दिल्ली (ख) मुंबई (ग) पुणे')] == ['क', 'ख', 'ग']
    # a '(2)' inside the stem before option (1) is not an option
    assert len(split_options('Repeat (2) times. (A) yes (B) no')) == 2
    assert script_of('एक कार 120 km चलती') == 'Devanagari'


# -------------------------------------------------------------------------------------------- paper
def test_parse_clean_paper():
    paper = parse_paper(english_paper())
    assert [q.number for q in paper.questions] == [1, 2, 3, 4, 5]
    assert all(len(q.options) == 4 for q in paper.questions)
    assert paper.declared_total == 20 and all(q.marks == 4 for q in paper.questions)
    assert paper.questions[1].stem_numerals == numerals('20 km 3 hours day 2')


def test_clean_paper_has_no_issues():
    assert verify(english_paper())['summary']['errors'] == 0
    assert verify(english_paper())['summary']['warnings'] == 0


def test_numbering_gap_and_repeat():
    assert 'NUMBER_GAP' in codes(verify(english_paper(numbers=[1, 2, 4, 5]))['issues'])
    assert 'NUMBER_REPEATED' in codes(verify(english_paper(numbers=[1, 2, 2, 3]))['issues'])


def test_duplicate_question_is_found():
    assert 'DUPLICATE_QUESTION' in codes(verify(english_paper(dup=3))['issues'])


def test_option_count_and_total():
    assert 'OPTION_COUNT' in codes(verify(english_paper(opts_override={3: 3}))['issues'])
    assert 'MARKS_TOTAL' in codes(verify(english_paper(total=25))['issues'])


# ------------------------------------------------------------------------------------------- parity
def test_parity_clean_pair():
    report = verify(hindi_paper(), source_pages=english_paper())
    assert report['summary']['errors'] == 0, report['issues']


def test_parity_finds_changed_number_missing_question_and_option():
    assert 'NUMBER_MISMATCH' in codes(verify(hindi_paper(changes={2: 200}), source_pages=english_paper())['issues'])
    assert 'MISSING_IN_TRANSLATION' in codes(verify(hindi_paper(drop=4), source_pages=english_paper())['issues'])
    assert 'OPTION_COUNT_MISMATCH' in codes(verify(hindi_paper(options=3), source_pages=english_paper())['issues'])


def test_alignment_without_numbers_uses_content():
    src, tgt = parse_paper(english_paper()), parse_paper(hindi_paper(drop=3))
    for q in src.questions + tgt.questions:
        q.number = None
    pairs = align(src.questions, tgt.questions)
    assert (2, None) in pairs  # source question 3 (index 2) has no partner
    issues = compare_versions(src, tgt)
    assert any(i.code == 'MISSING_IN_TRANSLATION' and i.question == 3 for i in issues)


def test_bilingual_single_document():
    eng, hin = english_paper()[0]['regions'], hindi_paper()[0]['regions']
    merged = [{**e, 'text': e['text'] + '\n' + h['text']} for e, h in zip(eng, hin)]
    streams = split_bilingual([page(merged)], {'Latin': 'english', 'Devanagari': 'hindi'})
    assert len(parse_paper(streams['hindi']).questions) == 5
    report = verify([page(merged)], split_scripts={'Latin': 'english', 'Devanagari': 'hindi'})
    assert report['summary']['errors'] == 0, report['issues']


def test_repeats_from_question_bank():
    bank = [{'text': 'A car travels 30 km in 4 hours. Find the average speed of the car on day 3.',
             'source': 'Paper 2025, Q3'}]
    issues = find_repeats(parse_paper(english_paper()), bank)
    assert [i.question for i in issues] == [3]


# ------------------------------------------------------------------------------------------- format
def test_profiles_and_format_check():
    assert {'question_paper', 'office_memorandum'} <= set(list_profiles())
    om = [page([region('header', 'No. A-12034/1/2026-Estt.\nGovernment of India\nMinistry of Education', 10),
                region('paragraph', 'New Delhi, the 12th March, 2026', 60),
                region('section-title', 'OFFICE MEMORANDUM', 90),
                region('paragraph', 'Subject: Holiday on 15 August.', 120),
                region('paragraph', 'The undersigned is directed to say that offices will remain closed.', 160),
                region('paragraph', '(A. Kumar)\nUnder Secretary to the Government of India', 300)])]
    issues = check_format(om, 'office_memorandum')
    assert 'FORMAT_MISSING' not in codes(issues)
    assert 'BILINGUAL' in codes(issues)  # English only: Official Languages Act s.3(3) asks for Hindi too
    om[0]['regions'].pop(3)  # remove the subject line
    assert any(i.code == 'FORMAT_MISSING' and i.details['element'] == 'subject' for i in check_format(om, 'office_memorandum'))


def test_question_paper_profile_and_report_html():
    report = verify(english_paper(numbers=[1, 2, 4, 5]), profile='question_paper')
    assert 'NUMBER_GAP' in codes(report['issues'])
    html = to_html(report)
    assert 'Do not print' in html and 'number gap' in html


def test_number_merged_with_stem_number_is_split():
    regs = [region('first-level-question', '4. A car travels 40 km in 5 hours. Find the speed on day 4.', 100),
            region('first-level-question', '5.16 V battery across 8 ohm. Find the current.', 160)]
    paper = parse_paper([page(regs)])
    assert [q.number for q in paper.questions] == [4, 5]
    assert paper.questions[1].stem_numerals == numerals('16 V 8 ohm')


def test_markers_glued_to_values_are_removed():
    from ml.verify.paper import strip_marker
    for raw, value in [('(3)12 m/s', '12 m/s'), ('2)8', '8'), ('(128', '28'), ('©2A', '2A'), ('M4A', '4A'),
                       ('(4) 18', '18'), ('226', '226')]:
        assert ' '.join(strip_marker(raw.split())) == value, raw


def test_ocr_tolerant_marks_and_numbers():
    from ml.verify.paper import fix_option_lookalikes
    from ml.verify.text import is_instruction, marks_schedule
    assert fix_option_lookalikes(['7A', '2A', 'SA', '8A']) == ['7A', '2A', '5A', '8A']
    assert fix_option_lookalikes(['Jupiter', 'Venus', 'Saturn', 'Mars']) == ['Jupiter', 'Venus', 'Saturn', 'Mars']
    assert numerals('A24V battery of 12 ohm') == numerals('A 24 V battery of 12 ohm')
    assert numerals('तर्क (2) : 180 अंश') == numerals('Reason (R) : 180 degrees')
    assert marks_in('What is the cost of the pens? (1]') == 1
    per_q, _ = marks_schedule('खण्ड - क में प्रश्न संख्या 1 । से 18 तक बहुविकल्पीय प्रश्न 1 अंक के हैं।')
    assert per_q.get(1) == 1 and per_q.get(18) == 1
    # 'this section' is the heading just above, not one several sentences back
    _, per_section = marks_schedule('SECTION - A\nThis section has 20 MCQs of 1 mark each.\nQ1. a\nQ2. b\nQ3. c\n'
                                    'Q4. d\nइस खण्ड में 6 प्रश्न हैं, जिनमें प्रत्येक के 3 अंक हैं।')
    assert per_section == {'A': 1}
    assert is_instruction('3a खण्ड में 3 प्रकरण अध्ययन आधारित प्रश्न हैं, जिनमें प्रत्येक के 4 अंक हैं।')


# ---------------------------------------------------------------- real CBSE 2025 paper structures
from ml.verify.checks import check_paper  # noqa: E402
from ml.verify.reader import text_layer_usable  # noqa: E402
from ml.verify.text import choice_group, is_instruction, marks_schedule  # noqa: E402


def parts_question(n_parts=6, opts=4, no_options=(5,)):
    """'1. Answer any 4 out of the given 6 questions ... 4×1=4' with parts (i)-(vi), each with its own (A)-(D)."""
    regs = [region('paragraph', 'Maximum Marks : 4', 10),
            region('first-level-question', '1. Answer any 4 out of the given 6 questions. 4×1=4', 50)]
    y = 100
    for k in range(1, n_parts + 1):
        regs.append(region('second-level-question', f'({"i" * k if k < 4 else ["iv", "v", "vi"][k - 4]}) '
                                                    f'Which value is correct for item {k}?', y)); y += 40
        if k not in no_options:
            regs.append(region('options', ' '.join(f'({m}) value {k}{m}' for m in 'ABCD'[:opts]), y)); y += 40
    return [page(regs, 'Latin')]


def test_choice_group_line_is_read():
    line = 'Answer any 4 out of the given 6 questions in 20-30 words each. 4 × 2 = 8'
    assert choice_group(line) == (4, 6, 2.0, 8.0)
    assert is_instruction(line)
    assert choice_group('दिए गए 6 प्रश्नों में से किन्हीं 4 प्रश्नों के उत्तर दीजिए। 4 × 2 = 8') == (4, 6, 2.0, 8.0)
    assert choice_group('1. Answer any 4 out of 6 questions. 4×1=4') == (4, 6, 1.0, 4.0)
    assert not is_instruction('1. Answer any 4 out of 6 questions. 4×1=4')   # numbered: it is a question
    assert choice_group('If 3 x 4 = 13, find the error.') is None            # a wrong product is not a group


def test_parts_with_their_own_options_are_checked_per_part():
    paper = parse_paper(parts_question())
    q = paper.questions[0]
    assert q.option_groups == [4, 4, 4, 4, 4]
    assert q.marks == 4 and q.answer_any == 4
    codes_found = codes(check_paper(paper))
    assert 'OPTION_COUNT' not in codes_found and 'OPTION_REPEATED' not in codes_found
    assert 'MARKS_TOTAL' not in codes_found and 'SUBMARKS_TOTAL' not in codes_found


def test_part_with_wrong_option_count_is_still_found():
    regs = parts_question()[0]['regions']
    regs[-1] = region('options', '(A) x (B) y (C) z', regs[-1]['box'][1])   # part (vi) lost option (D)
    issues = check_paper(parse_paper([page(regs, 'Latin')]))
    assert 'OPTION_COUNT' in codes(issues)


def test_unnumbered_choice_group_sets_marks_and_total():
    regs = [region('paragraph', 'Maximum Marks : 10', 10),
            region('first-level-question', '10. Define a data set. [2]', 50),
            region('paragraph', 'Answer any 4 out of the given 6 questions in 20-30 words each. 4 × 2 = 8', 90)]
    y = 130
    for n in range(11, 17):
        regs.append(region('first-level-question', f'{n}. Explain term number {n}.', y)); y += 40
    paper = parse_paper([page(regs, 'Latin')])
    assert [q.number for q in paper.questions] == list(range(10, 17))   # the group line is not glued to Q10
    assert all(q.marks == 2 for q in paper.questions[1:])
    assert 'MARKS_TOTAL' not in codes(check_paper(paper))               # 2 + 8 = 10, not 2 + 6 × 2


def test_section_total_is_not_each_questions_marks():
    _, per_section = marks_schedule('SECTION A\nSection A : Objective Type Questions (24 marks)')
    assert per_section == {}
    _, per_section = marks_schedule('SECTION B\nThis section comprises 5 questions of 2 marks each.')
    assert per_section == {'B': 2.0}


def test_private_use_text_layer_goes_to_ocr():
    words = [{'text': ''} for _ in range(30)]
    assert not text_layer_usable(words, None)
    assert text_layer_usable([{'text': 'question'} for _ in range(30)], 'Latin')


def test_reader_comparison_rows_for_the_ui():
    from ml.verify import _comparison
    from ml.verify.consensus import fuse
    pa, pb = parse_paper(english_paper()), parse_paper(english_paper())
    _, records = fuse(pa, pb)
    rows = _comparison({'a': {'paper': pa}, 'b': {'paper': pb}}, {'paper': records})
    assert len(rows) == 5 and all(r['outcome'] == 'both agree' for r in rows)
    assert rows[0]['reader_a'] == 'Q1 · 4 options · 4 marks' == rows[0]['reader_b']


def test_page_check_survives_mismatched_page_lists():
    paper = parse_paper(english_paper())
    paper.page_numbers, paper.page_indices = [1, 2, 3], [0]   # reader B's page numbers, reader A's sheets
    check_paper(paper)   # must not raise IndexError


def test_two_ranges_in_one_sentence_get_their_own_marks():
    per_q, _ = marks_schedule('खण्ड ग में प्रश्न संख्या 19 से 25 प्रत्येक 2 अंकों के तथा प्रश्न संख्या 26 से 29 प्रत्येक 3 अंकों के हैं ।')
    assert per_q[19] == 2 and per_q[25] == 2 and per_q[26] == 3 and per_q[29] == 3
    per_q, _ = marks_schedule('Section C has questions no. 19 to 25 of 2 marks each and questions no. 26 to 29 of 3 marks each.')
    assert per_q[25] == 2 and per_q[26] == 3


def test_brackets_inside_options_are_not_options():
    opts = split_options('(A) Both (A) and (R) are true (B) Both are false (C) (A) is true, (R) is false '
                         '(D) (A) is false, (R) is true', options_box=True)
    assert [m for m, _ in opts] == ['A', 'B', 'C', 'D']
    opts = split_options('(A) (i), (ii) and (iii) (B) (ii) and (iv) (C) (i) and (iv) (D) (iii) only', options_box=True)
    assert len(opts) == 4
