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
