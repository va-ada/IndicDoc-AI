"""
IndicDoc AI - OCR tests: word-to-region assignment, reading order, Markdown, and real Tesseract runs.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

FONT = Path('/System/Library/Fonts/Supplemental/Arial.ttf')


def _word(text, left, top, width=40, height=12, block=1, par=1, line=1, conf=90.0):
    return {'text': text, 'left': left, 'top': top, 'width': width, 'height': height,
            'block_num': block, 'par_num': par, 'line_num': line, 'conf': conf}


class TestAssignWords:
    def test_words_go_to_the_region_that_contains_them_in_line_order(self):
        from ml.ocr import assign_words
        regions = [{'box': [0, 0, 200, 50], 'label': 'section-title', 'score': 0.9},
                   {'box': [0, 60, 200, 200], 'label': 'paragraph', 'score': 0.8}]
        words = [_word('Results', 10, 10),
                 _word('world', 60, 100, line=2), _word('Hello', 10, 100, line=2),   # same line, given out of order
                 _word('again', 10, 130, line=3),
                 _word('stray', 300, 300)]                                          # outside every region
        out = assign_words(regions, words)
        assert out[0]['text'] == 'Results'
        assert out[1]['text'] == 'Hello world\nagain'

    def test_low_confidence_and_empty_words_are_dropped(self):
        from ml.ocr import assign_words
        regions = [{'box': [0, 0, 200, 50], 'label': 'paragraph', 'score': 0.9}]
        out = assign_words(regions, [_word('good', 5, 5), _word('', 50, 5), _word('noise', 100, 5, conf=-1)])
        assert out[0]['text'] == 'good'

    def test_word_counted_once_in_smallest_overlapping_region(self):
        """Nested boxes (a list inside a sidebar): the word belongs to the innermost one."""
        from ml.ocr import assign_words
        regions = [{'box': [0, 0, 400, 400], 'label': 'sidebar', 'score': 0.9},
                   {'box': [10, 10, 200, 60], 'label': 'unordered-list', 'score': 0.9}]
        out = assign_words(regions, [_word('item', 20, 20)])
        assert out[0]['text'] == '' and out[1]['text'] == 'item'


class TestReadingOrderAndMarkdown:
    def test_top_to_bottom_then_left_to_right(self):
        from ml.ocr import reading_order
        regions = [{'box': [300, 100, 500, 150]}, {'box': [0, 100, 250, 150]}, {'box': [0, 0, 500, 50]}]
        assert [r['box'][0:2] for r in reading_order(regions)] == [[0, 0], [0, 100], [300, 100]]

    def test_markdown_formats_by_label(self):
        from ml.ocr import to_markdown
        regions = [{'label': 'headline', 'text': 'Big News'}, {'label': 'sub-headline', 'text': 'Details'},
                   {'label': 'paragraph', 'text': 'Body text.'}, {'label': 'unordered-list', 'text': 'one\ntwo'},
                   {'label': 'figure', 'text': ''}, {'label': 'page-number', 'text': '7'}]
        md = to_markdown(regions)
        assert '# Big News' in md and '## Details' in md and 'Body text.' in md
        assert '- one\n- two' in md and '[figure]' in md and '7' not in md.split('---')[0]


class TestScriptDetection:
    @pytest.mark.parametrize('script,expected', [('Devanagari', 'hin+mar+eng'), ('Tamil', 'tam+eng'),
                                                 ('Latin', 'eng'), ('Klingon', 'eng')])
    def test_script_maps_to_tesseract_languages(self, script, expected):
        from ml.ocr import languages_for_script
        assert languages_for_script(script) == expected


needs_tesseract = pytest.mark.skipif(shutil.which('tesseract') is None, reason='tesseract not installed')


@needs_tesseract
class TestTesseract:
    @staticmethod
    def _page():
        from PIL import Image, ImageDraw, ImageFont
        if not FONT.exists():
            pytest.skip('no system font to render a test page')
        img = Image.new('RGB', (1000, 400), 'white')
        d = ImageDraw.Draw(img)
        d.text((40, 30), 'Quarterly Report', font=ImageFont.truetype(str(FONT), 48), fill='black')
        d.text((40, 150), 'Revenue grew strongly in every region.', font=ImageFont.truetype(str(FONT), 32), fill='black')
        return np.array(img)

    def test_extract_document_reads_text_into_detected_regions(self):
        from ml.ocr import TESSDATA_DIR, extract_document
        if not (TESSDATA_DIR / 'eng.traineddata').exists():
            pytest.skip('OCR models not downloaded (python scripts/download_models.py)')
        detections = {'boxes': np.array([[30, 20, 600, 95], [30, 140, 900, 200]]),
                      'labels': np.array([26, 21]), 'scores': np.array([0.9, 0.8])}  # section-title, paragraph
        doc = extract_document(self._page(), detections, languages='eng')
        assert doc['regions'][0]['label'] == 'section-title'
        assert 'Quarterly Report' in doc['regions'][0]['text']
        assert 'Revenue grew strongly' in doc['regions'][1]['text']
        assert '## Quarterly Report' in doc['markdown'] or '# Quarterly Report' in doc['markdown']

    def test_detect_script_by_probe_on_english_page(self):
        from ml.ocr import TESSDATA_DIR, detect_script_by_probe
        if not (TESSDATA_DIR / 'hin.traineddata').exists():
            pytest.skip('OCR models not downloaded (python scripts/download_models.py)')
        detections = {'boxes': np.array([[30, 140, 900, 200]]), 'labels': np.array([21])}  # the paragraph line
        assert detect_script_by_probe(self._page(), detections) == 'Latin'

    def test_formula_and_figure_regions_are_not_ocr_text(self):
        from ml.ocr import TESSDATA_DIR, extract_document
        if not (TESSDATA_DIR / 'eng.traineddata').exists():
            pytest.skip('OCR models not downloaded (python scripts/download_models.py)')
        detections = {'boxes': np.array([[30, 140, 900, 200]]), 'labels': np.array([13]), 'scores': np.array([0.9])}  # formula
        doc = extract_document(self._page(), detections, languages='eng')
        formula = next(r for r in doc['regions'] if r['label'] == 'formula')
        assert formula['text'] == '' and '[formula]' in doc['markdown']
        # the title outside every detected box is still read, as unlabelled text
        assert any(r['label'] == 'unlabelled-text' and 'Quarterly' in r['text'] for r in doc['regions'])


class TestLeftoverText:
    def test_words_outside_every_box_become_unlabelled_regions(self):
        from ml.ocr import leftover_regions
        regions = [{'box': [0, 0, 200, 50], 'label': 'paragraph', 'score': 0.9}]
        words = [_word('inside', 10, 10),
                 _word('Missed', 10, 300, block=2, par=1, line=1), _word('heading', 60, 300, block=2, par=1, line=1),
                 _word('Another', 10, 500, block=3, par=1, line=1)]
        extra = leftover_regions(regions, words)
        assert [r['text'] for r in extra] == ['Missed heading', 'Another']
        assert all(r['label'] == 'unlabelled-text' and r['detected'] is False for r in extra)
        assert extra[0]['box'] == [10, 300, 100, 312]


class TestBullets:
    @pytest.mark.parametrize('line', ['« Built a model', '• Built a model', '● Built a model', '- Built a model', '▪Built a model'])
    def test_ocr_bullet_glyphs_are_replaced_by_markdown_bullets(self, line):
        from ml.ocr import to_markdown
        assert to_markdown([{'label': 'unordered-list', 'text': line}]).strip() == '- Built a model'

    def test_numbered_items_keep_their_numbers(self):
        from ml.ocr import to_markdown
        assert to_markdown([{'label': 'ordered-list', 'text': '1. First\n2. Second'}]).strip() == '- 1. First\n- 2. Second'


class TestListItems:
    def test_wrapped_lines_join_their_item(self):
        from ml.ocr import to_markdown
        text = '+ If the system came equipped with an internal\nembedded wireless device, do not\n+ Wireless devices are not user serviceable.'
        assert to_markdown([{'label': 'unordered-list', 'text': text}]).strip() == (
            '- If the system came equipped with an internal embedded wireless device, do not\n'
            '- Wireless devices are not user serviceable.')

    def test_numbered_items_split_on_numbers(self):
        from ml.ocr import to_markdown
        text = '1. First item that\nwraps here\n2. Second\n(a) third'
        assert to_markdown([{'label': 'ordered-list', 'text': text}]).strip() == (
            '- 1. First item that wraps here\n- 2. Second\n- (a) third')

    def test_lines_without_markers_stay_one_item_each(self):
        from ml.ocr import to_markdown
        assert to_markdown([{'label': 'unordered-list', 'text': 'alpha\nbeta'}]).strip() == '- alpha\n- beta'


def _w(text, left, top=10, width=40, height=20):
    return {'text': text, 'conf': 90.0, 'left': left, 'top': top, 'width': width, 'height': height,
            'block_num': 1, 'par_num': 1, 'line_num': 1}


class TestNumberPass:
    """merge_number_reading: the English model's digits fix an Indic reading."""

    def test_fixes_misread_latin_digits(self):
        from ml.ocr import merge_number_reading
        hindi = [_w('भारत', 0), _w('947', 50), _w('में', 100)]
        english = [_w('wea', 0), _w('1947', 50), _w('#', 100)]
        assert [w['text'] for w in merge_number_reading(hindi, english)] == ['भारत', '1947', 'में']

    def test_inserts_missing_and_drops_unconfirmed_numbers(self):
        from ml.ocr import merge_number_reading
        hindi = [_w('प्रत्येक', 0), _w('प्रश्न', 50), _w('6', 150)]  # '1' missed, a letter read as '6'
        english = [_w('wre', 0), _w('war', 50), _w('1', 100), _w('Fl', 150)]
        texts = [w['text'] for w in merge_number_reading(hindi, english)]
        assert '1' in texts and '6' not in texts

    def test_keeps_markers(self):
        from ml.ocr import merge_number_reading
        tamil = [_w('(2)', 0), _w('(m2', 50), _w('40', 100)]
        english = [_w('2)', 0), _w('12', 50), _w('40', 100)]
        assert [w['text'] for w in merge_number_reading(tamil, english)] == ['(2)', '(?) 12', '40']

    def test_page_printed_in_indian_digits_keeps_them(self):
        from ml.ocr import merge_number_reading
        hindi = [_w('प्रश्न', 0), _w('१२०', 50), _w('४५', 100), _w('(क)', 150)]
        english = [_w('Wee', 0), _w('R20', 50), _w('84', 100), _w('(a)', 150)]
        assert [w['text'] for w in merge_number_reading(hindi, english)] == ['प्रश्न', '१२०', '४५', '(क)']

    def test_latin_digit_page_corrects_indian_digit_misreads(self):
        from ml.ocr import merge_number_reading
        hindi = [_w('पृष्ठ', 0), _w('120', 50), _w('2८0', 100), _w('45', 150)]  # '2A' read as '2८0'
        english = [_w('wa', 0), _w('120', 50), _w('2A', 100), _w('45', 150)]
        assert [w['text'] for w in merge_number_reading(hindi, english)] == ['पृष्ठ', '120', '2A', '45']

    def test_confident_digit_stands_against_an_unsure_english_reading(self):
        from ml.ocr import merge_number_reading
        hindi = [_w('को', 0), {**_w('9', 50), 'conf': 94.0}, _w('सेकंड', 100)]
        english = [_w('at', 0), {**_w('O', 50), 'conf': 77.0}, _w('aes', 100)]
        assert [w['text'] for w in merge_number_reading(hindi, english)] == ['को', '9', 'सेकंड']

    def test_option_read_as_one_english_token(self):
        from ml.ocr import merge_number_reading
        hindi = [_w('(0)', 0, width=40), _w('28', 77, width=23)]      # '(A) 1 A': marker and value both garbled
        english = [{**_w('(A)1A', 0, width=100), 'conf': 29.0}]
        assert [w['text'] for w in merge_number_reading(hindi, english)] == ['(A) 1A']
        hindi = [_w('(०)52', 0, width=97)]
        english = [_w('(C)SA', 0, width=97)]                           # 'S' for '5', confirmed by the Hindi digits
        assert [w['text'] for w in merge_number_reading(hindi, english)] == ['(C) 5A']

    def test_unit_read_as_a_digit_is_dropped(self):
        from ml.ocr import merge_number_reading
        hindi = [_w('24', 0, width=34), _w('५', 43, width=21), _w('की', 72)]   # '24 V की'
        english = [_w('24V', 2, width=60), _w('ht', 82)]
        assert [w['text'] for w in merge_number_reading(hindi, english)] == ['24', 'की']
