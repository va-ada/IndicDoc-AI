import io
import zipfile
from xml.dom import minidom

from ml.accessible import language_tag, to_epub, to_html

DOC = {'languages': 'tam+eng', 'regions': [
    {'label': 'chapter-title', 'text': 'பாடம் 1'},
    {'label': 'paragraph', 'text': 'ஒரு தாய் <மக்கள்>'},
    {'label': 'ordered-list', 'text': '1. one\n2. two'},
    {'label': 'figure', 'text': ''},
    {'label': 'page-number', 'text': '12'},
]}


def test_language_tag():
    assert language_tag('tam+eng') == 'ta'
    assert language_tag('ori') == 'or'
    assert language_tag(None) == 'en'


def test_html_is_semantic_and_escaped():
    out = to_html(DOC, 'Lesson')
    assert '<html lang="ta"' in out
    assert '<h1>பாடம் 1</h1>' in out
    assert '&lt;மக்கள்&gt;' in out
    assert '<ol><li>' in out and '[figure: no description yet]' in out
    # page furniture comes after the main text
    assert out.index('page-number') > out.index('ஒரு தாய்')


def test_epub_structure():
    data = to_epub([DOC, DOC], 'Lesson')
    z = zipfile.ZipFile(io.BytesIO(data))
    names = z.namelist()
    assert names[0] == 'mimetype' and z.getinfo('mimetype').compress_type == zipfile.ZIP_STORED
    assert z.read('mimetype') == b'application/epub+zip'
    for n in ('OEBPS/content.opf', 'OEBPS/nav.xhtml', 'OEBPS/page1.xhtml', 'OEBPS/page2.xhtml'):
        minidom.parseString(z.read(n))  # every XML file is well-formed
    assert b'<dc:language>ta</dc:language>' in z.read('OEBPS/content.opf')
