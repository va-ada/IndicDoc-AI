"""
Accessible output for blind and low-vision readers: semantic HTML and EPUB3 from extract_document() results.

Screen readers (NVDA, JAWS, TalkBack) navigate by headings, lists and tables, and pick the speech voice from the
`lang` attribute, so both are set from the detected layout and script. Page furniture (headers, page numbers ...)
is kept but moved after the main text, and figures get a visible placeholder so the reader knows one is there.
"""
import html
import io
import uuid
import zipfile
from typing import Dict, List

from ml.ocr import _FURNITURE, _HEADINGS, _LISTS, _NO_TEXT, _list_items

# Tesseract language code -> BCP 47 language tag
_BCP47 = {'asm': 'as', 'ben': 'bn', 'eng': 'en', 'guj': 'gu', 'hin': 'hi', 'kan': 'kn', 'mal': 'ml', 'mar': 'mr',
          'ori': 'or', 'pan': 'pa', 'tam': 'ta', 'tel': 'te'}
_ORDERED = {'ordered-list', 'sub-ordered-list', 'subsub-ordered-list', 'options'}


def language_tag(languages: str) -> str:
    """'tam+eng' -> 'ta' (the first, main language)."""
    first = (languages or 'eng').split('+')[0]
    return _BCP47.get(first, 'en')


def _esc(text: str) -> str:
    return html.escape(' '.join(text.split()))


def regions_to_html_body(regions: List[Dict]) -> str:
    """Regions in reading order -> semantic HTML fragment."""
    body, furniture = [], []
    for r in regions:
        label, text = r['label'], r.get('text', '').strip()
        if label in _NO_TEXT:
            body.append(f'<figure><p>[{html.escape(label)}: no description yet]</p></figure>')
            continue
        if not text:
            continue
        if label in _FURNITURE:
            furniture.append(f'<li>{html.escape(label)}: {_esc(text)}</li>')
        elif label in _HEADINGS:
            level = min(len(_HEADINGS[label]), 6)
            body.append(f'<h{level}>{_esc(text)}</h{level}>')
        elif label in _LISTS:
            tag = 'ol' if label in _ORDERED else 'ul'
            items = ''.join(f'<li>{_esc(i)}</li>' for i in _list_items(text))
            body.append(f'<{tag}>{items}</{tag}>')
        elif label == 'table':
            rows = ''.join(f'<tr><td>{_esc(line)}</td></tr>' for line in text.splitlines() if line.strip())
            body.append(f'<table>{rows}</table>')
        elif label in ('figure-caption', 'table-caption'):
            body.append(f'<p><em>{_esc(text)}</em></p>')
        elif label == 'quote':
            body.append(f'<blockquote><p>{_esc(text)}</p></blockquote>')
        else:
            body.append(f'<p>{_esc(text)}</p>')
    out = '\n'.join(body)
    if furniture:
        out += '\n<aside aria-label="Page details"><ul>' + ''.join(furniture) + '</ul></aside>'
    return out


def to_html(doc: Dict, title: str = 'Document') -> str:
    """One page (extract_document result) as a standalone accessible HTML page."""
    lang = language_tag(doc.get('languages'))
    return (f'<!doctype html>\n<html lang="{lang}" xml:lang="{lang}"><head><meta charset="utf-8">'
            f'<title>{html.escape(title)}</title></head>\n<body><main>\n'
            f'{regions_to_html_body(doc["regions"])}\n</main></body></html>\n')


def to_epub(pages: List[Dict], title: str = 'Document') -> bytes:
    """Pages (extract_document results, in order) as an EPUB3 book: one chapter per page, with a table of contents."""
    lang = language_tag(pages[0].get('languages') if pages else 'eng')
    book_id = f'urn:uuid:{uuid.uuid4()}'
    chapters = []
    for i, doc in enumerate(pages, start=1):
        xhtml = (f'<?xml version="1.0" encoding="utf-8"?>\n<html xmlns="http://www.w3.org/1999/xhtml" '
                 f'xmlns:epub="http://www.idpf.org/2007/ops" lang="{language_tag(doc.get("languages"))}" '
                 f'xml:lang="{language_tag(doc.get("languages"))}"><head><title>Page {i}</title></head><body>'
                 f'<section epub:type="chapter"><span epub:type="pagebreak" id="page{i}" role="doc-pagebreak" '
                 f'aria-label="Page {i}"/>\n{regions_to_html_body(doc["regions"])}\n</section></body></html>')
        chapters.append((f'page{i}.xhtml', xhtml))
    nav = ('<?xml version="1.0" encoding="utf-8"?>\n<html xmlns="http://www.w3.org/1999/xhtml" '
           f'xmlns:epub="http://www.idpf.org/2007/ops" lang="{lang}"><head><title>Contents</title></head><body>'
           '<nav epub:type="toc"><ol>' +
           ''.join(f'<li><a href="{name}">Page {i}</a></li>' for i, (name, _) in enumerate(chapters, 1)) +
           '</ol></nav><nav epub:type="page-list"><ol>' +
           ''.join(f'<li><a href="{name}#page{i}">{i}</a></li>' for i, (name, _) in enumerate(chapters, 1)) +
           '</ol></nav></body></html>')
    manifest = ''.join(f'<item id="p{i}" href="{n}" media-type="application/xhtml+xml"/>'
                       for i, (n, _) in enumerate(chapters, 1))
    spine = ''.join(f'<itemref idref="p{i}"/>' for i in range(1, len(chapters) + 1))
    opf = ('<?xml version="1.0" encoding="utf-8"?>\n<package xmlns="http://www.idpf.org/2007/opf" version="3.0" '
           'unique-identifier="id"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
           f'<dc:identifier id="id">{book_id}</dc:identifier><dc:title>{html.escape(title)}</dc:title>'
           f'<dc:language>{lang}</dc:language>'
           '<meta property="dcterms:modified">2026-01-01T00:00:00Z</meta>'
           '<meta property="schema:accessMode">textual</meta>'
           '<meta property="schema:accessibilityFeature">structuralNavigation</meta>'
           '<meta property="schema:accessibilityFeature">printPageNumbers</meta>'
           '<meta property="schema:accessibilityHazard">none</meta>'
           '</metadata><manifest><item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>'
           f'{manifest}</manifest><spine>{spine}</spine></package>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        z.writestr(zipfile.ZipInfo('mimetype'), 'application/epub+zip')  # first and uncompressed, as EPUB requires
        z.writestr('META-INF/container.xml',
                   '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                   '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
                   '</rootfiles></container>', compress_type=zipfile.ZIP_DEFLATED)
        z.writestr('OEBPS/content.opf', opf, compress_type=zipfile.ZIP_DEFLATED)
        z.writestr('OEBPS/nav.xhtml', nav, compress_type=zipfile.ZIP_DEFLATED)
        for name, xhtml in chapters:
            z.writestr(f'OEBPS/{name}', xhtml, compress_type=zipfile.ZIP_DEFLATED)
    return buf.getvalue()
