"""
The second, independent reading of a file for dual-reader verification (reader B), shared by the app and the API.

Reader A is the layout model + OCR (ml.ocr.extract_document); its pages are the input here. For every page, reader B
uses the PDF's own text layer when the file is born-digital and the layer is trustworthy (enough words, in the
script the page shows); otherwise it reads the page image again with another OCR configuration (SECOND_OCR, English
number pass on Indic pages), so that the two readings do not share their errors. Either way it is
parsed without the layout model (ml.verify.reader).
"""
from typing import Dict, List, Optional

from ml.verify.reader import read_pages, text_layer_usable
from ml.verify.text import script_of

# Reader B's OCR configuration for pages without a usable text layer: another segmentation mode and resolution than
# reader A's (psm 3 at the rendered size); on the development set's scanned Hindi pages it read 95% of the numbers
# exactly, against 93% for psm 6 on a binarised page
SECOND_OCR = {'psm': 4, 'scale': 1.5}


def text_layer_words(pdf_bytes: bytes, dpi: int) -> List[List[Dict]]:
    """Words of every page of a PDF's text layer, in the pixel coordinates of pages rendered at `dpi`."""
    import pymupdf
    doc = pymupdf.open(stream=pdf_bytes, filetype='pdf')
    s = dpi / 72.0
    return [[{'text': w[4], 'box': [w[0] * s, w[1] * s, w[2] * s, w[3] * s]} for w in page.get_text('words')]
            for page in doc]


def _page_script(page: Dict) -> Optional[str]:
    return page.get('script') or script_of(' '.join(r.get('text', '') or '' for r in page.get('regions', [])))


def second_reading(images: List, pages_a: List[Dict], pdf_bytes: Optional[bytes] = None, dpi: int = 200,
                   progress=None) -> List[Dict]:
    """
    Reader B pages for a file: images are the rendered pages (RGB arrays), pages_a reader A's pages of the same
    file (their OCR language and script are reused). progress(k, n): optional callback per page.
    """
    from ml.ocr import merge_number_reading, read_words
    layer = text_layer_words(pdf_bytes, dpi) if pdf_bytes else [None] * len(images)
    inputs, sources = [], []
    for k, (img, pa) in enumerate(zip(images, pages_a)):
        words = layer[k] if k < len(layer) else None
        if words and text_layer_usable(words, _page_script(pa)):
            inputs.append({'words': words, 'size': (img.shape[1], img.shape[0])})
            sources.append('text-layer')
        else:
            lang = pa.get('languages') or 'eng'
            indic = '+'.join(c for c in lang.split('+') if c != 'eng')
            eng = read_words(img, 'eng', **SECOND_OCR)
            words = merge_number_reading(read_words(img, indic, **SECOND_OCR), eng) if indic else eng
            inputs.append({'words': words, 'size': (img.shape[1], img.shape[0])})
            sources.append('ocr')
        if progress:
            progress(k + 1, len(images))
    pages = read_pages(inputs)
    for k, (page, src) in enumerate(zip(pages, sources)):
        page['source'] = src
        page['sheet'] = k
        page['languages'] = pages_a[k].get('languages')
    return pages


def version_langs(pages_a: List[Dict]) -> Optional[str]:
    """The OCR language most pages of a file were read with (for the third reading of that file's questions)."""
    from collections import Counter
    langs = Counter(p.get('languages') for p in pages_a if p.get('languages'))
    return langs.most_common(1)[0][0] if langs else None
