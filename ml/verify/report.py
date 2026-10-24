"""
The verification report a paper-setter or vetter reads: a verdict, the issues sorted by severity with the question
and page, and the pages with each problem outlined. Self-contained HTML (images embedded), safe to e-mail or print.
"""
import base64
import html
import io
from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw

SEVERITY_ORDER = {'error': 0, 'warning': 1, 'info': 2}
_COLOURS = {'error': (200, 30, 45), 'warning': (220, 140, 0), 'info': (60, 110, 200)}


def annotate(image: np.ndarray, numbered_issues: List) -> Image.Image:
    """Page image (RGB) with the box of every (number, issue) pair outlined in its severity colour and numbered."""
    img = Image.fromarray(image).convert('RGB')
    draw = ImageDraw.Draw(img)
    width = max(3, img.width // 300)
    for n, issue in numbered_issues:
        if not issue.get('box'):
            continue
        x1, y1, x2, y2 = issue['box']
        colour = _COLOURS[issue['severity']]
        draw.rectangle([x1 - 4, y1 - 4, x2 + 4, y2 + 4], outline=colour, width=width)
        tag = f' {n} '
        draw.rectangle([x1 - 4, y1 - 4 - 14 * width // 3, x1 - 4 + 9 * width * len(tag) // 3, y1 - 4], fill=colour)
        draw.text((x1 - 2, y1 - 4 - 13 * width // 3), tag, fill=(255, 255, 255))
    return img


def _png(img: Image.Image, max_width: int = 900) -> str:
    if img.width > max_width:
        img = img.resize((max_width, int(img.height * max_width / img.width)))
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def to_html(report: Dict, page_images: Optional[Dict[str, List[np.ndarray]]] = None, title: str = 'Verification report') -> str:
    """
    report: the dict returned by ml.verify.verify(). page_images: {'paper': [...], 'source': [...]} RGB arrays, to
    show where each issue is; issues of the translation ('target') are drawn on the 'paper' pages.
    """
    issues = sorted(report['issues'], key=lambda i: (SEVERITY_ORDER[i['severity']], i.get('question') or 0))
    numbered = list(enumerate(issues, 1))
    s = report['summary']
    verdict = ('Ready for printing: no problems found' if s['errors'] == 0 and s['warnings'] == 0 else
               'Check before printing' if s['errors'] == 0 else 'Do not print: errors found')
    colour = '#1B7F3B' if s['errors'] == 0 and s['warnings'] == 0 else '#B26A00' if s['errors'] == 0 else '#B3261E'
    rows = ''.join(
        f'<tr><td>{n}</td><td><span class="sev {i["severity"]}">{i["severity"]}</span></td>'
        f'<td>{html.escape(i["code"].replace("_", " ").lower())}</td><td>{i.get("question") or ""}</td>'
        f'<td>{"" if i.get("page") is None else i["page"] + 1}</td><td>{html.escape(i["version"])}</td>'
        f'<td>{html.escape(i["message"])}</td></tr>' for n, i in numbered)
    pages_html = ''
    for version, images in (page_images or {}).items():
        for p, image in enumerate(images):
            on_page = [(n, i) for n, i in numbered if i.get('page') == p and
                       (i['version'] == version or (version == 'paper' and i['version'] in ('paper', 'target')))]
            pages_html += (f'<figure><img src="{_png(annotate(image, on_page))}"><figcaption>{html.escape(version)} '
                           f'page {p + 1} &middot; {len(on_page)} issue(s) marked</figcaption></figure>')
    meta = ''.join(f'<span><b>{html.escape(k)}</b> {html.escape(str(v))}</span>' for k, v in report.get('meta', {}).items())
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body {{ font-family: Inter, -apple-system, 'Segoe UI', 'Noto Sans', sans-serif; color: #1E293B; margin: 28px; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }} .meta span {{ margin-right: 18px; font-size: 13px; color: #475569; }}
.verdict {{ margin: 16px 0; padding: 14px 18px; border-radius: 10px; color: #fff; background: {colour}; font-size: 18px; font-weight: 700; }}
.kpis {{ display: flex; gap: 12px; margin-bottom: 18px; }} .kpis div {{ flex: 1; border: 1px solid #E2E8F0; border-radius: 10px; padding: 10px 14px; }}
.kpis b {{ display: block; font-size: 24px; }} table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th {{ text-align: left; background: #14284B; color: #fff; padding: 7px; }} td {{ border-bottom: 1px solid #E2E8F0; padding: 6px 7px; vertical-align: top; }}
.sev {{ padding: 2px 8px; border-radius: 10px; color: #fff; font-size: 11px; text-transform: uppercase; font-weight: 700; }}
.sev.error {{ background: #B3261E; }} .sev.warning {{ background: #B26A00; }} .sev.info {{ background: #2F5FAF; }}
figure {{ margin: 18px 0; }} figure img {{ max-width: 100%; border: 1px solid #CBD5E1; }} figcaption {{ font-size: 12px; color: #475569; }}
</style></head><body>
<h1>{html.escape(title)}</h1><div class="meta">{meta}</div>
<div class="verdict">{verdict}</div>
<div class="kpis"><div><b>{s["questions"]}</b>questions read</div><div><b>{s["errors"]}</b>errors</div>
<div><b>{s["warnings"]}</b>warnings</div><div><b>{s["info"]}</b>notes</div></div>
<table><thead><tr><th>#</th><th>Severity</th><th>Check</th><th>Q</th><th>Page</th><th>Version</th><th>Finding</th></tr></thead>
<tbody>{rows or '<tr><td colspan="7">No issues.</td></tr>'}</tbody></table>
{pages_html}
<p style="font-size:12px;color:#64748B;margin-top:24px">IndicDoc Verify checks structure, numbers, marks and repeats. It does
not judge whether a question is correct or well translated in meaning; every finding is for a human vetter to confirm.</p>
</body></html>'''
