"""
Make half of the benchmark papers look scanned: slight blur, sensor noise and JPEG compression (the same kinds of
damage as the robustness study, at milder strength). The boxes stay valid because nothing moves.
Usage: python scripts/verify_bench/degrade.py data/verify_bench
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

root = Path(sys.argv[1] if len(sys.argv) > 1 else 'data/verify_bench')
cases = json.loads((root / 'cases.json').read_text())
rng = np.random.default_rng(7)
n = 0
for case in cases:
    if not case['degraded']:
        continue
    for f in sorted((root / 'pages').glob(f"{case['id']}_*.png")):
        img = Image.open(f).convert('L').filter(ImageFilter.GaussianBlur(0.7))
        arr = np.asarray(img, dtype=np.float32) + rng.normal(0, 8, (img.height, img.width))
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=55)
        Image.open(buf).convert('RGB').save(f)
        n += 1
print(f'degraded {n} pages')
