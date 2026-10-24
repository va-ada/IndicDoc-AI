import json
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image
from starlette.concurrency import run_in_threadpool

# Allow importing the ml package from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from ml.config import AUTHORS_CHECKPOINT, BASELINE_CHECKPOINT, CLASS_NAMES, METRICS_DIR, PROPOSED_CHECKPOINT, get_device
from ml.inference.predict import dedupe_boxes, demo_inference, load_model, predict_single
from ml.preprocessing.preprocess import decode_image, pdf_page_count, render_pdf_page
from ml.ocr import OCR_LANGUAGES, extract_document, tesseract_available
from ml.verify import list_profiles, verify
from ml.verify.report import to_html as verify_report_html
from ml.inference.predict import disable_runtime_installs

disable_runtime_installs()  # serving must never pip-install packages (see the function's docstring)

CHECKPOINTS = {'doclayout': AUTHORS_CHECKPOINT, 'yolov8': PROPOSED_CHECKPOINT, 'fasterrcnn': BASELINE_CHECKPOINT}
# Saved evaluation results of each model, written by scripts/evaluate.py
METRICS_NAMES = {'doclayout': 'authors_metrics', 'yolov8': 'proposed_metrics', 'fasterrcnn': 'baseline_metrics'}

# Deployment settings, overridable through environment variables
MAX_UPLOAD_BYTES = int(float(os.environ.get('INDICDOC_MAX_UPLOAD_MB', '20')) * 1024 * 1024)
MAX_PIXELS = int(float(os.environ.get('INDICDOC_MAX_MEGAPIXELS', '40')) * 1_000_000)
# One lock per model: Ultralytics predictors are not safe to share between threads
MODEL_LOCKS = {name: threading.Lock() for name in CHECKPOINTS}
OCR_CODES = set(OCR_LANGUAGES.values())
MAX_VERIFY_PAGES = int(os.environ.get('INDICDOC_MAX_VERIFY_PAGES', '60'))
# A whole scanned paper is much larger than one page, so /verify has its own upload limit
MAX_VERIFY_BYTES = int(float(os.environ.get('INDICDOC_MAX_VERIFY_MB', '100')) * 1024 * 1024)
VERIFY_DPI = 200  # every digit must be read exactly, so papers are rendered finer than for layout analysis
SCRIPTS = {'Latin', 'Devanagari', 'Bengali', 'Gurmukhi', 'Gujarati', 'Oriya', 'Tamil', 'Telugu', 'Kannada', 'Malayalam'}
CORS_ORIGINS = [o.strip() for o in os.environ.get('INDICDOC_CORS_ORIGINS', 'http://localhost:5173').split(',') if o.strip()]



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the available models before serving, so the first request does not wait ~20 s for a 166 MB checkpoint
    for model_type, path in CHECKPOINTS.items():
        if path.exists():
            await run_in_threadpool(get_model, model_type)
    yield


app = FastAPI(title="IndicDoc AI API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,  # set INDICDOC_CORS_ORIGINS to the sites allowed to call this API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=2)
def get_model(model_type: str):
    """Load each model once per process instead of on every request."""
    device = get_device()
    model = load_model(str(CHECKPOINTS[model_type]), model_type, device)
    disable_runtime_installs()  # again, now that ultralytics may have been imported
    return model, device


@app.get("/")
def read_root():
    return {"message": "Welcome to IndicDoc AI API"}


@app.post("/analyze")
async def analyze_document(
    file: UploadFile = File(...),
    model_type: Literal['doclayout', 'yolov8', 'fasterrcnn'] = 'doclayout',
    conf_threshold: Optional[float] = Query(None, ge=0.0, le=1.0,
                                            description="Defaults to the model's recommended threshold"),
    mode: Literal['balanced', 'all'] = Query('balanced', description="'all' = the model's high-recall threshold"),
    ocr: bool = Query(False, description="Also read the text of every detected region (Tesseract)"),
    ocr_lang: str = Query('auto', pattern=r'^(auto|[a-z]{3}(\+[a-z]{3})*)$',
                          description="'auto' or Tesseract codes joined by '+', e.g. 'hin+eng'"),
):
    # Read at most one byte past the limit, so an oversized upload is never loaded in full
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    try:
        image = decode_image(contents, max_pixels=MAX_PIXELS)
    except ValueError as e:
        too_large = 'too large' in str(e)
        raise HTTPException(status_code=413 if too_large else 400, detail=str(e) if too_large else "Invalid image file")
    if ocr:
        if ocr_lang != 'auto' and not set(ocr_lang.split('+')) <= OCR_CODES:
            raise HTTPException(status_code=400, detail=f"ocr_lang must be 'auto' or codes from {sorted(OCR_CODES)}")
        if not tesseract_available():
            raise HTTPException(status_code=503, detail="OCR is not available on this server")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if conf_threshold is None:
        conf_threshold = recommended_conf(model_type, mode)

    if CHECKPOINTS[model_type].exists():
        def infer():
            with MODEL_LOCKS[model_type]:
                model, device = get_model(model_type)
                start = time.time()
                # PIL keeps channel order unambiguous for both torchvision and ultralytics
                out = predict_single(model, Image.fromarray(image_rgb), model_type, device, conf=conf_threshold)
                out['inference_time'] = time.time() - start
                return out

        # Run in a worker thread so a slow page does not block other requests (health checks, metrics)
        results = await run_in_threadpool(infer)
        keep = results['scores'] >= conf_threshold
        for key in ('boxes', 'labels', 'scores'):
            results[key] = results[key][keep]
        results = dedupe_boxes(results)  # one box per page element, not one per candidate label
        results['is_demo'] = False
    else:
        # No trained checkpoint yet: synthetic boxes, flagged so clients can label them
        results = demo_inference(image_rgb)

    document = None
    if ocr:
        # Tesseract is a separate process; run it off the event loop like inference
        document = await run_in_threadpool(extract_document, image_rgb, results, ocr_lang)

    labels = [int(label) for label in results["labels"]]
    return {
        "status": "success",
        "results": {
            "boxes": results["boxes"].tolist(),
            "labels": labels,
            "class_names": [CLASS_NAMES[label] for label in labels],
            "scores": results["scores"].tolist(),
            "inference_time": results["inference_time"],
            "is_demo": results["is_demo"],
            "model_type": model_type,
            "mode": mode,
            "conf_threshold": conf_threshold,
            "document": document,
            "image_width": image.shape[1],
            "image_height": image.shape[0],
        },
    }


def _read_metrics(name: str) -> Optional[dict]:
    path = METRICS_DIR / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def recommended_conf(model_type: str, mode: str = 'balanced') -> float:
    """Threshold chosen on the validation pages by scripts/evaluate.py (best F1, or high recall for 'all'), else 0.5."""
    metrics = _read_metrics(METRICS_NAMES[model_type]) or {}
    balanced = metrics.get('recommended_conf', 0.5)
    return metrics.get('high_recall_conf', balanced) if mode == 'all' else balanced


@app.get("/metrics")
def get_metrics():
    # Saved evaluation results; null until a model has been evaluated
    return {
        "authors": _read_metrics("authors_metrics"),
        "baseline": _read_metrics("baseline_metrics"),
        "proposed": _read_metrics("proposed_metrics"),
    }


# ---------------------------------------------------------------------------------------------- verification
async def _read_upload(file: UploadFile, limit: int = MAX_VERIFY_BYTES) -> bytes:
    contents = await file.read(limit + 1)
    if len(contents) > limit:
        raise HTTPException(status_code=413, detail=f"File larger than {limit // (1024 * 1024)} MB")
    return contents


def _pages_rgb(contents: bytes) -> list:
    """Every page of an uploaded PDF (or the one image) as RGB arrays."""
    if contents[:5] == b'%PDF-':
        n = pdf_page_count(contents)
        if n > MAX_VERIFY_PAGES:
            raise HTTPException(status_code=413, detail=f"At most {MAX_VERIFY_PAGES} pages per paper")
        return [cv2.cvtColor(render_pdf_page(contents, i, dpi=VERIFY_DPI, max_pixels=MAX_PIXELS), cv2.COLOR_BGR2RGB)
                for i in range(n)]
    try:
        return [cv2.cvtColor(decode_image(contents, max_pixels=MAX_PIXELS), cv2.COLOR_BGR2RGB)]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _read_paper(images: list, lang: str) -> list:
    """Layout (DocLayout-YOLO, high-recall threshold) + OCR with the number pass, page by page."""
    conf = recommended_conf('doclayout', 'all')
    pages = []
    for image_rgb in images:
        with MODEL_LOCKS['doclayout']:
            model, device = get_model('doclayout')
            out = predict_single(model, Image.fromarray(image_rgb), 'doclayout', device, conf=conf)
        keep = out['scores'] >= conf
        out = dedupe_boxes({**out, 'boxes': out['boxes'][keep], 'labels': out['labels'][keep],
                            'scores': out['scores'][keep]})
        pages.append(extract_document(image_rgb, out, lang, number_pass=True))
    return pages


@app.get("/verify/profiles")
def verify_profiles():
    """Format profiles available to POST /verify (ml/verify/profiles/*.json)."""
    return list_profiles()


@app.post("/verify")
async def verify_paper(
    file: UploadFile = File(..., description="The paper to check (PDF or image)"),
    source: Optional[UploadFile] = File(None, description="The source-language version, for translation parity"),
    bank: Optional[UploadFile] = File(None, description="Earlier papers' questions: JSON [{'text', 'source'}]"),
    lang: str = Query('auto', pattern=r'^(auto|[a-z]{3}(\+[a-z]{3})*)$', description="OCR languages of `file`"),
    source_lang: str = Query('auto', pattern=r'^(auto|[a-z]{3}(\+[a-z]{3})*)$'),
    bilingual: Optional[str] = Query(None, description="One bilingual paper: its two scripts, e.g. 'Latin,Devanagari'"),
    profile: Optional[str] = Query(None, description="Format profile id from GET /verify/profiles"),
    output: Literal['json', 'html'] = Query('json'),
):
    """
    Pre-print verification of a question paper: numbering, options, marks, duplicates; with `source`, parity with the
    source-language version; with `bank`, repeats of earlier questions; with `profile`, the document's format.
    """
    if not CHECKPOINTS['doclayout'].exists() or not tesseract_available():
        raise HTTPException(status_code=503, detail="Verification needs the layout model and OCR on this server")
    for code in (lang, source_lang):
        if code != 'auto' and not set(code.split('+')) <= OCR_CODES:
            raise HTTPException(status_code=400, detail=f"OCR languages must be 'auto' or codes from {sorted(OCR_CODES)}")
    if profile is not None and profile not in list_profiles():
        raise HTTPException(status_code=400, detail=f"Unknown profile; choose from {sorted(list_profiles())}")
    split = None
    if bilingual:
        scripts = [s.strip() for s in bilingual.split(',')]
        if len(scripts) != 2 or not set(scripts) <= SCRIPTS or source is not None:
            raise HTTPException(status_code=400, detail=f"bilingual takes two scripts from {sorted(SCRIPTS)}, without `source`")
        split = {scripts[0]: scripts[0].lower(), scripts[1]: scripts[1].lower()}
    bank_items = None
    if bank is not None:
        try:
            bank_items = json.loads(await _read_upload(bank))
            assert isinstance(bank_items, list) and all(isinstance(b, dict) and 'text' in b for b in bank_items)
        except (ValueError, AssertionError):
            raise HTTPException(status_code=400, detail="bank must be a JSON list of {'text', 'source'} objects")

    images = _pages_rgb(await _read_upload(file))
    src_images = _pages_rgb(await _read_upload(source)) if source is not None else None
    pages = await run_in_threadpool(_read_paper, images, lang)
    src_pages = await run_in_threadpool(_read_paper, src_images, source_lang) if src_images else None
    meta = {'File': file.filename or 'paper', 'Pages': len(images)}
    if source is not None:
        meta['Checked against'] = source.filename or 'source'
    report = verify(pages, source_pages=src_pages, bank=bank_items, profile=profile, split_scripts=split, meta=meta)
    if output == 'html':
        imgs = {'paper': images, **({'source': src_images} if src_images else {})}
        return HTMLResponse(verify_report_html(report, imgs, title='IndicDoc Verify: pre-print report'))
    return report

