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
from PIL import Image
from starlette.concurrency import run_in_threadpool

# Allow importing the ml package from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from ml.config import AUTHORS_CHECKPOINT, BASELINE_CHECKPOINT, CLASS_NAMES, METRICS_DIR, PROPOSED_CHECKPOINT, get_device
from ml.inference.predict import dedupe_boxes, demo_inference, load_model, predict_single
from ml.preprocessing.preprocess import decode_image
from ml.ocr import OCR_LANGUAGES, extract_document, tesseract_available
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
