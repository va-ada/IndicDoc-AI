# Progress

## Done

### Data
- [x] IndicDLP access (Hugging Face, gated) and the official 42-label mapping (`data/indic.yaml` from AI4Bharat)
- [x] Language-balanced subset builder reading only the needed parquet row groups (`scripts/prepare_indicdlp.py`)
- [x] Subset built and checked: boxes overlaid on real pages, cleaning statistics recorded (`docs/DATASET.md`)

### Models and training
- [x] Faster R-CNN baseline: fixed labels, normalisation and augmentation; validation loss, best checkpoint,
      time budget, Apple-GPU support (`ml/training/train_baseline.py`)
- [x] YOLOv8s: fixed dataset paths, no flips, time budget, best weights copied for the app (`ml/training/train_yolo.py`)
- [x] Both trained on the IndicDLP subset

### Evaluation
- [x] COCO-standard evaluator (pycocotools), P/R/F1, IoU, per-label AP, per-language mAP, confusion matrix
- [x] Evaluator validated on hand-worked cases and cross-checked against Ultralytics' AP code on the same detections
- [x] All fabricated plots and metrics removed; every output now measured (`docs/RESULTS.md`)

### Product
- [x] Streamlit app shows real results, loss curves and per-language tables; PDF page picker at 150 dpi
- [x] FastAPI REST API with input validation, upload limit and CORS allow-list
- [x] Docker image (CPU PyTorch, non-root, models included) + `docker-compose.yml` for app and API
- [x] Pinned dependencies

### Third model and OCR
- [x] IndicDLP authors' DocLayout-YOLO integrated and evaluated on the same test pages; default model in app and API
- [x] OCR of detected regions with Tesseract (12 languages), automatic script detection, reading order, Markdown/JSON
- [x] Confidence slider filters live (inference keeps boxes down to 0.05)

### Quality
- [x] End-to-end audit, 30 bugs fixed (`docs/FINAL_AUDIT.md`)
- [x] pytest suite covering data conversion, training, evaluation, inference, API and app
      (count in `docs/RESULTS.md`)
- [x] Plain-language project report (`docs/report/IndicDoc_AI_Project_Report.pdf`)

## Not done / future work
- Training on the full ~96K-page training set (needs a cloud GPU)
- OCR of the detected regions; reading-order reconstruction
- Authentication and rate limiting (expected from a reverse proxy in front of the containers)
- The React frontend in `frontend/` stays an experimental prototype
