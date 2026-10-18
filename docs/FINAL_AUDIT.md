# Audit and Fix Log

An end-to-end audit (October 2026) read every file, ran everything, and tested each fix twice: with
controlled synthetic inputs, and on real data (IndicDLP pages, real PDFs, the live app and API).
This file records what was wrong and what was done about it. It replaces the earlier
`SELF_CRITIQUE.md` and `FINAL_SUBMISSION_REPORT.md`, whose claims did not hold up.

## State before the audit

- No model had been trained on real documents. The only training run used 30 synthetic pages (and the
  annotation file covered only 5 of them), so every metric was 0.0.
- Several outputs were not measured: the confusion matrix, class distribution, precision-recall curve and
  error-analysis images were generated with `np.random`; the YOLOv8 "results" were the baseline numbers
  multiplied by constants; "mAP" was precision x recall; the API's `/metrics` returned hard-coded 0.92/0.95.
- The 42 class names in the code were invented and did not match IndicDLP's labels.
- The 27 tests checked shapes, so none of the bugs below were caught.

## Bugs found and fixed

| # | Problem | Fix |
|---|---|---|
| 1 | `predict.load_model` crashed on the state_dicts the training scripts save | Rebuilds the model and loads the weights; labels returned 0-indexed for both models |
| 2 | App showed the wrong class name (off by one) for Faster R-CNN | Uses the shared `predict_single` |
| 3 | COCO -> YOLO converter appended, so re-runs duplicated every label | Writes each label file once |
| 4 | Deskew selected ~89% of a white page as "ink" and returned -90 degrees | Otsu ink mask + angle from rectangle corners; within 0.12 degrees on real pages |
| 5 | FastAPI backend crashed on import; returned invented metrics; turned its own 400 into a 500 | Rewritten: input validation, real model when a checkpoint exists, saved metrics, upload limit, CORS allow-list |
| 6 | React dashboard fell back to invented numbers | Shows a dash when a metric is missing |
| 7 | Faster R-CNN training: background-label clash (class 0), double normalisation, left-right flips, no validation, a checkpoint per epoch | `label_offset=1`, [0,1] inputs, no flips, validation loss, best checkpoint only |
| 8 | Sample annotations covered 5 of 30 images; generator unseeded | Seeded generator, regenerated |
| 9 | Docker build failed (`libgl1-mesa-glx` gone); 11.3 GB image; root user | `libgl1`, CPU-only PyTorch (3.5 GB), unprivileged user, models included, compose file |
| 10 | App results list lagged one interaction behind | Reruns after inference |
| 11 | YOLO training had never worked (relative dataset paths resolved against the YAML's folder) | Absolute paths, CLI options, best weights copied where the app loads them |
| 12 | Training squashed pages to 640x640 squares, inference did not | Resize longest side, aspect ratio kept |
| 13 | No real evaluation | `ml/evaluation/coco_eval.py` (pycocotools) + `scripts/evaluate.py` |
| 14 | Demo notebooks and sample scripts overwrote real checkpoints and plots | All write to `outputs/sample_run/` |
| 15 | GPU never used on Apple Silicon | `get_device()`: CUDA, then MPS, then CPU (about 4x faster than CPU here) |
| 16 | PDFs rendered at 72 dpi, page 1 only; corrupt uploads crashed the page | 150 dpi, page picker, friendly error |
| 17 | Unpinned dependencies; deprecated Streamlit arguments | Exact versions pinned; `width="stretch"` |
| 18 | App reloaded the model from disk on every click, and defaulted to "Demo" even with trained models | Cached loader; model list built from the checkpoints present |
| 19 | API ran inference on its event loop (one slow page blocked every request); YOLO silently ignored thresholds below 0.25 | Worker thread + per-model lock; threshold passed to the model |
| 20 | A fixed 0.5 confidence left 5% of test pages with no boxes at all | Recommended threshold = best F1 on validation pages, used by default in app and API |
| 21 | Uploads could be decompression bombs (tiny file declaring a huge image); huge PDF pages rendered unbounded | Pixel limit checked from the file header; oversized PDF pages rendered at lower dpi |
| 22 | App drew the previous page's boxes on a newly selected page; its samples were synthetic rectangles; a corrupt PDF crashed the page; Reset blanked the viewer | Results tied to page content; six real IndicDLP sample pages; friendly PDF error; Reset clears results properly |
| 23 | Ultralytics patches `PIL.Image.open`; any unreadable upload made the server `pip install pi-heif` from the internet at runtime | Runtime installs disabled when serving (`disable_runtime_installs()`); Docker sets `YOLO_AUTOINSTALL=False`, `YOLO_OFFLINE=True` |
| 24 | The first API request after a restart waited about 24 s while the 166 MB Faster R-CNN checkpoint loaded | Available models load at API startup (within the health-check grace period) |
| 25 | Loading the trained Faster R-CNN downloaded 98 MB of ImageNet weights (torchvision's `weights=None` still fetches the backbone), so it needed internet and failed offline | Untrained architecture built without any download; image verified with networking disabled |
| 26 | UI: once the sidebar was collapsed there was no way to reopen it (CSS hid the whole header that holds the button); in OS dark mode tab names, labels and the model dropdown were unreadable; the upload screen was the second tab | Header kept (transparent), only Deploy/menu hidden; light theme pinned in `.streamlit/config.toml`; dropdown restyled; Analyze is the first tab. Checked in a real browser in dark mode |
| 27 | Our models' weak results on real use (a CV gave 3 boxes) | Added the IndicDLP authors' DocLayout-YOLO (0.76 mAP@50 on our test pages) as the default model |
| 28 | Tesseract's own script detector picked the right script on only 70/120 test pages (0/10 for Gujarati, Odia, Punjabi) | Script detection by probing each script's model on a text block, counting only own-script characters |
| 29 | At low thresholds one heading came back as section-title, sub-section-title and subsub-section-title | Class-agnostic duplicate removal (IoU > 0.7) in app, API and OCR |
| 30 | Text outside every detected box was silently dropped from OCR; OCR'd bullets started with junk ('«') | Leftover words kept as 'unlabelled-text' regions; bullet glyphs replaced by Markdown bullets |

## Verification

- Automated tests: see the count in `docs/RESULTS.md` (`python -m pytest tests/`).
- Real data: IndicDLP subset built and its boxes checked by overlaying them on pages; both models trained
  and evaluated on held-out test pages (`docs/RESULTS.md`).
- Deployment: Docker image built, both services started with `docker compose`, and the app and API
  exercised over HTTP with real test pages (`docs/DEPLOYMENT.md`).
