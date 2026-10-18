# Limitations

## Data and training scale
- Our two models are trained on a **language-balanced subset** of IndicDLP (about 200 training pages per
  language, about 2.4K pages in total), not the full ~96K training pages. The IndicDLP paper's models
  trained on everything reach about 0.73 mAP@50, so the numbers here are lower and should be read as
  "what a laptop-scale run achieves".
- Training ran for a few epochs on an Apple M1 Pro laptop GPU. Longer training and the full dataset would
  improve both models, especially on rare labels.
- The labels are very unbalanced: `paragraph` is 31% of the training boxes, while `index` (6 boxes) and
  `sub-unordered-list` (13) are rare and `subsub-unordered-list` does not occur in the training subset at
  all (counts in `docs/DATASET.md`). Rare labels are learnt poorly. A label with no test examples gets
  "N/A", never zero.
- Per-language numbers rest on about 60 test pages per language (Hindi only 22, Assamese has 129 training
  pages instead of 200), so differences of a few points between languages are within noise.

## What the system does not do well
- **OCR quality is not measured**: IndicDLP has no transcriptions. Tesseract reads clean print well; handwriting, old
  or degraded prints and decorative fonts read badly. Formulas and figures are not read (marked `[formula]`/`[figure]`).
- **Script detection** picks the right script on about 88% of test pages; on the rest the user can pick the language.
- **OCR speed**: about 8 s per page natively on an Apple M1 Pro (Homebrew Tesseract); in the Docker container on the same
  Mac 8-45 s depending on how dense the page is (Debian's Tesseract build inside Docker's VM is slower). Layout detection
  itself takes about 2 s per page on CPU.
- **Side-by-side text on one line** (e.g. a degree and its dates in a CV) can be merged by OCR into one line.
- **Out-of-domain documents**: IndicDLP covers 12 document types; CVs, invoices, slides etc. are segmented less well.
- **No reading order or hierarchy**: the output is a flat list of boxes, not "section 2 contains these
  paragraphs".
- **PDFs are analysed one page at a time** (the app has a page picker); there is no whole-document batch mode.
- Pages are assumed roughly upright. Deskewing exists in `ml/preprocessing` but is not applied
  automatically.

## Deployment scope
- The deployed product is the Streamlit app plus the FastAPI REST API, in one Docker image running on CPU.
  Measured in the container on an Apple M1 Pro: about 0.15-0.35 s per page for YOLOv8s and 0.9-1.4 s for
  Faster R-CNN. Both models are loaded when the API starts (about 20 s), so the first request is not slow.
  A GPU host would be much faster.
- The API has an upload-size limit and a CORS allow-list, but **no authentication or rate limiting**. Put it
  behind a reverse proxy (e.g. nginx, Caddy, a cloud load balancer) that adds these before exposing it to
  the public internet.
- The React frontend in `frontend/` is an unfinished prototype and is not deployed.

## Mitigation / next steps
1. Train on more of IndicDLP (or all of it) on a cloud GPU; the scripts take a `--train_groups` setting.
2. Try the document-specialised DocLayout-YOLO weights published by the IndicDLP authors as a stronger
   starting point.
3. Measure OCR quality on a transcribed Indian-language set, and try stronger Indic OCR models.
4. Multi-page PDF processing and reading-order reconstruction.
