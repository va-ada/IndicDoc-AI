# IndicDoc AI · IndicDoc Verify

**Check an Indian question paper and its translations before it is printed.** Offline, 12 Indian languages.

Indian exams are set in one language and translated into Hindi and regional languages; every version is checked
by hand, and errors keep reaching candidates (NEET 2018: 49 wrong questions in Tamil; KPSC 2024: 59 errors and a
re-exam; UGC-NET June 2026: three papers re-tested). IndicDoc Verify reads the printed paper and reports:

- **Structure:** numbering gaps and repeats, the same question printed twice, a question with a missing or repeated
  option, marks that do not add up to the printed maximum, page numbering
- **Translation parity:** a question missing in a version, and every number, option, mark, sub-question or figure that
  differs from the source version, question by question, **without translating anything**
- **Repeats:** questions copied from earlier papers (a question bank)
- **Format:** any document type described as a JSON profile (e.g. an Office Memorandum, bilingual circulars)

Each finding points to the question and page and is outlined on the page (HTML / JSON report). It runs in one Docker
image with no internet access, because a pre-print paper must never leave the exam body.

**How it works.** A deep-learning layout detector (DocLayout-YOLO, fine-tuned on IndicDLP by its authors) finds the
questions, sub-questions, options and answers on each page; Tesseract reads the text, with an English *number pass*
that must agree on every digit; the questions are built from the layout and checked. Three layout detectors are
compared on the same IndicDLP test pages (DocLayout-YOLO, and a YOLOv8s and a Faster R-CNN we fine-tuned).

Product, evidence and market: [docs/PRODUCT.md](docs/PRODUCT.md). Benchmark: [Verification results](#verification-results).

<!-- VERIFY:START -->
## Verification results

Measured by `scripts/evaluate_verify.py` on the benchmark's test set: 132 paper pairs (English source +
Hindi or Tamil version, 18 questions each, rendered at 200 dpi), 108 with one injected error of 9
kinds that have happened in real exams, 24 clean; half degraded like scans; read with Tesseract.
An error counts as found only when a matching finding points at the right question. The parser was developed on a
separate development set; the test set was run once.

| | Errors found at the right question | False alarms per clean pair |
|---|---|---|
| Born-digital papers | 98% | 3.0 |
| Scan-degraded papers | 98% | 5.5 |
| **All** | **98%** | **4.5** |
| Ablation: no number pass | 77% | 13.8 |
| Ablation: options split by markers, not layout | 98% | 11.5 |
| Noisy layout (boxes shifted, 10% of question/option labels wrong) | 98% | 7.5 |

The two misses were Hindi maximum-marks lines read by OCR as "पूर्णाक"; after widening that pattern the same test set
gives 100% and 4.4 false alarms per clean pair (post-hoc). Layout boxes come from the
renderer; the layout model's AP@50 on question-paper labels (first-level question 0.85, options
0.84, answer 0.81) is in the table below.
<!-- VERIFY:END -->

## Layout results

Measured by `scripts/evaluate.py` on held-out IndicDLP test pages that neither model saw during training:

<!-- RESULTS:START -->
| Metric | DocLayout-YOLO (authors) | YOLOv8s (ours) | Faster R-CNN (ours) |
|---|---|---|---|
| mAP@50 | 0.759 | 0.259 | 0.148 |
| mAP@50:95 | 0.575 | 0.165 | 0.079 |
| mAP@75 | 0.633 | 0.172 | 0.075 |
| Precision (conf >= 0.5) | 0.891 | 0.806 | 0.546 |
| Recall (conf >= 0.5) | 0.708 | 0.362 | 0.409 |
| F1 (conf >= 0.5) | 0.789 | 0.500 | 0.468 |
| Average IoU (correct boxes) | 0.899 | 0.885 | 0.821 |
| Inference time (s/page) | 0.150 | 0.024 | 0.202 |
| Recommended confidence (best F1 on validation pages) | 0.35 | 0.25 | 0.45 |
| Precision at recommended confidence | 0.848 | 0.603 | 0.510 |
| Recall at recommended confidence | 0.775 | 0.508 | 0.435 |
| F1 at recommended confidence | 0.810 | 0.551 | 0.469 |
| "Find every part" confidence (lowest with precision >= 0.70 on validation) | 0.15 | 0.25 | 0.45 |
| Precision, find every part | 0.742 | 0.603 | 0.510 |
| Recall, find every part | 0.851 | 0.508 | 0.435 |
| Model size (MB) | 161.5 | 22.5 | 166.6 |
| Parameters (M) | 20.0 | 11.2 | 41.5 |

Test set: 682 pages, 12 languages. DocLayout-YOLO: trained by the IndicDLP authors on the full dataset. YOLOv8s and Faster R-CNN: trained here on 2293 pages (language-balanced subset) on an Apple M1 Pro.
<!-- RESULTS:END -->

Full results, per-class and per-language tables are in [docs/RESULTS.md](docs/RESULTS.md). There is also
a plain-language project report: [docs/report/IndicDoc_AI_Project_Report.pdf](docs/report/IndicDoc_AI_Project_Report.pdf), and the
Expt.-11 lab report: [docs/report/IndicDoc_AI_Expt11_Lab_Report.pdf](docs/report/IndicDoc_AI_Expt11_Lab_Report.pdf).
Everything in one print-ready A4 PDF (dataset, code, repository, both decks, paper; each part starts on a new sheet):
[docs/print/IndicDoc_AI_Complete_Submission.pdf](docs/print/IndicDoc_AI_Complete_Submission.pdf), built by `python scripts/build_print_bundle.py`.

## Quick start (Docker)

```bash
docker compose up --build -d
```

- Web app: http://localhost:8501
- REST API: http://localhost:8000 (docs at http://localhost:8000/docs)

```bash
curl -F file=@page.png "http://localhost:8000/analyze"                    # layout (DocLayout-YOLO)
curl -F file=@page.png "http://localhost:8000/analyze?ocr=true"           # layout + text, language auto-detected
curl -F file=@page.png "http://localhost:8000/analyze?ocr=true&ocr_lang=hin+eng"
curl -F file=@paper_hi.pdf -F source=@paper_en.pdf "http://localhost:8000/verify?lang=hin&source_lang=eng"
curl -F file=@paper_hi.pdf -F source=@paper_en.pdf "http://localhost:8000/verify?lang=hin&source_lang=eng&output=html" > report.html
```

Configuration, health checks and production notes: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Local setup

```bash
python3.11 -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_models.py     # authors' DocLayout-YOLO + Tesseract language models (~300 MB, once)
brew install tesseract                # or: apt install tesseract-ocr (the OCR engine itself)

streamlit run app/streamlit_app.py    # web app
uvicorn backend.main:app --port 8000  # REST API
python -m pytest tests/               # test suite
```

## Reproduce the results

1. **Get the data.** Accept the terms of [ai4bharat/indicdlp](https://huggingface.co/datasets/ai4bharat/indicdlp)
   on Hugging Face and log in (`huggingface-cli login`, or put a read token in `~/.cache/huggingface/token`).
   Then build the language-balanced subset (reads only the parquet row groups it needs):

   ```bash
   python scripts/prepare_indicdlp.py     # -> data/raw/indicdlp/{train,val,test}
   ```

2. **Train.** Both scripts use the GPU if there is one (CUDA, or Apple Silicon via MPS):

   ```bash
   D=data/raw/indicdlp
   python -m ml.training.train_yolo --train_ann $D/train/annotations.json --train_img $D/train/images \
       --val_ann $D/val/annotations.json --val_img $D/val/images --convert_annotations --batch 8 --hours 1
   python -m ml.training.train_baseline --train_ann $D/train/annotations.json --train_img $D/train/images \
       --val_ann $D/val/annotations.json --val_img $D/val/images --batch_size 4 --epochs 20 --max_minutes 110
   ```

3. **Evaluate** on the test pages (writes `outputs/metrics`, `outputs/plots`, `outputs/predictions`):

   ```bash
   V="--val_ann $D/val/annotations.json --val_img $D/val/images"   # validation pages pick each model's threshold
   python scripts/evaluate.py --model_type doclayout --name authors --ann $D/test/annotations.json --img_dir $D/test/images $V
   python scripts/evaluate.py --model_type yolov8 --name proposed --ann $D/test/annotations.json --img_dir $D/test/images $V
   python scripts/evaluate.py --model_type fasterrcnn --name baseline --ann $D/test/annotations.json --img_dir $D/test/images $V
   python scripts/evaluate_ocr.py     # script detection, OCR coverage and speed
   ```

4. **Verification benchmark** (rendered English/Hindi and English/Tamil paper pairs with injected errors, read with
   Tesseract; needs Node.js):

   ```bash
   (cd scripts/verify_bench && npm install && python generate.py --out ../../data/verify_bench --per_type 6 --clean 12 --seed 42)
   node scripts/verify_bench/render.js data/verify_bench          # pages at 200 dpi + region boxes
   python scripts/verify_bench/degrade.py data/verify_bench       # half of the pairs look scanned
   node scripts/verify_bench/ocr.js data/verify_bench             # Tesseract (tesseract.js) + number pass
   python scripts/evaluate_verify.py --bench data/verify_bench    # + --layout noisy, --no_number_pass, --no_geometry
   python scripts/verify_figures.py
   ```
   Copy the language models into `scripts/verify_bench/tessdata/` (eng, hin, tam `.traineddata.gz`, from the
   `@tesseract.js-data/*` packages). The development set used to build the parser: `--seed 7 --per_type 2 --clean 6`.

5. **Documents:** `python scripts/build_slides.py && python scripts/slides_to_pptx.py` (Phase-2 deck),
   `python scripts/build_lab_report.py` (Expt.-11 report), `python scripts/build_verify_paper.py` (IEEE paper),
   `python scripts/build_report.py` (plain-language report)

No real data? `python scripts/run_sample_training.py` runs the whole pipeline on 30 synthetic pages
(outputs go to `outputs/sample_run/`; the numbers are meaningless, it only checks the plumbing).

## Project structure

```
app/streamlit_app.py        Streamlit web app (the product UI)
app/assets/samples/         six real IndicDLP test pages to try the app with
backend/main.py             FastAPI REST API
ml/
  config.py                 paths, the 42 IndicDLP labels, defaults, get_device()
  datasets/                 COCO dataset loader, synthetic sample generator
  preprocessing/            transforms, deskew/CLAHE, PDF page rendering
  models/                   Faster R-CNN and YOLOv8 builders, COCO->YOLO converter
  training/                 train_baseline.py, train_yolo.py
  evaluation/coco_eval.py   COCO mAP (pycocotools), P/R/F1, confusion matrix, per-language
  inference/predict.py      load a checkpoint (Faster R-CNN, YOLOv8, DocLayout-YOLO), predict
  ocr/                      Tesseract OCR of detected regions, script detection, number pass, reading order
  verify/                   IndicDoc Verify: question parsing, checks, translation parity, repeats,
                            format profiles (profiles/*.json), HTML report
scripts/
  prepare_indicdlp.py       build the IndicDLP subset from Hugging Face
  download_models.py        fetch the authors' DocLayout-YOLO and the Tesseract language models
  evaluate_ocr.py           measure script detection, OCR coverage and speed
  evaluate.py               evaluate a checkpoint, write metrics + plots
  build_report.py           PDF project report + docs/RESULTS.md
  build_lab_report.py       Expt.-11 lab report (IEEE tables/figures/references)
  build_slides.py           Phase-2 deck (PDF); slides_to_pptx.py makes the PPTX
  build_verify_paper.py     IEEE research paper on IndicDoc Verify
  evaluate_verify.py        verification benchmark: errors found, false alarms, ablations
  verify_bench/             benchmark generator (questions in English/Hindi/Tamil), renderer, OCR
  run_sample_training.py    pipeline smoke run on synthetic pages
outputs/                    trained models, metrics, plots, example predictions
docs/                       dataset, architecture, methodology, evaluation, results, deployment, audit
tests/                      pytest suite
notebooks/                  demo scripts on the synthetic sample (write to outputs/sample_run/)
frontend/                   React dashboard prototype (experimental, not deployed)
Dockerfile, docker-compose.yml
```

## Documentation

| Document | Contents |
|---|---|
| [docs/PRODUCT.md](docs/PRODUCT.md) | IndicDoc Verify: evidence of the problem, competitors, USP, customers |
| [paper/IndicDoc_Verify_Research_Paper.pdf](paper/IndicDoc_Verify_Research_Paper.pdf) | IEEE paper on IndicDoc Verify |
| [docs/report/IndicDoc_AI_Expt11_Lab_Report.pdf](docs/report/IndicDoc_AI_Expt11_Lab_Report.pdf) | Expt.-11 lab report |
| [docs/presentation/](docs/presentation/) | Phase-1 and Phase-2 slides (Phase-2 as PDF and PPTX) |
| [docs/DATASET.md](docs/DATASET.md) | IndicDLP, the 42 labels, the subset used |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Pipeline, models, code layout, deployment architecture |
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | Data selection, preprocessing, training, evaluation protocol |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Metric definitions and evaluator validation |
| [docs/RESULTS.md](docs/RESULTS.md) | All measured results |
| [docs/LIMITATIONS.md](docs/LIMITATIONS.md) | What the system does not do, and caveats |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Running and configuring the Docker deployment |
| [docs/FINAL_AUDIT.md](docs/FINAL_AUDIT.md) | Bugs found and fixed in the end-to-end audit |
| [docs/VIVA_PREPARATION.md](docs/VIVA_PREPARATION.md) | Short answers to likely viva questions |

## Dataset and references

- O. Nath, S. Kukkala, M. Khapra, R. K. Sarvadevabhatla. *IndicDLP: A Foundational Dataset for Multi-lingual
  and Multi-domain Document Layout Parsing.* ICDAR 2025.
- Z. Zhao et al. *DocLayout-YOLO.* arXiv:2410.12628, 2024. R. Smith. *An Overview of the Tesseract OCR Engine.* ICDAR 2007.
- S. Ren et al. *Faster R-CNN.* NeurIPS 2015. G. Jocher et al. *Ultralytics YOLOv8.* 2023.
  T.-Y. Lin et al. *Microsoft COCO.* ECCV 2014.

*B.E. Artificial Intelligence & Machine Learning, Deep Learning mini-project. SDG 9: Industry, Innovation and Infrastructure.*
