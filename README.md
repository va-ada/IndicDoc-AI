# IndicDoc AI

**Deep-learning layout detection for multilingual Indian documents.**

Give it a photo, scan or PDF page in any of 12 Indian languages. It draws a box around every structural
element (paragraphs, headings, tables, figures, page numbers, lists, questions...), labels each one with one of
the 42 layout classes of the IndicDLP dataset, and then **reads the text** of every box (OCR), returning the page as
an organised Markdown/JSON document.

Three layout models are compared on the same real IndicDLP test pages:
- **DocLayout-YOLO (IndicDLP authors)**: trained by the dataset's authors on the full dataset, downloaded from their
  public release (Zenodo, CC-BY 4.0). Best by far; the app's default.
- **YOLOv8s (proposed)** and **Faster R-CNN (baseline)**: trained in this project on a language-balanced subset.

Text is read with Tesseract (12 Indian languages), with the page's script detected automatically.
The product is a Streamlit web app plus a REST API, shipped as one Docker image that works offline.

## Results

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

4. **Report:** `python scripts/build_report.py`

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
  ocr/                      Tesseract OCR of detected regions, script detection, reading order, Markdown
scripts/
  prepare_indicdlp.py       build the IndicDLP subset from Hugging Face
  download_models.py        fetch the authors' DocLayout-YOLO and the Tesseract language models
  evaluate_ocr.py           measure script detection, OCR coverage and speed
  evaluate.py               evaluate a checkpoint, write metrics + plots
  build_report.py           PDF project report + docs/RESULTS.md
  build_lab_report.py       Expt.-11 lab report (IEEE tables/figures/references)
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
