# Methodology

## 1. Problem Formulation

Document Layout Parsing is treated as **object detection**:
- **Input**: an image of one document page (scanned or digital-born), in any of 12 languages
- **Output**: a list of (bounding box, layout label, confidence score), one per element on the page
- **Labels**: the 42 IndicDLP region labels, from generic ones (`paragraph`, `table`, `figure`, `header`)
  to domain-specific ones (`jumpline` in newspapers, `first-level-question` in question papers)

## 2. Data

### 2.1 Source
IndicDLP (Nath et al., ICDAR 2025), downloaded from `huggingface.co/datasets/ai4bharat/indicdlp`
(gated: the account must accept the dataset terms). It is distributed as parquet shards (train 73 GB,
validation 9 GB, test 9 GB). Each row holds one PNG page, its boxes as `[x, y, width, height]` in pixels,
and integer class ids 0..41. Class names come from the authors' `data/indic.yaml`.

### 2.2 Subset used here
Training on all ~120K pages is not feasible on a laptop, so `scripts/prepare_indicdlp.py` builds a
**language-balanced subset**:

| Split | Pages per language | Source split |
|---|---|---|
| Train | up to 200 | official train |
| Validation | up to 30 | official validation |
| Test | up to 60 | official test |

Actual subset (some row groups held fewer pages of a language than the cap):

| Language | Train pages | Val pages | Test pages |
|---|---|---|---|
| Assamese | 129 | 30 | 60 |
| Bengali | 200 | 30 | 60 |
| English | 188 | 30 | 60 |
| Gujarati | 200 | 30 | 60 |
| Hindi | 200 | 21 | 22 |
| Kannada | 200 | 21 | 60 |
| Malayalam | 200 | 30 | 60 |
| Marathi | 200 | 30 | 60 |
| Odia | 200 | 30 | 60 |
| Punjabi | 188 | 30 | 60 |
| Tamil | 200 | 30 | 60 |
| Telugu | 188 | 30 | 60 |
| **Total** | **2293** | **342** | **682** |

How the pages are chosen:
- The shards are sorted by language, and a shard can straddle two languages. So every parquet row group
  (~100 pages) is labelled by the languages of its own pages, read from the file names
  (`<domain>_<language>_<id>_<n>.png`).
- For each language, row groups are picked from as many different shards as possible (more variety of
  documents), with a fixed random seed so the subset can be rebuilt exactly.
- Only the parquet row groups that are needed are downloaded (HTTP range requests), a few GB instead of 91.
- Train, validation and test pages come from the official splits, so they never overlap.

### 2.3 Cleaning
Each box is clipped to the page. Boxes left under 1 px wide or tall after clipping are dropped, and an
unknown class id stops the conversion with an error. Counts are reported in the data summary
(`docs/DATASET.md`).

## 3. Preprocessing and Augmentation

| Step | Where | Why |
|---|---|---|
| Resize longest side to 640 px, keep aspect ratio | Faster R-CNN data pipeline | Same page shape in training and at inference |
| Letterbox to 640 px | YOLOv8 (ultralytics) | Standard YOLO input |
| Scale pixels to [0, 1] | Faster R-CNN data pipeline | The model applies ImageNet normalisation itself |
| Colour jitter, Gaussian blur | Faster R-CNN training | Different scan quality, paper colour, focus |
| Mosaic, HSV, translate/scale | YOLOv8 training | Ultralytics defaults |
| No horizontal flips | Both | A mirrored page is not a realistic document |

`ml/preprocessing/preprocess.py` also provides deskewing (minimum-area rectangle of the ink pixels,
accurate to about 0.1 degrees on tilted real pages) and CLAHE contrast enhancement for poor scans. They are
optional page-cleaning tools and are not part of the training pipeline.

## 4. Models and Training

Both detectors start from COCO-pretrained weights and get a new 42-class head (transfer learning).
Details are in `docs/ARCHITECTURE.md`.

| | Faster R-CNN (baseline) | YOLOv8s (proposed) |
|---|---|---|
| Script | `ml/training/train_baseline.py` | `ml/training/train_yolo.py` |
| Hardware | Apple M1 Pro GPU (PyTorch MPS) | Apple M1 Pro GPU (PyTorch MPS) |
| Epochs | As many as fit a 140-minute budget (measured after epoch 1) | As many as fit a 1-hour budget (Ultralytics `time=1.0`) |
| Batch size | 4 | 8 |
| Optimizer | SGD (lr 0.005, momentum 0.9, wd 5e-4), cosine schedule | Ultralytics automatic choice |
| Checkpoint kept | Lowest validation loss | Best validation fitness (mAP-based) |
| Seed | 42 | 42 |

The actual epoch counts and wall-clock times are recorded in `docs/RESULTS.md`.

## 5. Evaluation Protocol

All numbers come from `scripts/evaluate.py` on the held-out **test** pages, which neither model saw during
training or model selection.

| Metric | Definition |
|---|---|
| **mAP@50** | COCO-style mean Average Precision at IoU >= 0.50, averaged over the classes present (pycocotools) |
| **mAP@50:95** | The same, averaged over IoU thresholds 0.50, 0.55, ..., 0.95: rewards tight boxes |
| **Precision / Recall / F1** | At one operating point: confidence >= 0.5, matched to same-label truth at IoU >= 0.5 |
| **Average IoU** | Mean overlap of the correctly matched boxes |
| **Per-class AP@50** | Which layout elements are easy or hard |
| **Per-language mAP@50** | mAP computed separately on each language's test pages |
| **Confusion matrix** | Which labels get mixed up, plus missed elements and false alarms |
| **Inference time** | Seconds per page on the evaluation device |

Up to 300 detections per page are scored (COCO's default of 100 is too low for dense newspaper pages).
The evaluator is tested against hand-worked examples (`tests/test_coco_eval.py`). Its mAP is also
cross-checked against Ultralytics' own AP code on the same YOLO detections (`docs/EVALUATION.md`).

## 6. Third model and OCR

- **DocLayout-YOLO (IndicDLP authors)** is downloaded (`scripts/download_models.py`), not trained here, and evaluated on
  the same test pages with the same code; its recommended threshold is chosen on the same validation pages.
- **OCR** (`scripts/evaluate_ocr.py`): with the deployed layout model, 10 test pages per language. Measured: whether
  the detected script matches the page's language, the share of OCR words falling inside detected regions, and seconds
  per page. Character accuracy cannot be measured because IndicDLP has no transcriptions.

## 7. Reproducibility

- Fixed seed (42) for data selection, splits and training
- Subset construction is deterministic (`scripts/prepare_indicdlp.py`)
- Exact package versions are pinned in `requirements.txt`
- Training histories, metrics and plots are written to `outputs/`
