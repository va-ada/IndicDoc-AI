# Architecture

## Pipeline Overview

```
Document image (PNG/JPG, or page 1 of a PDF)
      |
      v
Resize so the longest side is 640 px (aspect ratio kept)
      |
      v
Detector: DocLayout-YOLO (default, IndicDLP authors) | YOLOv8s (ours) | Faster R-CNN (ours)
      |
      v
Boxes + one of 42 IndicDLP layout labels + confidence score
      |
      v
Script detection (probe each script's OCR model on one text block)  ->  Tesseract OCR of the page
      |
      v
Words assigned to boxes, boxes put in reading order  ->  Markdown + JSON
      |
      v
Streamlit app (boxes, text, downloads) or JSON (FastAPI /analyze, ocr=true)
```

## Code Layout

| Path | Role |
|---|---|
| `ml/config.py` | Paths, the 42 class names (official IndicDLP order), training defaults, `get_device()` |
| `ml/datasets/coco_dataset.py` | PyTorch `Dataset` over COCO annotations (`label_offset=1` for Faster R-CNN) |
| `ml/preprocessing/` | Albumentations transforms (resize longest side, colour jitter, blur) and page utilities (CLAHE, deskew) |
| `ml/models/` | Model builders and the COCO -> YOLO label converter |
| `ml/training/` | `train_baseline.py` (Faster R-CNN), `train_yolo.py` (YOLOv8) |
| `ml/evaluation/coco_eval.py` | COCO mAP via pycocotools, P/R/F1, confusion matrix, per-language breakdown |
| `ml/inference/predict.py` | Load a checkpoint (Faster R-CNN, YOLOv8, DocLayout-YOLO) and predict; labels always 0-indexed class ids |
| `ml/ocr/__init__.py` | Tesseract OCR: script detection, word-to-region assignment, reading order, Markdown |
| `scripts/prepare_indicdlp.py` | Builds the language-balanced IndicDLP subset from the Hugging Face parquet shards |
| `scripts/evaluate.py` | Evaluates a checkpoint on the test pages and writes metrics + plots to `outputs/` |
| `app/streamlit_app.py` | The user-facing app |
| `backend/main.py` | REST API (`/analyze`, `/metrics`) |
| `frontend/` | React dashboard prototype (experimental, not part of the deployment) |

## Baseline: Faster R-CNN with ResNet-50 FPN

| Property | Value |
|---|---|
| Architecture | Faster R-CNN (Ren et al., NeurIPS 2015), torchvision `fasterrcnn_resnet50_fpn` |
| Backbone | ResNet-50 (He et al., CVPR 2016) + Feature Pyramid Network (Lin et al., CVPR 2017) |
| Detector type | Two-stage: a Region Proposal Network proposes boxes, a second head classifies and refines them |
| Starting weights | COCO-pretrained (80 everyday-object classes) |
| Output classes | 42 IndicDLP labels + 1 background = 43 |
| Parameters | 41.5 million |
| Internal resize | Shortest side 640 px, longest at most 1024 px (torchvision default 800/1333 lowered to fit a laptop GPU) |
| Normalisation | ImageNet mean/std, applied inside the model; the data pipeline only scales pixels to [0, 1] |

**Training configuration** (`ml/training/train_baseline.py`):
- Optimizer: SGD, learning rate 0.005, momentum 0.9, weight decay 5e-4
- Schedule: cosine annealing over the epochs
- Gradient clipping at norm 10; batches with a non-finite loss are skipped
- Augmentation: colour jitter, mild Gaussian blur. No flips: a mirrored page is not a realistic document
- The checkpoint with the lowest validation loss is kept

## Proposed: YOLOv8s

| Property | Value |
|---|---|
| Architecture | YOLOv8 small (Jocher et al., Ultralytics 2023) |
| Backbone / neck | CSPDarknet with C2f blocks / PAN feature pyramid |
| Detector type | Single-stage, anchor-free: one network pass predicts boxes and classes directly |
| Starting weights | COCO-pretrained |
| Output classes | 42 IndicDLP labels |
| Parameters | 11.2 million |
| Input | 640 px, letterboxed (aspect ratio kept, padded) |

**Training configuration** (`ml/training/train_yolo.py`): Ultralytics trainer with its default optimizer selection,
warm-up and learning-rate decay; mosaic, HSV and translate/scale augmentation; horizontal flips disabled
(`fliplr=0`). The weights with the best validation fitness are copied to `outputs/models/yolov8_best.pt`.

## Deployed default: DocLayout-YOLO (IndicDLP authors)

| Property | Value |
|---|---|
| Architecture | DocLayout-YOLO (Zhao et al., arXiv:2410.12628, 2024), a YOLOv10-based detector for documents |
| Training | By the IndicDLP authors: pre-trained on DocSynth-300K synthetic pages, fine-tuned on the full IndicDLP training set (100 epochs, 1024 px) |
| Source | Zenodo record 15881917, `doclayout_yolo_indicdlp.pt`, licence CC-BY 4.0; library `doclayout-yolo` (AGPL-3.0, like Ultralytics) |
| Parameters | 20.0 million |
| Labels | The same 42 labels in the same order as `ml/config.py` (checked when loading) |

It is not trained in this project. It is evaluated with exactly the same test pages and code as our two models
(`docs/RESULTS.md`), and it is the app's and API's default because it is by far the most accurate.

## OCR stage

`ml/ocr/__init__.py`, using Tesseract 5 with tessdata_best models for the 12 IndicDLP languages:
1. **Script detection:** the largest detected text block is read with each script's model (in parallel), counting
   only characters of that model's own script; the most confident script wins. Tesseract's own OSD detector was
   measured and rejected (70/120 test pages right).
2. **One OCR pass per page** (TSV output: every word with its box); pages under 1600 px are upscaled 2x first.
3. **Assignment:** each word goes to the smallest detected region containing its centre; lines are rebuilt from
   Tesseract's line ids.
4. **Reading order:** top to bottom; regions overlapping vertically by at least half form a row, read left to right.
5. **Markdown:** headings by label level, lists, tables in code blocks, captions in italics, `[figure]`/`[formula]`
   placeholders (OCR cannot read them), page furniture (header, footer, page number...) listed at the end.

## Why These Two Models

- **Faster R-CNN** is the standard two-stage reference detector, with a mature torchvision implementation.
  It looks at each proposed region separately, which tends to help with small and tightly packed elements.
- **YOLOv8** is a modern single-stage detector: about a quarter of the parameters, one pass per image,
  and a training pipeline built for detection. The IndicDLP authors report strong results with YOLO-family
  models (YOLOv10x and DocLayout-YOLO reach about 0.735 mAP@50 on the full dataset).

Both models use **transfer learning**: they start from COCO weights, get a new 42-class output head and are
fine-tuned end to end. This converges far faster than training from scratch, which matters on a laptop.

## Deployment Architecture

One Docker image holds the code, the two trained checkpoints and the evaluation results:

```
docker compose up --build
  app  -> streamlit run app/streamlit_app.py   (port 8501)
  api  -> uvicorn backend.main:app             (port 8000)
```

- The image uses CPU-only PyTorch (3.5 GB instead of 11 GB with the CUDA libraries).
- It runs as an unprivileged user.
- The API limits uploads (`INDICDOC_MAX_UPLOAD_MB`, default 20) and only allows the browser origins listed in
  `INDICDOC_CORS_ORIGINS`.
