# Evaluation

## Protocol

- **Test set:** 682 pages from IndicDLP's official test split (10513 boxes, all 12 languages,
  all 12 domains). Never used for training or for choosing checkpoints (that is what the 342 validation
  pages are for).
- **Detections:** every detection with confidence >= 0.001 is kept for mAP (up to 300 per page; dense newspaper pages
  exceed COCO's default of 100).
- **Tool:** `ml/evaluation/coco_eval.py`, built on pycocotools (the reference COCO implementation).

## Metrics

| Metric | Meaning |
|---|---|
| IoU | Overlap of a predicted and a true box, divided by the area they cover together (1 = identical) |
| mAP@50 | Area under the precision-recall curve (COCO 101-point interpolation) at IoU >= 0.50, averaged over labels |
| mAP@75 | Same at IoU >= 0.75 (needs tight boxes) |
| mAP@50:95 | Average of mAP over IoU 0.50, 0.55, ..., 0.95 |
| Precision / Recall / F1 | At one operating point: confidence >= 0.5, same-label match at IoU >= 0.5, greedy by score |
| Recommended confidence | The threshold (0.05 to 0.95 in steps of 0.05) with the best F1 on the **validation** pages; test P/R/F1 are also reported at it. The app and API use it by default |
| Average IoU | Mean IoU of the correctly matched boxes |
| Confusion matrix | Label-agnostic matching at IoU >= 0.5: true label vs predicted label, plus missed and spurious boxes |
| Per-language mAP@50 | mAP restricted to each language's test pages |
| Inference time | Seconds per page, one page at a time, first page excluded as warm-up |

## Is the evaluator right?

1. **Hand-worked unit tests** (`tests/test_coco_eval.py`): perfect detections give mAP 1.0; no detections give 0;
   one wrong label gives exactly 2 TP / 1 FP / 1 FN and the expected confusion-matrix cell; boxes 60% too wide
   (IoU 0.625) pass at IoU 0.5 and fail at 0.75; per-language splits are separated correctly.
2. **Independent cross-check of the maths:** Ultralytics' own matching and AP code, run on exactly the detections this
   project scores, gives mAP@50 0.2576 against this project's
   0.2587 (mAP@50:95 0.1634 vs
   0.1646).
3. **Why Ultralytics' validator reports more** (0.2787): its validation NMS is multi-label
   (`multi_label=True` in `ultralytics/models/yolo/detect/val.py`), so one box can also be kept under its second-best
   label, which adds recall. Normal prediction keeps one label per box. That is what users of the app and API get, so
   it is what this project reports.

Results: [RESULTS.md](RESULTS.md).
