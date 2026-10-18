# Viva Preparation Guide

Short, accurate answers to likely viva questions. Every claim here matches the code and the measured
results in `docs/RESULTS.md`.

**1. What problem does IndicDoc AI solve?**
Given a photo or scan of a document page in one of 12 Indian languages, it finds every structural element
(paragraph, table, figure, header, page number, question, list, ...) and labels it with one of 42 layout
classes. This is the step before OCR and digitisation: you first need to know where the table or the
footnote is.

**2. Why this problem?**
India has huge archives of documents in many scripts. Most layout tools are trained on English documents
and fail on Indic scripts and layouts (newspapers with jumplines, question papers with nested questions).

**3. Which dataset, and where is it from?**
IndicDLP (ICDAR 2025, AI4Bharat / IIT Madras / IIIT Hyderabad): 119,806 annotated pages, 12 languages,
12 document domains, 42 labels. It is listed on AIKosh and downloaded from Hugging Face. We trained on a
language-balanced subset (about 200 pages per language) because the full 91 GB is beyond a laptop.

**4. Why deep learning?**
Rule-based layout analysis (whitespace, lines) breaks on varied, noisy and multi-column layouts. A CNN
detector learns visual patterns (text density, font size, position) that generalise across scripts.

**5. Why these two models?**
Faster R-CNN is the classic two-stage detector and a strong, well-understood baseline. YOLOv8 is a
modern single-stage detector: about a quarter of the parameters (11.2M vs 41.5M) and faster inference.
The comparison asks whether the lighter, faster model loses accuracy on document layouts.

**6. What is the input and the output?**
Input: one page image (or a PDF page rendered at 150 dpi). Output: a list of boxes, each with a label and
a confidence score.

**7. What is transfer learning here?**
Both models start from COCO-pretrained weights (80 everyday-object classes). We replace the
classification head with a 42-class one and fine-tune the whole network. Pretrained edge and texture
features make training converge with a few thousand pages instead of hundreds of thousands.

**8. What preprocessing is done?**
Resize so the longest side is 640 px, keeping the aspect ratio (so training and inference see pages the
same shape), and scale pixels to [0, 1]. Faster R-CNN normalises with ImageNet statistics inside the model.
Augmentation: colour jitter and blur for Faster R-CNN; mosaic, HSV and translate/scale for YOLO. No
horizontal flips, because a mirrored page is not a realistic document.

**9. What loss functions are used?**
Faster R-CNN: RPN objectness (binary cross-entropy) + RPN box regression (smooth L1) + classifier
cross-entropy + box regression (smooth L1). YOLOv8: classification BCE + CIoU box loss + distribution
focal loss (DFL).

**10. Which optimizer?**
Faster R-CNN: SGD with momentum 0.9, learning rate 0.005, weight decay 5e-4 and a cosine schedule.
YOLOv8: the Ultralytics trainer picks the optimizer and learning rate automatically.

**11. What is IoU?**
Intersection over Union: the overlap area of the predicted and true box divided by the area of their union.
1 is a perfect match. A prediction counts as correct at IoU >= 0.5.

**12. What is mAP?**
For each class, sort predictions by confidence and trace the precision-recall curve. AP is the area under
it (COCO's 101-point interpolation). mAP is the mean over classes. mAP@50 uses IoU 0.5; mAP@50:95 averages
over IoU 0.50 to 0.95 and rewards tight boxes. We compute it with pycocotools, the reference COCO
implementation.

**13. Precision vs recall?**
Precision: of the boxes the model drew, how many were right. Recall: of the true elements on the page, how
many the model found. We report both at confidence >= 0.5.

**14. How was the evaluation kept honest?**
Test pages come from IndicDLP's official test split, which no model saw during training or checkpoint
selection. The evaluator is unit-tested on hand-worked examples, and its mAP matches Ultralytics' own AP
code run on the same detections. (Ultralytics' full validator reports a little more because its validation NMS
keeps several labels per box; the app returns one label per box, so that is what we report.)

**15. What are the results?**
See `docs/RESULTS.md`: mAP@50, mAP@50:95, P/R/F1, per-class AP, per-language mAP and inference time for both
models on the same test pages.

**16. Major limitations?**
Subset-scale training on a laptop (the paper's full-data models reach about 0.73 mAP@50), rare labels with
little data, no OCR, PDFs are analysed one page at a time. See `docs/LIMITATIONS.md`.

**17. How is it deployed?**
One Docker image with both trained models: a Streamlit web app on port 8501 and a REST API (`POST /analyze`)
on port 8000, started together with `docker compose up --build`.

**18. Future work?**
Train on the full dataset on a cloud GPU, start from document-specialised weights (DocLayout-YOLO), add OCR
per region, multi-page processing and reading-order reconstruction.

**19. SDG relevance?**
SDG 9 (Industry, Innovation and Infrastructure): a building block for digitising government, legal and
educational documents in Indian languages.

**20. Why does the app use a model you did not train?**
The IndicDLP authors published DocLayout-YOLO trained on the full dataset (about 40x our training data, 100 epochs).
We evaluated it with exactly the same test pages and code as our own two models: it scores about three times higher.
Using the best available model for the product, while reporting our own models honestly as the comparison, is
standard practice. Our contribution is the pipeline, the evaluation, the OCR stage and the deployable system.

**21. How does the OCR work, and how do you pick the language?**
Tesseract reads the whole page once and reports each word with its position; each word is placed in the detected box
that contains it, and boxes are put in reading order. To pick the language, one text block is read with each script's
model and the most confident reading wins (88% right on test pages; Tesseract's own detector got 58%).

**22. Why is this genuinely deep learning and not an API wrapper?**
We build the dataset pipeline, fine-tune two CNN detectors with backpropagation on real annotated pages,
and evaluate them with standard detection metrics on held-out data. No external AI API is called.
