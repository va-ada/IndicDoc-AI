# IndicDoc AI — Academic Report Content

## 1. Abstract

IndicDoc AI addresses the challenge of automated document layout understanding for multilingual Indian documents. We develop a deep learning-based system that detects and classifies 42 types of document structural elements (paragraphs, tables, headers, figures, etc.) in scanned document images across 12 Indian languages. Using a language-balanced subset of the IndicDLP dataset (AIKosh / Hugging Face; 119,806 annotated pages in total), we implement and compare two object detection architectures: Faster R-CNN with ResNet-50 FPN (baseline) and YOLOv8 (proposed model). Both models leverage transfer learning from COCO-pretrained weights. The system includes a complete pipeline from data preprocessing and augmentation through model training, evaluation, and an interactive Streamlit-based demo application. This work contributes to SDG 9 (Industry, Innovation and Infrastructure) by advancing AI-based document digitization infrastructure for Indian languages.

## 2. Introduction

Document layout analysis is a fundamental task in document understanding that involves identifying and classifying the structural components of document pages. With India's rich diversity of languages and scripts, automated document processing faces unique challenges: multiple scripts, varied document formats, and diverse layout conventions.

The proliferation of digital government services, educational platforms, and administrative systems in India has created an urgent need for automated document understanding tools that can handle multilingual content. Traditional rule-based approaches are insufficient for this diversity, making deep learning-based solutions essential.

This project develops IndicDoc AI, a deep learning system for document layout detection specifically designed for Indian multilingual documents. We leverage the IndicDLP dataset, a large-scale benchmark hosted on AIKosh (India's national AI dataset platform), and compare two state-of-the-art object detection approaches.

## 3. Problem Statement

Given a scanned document image in any of 12 Indian languages, automatically detect and classify all structural elements (paragraphs, tables, headers, footers, figures, captions, etc.) with their spatial locations (bounding boxes) and element types (class labels) with associated confidence scores.

**Input**: Document image (scanned/photographed)
**Output**: Set of (bounding box, class label, confidence score) tuples

## 4. Motivation

- Over 1.4 billion people in India use documents in 22+ officially recognized languages
- Government digitization initiatives require automated document processing
- Manual document analysis is time-consuming and error-prone
- Existing document analysis tools are primarily designed for English/Latin script documents
- AIKosh provides authentic Indian document datasets enabling focused research

## 5. Objectives

1. Build a robust deep learning pipeline for document layout detection on multilingual Indian documents
2. Implement a baseline model (Faster R-CNN) and a proposed model (YOLOv8) for comparative analysis
3. Evaluate detection performance across 42 document element classes using standard metrics (mAP, Precision, Recall, F1)
4. Develop an interactive demonstration application for real-time document analysis
5. Perform error analysis to understand model limitations and failure modes

## 6. Dataset

### IndicDLP (Indic Document Layout Parsing)

- **Source**: AIKosh (aikosh.indiaai.gov.in) / HuggingFace (IndicDLP/IndicDLP-dataset)
- **Origin**: AI4Bharat research collaboration
- **Size**: 119,806 document images
- **Languages**: 12 — Assamese, Bengali, English, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu
- **Domains**: 12 — Novels, Textbooks, Magazines, Acts & Rules, Research Papers, Manuals, Brochures, Syllabi, Question Papers, Notices, Forms, Newspapers
- **Annotations**: COCO-style JSON with bounding box coordinates
- **Classes**: 42 physical and logical layout classes

## 7. Data Preprocessing

1. **Subset construction**: a language-balanced subset of IndicDLP (about 200 training, 30 validation and 60 test
   pages per language) built from the official splits by `scripts/prepare_indicdlp.py`; boxes clipped to the page,
   degenerate boxes dropped (details in `docs/DATASET.md`)
2. **Resizing**: longest side to 640 px with the aspect ratio kept (Faster R-CNN); 640 px letterbox (YOLOv8)
3. **Normalisation**: pixels scaled to [0, 1]; Faster R-CNN applies ImageNet mean/std inside the model
4. **Augmentation** (training only): colour jitter and Gaussian blur (Faster R-CNN); mosaic, HSV and
   translate/scale (YOLOv8). Horizontal flips are disabled because mirrored text is not a realistic page
5. **Optional page cleaning** (not used in training): CLAHE contrast enhancement and deskewing

## 8. Methodology

The methodology follows a standard object detection pipeline:

1. **Data Loading**: Custom PyTorch Dataset class parses COCO-format annotations
2. **Preprocessing**: Albumentations-based transform pipeline handles both image and bounding box transformations
3. **Model Selection**: Two architectures selected based on the detection task requirements
4. **Transfer Learning**: Both models initialized with COCO-pretrained weights, detection heads modified for 42 classes
5. **Training**: PyTorch training loop with mixed precision support, learning rate scheduling, and checkpoint saving
6. **Evaluation**: IoU-based matching between predictions and ground truth, per-class AP computation
7. **Error Analysis**: Investigation of failure modes across document types, languages, and element sizes

## 9. Model Architecture

### Baseline: Faster R-CNN (ResNet-50 FPN)
- **Two-stage detector**: Region Proposal Network (RPN) generates proposals, followed by classification and regression
- **Backbone**: ResNet-50 with Feature Pyramid Network for multi-scale features
- **Detection Head**: FastRCNNPredictor modified for 43 classes (42 + background)
- **Anchor Sizes**: Default (32, 64, 128, 256, 512) with aspect ratios (0.5, 1.0, 2.0)

### Proposed: YOLOv8s
- **Single-stage detector**: Directly predicts bounding boxes and class probabilities
- **Backbone**: CSPDarknet with C2f modules
- **Neck**: PANet feature aggregation
- **Head**: Decoupled head for classification and regression
- **Advantages**: Faster inference, more suitable for real-time applications

## 10. Training

| Parameter | Faster R-CNN | YOLOv8s |
|-----------|-------------|---------|
| Optimizer | SGD (momentum 0.9, weight decay 5e-4) | Chosen by Ultralytics (AdamW for this run length) |
| Learning rate | 0.005, cosine schedule | Chosen by Ultralytics, with warm-up |
| Batch size | 4 | 8 |
| Epochs | As many as fit a 110-minute budget (see `docs/RESULTS.md`) | As many as fit a 1-hour budget |
| Input size | 640 px longest side | 640 px |
| Hardware | Apple M1 Pro GPU (PyTorch MPS) | Apple M1 Pro GPU (PyTorch MPS) |
| Checkpoint selection | Lowest validation loss | Best validation fitness |
| Random seed | 42 | 42 |

## 11. Evaluation Metrics

- **IoU (Intersection over Union)**: Measures overlap between predicted and ground truth bounding boxes
- **Precision**: TP / (TP + FP) — correctness of positive predictions
- **Recall**: TP / (TP + FN) — completeness of ground truth detection
- **F1-Score**: Harmonic mean of Precision and Recall
- **mAP@50**: Mean AP at IoU threshold of 0.50
- **mAP@50:95**: Mean AP averaged over IoU thresholds from 0.50 to 0.95 (step 0.05)
- **Inference Time**: Average time per image for model prediction

## 12. Results

All results are measured on held-out IndicDLP test pages and are listed in full in `docs/RESULTS.md`
(generated from `outputs/metrics/` by `scripts/build_report.py`): overall mAP@50, mAP@50:95, precision, recall,
F1 and IoU for both models, AP per label, mAP per language, inference time, and training curves. The evaluator is
validated by hand-worked unit tests and cross-checked against Ultralytics' own AP code on the same detections (`docs/EVALUATION.md`).

## 13. Baseline Comparison

The comparison evaluates:
- Detection accuracy (mAP, Precision, Recall)
- Inference speed (ms per image)
- Model size and complexity
- Per-class performance differences
- Speed-accuracy trade-off analysis

## 14. Error Analysis

Error categories investigated:
1. **False Negatives**: Missed small elements, low-contrast elements, occluded elements
2. **False Positives**: Background noise, decorative elements, scanning artifacts
3. **Localization Errors**: Imprecise bounding boxes, partial element coverage
4. **Classification Errors**: Similar class confusion (e.g., Header vs Title, Figure vs Image)
5. **Script-specific Errors**: Performance variation across different Indian scripts

## 15. SDG Relevance

**SDG 9 — Industry, Innovation and Infrastructure**

- **Document Digitization**: Enables automated processing of government, educational, and legal documents in Indian languages
- **AI Infrastructure**: Contributes to building AI capabilities for Indian language processing
- **Innovation**: Applies state-of-the-art deep learning techniques to a real-world Indian problem
- **Automation**: Reduces manual effort and improves efficiency in document management workflows
- **Digital Inclusion**: Supports multilingual document processing, enabling digital services for non-English-speaking populations

This project does not exaggerate its social impact. It is a focused technical contribution to document understanding infrastructure.

## 16. Limitations

1. Training on the full IndicDLP dataset (119K images) requires significant GPU resources beyond typical student hardware
2. Models use transfer learning from COCO rather than being trained from scratch on document data
3. Limited hyperparameter search due to computational constraints
4. OCR is not integrated — the system detects layout elements but does not extract text
5. Some of the 42 classes may have imbalanced representation, affecting per-class performance
6. Evaluation on real-world documents outside the IndicDLP distribution may show lower performance
7. The system assumes document images are roughly upright; heavily rotated documents may require additional preprocessing

## 17. Future Scope

1. **Full Dataset Training**: Train on the complete IndicDLP dataset (119,806 images) with GPU resources
2. **OCR Integration**: Integrate PaddleOCR or Tesseract for end-to-end document understanding
3. **LayoutLMv3**: Explore multimodal models that jointly process layout and text
4. **Document Structure**: Reconstruct hierarchical document structure (sections, subsections) from detected elements
5. **Real-time Processing**: Optimize for real-time inference on edge devices
6. **Mobile Deployment**: Convert models to ONNX/TFLite for mobile deployment
7. **Handwritten Documents**: Extend to handwritten document analysis
8. **Active Learning**: Implement active learning for efficient dataset expansion

## 18. Conclusion

IndicDoc AI demonstrates the feasibility of deep learning-based document layout detection for multilingual Indian documents. By leveraging the IndicDLP dataset from AIKosh and comparing Faster R-CNN and YOLOv8 architectures, the project provides a complete pipeline from data preprocessing through model training, evaluation, and interactive demonstration. The system's modular design allows for future extension with OCR integration, additional model architectures, and deployment optimization. This work contributes to the growing ecosystem of AI tools for Indian language document processing, aligned with SDG 9's goals of building resilient infrastructure and promoting innovation.

## 19. References

1. AI4Bharat. "IndicDLP: A Large-Scale Multilingual Document Layout Parsing Dataset." AIKosh / HuggingFace.
2. Ren, S., He, K., Girshick, R., & Sun, J. (2015). "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks." NeurIPS 2015.
3. Jocher, G., Chaurasia, A., & Qiu, J. (2023). "Ultralytics YOLOv8." GitHub.
4. Lin, T.Y., Maire, M., Belongie, S., et al. (2014). "Microsoft COCO: Common Objects in Context." ECCV 2014.
5. He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition." CVPR 2016.
6. Lin, T.Y., Dollar, P., Girshick, R., He, K., Hariharan, B., & Belongie, S. (2017). "Feature Pyramid Networks for Object Detection." CVPR 2017.
7. Ministry of Electronics and Information Technology. "AIKosh — IndiaAI Datasets Platform." Government of India.
8. Zhong, X., Tang, J., & Yepes, A.J. (2019). "PubLayNet: Largest Dataset Ever for Document Layout Analysis." ICDAR 2019.
9. Xu, Y., Xu, Y., Lv, T., et al. (2022). "LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking." ACM MM 2022.
