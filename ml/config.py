import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'
MODEL_DIR = PROJECT_ROOT / 'outputs' / 'models'
PREDICTION_DIR = OUTPUT_DIR / 'predictions'
METRICS_DIR = OUTPUT_DIR / 'metrics'
PLOTS_DIR = OUTPUT_DIR / 'plots'
BASELINE_CHECKPOINT = MODEL_DIR / 'baseline_frcnn_best.pth'
PROPOSED_CHECKPOINT = MODEL_DIR / 'yolov8_best.pt'
# DocLayout-YOLO trained on the full IndicDLP training set by the dataset's authors (Zenodo 15881917, CC-BY 4.0)
AUTHORS_CHECKPOINT = MODEL_DIR / 'indicdlp_authors' / 'doclayout_yolo_indicdlp.pt'
AUTHORS_IMG_SIZE = 1024  # resolution it was trained at
# Demo notebooks and the sample-data pipeline write here, never over real results
SAMPLE_RUN_DIR = OUTPUT_DIR / 'sample_run'

# Dataset config
NUM_CLASSES = 42  # IndicDLP classes
# Official IndicDLP labels, in class-id order (0..41), from AI4Bharat/IndicDLP data/indic.yaml
CLASS_NAMES = [
    'advertisement', 'answer', 'author', 'chapter-title', 'contact-info', 'dateline',
    'figure', 'figure-caption', 'first-level-question', 'flag', 'folio', 'footer',
    'footnote', 'formula', 'header', 'headline', 'index', 'jumpline',
    'options', 'ordered-list', 'page-number', 'paragraph', 'placeholder-text', 'quote',
    'reference', 'second-level-question', 'section-title', 'sidebar', 'sub-headline', 'sub-ordered-list',
    'sub-section-title', 'subsub-ordered-list', 'subsub-section-title', 'sub-unordered-list', 'subsub-headline', 'subsub-unordered-list',
    'table', 'table-caption', 'table-of-contents', 'third-level-question', 'unordered-list', 'website-link',
]

# Training config
BATCH_SIZE = 4
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
IMG_SIZE = 640
SEED = 42
NUM_WORKERS = 2
MIXED_PRECISION = True
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Model config
BASELINE_MODEL = 'fasterrcnn'  
PROPOSED_MODEL = 'yolov8'
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5


def get_device() -> str:
    """Fastest available device: NVIDIA GPU, then Apple Silicon GPU (MPS), then CPU."""
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'
