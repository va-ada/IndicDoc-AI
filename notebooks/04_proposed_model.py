"""
04 - Proposed Model (YOLOv8)
IndicDoc AI - Document Layout Detection

This notebook demonstrates the proposed YOLOv8 model:
- Model architecture overview
- COCO to YOLO format conversion
- Model initialization
- Architecture comparison with baseline

Note: Full YOLOv8 training uses the ultralytics training framework.
This notebook shows the setup and conversion steps.
"""
import sys
import json
import numpy as np
from pathlib import Path

import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path('.').resolve()))
from ml.config import (
    SAMPLE_RUN_DIR, PROCESSED_DATA_DIR, MODEL_DIR, METRICS_DIR, PLOTS_DIR,
    NUM_CLASSES, CLASS_NAMES, IMG_SIZE
)

# Demo notebook on synthetic sample data: keep its outputs away from real results
MODEL_DIR, METRICS_DIR, PLOTS_DIR, PREDICTION_DIR = (SAMPLE_RUN_DIR / d for d in ('models', 'metrics', 'plots', 'predictions'))
from ml.models.proposed_yolov8 import get_proposed_model, coco_to_yolo
from ml.models.baseline_frcnn import get_baseline_model
from ml.evaluation.metrics import save_metrics_report

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ========== 1. COCO to YOLO Conversion ==========
print("=" * 60)
print("1. COCO TO YOLO FORMAT CONVERSION")
print("=" * 60)

ann_file = PROCESSED_DATA_DIR / 'sample' / 'annotations.json'
img_dir = PROCESSED_DATA_DIR / 'sample' / 'images'
label_dir = SAMPLE_RUN_DIR / 'labels'

coco_to_yolo(str(ann_file), str(label_dir))

txt_files = list(label_dir.glob('*.txt'))
print(f"Converted {len(txt_files)} YOLO label files")

# Show example
if txt_files:
    print(f"\nExample label file ({txt_files[0].name}):")
    with open(txt_files[0]) as f:
        for i, line in enumerate(f.readlines()[:5]):
            print(f"  {line.strip()}")
        print("  ...")

# ========== 2. Model Architecture ==========
print("\n" + "=" * 60)
print("2. YOLOV8 MODEL ARCHITECTURE")
print("=" * 60)

try:
    yolo_model = get_proposed_model(model_size='n', weights_dir=str(MODEL_DIR))  # nano for quick demo
    print(f"YOLOv8n model loaded successfully")
    print(f"Model task: {yolo_model.task}")
    
    # Model info
    yolo_params = sum(p.numel() for p in yolo_model.model.parameters())
    print(f"Total parameters: {yolo_params:,}")
    print(f"Model size: ~{yolo_params * 4 / 1e6:.1f} MB (float32)")
except Exception as e:
    print(f"YOLOv8 model loading skipped: {e}")
    yolo_params = 3_200_000  # Approximate for YOLOv8n

# ========== 3. Architecture Comparison ==========
print("\n" + "=" * 60)
print("3. ARCHITECTURE COMPARISON")
print("=" * 60)

# Baseline parameters
baseline_model = get_baseline_model(NUM_CLASSES + 1, pretrained=False)
baseline_params = sum(p.numel() for p in baseline_model.parameters())
del baseline_model

comparison = {
    'Faster R-CNN': {
        'type': 'Two-stage detector',
        'backbone': 'ResNet-50 + FPN',
        'parameters': baseline_params,
        'size_mb': round(baseline_params * 4 / 1e6, 1),
        'pretrained': 'COCO (torchvision)',
        'strengths': 'Higher accuracy on small objects',
        'weaknesses': 'Slower inference'
    },
    'YOLOv8n': {
        'type': 'Single-stage detector',
        'backbone': 'CSPDarknet + C2f',
        'parameters': yolo_params,
        'size_mb': round(yolo_params * 4 / 1e6, 1),
        'pretrained': 'COCO (ultralytics)',
        'strengths': 'Fast inference, efficient',
        'weaknesses': 'May miss small objects'
    }
}

for name, info in comparison.items():
    print(f"\n{name}:")
    for k, v in info.items():
        if k == 'parameters':
            print(f"  {k}: {v:,}")
        else:
            print(f"  {k}: {v}")

# Bar chart comparison
fig, ax = plt.subplots(figsize=(8, 5))
models = list(comparison.keys())
params = [comparison[m]['parameters'] / 1e6 for m in models]
ax.bar(models, params, color=['#6c757d', '#e94560'])
ax.set_ylabel('Parameters (Millions)')
ax.set_title('Model Size Comparison')
for i, v in enumerate(params):
    ax.text(i, v + 0.5, f'{v:.1f}M', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(str(PLOTS_DIR / 'model_comparison.png'), dpi=150)
print(f"\nSaved comparison plot to {PLOTS_DIR / 'model_comparison.png'}")
plt.close()

# Save comparison metrics
save_metrics_report(comparison, str(METRICS_DIR / 'model_comparison.json'))

print("\n" + "=" * 60)
print("PROPOSED MODEL ANALYSIS COMPLETE")
print("=" * 60)
