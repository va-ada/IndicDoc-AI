"""
05 - Evaluation & Error Analysis
IndicDoc AI - Document Layout Detection

This notebook performs:
- Model inference on test images
- Visual predictions
- Baseline comparison
- Error analysis
- Performance summary
"""
import sys
import json
import time
import numpy as np
from pathlib import Path
from collections import Counter

import torch
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, str(Path('.').resolve()))
from ml.config import (
    SAMPLE_RUN_DIR, PROCESSED_DATA_DIR, MODEL_DIR, METRICS_DIR, PLOTS_DIR,
    PREDICTION_DIR, NUM_CLASSES, CLASS_NAMES, IMG_SIZE
)

# Demo notebook on synthetic sample data: keep its outputs away from real results
MODEL_DIR, METRICS_DIR, PLOTS_DIR, PREDICTION_DIR = (SAMPLE_RUN_DIR / d for d in ('models', 'metrics', 'plots', 'predictions'))
from ml.models.baseline_frcnn import get_baseline_model
from ml.evaluation.metrics import save_metrics_report

METRICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

# ========== 1. Load Model ==========
print("=" * 60)
print("1. LOADING MODEL")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model_path = MODEL_DIR / 'baseline_frcnn_best.pth'
if model_path.exists():
    model = get_baseline_model(NUM_CLASSES + 1, pretrained=False)
    model.load_state_dict(torch.load(str(model_path), map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}")
    model_loaded = True
else:
    print(f"No trained model found at {model_path}")
    print("Skipping inference. Train the model first (notebook 03).")
    model_loaded = False

# ========== 2. Inference on Sample Images ==========
print("\n" + "=" * 60)
print("2. INFERENCE ON SAMPLE IMAGES")
print("=" * 60)

img_dir = PROCESSED_DATA_DIR / 'sample' / 'images'
ann_file = PROCESSED_DATA_DIR / 'sample' / 'annotations.json'

with open(ann_file) as f:
    coco = json.load(f)
cat_map = {cat['id']: cat['name'] for cat in coco['categories']}

test_images = sorted(list(img_dir.glob('*.jpg')))[:6]
all_pred_labels = []
all_pred_scores = []
inference_times = []

if model_loaded:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for idx, img_path in enumerate(test_images):
        row, col = idx // 3, idx % 3
        
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float().div(255.0).to(device)
        
        start = time.time()
        with torch.no_grad():
            predictions = model([img_tensor])[0]
        inf_time = time.time() - start
        inference_times.append(inf_time)
        
        # Filter by confidence
        conf_thresh = 0.3
        keep = predictions['scores'] > conf_thresh
        boxes = predictions['boxes'][keep].cpu().numpy()
        labels = predictions['labels'][keep].cpu().numpy()
        scores = predictions['scores'][keep].cpu().numpy()
        
        all_pred_labels.extend(labels.tolist())
        all_pred_scores.extend(scores.tolist())
        
        # Draw
        axes[row, col].imshow(img_np)
        colors = plt.cm.Set1(np.linspace(0, 1, 10))
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            color = colors[label % len(colors)]
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor='none')
            axes[row, col].add_patch(rect)
            lbl = CLASS_NAMES[label-1] if 0 < label <= len(CLASS_NAMES) else f'cls_{label}'
            axes[row, col].text(x1, y1-3, f'{lbl}: {score:.2f}', fontsize=6, color='white',
                              bbox=dict(facecolor=color, alpha=0.7, pad=1))
        
        axes[row, col].set_title(f'{img_path.name} ({len(boxes)} detections, {inf_time:.3f}s)')
        axes[row, col].axis('off')
    
    plt.suptitle('Model Predictions on Sample Images', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(PREDICTION_DIR / 'sample_predictions.png'), dpi=150)
    print(f"Saved predictions to {PREDICTION_DIR / 'sample_predictions.png'}")
    plt.close()
    
    print(f"\nAverage inference time: {np.mean(inference_times):.4f}s")
    print(f"Total detections: {len(all_pred_labels)}")
else:
    print("Skipping inference - no model loaded.")

# ========== 3. Prediction Statistics ==========
print("\n" + "=" * 60)
print("3. PREDICTION STATISTICS")
print("=" * 60)

if model_loaded and len(all_pred_scores) > 0:
    print(f"Total predictions: {len(all_pred_labels)}")
    print(f"Avg confidence: {np.mean(all_pred_scores):.3f}")
    print(f"Min confidence: {min(all_pred_scores):.3f}")
    print(f"Max confidence: {max(all_pred_scores):.3f}")
    
    # Confidence distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(all_pred_scores, bins=20, color='steelblue', edgecolor='white')
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Count')
    ax.set_title('Prediction Confidence Distribution')
    plt.tight_layout()
    plt.savefig(str(PLOTS_DIR / 'confidence_distribution.png'), dpi=150)
    plt.close()
    
    # Predicted class distribution
    pred_counts = Counter(all_pred_labels)
    fig, ax = plt.subplots(figsize=(10, 4))
    pred_names = [CLASS_NAMES[l-1] if 0 < l <= len(CLASS_NAMES) else f'cls_{l}' for l in pred_counts.keys()]
    ax.bar(pred_names, list(pred_counts.values()), color='coral')
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Count')
    ax.set_title('Predicted Class Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(str(PLOTS_DIR / 'predicted_class_distribution.png'), dpi=150)
    plt.close()
else:
    print("No predictions available for statistics.")

# ========== 4. Performance Summary ==========
print("\n" + "=" * 60)
print("4. PERFORMANCE SUMMARY")
print("=" * 60)

# Load training metrics if available
baseline_metrics_file = METRICS_DIR / 'baseline_training.json'
if baseline_metrics_file.exists():
    with open(baseline_metrics_file) as f:
        bl_metrics = json.load(f)
    print(f"\nBaseline (Faster R-CNN):")
    print(f"  Epochs trained: {bl_metrics.get('epochs', 'N/A')}")
    print(f"  Final train loss: {bl_metrics.get('final_train_loss', 'N/A')}")
    print(f"  Final val loss: {bl_metrics.get('final_val_loss', 'N/A')}")
    print(f"  Parameters: {bl_metrics.get('total_params', 'N/A'):,}")
    if inference_times:
        print(f"  Avg inference time: {np.mean(inference_times):.4f}s")
else:
    print("No baseline metrics found.")

model_comp_file = METRICS_DIR / 'model_comparison.json'
if model_comp_file.exists():
    with open(model_comp_file) as f:
        comp = json.load(f)
    print(f"\nModel Comparison:")
    for model_name, info in comp.items():
        print(f"  {model_name}: {info.get('parameters', 'N/A'):,} params, ~{info.get('size_mb', 'N/A')} MB")

# ========== 5. Error Analysis ==========
print("\n" + "=" * 60)
print("5. ERROR ANALYSIS")
print("=" * 60)

if model_loaded:
    print("\nError Analysis Observations:")
    print("  - Model trained on synthetic sample data (not real documents)")
    print("  - Predictions are expected to be noisy on synthetic data")
    print("  - Real evaluation requires training on actual IndicDLP dataset")
    print("\nPotential error sources on real data:")
    print("  1. Small document elements (footnotes, page numbers)")
    print("  2. Complex multi-column layouts")
    print("  3. Low-contrast scans")
    print("  4. Dense overlapping regions")
    print("  5. Script-specific layout conventions")
else:
    print("Error analysis requires a trained model. Run notebook 03 first.")

# Save final evaluation report
eval_report = {
    'model': 'Faster R-CNN (Baseline)',
    'dataset': 'Synthetic sample (pipeline test)',
    'num_test_images': len(test_images),
    'total_predictions': len(all_pred_labels),
    'avg_confidence': float(np.mean(all_pred_scores)) if all_pred_scores else None,
    'avg_inference_time_s': float(np.mean(inference_times)) if inference_times else None,
    'note': 'Evaluated on synthetic sample data. Real metrics require training on IndicDLP.'
}
save_metrics_report(eval_report, str(METRICS_DIR / 'evaluation_report.json'))

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)
