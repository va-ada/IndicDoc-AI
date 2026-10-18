"""
01 - Dataset Analysis
IndicDoc AI - Document Layout Detection

This notebook analyzes the IndicDLP dataset:
- Dataset loading and statistics
- Class distribution
- Language/script distribution
- Sample image visualization
- Annotation visualization
- Quality checks
"""
import sys
import json
import os
import numpy as np
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Setup
sys.path.insert(0, str(Path('.').resolve()))
from ml.config import SAMPLE_RUN_DIR, PROCESSED_DATA_DIR, CLASS_NAMES, PLOTS_DIR

# Demo notebook on synthetic sample data: keep its outputs away from real results
MODEL_DIR, METRICS_DIR, PLOTS_DIR, PREDICTION_DIR = (SAMPLE_RUN_DIR / d for d in ('models', 'metrics', 'plots', 'predictions'))

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ========== 1. Load Dataset ==========
print("=" * 60)
print("1. LOADING DATASET")
print("=" * 60)

ann_file = PROCESSED_DATA_DIR / 'sample' / 'annotations.json'
img_dir = PROCESSED_DATA_DIR / 'sample' / 'images'

with open(ann_file, 'r') as f:
    coco = json.load(f)

print(f"Number of images: {len(coco['images'])}")
print(f"Number of annotations: {len(coco['annotations'])}")
print(f"Number of categories: {len(coco['categories'])}")

# ========== 2. Dataset Statistics ==========
print("\n" + "=" * 60)
print("2. DATASET STATISTICS")
print("=" * 60)

# Image sizes
widths = [img['width'] for img in coco['images']]
heights = [img['height'] for img in coco['images']]
print(f"Image width  - Min: {min(widths)}, Max: {max(widths)}, Mean: {np.mean(widths):.0f}")
print(f"Image height - Min: {min(heights)}, Max: {max(heights)}, Mean: {np.mean(heights):.0f}")

# Annotations per image
anns_per_img = Counter(ann['image_id'] for ann in coco['annotations'])
print(f"\nAnnotations per image:")
print(f"  Min: {min(anns_per_img.values())}")
print(f"  Max: {max(anns_per_img.values())}")
print(f"  Mean: {np.mean(list(anns_per_img.values())):.1f}")

# ========== 3. Class Distribution ==========
print("\n" + "=" * 60)
print("3. CLASS DISTRIBUTION")
print("=" * 60)

cat_map = {cat['id']: cat['name'] for cat in coco['categories']}
class_counts = Counter(ann['category_id'] for ann in coco['annotations'])

print(f"\n{'Class':<25} {'Count':>6}")
print("-" * 35)
for cat_id, count in sorted(class_counts.items(), key=lambda x: -x[1]):
    name = cat_map.get(cat_id, f'Unknown_{cat_id}')
    print(f"{name:<25} {count:>6}")

# Plot class distribution
fig, ax = plt.subplots(figsize=(14, 6))
names = [cat_map.get(cid, str(cid)) for cid in class_counts.keys()]
counts = list(class_counts.values())
ax.bar(range(len(names)), counts, color='steelblue')
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
ax.set_xlabel('Class')
ax.set_ylabel('Count')
ax.set_title('Class Distribution in Dataset')
plt.tight_layout()
plt.savefig(str(PLOTS_DIR / 'class_distribution.png'), dpi=150)
print(f"\nSaved class distribution plot to {PLOTS_DIR / 'class_distribution.png'}")
plt.close()

# ========== 4. Bounding Box Statistics ==========
print("\n" + "=" * 60)
print("4. BOUNDING BOX STATISTICS")
print("=" * 60)

box_widths = [ann['bbox'][2] for ann in coco['annotations']]
box_heights = [ann['bbox'][3] for ann in coco['annotations']]
areas = [ann.get('area', ann['bbox'][2] * ann['bbox'][3]) for ann in coco['annotations']]

print(f"Box width  - Min: {min(box_widths):.0f}, Max: {max(box_widths):.0f}, Mean: {np.mean(box_widths):.0f}")
print(f"Box height - Min: {min(box_heights):.0f}, Max: {max(box_heights):.0f}, Mean: {np.mean(box_heights):.0f}")
print(f"Area       - Min: {min(areas):.0f}, Max: {max(areas):.0f}, Mean: {np.mean(areas):.0f}")

# Box size distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(box_widths, bins=20, color='steelblue', edgecolor='white')
axes[0].set_title('Box Width Distribution')
axes[0].set_xlabel('Width (px)')
axes[1].hist(box_heights, bins=20, color='coral', edgecolor='white')
axes[1].set_title('Box Height Distribution')
axes[1].set_xlabel('Height (px)')
axes[2].hist(areas, bins=20, color='seagreen', edgecolor='white')
axes[2].set_title('Box Area Distribution')
axes[2].set_xlabel('Area (px²)')
plt.tight_layout()
plt.savefig(str(PLOTS_DIR / 'bbox_statistics.png'), dpi=150)
print(f"Saved bbox statistics plot to {PLOTS_DIR / 'bbox_statistics.png'}")
plt.close()

# ========== 5. Sample Visualization ==========
print("\n" + "=" * 60)
print("5. SAMPLE VISUALIZATION")
print("=" * 60)

# Show first 4 images with annotations
fig, axes = plt.subplots(2, 2, figsize=(12, 15))
colors = plt.cm.Set3(np.linspace(0, 1, len(coco['categories'])))

for idx, ax in enumerate(axes.flatten()):
    if idx >= len(coco['images']):
        break
    img_info = coco['images'][idx]
    img = Image.open(img_dir / img_info['file_name'])
    ax.imshow(img)
    
    img_anns = [a for a in coco['annotations'] if a['image_id'] == img_info['id']]
    for ann in img_anns:
        x, y, w, h = ann['bbox']
        color = colors[ann['category_id'] % len(colors)]
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        label = cat_map.get(ann['category_id'], str(ann['category_id']))
        ax.text(x, y - 2, label, fontsize=6, color='white',
                bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=1))
    
    ax.set_title(f"Image: {img_info['file_name']} ({len(img_anns)} annotations)")
    ax.axis('off')

plt.tight_layout()
plt.savefig(str(PLOTS_DIR / 'sample_annotations.png'), dpi=150)
print(f"Saved sample annotations to {PLOTS_DIR / 'sample_annotations.png'}")
plt.close()

# ========== 6. Quality Checks ==========
print("\n" + "=" * 60)
print("6. QUALITY CHECKS")
print("=" * 60)

# Check for images without annotations
imgs_with_anns = set(ann['image_id'] for ann in coco['annotations'])
all_img_ids = set(img['id'] for img in coco['images'])
imgs_without_anns = all_img_ids - imgs_with_anns
print(f"Images with annotations: {len(imgs_with_anns)}")
print(f"Images without annotations: {len(imgs_without_anns)}")

# Check for negative/zero bbox dimensions
invalid_boxes = [ann for ann in coco['annotations'] if ann['bbox'][2] <= 0 or ann['bbox'][3] <= 0]
print(f"Invalid bounding boxes (w<=0 or h<=0): {len(invalid_boxes)}")

# Check for annotations outside image bounds
out_of_bounds = 0
img_dict = {img['id']: img for img in coco['images']}
for ann in coco['annotations']:
    img = img_dict[ann['image_id']]
    x, y, w, h = ann['bbox']
    if x < 0 or y < 0 or x + w > img['width'] or y + h > img['height']:
        out_of_bounds += 1
print(f"Annotations out of image bounds: {out_of_bounds}")

print("\n" + "=" * 60)
print("DATASET ANALYSIS COMPLETE")
print("=" * 60)
