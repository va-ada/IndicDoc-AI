"""
02 - Preprocessing Pipeline
IndicDoc AI - Document Layout Detection

This notebook demonstrates the preprocessing pipeline:
- Image loading
- Resizing
- Normalization
- Contrast enhancement (CLAHE)
- Deskewing
- Data augmentation
- Before/after visualization
"""
import sys
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path('.').resolve()))
from ml.config import SAMPLE_RUN_DIR, PROCESSED_DATA_DIR, PLOTS_DIR, IMG_SIZE

# Demo notebook on synthetic sample data: keep its outputs away from real results
MODEL_DIR, METRICS_DIR, PLOTS_DIR, PREDICTION_DIR = (SAMPLE_RUN_DIR / d for d in ('models', 'metrics', 'plots', 'predictions'))
from ml.preprocessing.preprocess import (
    resize_image, normalize_image, enhance_contrast, 
    deskew_image, preprocess_document
)
from ml.preprocessing.transforms import (
    get_train_transforms, get_val_transforms, get_inference_transforms
)

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ========== 1. Load Sample Image ==========
print("=" * 60)
print("1. LOADING SAMPLE IMAGE")
print("=" * 60)

img_dir = PROCESSED_DATA_DIR / 'sample' / 'images'
sample_img_path = list(img_dir.glob('*.jpg'))[0]
img = np.array(Image.open(sample_img_path).convert('RGB'))
print(f"Original image shape: {img.shape}")
print(f"Original dtype: {img.dtype}")
print(f"Pixel range: [{img.min()}, {img.max()}]")

# ========== 2. Preprocessing Steps ==========
print("\n" + "=" * 60)
print("2. INDIVIDUAL PREPROCESSING STEPS")
print("=" * 60)

# Resize
resized = resize_image(img, (IMG_SIZE, IMG_SIZE))
print(f"Resized shape: {resized.shape}")

# Normalize
normalized = normalize_image(img)
print(f"Normalized range: [{normalized.min():.4f}, {normalized.max():.4f}]")

# Enhance contrast
enhanced = enhance_contrast(img)
print(f"Enhanced shape: {enhanced.shape}")

# Deskew
deskewed = deskew_image(img)
print(f"Deskewed shape: {deskewed.shape}")

# Full pipeline
full_processed = preprocess_document(img)
print(f"Full pipeline output shape: {full_processed.shape}")

# ========== 3. Visualization ==========
print("\n" + "=" * 60)
print("3. PREPROCESSING VISUALIZATION")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(img)
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

axes[0, 1].imshow(resized)
axes[0, 1].set_title(f'Resized ({IMG_SIZE}x{IMG_SIZE})')
axes[0, 1].axis('off')

axes[0, 2].imshow(normalized)
axes[0, 2].set_title('Normalized [0,1]')
axes[0, 2].axis('off')

axes[1, 0].imshow(enhanced)
axes[1, 0].set_title('CLAHE Enhanced')
axes[1, 0].axis('off')

axes[1, 1].imshow(deskewed)
axes[1, 1].set_title('Deskewed')
axes[1, 1].axis('off')

axes[1, 2].imshow(full_processed)
axes[1, 2].set_title('Full Pipeline')
axes[1, 2].axis('off')

plt.suptitle('Preprocessing Pipeline Steps', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(str(PLOTS_DIR / 'preprocessing_steps.png'), dpi=150)
print(f"Saved preprocessing visualization to {PLOTS_DIR / 'preprocessing_steps.png'}")
plt.close()

# ========== 4. Augmentation ==========
print("\n" + "=" * 60)
print("4. DATA AUGMENTATION")
print("=" * 60)

train_tf = get_train_transforms(IMG_SIZE)
val_tf = get_val_transforms(IMG_SIZE)

# Demo augmentations
sample_boxes = [[100, 100, 250, 200], [300, 300, 500, 450]]
sample_labels = [0, 1]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i in range(8):
    row, col = i // 4, i % 4
    try:
        augmented = train_tf(
            image=img,
            bboxes=sample_boxes,
            category_ids=sample_labels
        )
        aug_img = augmented['image']
        if hasattr(aug_img, 'numpy'):
            # Transforms output [0, 1] floats (Faster R-CNN normalizes internally), so no de-normalization needed
            aug_img = aug_img.permute(1, 2, 0).numpy()
        axes[row, col].imshow(aug_img)
    except Exception as e:
        axes[row, col].text(0.5, 0.5, f'Error: {str(e)[:30]}', transform=axes[row, col].transAxes, ha='center')
    axes[row, col].set_title(f'Augmentation {i+1}')
    axes[row, col].axis('off')

plt.suptitle('Training Data Augmentation Examples', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(str(PLOTS_DIR / 'augmentation_examples.png'), dpi=150)
print(f"Saved augmentation examples to {PLOTS_DIR / 'augmentation_examples.png'}")
plt.close()

print("\n" + "=" * 60)
print("PREPROCESSING PIPELINE DEMO COMPLETE")
print("=" * 60)
