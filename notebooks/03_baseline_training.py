"""
03 - Baseline Model Training
IndicDoc AI - Document Layout Detection

This notebook demonstrates training the baseline Faster R-CNN model:
- Model architecture
- Training loop on sample data
- Loss tracking
- Checkpoint saving
- Validation
"""
import sys
import json
import random
import time
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path('.').resolve()))
from ml.config import (
    SAMPLE_RUN_DIR, PROCESSED_DATA_DIR, MODEL_DIR, METRICS_DIR, PLOTS_DIR,
    NUM_CLASSES, CLASS_NAMES, BATCH_SIZE, SEED, IMG_SIZE, LEARNING_RATE
)

# Demo notebook on synthetic sample data: keep its outputs away from real results
MODEL_DIR, METRICS_DIR, PLOTS_DIR, PREDICTION_DIR = (SAMPLE_RUN_DIR / d for d in ('models', 'metrics', 'plots', 'predictions'))
from ml.datasets.coco_dataset import COCODataset, collate_fn
from ml.models.baseline_frcnn import get_baseline_model
from ml.evaluation.metrics import save_metrics_report

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ========== 1. Setup ==========
print("=" * 60)
print("1. SETUP")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

ann_file = PROCESSED_DATA_DIR / 'sample' / 'annotations.json'
img_dir = PROCESSED_DATA_DIR / 'sample' / 'images'

dataset = COCODataset(str(ann_file), str(img_dir))
print(f"Dataset size: {len(dataset)}")

# Split into train/val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                          generator=torch.Generator().manual_seed(SEED))

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# ========== 2. Model ==========
print("\n" + "=" * 60)
print("2. MODEL ARCHITECTURE")
print("=" * 60)

num_classes = NUM_CLASSES + 1  # +1 for background
model = get_baseline_model(num_classes=num_classes, pretrained=False)
model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model size: ~{total_params * 4 / 1e6:.1f} MB (float32)")

# ========== 3. Training ==========
print("\n" + "=" * 60)
print("3. TRAINING")
print("=" * 60)

NUM_EPOCHS = 3  # Small number for demo
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

train_losses = []
val_losses = []

for epoch in range(NUM_EPOCHS):
    # Train
    model.train()
    epoch_loss = 0
    num_batches = 0
    
    for images, targets in train_loader:
        images = [img.to(device) if isinstance(img, torch.Tensor) else 
                  torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div(255.0).to(device)
                  for img in images]
        
        # Fix labels: Faster RCNN expects labels >= 1 (0 is background)
        fixed_targets = []
        for t in targets:
            new_t = {k: v.to(device) for k, v in t.items()}
            new_t['labels'] = new_t['labels'] + 1  # Shift labels to start from 1
            fixed_targets.append(new_t)
        
        try:
            loss_dict = model(images, fixed_targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            num_batches += 1
        except Exception as e:
            print(f"  Batch error: {e}")
            continue
    
    avg_train_loss = epoch_loss / max(num_batches, 1)
    train_losses.append(avg_train_loss)
    scheduler.step()
    
    # Simple validation loss
    model.train()  # FRCNN needs train mode to compute losses
    val_loss = 0
    val_batches = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) if isinstance(img, torch.Tensor) else
                      torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div(255.0).to(device)
                      for img in images]
            fixed_targets = []
            for t in targets:
                new_t = {k: v.to(device) for k, v in t.items()}
                new_t['labels'] = new_t['labels'] + 1
                fixed_targets.append(new_t)
            try:
                loss_dict = model(images, fixed_targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
                val_batches += 1
            except Exception:
                continue
    
    avg_val_loss = val_loss / max(val_batches, 1)
    val_losses.append(avg_val_loss)
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

# Save model
model_path = MODEL_DIR / 'baseline_frcnn_best.pth'
torch.save(model.state_dict(), model_path)
print(f"\nModel saved to {model_path}")

# ========== 4. Loss Curves ==========
print("\n" + "=" * 60)
print("4. TRAINING CURVES")
print("=" * 60)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, len(train_losses)+1), train_losses, 'b-o', label='Train Loss')
ax.plot(range(1, len(val_losses)+1), val_losses, 'r-o', label='Val Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Baseline (Faster R-CNN) Training Curves')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(str(PLOTS_DIR / 'baseline_training_loss.png'), dpi=150)
print(f"Saved training curves to {PLOTS_DIR / 'baseline_training_loss.png'}")
plt.close()

# Save metrics
metrics = {
    'model': 'Faster R-CNN (ResNet-50 FPN)',
    'epochs': NUM_EPOCHS,
    'final_train_loss': train_losses[-1],
    'final_val_loss': val_losses[-1],
    'train_losses': train_losses,
    'val_losses': val_losses,
    'total_params': total_params,
    'trainable_params': trainable_params,
    'device': str(device),
    'note': 'Trained on synthetic sample data for pipeline verification'
}
save_metrics_report(metrics, str(METRICS_DIR / 'baseline_training.json'))
print(f"Saved metrics to {METRICS_DIR / 'baseline_training.json'}")

print("\n" + "=" * 60)
print("BASELINE TRAINING COMPLETE")
print("=" * 60)
