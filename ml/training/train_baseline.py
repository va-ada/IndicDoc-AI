import argparse
import json
import random
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from ml.config import *
from ml.datasets.coco_dataset import COCODataset, collate_fn
from ml.preprocessing.transforms import get_train_transforms, get_val_transforms
from ml.models.baseline_frcnn import get_baseline_model

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def epochs_within_budget(max_epochs: int, budget_s, first_epoch_s: float) -> int:
    """How many epochs fit in the time budget, judging by the first one (never fewer than 1, never more than max)."""
    if not budget_s:
        return max_epochs
    return max(1, min(max_epochs, int(budget_s // first_epoch_s)))

def train_one_epoch(model, loader, optimizer, device, scaler, desc: str) -> float:
    """One pass over the training set. Returns the mean loss over the batches that were used."""
    model.train()
    total, n_batches = 0.0, 0
    params = [p for p in model.parameters() if p.requires_grad]
    progress_bar = tqdm(loader, desc=desc)

    for images, targets in progress_bar:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # A single exploding batch should not wreck the weights
        if not torch.isfinite(losses):
            print(f"  [WARN] non-finite loss, skipping batch")
            continue

        optimizer.zero_grad()
        if scaler:
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
            optimizer.step()

        total += losses.item()
        n_batches += 1
        progress_bar.set_postfix({'loss': losses.item()})

    return total / max(n_batches, 1)

@torch.no_grad()
def validation_loss(model, loader, device) -> float:
    """
    Mean loss on the validation set. torchvision detection models only return losses in
    train mode, so train mode is used under no_grad (batch-norm layers are frozen in this model).
    """
    model.train()
    total, n_batches = 0.0, 0
    for images, targets in loader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        losses = sum(loss for loss in model(images, targets).values())
        if torch.isfinite(losses):
            total += losses.item()
            n_batches += 1
    return total / max(n_batches, 1)

def main(args) -> dict:
    set_seed(SEED)
    device = torch.device(get_device())
    print(f"Using device: {device}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # label_offset=1: Faster R-CNN reserves label 0 for background
    train_dataset = COCODataset(args.train_ann, args.train_img,
                                transforms=get_train_transforms(args.img_size), label_offset=1)
    val_dataset = COCODataset(args.val_ann, args.val_img,
                              transforms=get_val_transforms(args.img_size), label_offset=1)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn)

    # Model (classes + 1 for background)
    model = get_baseline_model(num_classes=NUM_CLASSES + 1, pretrained=args.pretrained)
    model.to(device)

    # SGD with momentum, as in the torchvision detection reference recipe
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.amp.GradScaler('cuda') if MIXED_PRECISION and device.type == 'cuda' else None

    if MLFLOW_AVAILABLE and args.use_mlflow:
        mlflow.start_run()
        mlflow.log_params({"lr": args.lr, "batch_size": args.batch_size, "epochs": args.epochs})

    history = {'train_losses': [], 'val_losses': [], 'epoch_end_seconds': [],
               'device': device.type, 'batch_size': args.batch_size}
    best_val = float('inf')
    history_path = getattr(args, 'history', None)

    budget_s = getattr(args, 'max_minutes', None) and args.max_minutes * 60
    total_epochs = args.epochs
    start = time.time()

    for epoch in range(args.epochs):
        if epoch >= total_epochs:
            break
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler,
                                         desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        avg_val_loss = validation_loss(model, val_loader, device)
        scheduler.step()

        if epoch == 0 and budget_s:
            # Fit the run to the time budget and shorten the cosine schedule to match
            total_epochs = epochs_within_budget(args.epochs, budget_s, time.time() - start)
            scheduler.T_max = total_epochs
            print(f"  First epoch took {(time.time() - start) / 60:.1f} min: training {total_epochs} epochs in the budget")

        history['train_losses'].append(avg_train_loss)
        history['val_losses'].append(avg_val_loss)
        history['epoch_end_seconds'].append(round(time.time() - start, 1))
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}  Val Loss: {avg_val_loss:.4f}")

        if MLFLOW_AVAILABLE and args.use_mlflow:
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

        if history_path:  # written every epoch so the curve survives an interrupted run
            Path(history_path).parent.mkdir(parents=True, exist_ok=True)
            Path(history_path).write_text(json.dumps(history, indent=2))

        # Keep only the checkpoint with the lowest validation loss
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            torch.save(model.state_dict(), output)
            print(f"  Saved new best checkpoint to {output}")

    if MLFLOW_AVAILABLE and args.use_mlflow:
        mlflow.end_run()

    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_ann', type=str, required=True, help="Path to train annotations JSON")
    parser.add_argument('--train_img', type=str, required=True, help="Path to train images directory")
    parser.add_argument('--val_ann', type=str, required=True, help="Path to val annotations JSON")
    parser.add_argument('--val_img', type=str, required=True, help="Path to val images directory")
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--img_size', type=int, default=IMG_SIZE)
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS)
    parser.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=True,
                        help="Start from COCO weights (--no-pretrained to train from scratch)")
    parser.add_argument('--output', type=str, default=str(BASELINE_CHECKPOINT), help="Where to save the best checkpoint")
    parser.add_argument('--history', type=str, default=str(METRICS_DIR / 'baseline_training_history.json'),
                        help="Where to save per-epoch train/val losses")
    parser.add_argument('--use_mlflow', action='store_true', help="Enable MLflow logging")
    parser.add_argument('--max_minutes', type=float, default=None,
                        help="Time budget: after epoch 1, run only as many epochs as fit")
    args = parser.parse_args()
    main(args)
