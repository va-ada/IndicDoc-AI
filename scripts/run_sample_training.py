"""
IndicDoc AI - Sample-data smoke run.

Trains Faster R-CNN for a few epochs on the 30 synthetic sample pages, then evaluates it with the
same evaluator used for real results. It only proves the pipeline runs end to end: the sample pages
are random rectangles, so the numbers say nothing about document layout performance.

Outputs go to outputs/sample_run/ and never overwrite real results.

Usage:
    python scripts/run_sample_training.py
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import NUM_WORKERS, PROCESSED_DATA_DIR, SAMPLE_RUN_DIR
from ml.training.train_baseline import main as train_baseline

sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
from evaluate import main as evaluate


def ensure_sample_data():
    sample_dir = PROCESSED_DATA_DIR / "sample"
    ann_file = sample_dir / "annotations.json"
    if not ann_file.exists():
        print("Creating sample data...")
        from ml.datasets.create_sample_data import create_sample_dataset
        create_sample_dataset(num_images=30)
    return str(ann_file), str(sample_dir / "images")


def main(epochs: int = 3):
    ann_file, img_dir = ensure_sample_data()
    checkpoint = SAMPLE_RUN_DIR / 'models' / 'baseline_frcnn_sample.pth'

    print("=" * 60)
    print("  IndicDoc AI - sample-data smoke run (synthetic pages)")
    print("=" * 60)
    # Same pages for training and evaluation: this checks the plumbing, not generalisation
    history = train_baseline(argparse.Namespace(
        train_ann=ann_file, train_img=img_dir, val_ann=ann_file, val_img=img_dir,
        epochs=epochs, lr=0.005, batch_size=2, img_size=640, num_workers=NUM_WORKERS,
        pretrained=True, output=str(checkpoint), use_mlflow=False,
        history=str(SAMPLE_RUN_DIR / 'metrics' / 'baseline_training_history.json')))
    print(f"Train losses: {[round(v, 3) for v in history['train_losses']]}")

    evaluate(argparse.Namespace(
        model_type='fasterrcnn', checkpoint=str(checkpoint), ann=ann_file, img_dir=img_dir,
        name='baseline', conf=0.5, group_key='language', num_examples=3, out_dir=str(SAMPLE_RUN_DIR),
        dataset_name='Synthetic sample pages'))
    print(f"[DONE] Outputs in {SAMPLE_RUN_DIR}")


if __name__ == "__main__":
    main()
