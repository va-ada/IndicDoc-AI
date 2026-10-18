"""
IndicDoc AI - Evaluate a trained model on a COCO-format test set.

Writes measured metrics and plots only; nothing here is estimated or randomly generated.

Usage:
    python scripts/evaluate.py --model_type fasterrcnn --name baseline \
        --ann data/indicdlp/test/annotations.json --img_dir data/indicdlp/test/images
    python scripts/evaluate.py --model_type yolov8 --name proposed \
        --ann data/indicdlp/test/annotations.json --img_dir data/indicdlp/test/images
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import (AUTHORS_CHECKPOINT, BASELINE_CHECKPOINT, CLASS_NAMES, CONFIDENCE_THRESHOLD, OUTPUT_DIR,
                       PROPOSED_CHECKPOINT, get_device)
from ml.evaluation.coco_eval import (best_f1_threshold, collect_detections, dedupe_detections, evaluate_detections,
                                     high_recall_threshold, match_detections, precision_recall_at)
from ml.inference.predict import load_model

INDIGO, GOLD, GREEN, RED = '#1E2A5A', '#C89B3C', '#2E7D32', '#C62828'
MODEL_LABELS = {'fasterrcnn': 'Faster R-CNN (ResNet-50 FPN)', 'yolov8': 'YOLOv8s',
                'doclayout': 'DocLayout-YOLO (IndicDLP authors)'}


def count_parameters(model, model_type: str) -> int:
    net = model.model if model_type in ('yolov8', 'doclayout') else model
    return sum(p.numel() for p in net.parameters())


def draw_boxes(ax, image, items, color, title, with_scores=False):
    ax.imshow(image)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')
    for it in items:
        x, y, w, h = it['bbox']
        ax.add_patch(plt.Rectangle((x, y), w, h, linewidth=1.5, edgecolor=color, facecolor='none'))
        text = CLASS_NAMES[it['category_id']] + (f" {it['score']:.2f}" if with_scores else '')
        ax.text(x, y - 2, text, fontsize=6, color='white', bbox=dict(facecolor=color, alpha=0.8, pad=0.5, edgecolor='none'))


def save_comparison(gt, img_dir, img, dets, conf, title, path):
    image = Image.open(Path(img_dir) / img['file_name']).convert('RGB')
    gts = [a for a in gt['annotations'] if a['image_id'] == img['id']]
    preds = [d for d in dets if d['image_id'] == img['id'] and d['score'] >= conf]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 9))
    draw_boxes(ax1, image, gts, GREEN, f"Ground truth ({len(gts)})")
    draw_boxes(ax2, image, preds, RED, f"Prediction ({len(preds)}, conf >= {conf})", with_scores=True)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def make_plots(gt, metrics, dets, args, plots_dir, pred_dir):
    name = args.name
    gt_counts = Counter(a['category_id'] for a in gt['annotations'])

    # Class distribution of the evaluation set
    order = [k for k, _ in gt_counts.most_common()]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(order)), [gt_counts[k] for k in order], color=INDIGO)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([CLASS_NAMES[k] for k in order], rotation=90, fontsize=8)
    ax.set_ylabel('Ground-truth boxes')
    ax.set_title(f"Class distribution ({len(gt['images'])} evaluation pages)")
    fig.tight_layout()
    fig.savefig(plots_dir / 'class_distribution.png', dpi=130)
    plt.close(fig)

    # Per-class AP@50 (classes present in the evaluation set)
    ap = {c: v for c, v in metrics['per_class_ap50'].items() if v is not None}
    names = sorted(ap, key=ap.get, reverse=True)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(names)), [ap[c] for c in names], color=GOLD)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel('AP@50')
    ax.set_title(f"{MODEL_LABELS[args.model_type]}: AP@50 per class (mAP@50 = {metrics['map50']:.3f})")
    fig.tight_layout()
    fig.savefig(plots_dir / f'{name}_per_class_ap.png', dpi=130)
    plt.close(fig)

    # Precision-recall curve at IoU 0.50, averaged over classes
    if metrics['pr_curve']['recall']:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(metrics['pr_curve']['recall'], metrics['pr_curve']['precision'], color=INDIGO, linewidth=2)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{MODEL_LABELS[args.model_type]}: precision-recall (IoU 0.50, mean over classes)")
        fig.tight_layout()
        fig.savefig(plots_dir / f'{name}_pr_curve.png', dpi=130)
        plt.close(fig)

    # Confusion matrix for the 12 most frequent classes plus background
    top = order[:12]
    cm = np.array(metrics['confusion_matrix'])
    idx = top + [len(CLASS_NAMES)]
    sub = cm[np.ix_(idx, idx)]
    labels = [CLASS_NAMES[k] for k in top] + ['(missed / background)']
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(sub, cmap='YlOrBr')
    ax.set_xticks(range(len(idx)))
    ax.set_yticks(range(len(idx)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(idx)):
        for j in range(len(idx)):
            ax.text(j, i, sub[i, j], ha='center', va='center', fontsize=7)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f"{MODEL_LABELS[args.model_type]}: confusion matrix (IoU >= 0.5, conf >= {args.conf})")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(plots_dir / f'{name}_confusion_matrix.png', dpi=130)
    plt.close(fig)

    # Per-language mAP@50
    groups = metrics.get('per_group') or {}
    if len(groups) > 1:
        langs = sorted(groups, key=lambda g: groups[g]['map50'], reverse=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(langs, [groups[g]['map50'] for g in langs], color=INDIGO)
        for i, g in enumerate(langs):
            ax.text(i, groups[g]['map50'] + 0.01, f"n={groups[g]['images']}", ha='center', fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel('mAP@50')
        ax.set_title(f"{MODEL_LABELS[args.model_type]}: mAP@50 per {args.group_key}")
        plt.xticks(rotation=30)
        fig.tight_layout()
        fig.savefig(plots_dir / f'{name}_per_{args.group_key}.png', dpi=130)
        plt.close(fig)

    # Real prediction examples: the first pages of the evaluation set
    for i, img in enumerate(gt['images'][:args.num_examples], start=1):
        save_comparison(gt, args.img_dir, img, dets, args.conf, f"{MODEL_LABELS[args.model_type]} on {img['file_name']}",
                        pred_dir / f'{name}_prediction_{i}.png')

    # Error analysis: the page with the most missed elements and the page with the most false alarms
    per_image = []
    for img in gt['images']:
        gts = [a for a in gt['annotations'] if a['image_id'] == img['id']]
        preds = [d for d in dets if d['image_id'] == img['id'] and d['score'] >= args.conf]
        _, false_alarms, missed = match_detections(gts, preds, class_aware=True, iou_threshold=0.5)
        per_image.append((img, len(missed), len(false_alarms)))
    (pred_dir / 'errors').mkdir(exist_ok=True)
    worst_missed = max(per_image, key=lambda t: t[1])
    worst_false = max(per_image, key=lambda t: t[2])
    save_comparison(gt, args.img_dir, worst_missed[0], dets, args.conf,
                    f"Most missed elements: {worst_missed[1]} on {worst_missed[0]['file_name']}",
                    pred_dir / 'errors' / f'{name}_most_missed.png')
    save_comparison(gt, args.img_dir, worst_false[0], dets, args.conf,
                    f"Most false detections: {worst_false[2]} on {worst_false[0]['file_name']}",
                    pred_dir / 'errors' / f'{name}_most_false.png')


def plot_model_comparison(metrics_dir: Path, plots_dir: Path):
    """Bar chart of every evaluated model (baseline, proposed, authors) on the same test pages."""
    names = [n for n in ('baseline', 'proposed', 'authors') if (metrics_dir / f'{n}_metrics.json').exists()]
    if len(names) < 2:
        return
    m = {n: json.loads((metrics_dir / f'{n}_metrics.json').read_text()) for n in names}
    colors = {'baseline': INDIGO, 'proposed': GOLD, 'authors': GREEN}
    keys = ['precision', 'recall', 'f1', 'map50', 'map50_95']
    x = np.arange(len(keys))
    width = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, n in enumerate(names):
        offset = (i - (len(names) - 1) / 2) * width
        vals = [m[n][k] for k in keys]
        ax.bar(x + offset, vals, width, label=m[n]['model'], color=colors[n])
        for xi, v in zip(x + offset, vals):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha='center', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(['Precision', 'Recall', 'F1', 'mAP@50', 'mAP@50:95'])
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title('Model comparison on the same test pages (P/R/F1 at confidence >= 0.5)')
    fig.tight_layout()
    fig.savefig(plots_dir / 'model_comparison.png', dpi=130)
    plt.close(fig)


def main(args) -> dict:
    out = Path(args.out_dir)
    metrics_dir, plots_dir, pred_dir = out / 'metrics', out / 'plots', out / 'predictions'
    for d in (metrics_dir, plots_dir, pred_dir):
        d.mkdir(parents=True, exist_ok=True)

    gt = json.loads(Path(args.ann).read_text())
    device = get_device()
    model = load_model(args.checkpoint, args.model_type, device)
    print(f"Evaluating {args.model_type} on {len(gt['images'])} pages ({device})...")

    dets, seconds = collect_detections(model, args.model_type, gt, args.img_dir, device)
    metrics = evaluate_detections(gt, dets, CLASS_NAMES, conf_threshold=args.conf, group_key=args.group_key)

    timed = seconds[1:] if len(seconds) > 1 else seconds  # first image includes warm-up
    metrics.update({
        'model': MODEL_LABELS[args.model_type],
        'inference_time': float(np.mean(timed)),
        'device': device,
        'model_size_mb': round(Path(args.checkpoint).stat().st_size / 1e6, 1),
        'parameters': count_parameters(model, args.model_type),
        'total_predictions': metrics['true_positives'] + metrics['false_positives'],
        'evaluation_set': str(args.ann),
        'dataset': args.dataset_name,
        'note': f"Measured on {len(gt['images'])} held-out pages. P/R/F1 at confidence >= {args.conf}, IoU >= 0.5.",
    })
    # Operating points, chosen on the validation pages (never tuned on the test pages), measured on what the app
    # and API return: duplicates (same box, several labels) removed
    if getattr(args, 'val_ann', None):
        val_gt = json.loads(Path(args.val_ann).read_text())
        val_dets, _ = collect_detections(model, args.model_type, val_gt, args.val_img, device, min_score=0.05)
        val_dets, test_dets = dedupe_detections(val_dets), dedupe_detections(dets)
        conf, val_f1 = best_f1_threshold(val_gt, val_dets)
        # 'Find every part': lowest threshold still >= 70% precision, but never above the balanced one (for a weak
        # model the precision floor is only reached at a HIGH threshold, which would find fewer parts, not more)
        high = min(high_recall_threshold(val_gt, val_dets, min_precision=0.7), conf)
        for key, thr in (('recommended_conf', conf), ('high_recall_conf', high)):
            at = precision_recall_at(gt, test_dets, thr)
            metrics[key] = thr
            metrics[f'at_{key}'] = {k: at[k] for k in ('precision', 'recall', 'f1', 'avg_iou')}
            print(f"{key} {thr}: test P {at['precision']:.3f} R {at['recall']:.3f} F1 {at['f1']:.3f}")
    (metrics_dir / f'{args.name}_metrics.json').write_text(json.dumps(metrics, indent=2))

    make_plots(gt, metrics, dets, args, plots_dir, pred_dir)
    plot_model_comparison(metrics_dir, plots_dir)

    print(f"mAP@50 {metrics['map50']:.4f} | mAP@50:95 {metrics['map50_95']:.4f} | "
          f"P {metrics['precision']:.4f} R {metrics['recall']:.4f} F1 {metrics['f1']:.4f} | "
          f"{metrics['inference_time']:.3f}s/page")
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', choices=['fasterrcnn', 'yolov8', 'doclayout'], required=True)
    parser.add_argument('--checkpoint', type=str, default=None, help="Defaults to the app's checkpoint path")
    parser.add_argument('--ann', type=str, required=True, help="COCO annotations of the test set")
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--name', choices=['baseline', 'proposed', 'authors'], required=True)
    parser.add_argument('--conf', type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument('--group_key', type=str, default='language', help="Image field for the per-group breakdown")
    parser.add_argument('--num_examples', type=int, default=4)
    parser.add_argument('--out_dir', type=str, default=str(OUTPUT_DIR))
    parser.add_argument('--val_ann', type=str, default=None, help="Validation annotations, to choose the recommended threshold")
    parser.add_argument('--val_img', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='IndicDLP test subset', help="Shown in the app next to the page count")
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = str({'fasterrcnn': BASELINE_CHECKPOINT, 'yolov8': PROPOSED_CHECKPOINT,
                               'doclayout': AUTHORS_CHECKPOINT}[args.model_type])
    main(args)
