"""
Detection evaluation for IndicDoc AI.

mAP follows the COCO protocol, computed with pycocotools' COCOeval (Lin et al., "Microsoft COCO:
Common Objects in Context", ECCV 2014): AP is the area under the 101-point interpolated
precision-recall curve, averaged over classes, at IoU 0.50 (mAP@50), IoU 0.75 (mAP@75) and
averaged over IoU 0.50:0.05:0.95 (mAP@50:95).

Precision, recall, F1 and average IoU describe one operating point: detections with
score >= conf_threshold, matched greedily (highest score first) to same-class ground truth at IoU >= 0.5.
"""
import contextlib
import io
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ml.evaluation.metrics import calculate_iou
from ml.inference.predict import predict_single

# Newspaper pages can hold well over COCO's default limit of 100 elements per image
MAX_DETS = 300


def _xyxy(bbox: List[float]) -> List[float]:
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def collect_detections(model: Any, model_type: str, gt: Dict, img_dir: str, device: str,
                       min_score: float = 0.001) -> Tuple[List[Dict], List[float]]:
    """Run the model on every ground-truth image. Returns COCO-format detections and per-image seconds."""
    detections, seconds = [], []
    for img in gt['images']:
        image = Image.open(Path(img_dir) / img['file_name']).convert('RGB')
        start = time.time()
        out = predict_single(model, image, model_type, device, conf=min_score)
        seconds.append(time.time() - start)
        for box, label, score in zip(out['boxes'], out['labels'], out['scores']):
            x1, y1, x2, y2 = (float(v) for v in box)
            detections.append({'image_id': img['id'], 'category_id': int(label),
                               'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': float(score)})
    return detections, seconds


def _run_cocoeval(coco_gt: COCO, detections: List[Dict], img_ids: List[int]) -> Optional[COCOeval]:
    if not detections:
        return None
    with contextlib.redirect_stdout(io.StringIO()):  # COCOeval prints a large table
        coco_dt = coco_gt.loadRes(detections)
        ev = COCOeval(coco_gt, coco_dt, 'bbox')
        ev.params.maxDets = [1, 100, MAX_DETS]
        ev.params.imgIds = img_ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return ev


def match_detections(gts: List[Dict], dets: List[Dict], class_aware: bool, iou_threshold: float):
    """Greedy matching, highest score first. Returns (pairs as (gt, det, iou), unmatched dets, unmatched gts)."""
    matched, pairs, unmatched_dets = set(), [], []
    for det in sorted(dets, key=lambda d: -d['score']):
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gts):
            if j in matched or (class_aware and gt['category_id'] != det['category_id']):
                continue
            iou = calculate_iou(_xyxy(det['bbox']), _xyxy(gt['bbox']))
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_threshold:
            matched.add(best_j)
            pairs.append((gts[best_j], det, best_iou))
        else:
            unmatched_dets.append(det)
    unmatched_gts = [g for j, g in enumerate(gts) if j not in matched]
    return pairs, unmatched_dets, unmatched_gts


def _group(gt: Dict, detections: List[Dict], conf_threshold: float):
    gts_by_img, dets_by_img = defaultdict(list), defaultdict(list)
    for a in gt['annotations']:
        gts_by_img[a['image_id']].append(a)
    for d in detections:
        if d['score'] >= conf_threshold:
            dets_by_img[d['image_id']].append(d)
    return gts_by_img, dets_by_img


def precision_recall_at(gt: Dict, detections: List[Dict], conf_threshold: float, iou_threshold: float = 0.5) -> Dict:
    """Precision, recall, F1 and mean IoU of detections with score >= conf_threshold (same-label matches)."""
    gts_by_img, dets_by_img = _group(gt, detections, conf_threshold)
    tp = fp = fn = 0
    ious = []
    for img in gt['images']:
        pairs, extra_dets, missed = match_detections(gts_by_img[img['id']], dets_by_img[img['id']],
                                                     class_aware=True, iou_threshold=iou_threshold)
        tp, fp, fn = tp + len(pairs), fp + len(extra_dets), fn + len(missed)
        ious.extend(iou for _, _, iou in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            'avg_iou': float(np.mean(ious)) if ious else 0.0,
            'true_positives': tp, 'false_positives': fp, 'false_negatives': fn}


def best_f1_threshold(gt: Dict, detections: List[Dict], iou_threshold: float = 0.5,
                      thresholds=None) -> Tuple[float, float]:
    """
    Confidence threshold with the highest F1 (ties go to the higher threshold: fewer boxes, same quality).
    Run this on validation pages, never on the test pages that are being reported.
    """
    thresholds = thresholds if thresholds is not None else [round(t, 2) for t in np.arange(0.05, 0.951, 0.05)]
    best = (0.5, -1.0)
    for t in thresholds:
        f1 = precision_recall_at(gt, detections, t, iou_threshold)['f1']
        if f1 >= best[1]:
            best = (t, f1)
    return best


def evaluate_detections(gt: Dict, detections: List[Dict], class_names: List[str],
                        conf_threshold: float = 0.5, iou_threshold: float = 0.5,
                        group_key: Optional[str] = None) -> Dict:
    """
    Score COCO-format detections against COCO-format ground truth.
    Category ids must be 0..len(class_names)-1, in class_names order.
    group_key: optional image field (e.g. 'language') to report mAP per group.
    """
    cat_ids = sorted(c['id'] for c in gt['categories'])
    if cat_ids != list(range(len(class_names))):
        raise ValueError(f"category ids must be 0..{len(class_names) - 1}, got {cat_ids[:5]}...")

    coco_gt = COCO()
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.dataset = {**gt, 'annotations': [{**a, 'area': a.get('area', a['bbox'][2] * a['bbox'][3]),
                                                  'iscrowd': a.get('iscrowd', 0)} for a in gt['annotations']]}
        coco_gt.createIndex()
    img_ids = [img['id'] for img in gt['images']]
    ev = _run_cocoeval(coco_gt, detections, img_ids)

    metrics: Dict[str, Any] = {
        'num_images': len(img_ids), 'num_ground_truth': len(gt['annotations']), 'num_detections': len(detections),
        'map50_95': 0.0, 'map50': 0.0, 'map75': 0.0,
        'per_class_ap50': {}, 'pr_curve': {'recall': [], 'precision': []},
    }
    has_gt = {a['category_id'] for a in gt['annotations']}
    if ev is not None:
        metrics['map50_95'], metrics['map50'], metrics['map75'] = (max(0.0, float(v)) for v in ev.stats[:3])
        precision = ev.eval['precision'][0, :, :, 0, -1]  # IoU 0.50, all areas, max detections -> [recall, class]
        for k, name in enumerate(class_names):
            p = precision[:, k]
            metrics['per_class_ap50'][name] = float(p[p > -1].mean()) if (p > -1).any() else None
        valid = (precision > -1).all(axis=0)
        if valid.any():
            metrics['pr_curve'] = {'recall': ev.params.recThrs.tolist(),
                                   'precision': precision[:, valid].mean(axis=1).tolist()}
    else:
        metrics['per_class_ap50'] = {name: (0.0 if k in has_gt else None) for k, name in enumerate(class_names)}

    # Operating-point metrics and confusion matrix
    op = precision_recall_at(gt, detections, conf_threshold, iou_threshold)
    n = len(class_names)
    confusion = np.zeros((n + 1, n + 1), dtype=int)  # rows = truth, cols = prediction; index n = background
    gts_by_img, dets_by_img = _group(gt, detections, conf_threshold)
    for img_id in img_ids:
        pairs, extra_dets, missed = match_detections(gts_by_img[img_id], dets_by_img[img_id], class_aware=False,
                                                     iou_threshold=iou_threshold)
        for g, d, _ in pairs:
            confusion[g['category_id'], d['category_id']] += 1
        for d in extra_dets:
            confusion[n, d['category_id']] += 1
        for g in missed:
            confusion[g['category_id'], n] += 1
    metrics.update({'conf_threshold': conf_threshold, **op, 'confusion_matrix': confusion.tolist()})

    if group_key:
        groups = defaultdict(list)
        for img in gt['images']:
            groups[img.get(group_key, 'unknown')].append(img['id'])
        metrics['per_group'] = {}
        for group, ids in sorted(groups.items()):
            id_set = set(ids)
            sub = _run_cocoeval(coco_gt, [d for d in detections if d['image_id'] in id_set], ids)
            stats = [max(0.0, float(v)) for v in sub.stats[:2]] if sub is not None else [0.0, 0.0]
            metrics['per_group'][group] = {'images': len(ids), 'map50_95': stats[0], 'map50': stats[1]}

    return metrics


def dedupe_detections(detections: List[Dict], iou_threshold: float = 0.7) -> List[Dict]:
    """Per page, drop boxes that duplicate a more confident box (IoU > threshold), as the app and API do."""
    from ml.inference.predict import dedupe_boxes
    by_img = defaultdict(list)
    for d in detections:
        by_img[d['image_id']].append(d)
    out = []
    for img_id, dets in by_img.items():
        r = dedupe_boxes({'boxes': np.array([_xyxy(d['bbox']) for d in dets]),
                          'labels': np.array([d['category_id'] for d in dets]),
                          'scores': np.array([d['score'] for d in dets])}, iou_threshold)
        out += [{'image_id': img_id, 'category_id': int(l), 'bbox': [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                 'score': float(s)} for b, l, s in zip(r['boxes'], r['labels'], r['scores'])]
    return out


def high_recall_threshold(gt: Dict, detections: List[Dict], min_precision: float = 0.7, thresholds=None) -> float:
    """
    'Find every part' operating point: the lowest confidence threshold whose precision is still >= min_precision.
    Run this on validation pages, never on the test pages that are being reported.
    """
    thresholds = thresholds if thresholds is not None else [round(t, 2) for t in np.arange(0.05, 0.951, 0.05)]
    for t in thresholds:
        if precision_recall_at(gt, detections, t)['precision'] >= min_precision:
            return t
    return thresholds[-1]
