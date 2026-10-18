"""
IndicDoc AI - Cross-check the project's evaluator against Ultralytics' own validator.

Two checks on the same trained YOLOv8 model and test pages:
1. Ultralytics' AP code applied to the exact detections scripts/evaluate.py scores. This isolates the maths:
   it should match this project's pycocotools number closely.
2. Ultralytics' full validator. Its NMS is multi-label (one box may be kept under several labels), unlike
   normal prediction (one label per box, which is what the app and API return), so it scores somewhat higher.

Usage:
    python scripts/crosscheck_eval.py --ann data/raw/indicdlp/test/annotations.json --img_dir data/raw/indicdlp/test/images
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import CLASS_NAMES, METRICS_DIR, NUM_CLASSES, OUTPUT_DIR, PROPOSED_CHECKPOINT, get_device
from ml.models.proposed_yolov8 import coco_to_yolo
from ml.training.train_yolo import create_yolo_yaml


def ultralytics_ap_on(detections: list, gt: dict) -> tuple:
    """mAP@50 and mAP@50:95 of COCO-format detections, computed with Ultralytics' matching and AP code."""
    from collections import defaultdict
    import numpy as np
    import torch
    from ultralytics.utils.metrics import ap_per_class, box_iou

    iouv = np.linspace(0.5, 0.95, 10)
    xyxy = lambda b: [b[0], b[1], b[0] + b[2], b[1] + b[3]]
    g_by, d_by = defaultdict(list), defaultdict(list)
    for a in gt['annotations']:
        g_by[a['image_id']].append(a)
    for d in detections:
        d_by[d['image_id']].append(d)
    tp, conf, pred_cls, target_cls = [], [], [], []
    for img in gt['images']:
        g, d = g_by[img['id']], d_by[img['id']]
        target_cls += [a['category_id'] for a in g]
        if not d:
            continue
        correct = np.zeros((len(d), len(iouv)), dtype=bool)
        if g:
            iou = box_iou(torch.tensor([xyxy(a['bbox']) for a in g]), torch.tensor([xyxy(x['bbox']) for x in d]))
            same = torch.tensor([a['category_id'] for a in g])[:, None] == torch.tensor([x['category_id'] for x in d])[None]
            iou = (iou * same).numpy()
            for i, t in enumerate(iouv):  # same one-to-one matching as ultralytics' BaseValidator.match_predictions
                m = np.array(np.nonzero(iou >= t)).T
                if m.shape[0]:
                    if m.shape[0] > 1:
                        m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                        m = m[np.unique(m[:, 1], return_index=True)[1]]
                        m = m[np.unique(m[:, 0], return_index=True)[1]]
                    correct[m[:, 1].astype(int), i] = True
        tp.append(correct)
        conf += [x['score'] for x in d]
        pred_cls += [x['category_id'] for x in d]
    ap = ap_per_class(np.concatenate(tp), np.array(conf), np.array(pred_cls), np.array(target_cls),
                      names={i: n for i, n in enumerate(CLASS_NAMES)})[5]
    return float(ap[:, 0].mean()), float(ap.mean())


def main(args) -> dict:
    from ultralytics import YOLO

    img_dir = Path(args.img_dir)
    coco_to_yolo(args.ann, str(img_dir.parent / 'labels'))
    yaml_path = OUTPUT_DIR / 'yolo_runs' / 'crosscheck_test.yaml'
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    create_yolo_yaml(str(img_dir), str(img_dir), NUM_CLASSES, CLASS_NAMES, str(yaml_path))

    results = YOLO(args.checkpoint).val(data=str(yaml_path), split='val', imgsz=640, conf=0.001, iou=0.7,
                                        max_det=300, device=get_device(), plots=False, verbose=False,
                                        project=str(OUTPUT_DIR / 'yolo_runs'), name='crosscheck', exist_ok=True)
    ours = json.loads((METRICS_DIR / 'proposed_metrics.json').read_text())
    from ml.evaluation.coco_eval import collect_detections
    from ml.inference.predict import load_model
    gt = json.loads(Path(args.ann).read_text())
    device = get_device()
    dets, _ = collect_detections(load_model(args.checkpoint, 'yolov8', device), 'yolov8', gt, args.img_dir, device)
    u50, u5095 = ultralytics_ap_on(dets, gt)
    report = {
        'ultralytics_ap_on_our_detections_map50': u50,
        'ultralytics_ap_on_our_detections_map50_95': u5095,
        'ultralytics_map50': float(results.box.map50),
        'ultralytics_map50_95': float(results.box.map),
        'ours_map50': ours['map50'],
        'ours_map50_95': ours['map50_95'],
        'note': ('Same YOLOv8 checkpoint and test pages. ultralytics_ap_on_our_detections_* = Ultralytics AP code on '
                 'the detections scripts/evaluate.py scored (checks the maths). ultralytics_map* = Ultralytics validator, '
                 'whose multi-label NMS differs from normal one-label-per-box prediction.'),
    }
    (METRICS_DIR / 'evaluator_crosscheck.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann', required=True)
    parser.add_argument('--img_dir', required=True)
    parser.add_argument('--checkpoint', default=str(PROPOSED_CHECKPOINT))
    main(parser.parse_args())
