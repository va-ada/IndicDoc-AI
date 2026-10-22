"""
IndicDoc AI - Study: how well do the detectors hold up on damaged scans, and how even is accuracy
across languages and document domains?

For every model it scores the same 682 test pages clean and under five kinds of scan damage, and reports
mAP per language and per domain. Two summary numbers are defined in docs/RESEARCH_STUDY.md:
  Robustness Index (RI)      = mAP@50 on damaged pages / mAP@50 on clean pages
  Consistency Index (CI)     = lowest group mAP@50 / highest group mAP@50   (per language or per domain)

Usage:
    python scripts/study_robustness.py                  # all models found in outputs/models
    python scripts/study_robustness.py --models doclayout yolov8
Writes outputs/metrics/study_robustness.json.
"""
import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import (AUTHORS_CHECKPOINT, BASELINE_CHECKPOINT, CLASS_NAMES, METRICS_DIR, OUTPUT_DIR,
                       PROPOSED_CHECKPOINT, get_device)
from ml.evaluation.coco_eval import collect_detections, evaluate_detections
from ml.inference.predict import load_model

CHECKPOINTS = {'doclayout': AUTHORS_CHECKPOINT, 'yolov8': PROPOSED_CHECKPOINT, 'fasterrcnn': BASELINE_CHECKPOINT}
SEED = 42


def blur(img, rng):        # out-of-focus phone photo or a soft scan
    return img.filter(ImageFilter.GaussianBlur(radius=2))


def noise(img, rng):       # sensor / photocopier grain
    a = np.asarray(img, dtype=np.float32)
    a = a + rng.normal(0, 20, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def jpeg(img, rng):        # heavily compressed image (WhatsApp forward)
    buf = io.BytesIO()
    img.save(buf, 'JPEG', quality=15)
    return Image.open(io.BytesIO(buf.getvalue())).convert('RGB')


def low_res(img, rng):     # scanned at about half the resolution, then enlarged
    w, h = img.size
    return img.resize((w // 2, h // 2), Image.BILINEAR).resize((w, h), Image.BILINEAR)


def faded(img, rng):       # faded print / washed-out photocopy
    img = ImageEnhance.Contrast(img).enhance(0.45)
    return ImageEnhance.Brightness(img).enhance(1.25)


CORRUPTIONS = {'blur': blur, 'noise': noise, 'jpeg': jpeg, 'low_res': low_res, 'faded': faded}


def make_corrupted(gt, img_dir: Path, out_dir: Path, name: str) -> Path:
    """Write the damaged copy of every test page once (same random seed every run)."""
    target = out_dir / name
    if target.exists() and len(list(target.glob('*.png'))) == len(gt['images']):
        return target
    target.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    for img in gt['images']:
        page = Image.open(img_dir / img['file_name']).convert('RGB')
        CORRUPTIONS[name](page, rng).save(target / img['file_name'])
    return target


def consistency(groups: dict) -> float:
    vals = [g['map50'] for g in groups.values()]
    return min(vals) / max(vals) if vals and max(vals) > 0 else 0.0


def summarise(metrics: dict) -> dict:
    return {k: metrics[k] for k in ('map50', 'map50_95', 'precision', 'recall', 'f1')}


def main(args):
    ann = Path(args.ann)
    img_dir = Path(args.img_dir)
    gt = json.loads(ann.read_text())
    device = get_device()
    out_path = Path(args.out)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    corrupt_root = OUTPUT_DIR / 'study' / 'corrupted'
    dirs = {'clean': img_dir}
    for name in CORRUPTIONS:
        print(f"Preparing {name} pages...", flush=True)
        dirs[name] = make_corrupted(gt, img_dir, corrupt_root, name)

    for model_type in args.models:
        ckpt = CHECKPOINTS[model_type]
        if not Path(ckpt).exists():
            print(f"skip {model_type}: no checkpoint at {ckpt}")
            continue
        model = load_model(str(ckpt), model_type, device)
        res = results.setdefault(model_type, {})
        for cond, d in dirs.items():
            if cond in res and not args.force:
                print(f"{model_type}/{cond}: cached mAP50 {res[cond]['map50']:.3f}")
                continue
            t0 = time.time()
            dets, _ = collect_detections(model, model_type, gt, str(d), device)
            m = evaluate_detections(gt, dets, CLASS_NAMES, conf_threshold=0.5, group_key='language')
            entry = summarise(m)
            entry['per_language'] = {g: v['map50'] for g, v in m['per_group'].items()}
            if cond == 'clean':
                dm = evaluate_detections(gt, dets, CLASS_NAMES, conf_threshold=0.5, group_key='domain')
                entry['per_domain'] = {g: {'images': v['images'], 'map50': v['map50']}
                                       for g, v in dm['per_group'].items()}
                entry['language_consistency'] = consistency(m['per_group'])
                entry['domain_consistency'] = consistency(dm['per_group'])
                entry['per_class_ap50'] = m['per_class_ap50']
            res[cond] = entry
            out_path.write_text(json.dumps(results, indent=2))  # save after every condition
            print(f"{model_type}/{cond}: mAP50 {entry['map50']:.3f} ({time.time() - t0:.0f}s)", flush=True)

        clean = res['clean']['map50']
        res['robustness_index'] = {c: (res[c]['map50'] / clean if clean else 0.0) for c in CORRUPTIONS if c in res}
        res['mean_robustness_index'] = float(np.mean(list(res['robustness_index'].values())))
        out_path.write_text(json.dumps(results, indent=2))
        print(f"{model_type}: mean RI {res['mean_robustness_index']:.3f}", flush=True)
        del model


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', default=['doclayout', 'yolov8', 'fasterrcnn'])
    p.add_argument('--ann', default='data/raw/indicdlp/test/annotations.json')
    p.add_argument('--img_dir', default='data/raw/indicdlp/test/images')
    p.add_argument('--out', default=str(METRICS_DIR / 'study_robustness.json'))
    p.add_argument('--force', action='store_true')
    main(p.parse_args())
