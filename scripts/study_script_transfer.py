"""
IndicDoc AI - Study: does a layout detector work on a script it never saw during training?

Controlled experiment with two YOLOv8s models trained the same way (same page count, time budget, settings):
  seen     : trained on pages from all 12 languages
  held_out : trained with NO pages in the four Dravidian scripts (Kannada, Malayalam, Tamil, Telugu)
Both are tested on the same 682 test pages. Because the page count is matched, any drop on the Dravidian pages
comes from not having seen those scripts, not from having less data.

  Cross-Script Transfer Ratio (CSTR) = mAP@50 of held_out on the Dravidian pages
                                       / mAP@50 of seen on the Dravidian pages
CSTR near 1 means layout knowledge carries over to new scripts; well below 1 means it does not.

Usage:
    python scripts/study_script_transfer.py --hours 0.75
Writes outputs/metrics/study_script_transfer.json.
"""
import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import CLASS_NAMES, METRICS_DIR, MODEL_DIR, NUM_CLASSES, OUTPUT_DIR, SEED, get_device
from ml.evaluation.coco_eval import collect_detections, evaluate_detections
from ml.inference.predict import load_model
from ml.models.proposed_yolov8 import get_proposed_model

HELD_OUT = ['kannada', 'malayalam', 'tamil', 'telugu']
DATA = PROJECT_ROOT / 'data' / 'raw' / 'indicdlp'


def write_list(images, img_dir: Path, path: Path):
    path.write_text(''.join(f"{(img_dir / im['file_name']).resolve()}\n" for im in images))


def build_splits(work: Path) -> dict:
    train = json.loads((DATA / 'train' / 'annotations.json').read_text())
    val = json.loads((DATA / 'val' / 'annotations.json').read_text())
    unseen = [im for im in train['images'] if im['language'] not in HELD_OUT]
    rng = random.Random(SEED)
    matched = rng.sample(train['images'], len(unseen))  # same page count, all languages
    # Model selection uses only scripts both models have seen
    val_seen = [im for im in val['images'] if im['language'] not in HELD_OUT]
    work.mkdir(parents=True, exist_ok=True)
    write_list(val_seen, DATA / 'val' / 'images', work / 'val.txt')
    splits = {}
    for name, images in (('seen', matched), ('held_out', unseen)):
        write_list(images, DATA / 'train' / 'images', work / f'{name}_train.txt')
        cfg = {'train': str(work / f'{name}_train.txt'), 'val': str(work / 'val.txt'),
               'nc': NUM_CLASSES, 'names': CLASS_NAMES}
        (work / f'{name}.yaml').write_text(yaml.dump(cfg, sort_keys=False))
        langs = sorted({im['language'] for im in images})
        splits[name] = {'train_pages': len(images), 'train_languages': langs}
    splits['val_pages'] = len(val_seen)
    return splits


def train(name: str, work: Path, hours: float, device: str) -> Path:
    out = MODEL_DIR / 'study' / f'yolov8s_{name}.pt'
    if out.exists():
        print(f"{name}: using existing {out}")
        return out
    model = get_proposed_model('s', weights_dir=str(MODEL_DIR))
    model.train(data=str(work / f'{name}.yaml'), epochs=100, time=hours, imgsz=640, batch=8, device=device,
                project=str(work / 'runs'), name=name, exist_ok=True, seed=SEED, workers=2, fliplr=0.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(model.trainer.save_dir) / 'weights' / 'best.pt', out)
    shutil.copy(Path(model.trainer.save_dir) / 'results.csv', work / f'{name}_results.csv')
    return out


def subset(gt: dict, keep) -> dict:
    images = [im for im in gt['images'] if keep(im)]
    ids = {im['id'] for im in images}
    return {**gt, 'images': images, 'annotations': [a for a in gt['annotations'] if a['image_id'] in ids]}


def main(args):
    work = OUTPUT_DIR / 'study' / 'script_transfer'
    device = get_device()
    result = {'held_out_languages': HELD_OUT, 'splits': build_splits(work), 'hours_per_model': args.hours}
    test = json.loads((DATA / 'test' / 'annotations.json').read_text())
    groups = {'dravidian (held out)': subset(test, lambda im: im['language'] in HELD_OUT),
              'other scripts': subset(test, lambda im: im['language'] not in HELD_OUT)}
    for name in ('seen', 'held_out'):
        ckpt = train(name, work, args.hours, device)
        model = load_model(str(ckpt), 'yolov8', device)
        dets, _ = collect_detections(model, 'yolov8', test, str(DATA / 'test' / 'images'), device)
        full = evaluate_detections(test, dets, CLASS_NAMES, group_key='language')
        entry = {'map50': full['map50'], 'map50_95': full['map50_95'],
                 'per_language': {g: v['map50'] for g, v in full['per_group'].items()}}
        for gname, g in groups.items():
            ids = {im['id'] for im in g['images']}
            m = evaluate_detections(g, [d for d in dets if d['image_id'] in ids], CLASS_NAMES)
            entry[gname] = {'pages': len(ids), 'map50': m['map50'], 'map50_95': m['map50_95']}
        csv = work / f'{name}_results.csv'
        if csv.exists():
            entry['epochs'] = len(csv.read_text().strip().splitlines()) - 1
        result[name] = entry
        print(name, json.dumps({k: v for k, v in entry.items() if k != 'per_language'}), flush=True)
        del model

    d = 'dravidian (held out)'
    o = 'other scripts'
    result['cstr'] = result['held_out'][d]['map50'] / result['seen'][d]['map50']
    result['cstr_other_scripts'] = result['held_out'][o]['map50'] / result['seen'][o]['map50']
    out = Path(args.out)
    out.write_text(json.dumps(result, indent=2))
    print(f"CSTR (Dravidian) = {result['cstr']:.3f}; same ratio on other scripts = {result['cstr_other_scripts']:.3f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--hours', type=float, default=0.75, help="Training time budget per model")
    p.add_argument('--out', default=str(METRICS_DIR / 'study_script_transfer.json'))
    main(p.parse_args())
