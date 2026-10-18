"""
IndicDoc AI - Build a language-balanced IndicDLP subset in COCO format.

The dataset (huggingface.co/datasets/ai4bharat/indicdlp, gated: accept its terms first) is ~91 GB
of parquet shards sorted by language. This script reads only the parquet row groups it needs over
HTTP range requests, so a subset covering all 12 languages costs a few GB instead of 91.

Output (git-ignored):
    data/raw/indicdlp/{train,val,test}/images/*.png
    data/raw/indicdlp/{train,val,test}/annotations.json   (COCO, category ids 0..41, image
                                                           fields 'language' and 'domain')

Usage:
    python scripts/prepare_indicdlp.py --train_groups 2 --val_groups 1 --test_groups 1
"""
import argparse
import io
import json
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import CLASS_NAMES, RAW_DATA_DIR, SEED

REPO = 'ai4bharat/indicdlp'
SPLITS = {'train': 'train', 'val': 'validation', 'test': 'test'}

# File names look like '<domain>_<language>_<id>_<n>.png'; codes from the dataset card
LANGUAGES = {'as': 'assamese', 'bn': 'bengali', 'en': 'english', 'gu': 'gujarati', 'hi': 'hindi',
             'kn': 'kannada', 'ml': 'malayalam', 'mr': 'marathi', 'or': 'odia', 'pa': 'punjabi',
             'ta': 'tamil', 'te': 'telugu'}
DOMAINS = {'ar': 'acts & rules', 'br': 'brochures', 'fm': 'forms', 'mg': 'magazines', 'mn': 'manuals',
           'np': 'newspapers', 'nt': 'notices', 'nv': 'novels', 'qp': 'question papers',
           'rp': 'research papers', 'sy': 'syllabi', 'tb': 'textbooks'}


def parse_filename(name: str) -> Tuple[str, str]:
    """(domain, language) from an IndicDLP file name; unknown codes are kept as-is rather than dropped."""
    domain, lang = name.split('_')[:2]
    return DOMAINS.get(domain, domain), LANGUAGES.get(lang, lang)


def rows_to_coco(rows: List[Dict], img_dir: Path, start_image_id: int = 1, start_ann_id: int = 1) -> Tuple[Dict, Dict]:
    """
    Convert parquet rows (image bytes, [x, y, w, h] boxes, class ids) to COCO, writing images to img_dir.
    Boxes are clipped to the page; boxes left under 1 px wide or tall are dropped.
    """
    img_dir.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    stats = {'images': 0, 'boxes_in': 0, 'boxes_kept': 0, 'boxes_clipped': 0, 'boxes_dropped': 0}
    image_id, ann_id = start_image_id, start_ann_id

    for row in rows:
        name = Path(row['image']['path']).name
        data = row['image']['bytes']
        with Image.open(io.BytesIO(data)) as im:
            width, height = im.size
        (img_dir / name).write_bytes(data)
        domain, language = parse_filename(name)
        images.append({'id': image_id, 'file_name': name, 'width': width, 'height': height,
                       'language': language, 'domain': domain})
        stats['images'] += 1

        for (x, y, w, h), cat in zip(row['bboxes'], row['category_ids']):
            if not 0 <= cat < len(CLASS_NAMES):
                raise ValueError(f"{name}: class id {cat} outside 0..{len(CLASS_NAMES) - 1}")
            stats['boxes_in'] += 1
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(float(width), x + w), min(float(height), y + h)
            if x2 - x1 < 1 or y2 - y1 < 1:
                stats['boxes_dropped'] += 1
                continue
            if (x1, y1, x2, y2) != (x, y, x + w, y + h):
                stats['boxes_clipped'] += 1
            bbox = [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]
            annotations.append({'id': ann_id, 'image_id': image_id, 'category_id': int(cat), 'bbox': bbox,
                                'area': round(bbox[2] * bbox[3], 2), 'iscrowd': 0})
            ann_id += 1
            stats['boxes_kept'] += 1
        image_id += 1

    coco = {'images': images, 'annotations': annotations,
            'categories': [{'id': i, 'name': n, 'supercategory': 'layout'} for i, n in enumerate(CLASS_NAMES)]}
    return coco, stats


def language_candidates(row_group_paths: Dict[Tuple[str, int], List[str]],
                        min_pages: int = 20) -> Dict[str, List[Tuple[str, int]]]:
    """
    For each language, the (shard_url, row_group) pairs holding at least min_pages of its pages.
    Shards are sorted by language but can straddle a boundary, so each row group is labelled by
    its own pages rather than by the first page of the shard.
    """
    from collections import Counter
    candidates: Dict[str, List[Tuple[str, int]]] = {}
    for key in sorted(row_group_paths):
        for lang, n in Counter(parse_filename(p)[1] for p in row_group_paths[key]).items():
            if n >= min_pages:
                candidates.setdefault(lang, []).append(key)
    return candidates


def plan_selection(candidates: Dict[str, List[Tuple[str, int]]], groups_per_language: int,
                   seed: int = SEED) -> List[Tuple[str, str, int]]:
    """
    Pick (language, shard_url, row_group) triples: per language, spread the picks over different
    shards first (different documents), then over different row groups. Seeded, so reproducible.
    """
    rng = random.Random(seed)
    plan = []
    for lang in sorted(candidates):
        by_shard: Dict[str, List[int]] = {}
        for url, rg in candidates[lang]:
            by_shard.setdefault(url, []).append(rg)
        for rgs in by_shard.values():
            rng.shuffle(rgs)
        urls = sorted(by_shard)
        rng.shuffle(urls)
        picks, round_ = [], 0
        while len(picks) < groups_per_language and round_ < max(len(v) for v in by_shard.values()):
            for url in urls:
                if round_ < len(by_shard[url]) and len(picks) < groups_per_language:
                    picks.append((lang, url, by_shard[url][round_]))
            round_ += 1
        plan.extend(picks)
    return plan


def _filesystem():
    import fsspec
    token = (Path.home() / '.cache' / 'huggingface' / 'token').read_text().strip()
    return fsspec.filesystem('https', headers={'Authorization': f'Bearer {token}'}, block_size=2 ** 20)


def list_shards(split: str) -> List[str]:
    tree = json.load(urllib.request.urlopen(f'https://huggingface.co/api/datasets/{REPO}/tree/main/data'))
    names = sorted(x['path'] for x in tree if x['path'].split('/')[-1].startswith(SPLITS[split] + '-'))
    return [f'https://huggingface.co/datasets/{REPO}/resolve/main/{n}' for n in names]


def row_group_paths(fs, url: str) -> Dict[Tuple[str, int], List[str]]:
    """File names in every row group of a shard, reading only the parquet footer and the name column."""
    import pyarrow.parquet as pq
    with fs.open(url, 'rb') as f:
        pf = pq.ParquetFile(f)
        return {(url, rg): [x['path'] for x in pf.read_row_group(rg, columns=['image.path']).column('image').to_pylist()]
                for rg in range(pf.num_row_groups)}


def read_row_group(fs, url: str, rg: int) -> List[Dict]:
    import pyarrow.parquet as pq
    with fs.open(url, 'rb') as f:
        return pq.ParquetFile(f).read_row_group(rg).to_pylist()


def build_split(split: str, groups_per_language: int, max_per_group: int, out_root: Path, fs) -> Dict:
    print(f"[{split}] reading page names from every shard...")
    paths: Dict[Tuple[str, int], List[str]] = {}
    with ThreadPoolExecutor(8) as ex:
        for part in ex.map(lambda u: row_group_paths(fs, u), list_shards(split)):
            paths.update(part)
    candidates = language_candidates(paths, min_pages=min(20, max_per_group))
    print(f"[{split}] {len(paths)} row groups; candidates per language: " +
          ', '.join(f"{k}={len(v)}" for k, v in sorted(candidates.items())))

    plan = plan_selection(candidates, groups_per_language, seed=SEED + list(SPLITS).index(split))
    out = out_root / split
    rng = random.Random(SEED)

    def fetch(item):
        lang, url, rg = item
        # Keep only this language's pages: a row group on a language boundary also holds the next language
        rows = [r for r in read_row_group(fs, url, rg) if parse_filename(r['image']['path'])[1] == lang]
        random.Random(f"{url}:{rg}").shuffle(rows)
        return rows[:max_per_group]

    with ThreadPoolExecutor(4) as ex:
        groups = list(ex.map(fetch, plan))

    rows = [r for g in groups for r in g]
    rng.shuffle(rows)
    coco, stats = rows_to_coco(rows, out / 'images')
    (out / 'annotations.json').write_text(json.dumps(coco))
    print(f"[{split}] {stats}")
    return stats


def main(args):
    out_root = Path(args.out_dir)
    fs = _filesystem()
    for split, groups, cap in (('val', args.val_groups, args.val_per_group),
                               ('test', args.test_groups, args.test_per_group),
                               ('train', args.train_groups, args.train_per_group)):
        build_split(split, groups, cap, out_root, fs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default=str(RAW_DATA_DIR / 'indicdlp'))
    parser.add_argument('--train_groups', type=int, default=2, help="Row groups (~100 pages each) per language")
    parser.add_argument('--val_groups', type=int, default=1)
    parser.add_argument('--test_groups', type=int, default=1)
    parser.add_argument('--train_per_group', type=int, default=100)
    parser.add_argument('--val_per_group', type=int, default=30)
    parser.add_argument('--test_per_group', type=int, default=60)
    main(parser.parse_args())
