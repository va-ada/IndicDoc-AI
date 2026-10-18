"""
IndicDoc AI - Evaluation tests with hand-made cases whose correct answers are known exactly.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

CLASSES = ['paragraph', 'table', 'figure']


def _gt():
    """Two 200x200 pages: page 1 (Hindi) has a paragraph + a table, page 2 (Tamil) has a figure."""
    images = [
        {'id': 1, 'width': 200, 'height': 200, 'file_name': 'a.png', 'language': 'hindi'},
        {'id': 2, 'width': 200, 'height': 200, 'file_name': 'b.png', 'language': 'tamil'},
    ]
    anns = [
        {'id': 1, 'image_id': 1, 'category_id': 0, 'bbox': [10, 10, 100, 40], 'area': 4000, 'iscrowd': 0},
        {'id': 2, 'image_id': 1, 'category_id': 1, 'bbox': [10, 100, 100, 60], 'area': 6000, 'iscrowd': 0},
        {'id': 3, 'image_id': 2, 'category_id': 2, 'bbox': [50, 50, 80, 80], 'area': 6400, 'iscrowd': 0},
    ]
    cats = [{'id': i, 'name': n} for i, n in enumerate(CLASSES)]
    return {'images': images, 'annotations': anns, 'categories': cats}


def _perfect(score=0.9):
    return [{'image_id': a['image_id'], 'category_id': a['category_id'], 'bbox': list(a['bbox']), 'score': score}
            for a in _gt()['annotations']]


class TestEvaluateDetections:
    def test_perfect_detections(self):
        from ml.evaluation.coco_eval import evaluate_detections
        m = evaluate_detections(_gt(), _perfect(), CLASSES)
        assert m['map50'] == pytest.approx(1.0)
        assert m['map50_95'] == pytest.approx(1.0)
        assert (m['precision'], m['recall'], m['f1'], m['avg_iou']) == pytest.approx((1, 1, 1, 1))
        assert m['per_class_ap50'] == pytest.approx({'paragraph': 1.0, 'table': 1.0, 'figure': 1.0})

    def test_no_detections(self):
        from ml.evaluation.coco_eval import evaluate_detections
        m = evaluate_detections(_gt(), [], CLASSES)
        assert m['map50'] == 0.0
        assert (m['precision'], m['recall'], m['f1']) == (0.0, 0.0, 0.0)
        assert (m['true_positives'], m['false_positives'], m['false_negatives']) == (0, 0, 3)

    def test_wrong_class_counts_and_confusion(self):
        """Table predicted as figure: 2 TP, 1 FP, 1 FN; confusion shows table->figure."""
        from ml.evaluation.coco_eval import evaluate_detections
        dets = _perfect()
        dets[1]['category_id'] = 2
        m = evaluate_detections(_gt(), dets, CLASSES)
        assert (m['true_positives'], m['false_positives'], m['false_negatives']) == (2, 1, 1)
        assert m['precision'] == pytest.approx(2 / 3)
        assert m['recall'] == pytest.approx(2 / 3)
        cm = np.array(m['confusion_matrix'])  # rows = truth, cols = prediction, last = background
        assert cm.shape == (4, 4)
        assert cm[1, 2] == 1 and cm[0, 0] == 1 and cm[2, 2] == 1
        assert cm.sum() == 3

    def test_low_confidence_excluded_from_precision_but_not_map(self):
        from ml.evaluation.coco_eval import evaluate_detections
        m = evaluate_detections(_gt(), _perfect(score=0.2), CLASSES, conf_threshold=0.5)
        assert m['map50'] == pytest.approx(1.0)
        assert m['recall'] == 0.0

    def test_loose_boxes_pass_map50_but_fail_map75(self):
        """Each box 60% wider than the truth: IoU 0.625, which passes at IoU 0.50 but fails at 0.75."""
        from ml.evaluation.coco_eval import evaluate_detections
        dets = _perfect()
        for d in dets:
            d['bbox'][2] *= 1.6
        m = evaluate_detections(_gt(), dets, CLASSES)
        assert m['map50'] == pytest.approx(1.0)
        assert m['map75'] == pytest.approx(0.0)
        assert m['avg_iou'] == pytest.approx(1 / 1.6)

    def test_per_language_breakdown(self):
        from ml.evaluation.coco_eval import evaluate_detections
        dets = [d for d in _perfect() if d['image_id'] == 1]  # Hindi page found, Tamil page missed
        m = evaluate_detections(_gt(), dets, CLASSES, group_key='language')
        assert m['per_group']['hindi']['map50'] == pytest.approx(1.0)
        assert m['per_group']['tamil']['map50'] == pytest.approx(0.0)
        assert m['per_group']['hindi']['images'] == 1


class TestCollectDetections:
    def test_collects_coco_format_from_frcnn(self, tmp_path):
        """Model outputs (xyxy, labels shifted by +1) become COCO detections (xywh, 0-indexed)."""
        import torch
        from PIL import Image
        from ml.evaluation.coco_eval import collect_detections

        Image.new('RGB', (200, 200), 'white').save(tmp_path / 'a.png')
        Image.new('RGB', (200, 200), 'white').save(tmp_path / 'b.png')

        class FakeFRCNN(torch.nn.Module):
            def forward(self, images):
                return [{'boxes': torch.tensor([[10., 20., 110., 60.]]), 'labels': torch.tensor([2]),
                         'scores': torch.tensor([0.8])}]

        dets, times = collect_detections(FakeFRCNN(), 'fasterrcnn', _gt(), str(tmp_path), 'cpu')
        assert len(dets) == 2 and len(times) == 2
        assert dets[0] == {'image_id': 1, 'category_id': 1, 'bbox': pytest.approx([10, 20, 100, 40]),
                           'score': pytest.approx(0.8)}


class TestEvaluateScript:
    def test_writes_metrics_and_real_plots(self, tmp_path):
        """End-to-end on 2 sample pages with an untrained model: files appear, numbers are measured, not invented."""
        import argparse
        import json
        import torch
        from ml.config import NUM_CLASSES, PROCESSED_DATA_DIR
        from ml.models.baseline_frcnn import get_baseline_model
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from evaluate import main

        src = PROCESSED_DATA_DIR / 'sample'
        data = json.loads((src / 'annotations.json').read_text())
        data['images'] = data['images'][:2]
        data['annotations'] = [a for a in data['annotations'] if a['image_id'] in {1, 2}]
        ann = tmp_path / 'ann.json'
        ann.write_text(json.dumps(data))
        ckpt = tmp_path / 'm.pth'
        torch.save(get_baseline_model(NUM_CLASSES + 1, pretrained=False).state_dict(), ckpt)

        metrics = main(argparse.Namespace(
            model_type='fasterrcnn', checkpoint=str(ckpt), ann=str(ann), img_dir=str(src / 'images'),
            name='baseline', conf=0.5, group_key='language', num_examples=2, out_dir=str(tmp_path / 'out'),
            dataset_name='Synthetic sample pages', val_ann=str(ann), val_img=str(src / 'images')))

        out = tmp_path / 'out'
        saved = json.loads((out / 'metrics' / 'baseline_metrics.json').read_text())
        assert saved['num_images'] == 2 and saved['num_ground_truth'] == len(data['annotations'])
        assert saved['dataset'] == 'Synthetic sample pages'
        assert 0 < saved['recommended_conf'] < 1                      # chosen on the validation pages
        assert set(saved['at_recommended_conf']) >= {'precision', 'recall', 'f1'}
        assert 0 < saved['high_recall_conf'] <= saved['recommended_conf'] or saved['high_recall_conf'] == 0.95
        assert set(saved['at_high_recall_conf']) >= {'precision', 'recall', 'f1'}
        assert saved['model_size_mb'] == pytest.approx(ckpt.stat().st_size / 1e6, abs=0.1)
        assert saved['inference_time'] > 0
        assert saved == json.loads(json.dumps(metrics))
        for f in ['baseline_confusion_matrix.png', 'baseline_per_class_ap.png', 'class_distribution.png']:
            assert (out / 'plots' / f).exists(), f
        assert len(list((out / 'predictions').glob('baseline_prediction_*.png'))) == 2


class TestBestThreshold:
    def test_picks_threshold_with_best_f1(self):
        """3 correct boxes scored 0.3, 2 wrong boxes scored 0.6: F1 is 0.75 at <= 0.3 and 0 above, so pick 0.3."""
        from ml.evaluation.coco_eval import best_f1_threshold
        dets = _perfect(score=0.3) + [
            {'image_id': 1, 'category_id': 2, 'bbox': [150, 150, 20, 20], 'score': 0.6},
            {'image_id': 2, 'category_id': 0, 'bbox': [0, 150, 20, 20], 'score': 0.6}]
        thr, f1 = best_f1_threshold(_gt(), dets)
        assert thr == pytest.approx(0.3)
        assert f1 == pytest.approx(0.75)

    def test_precision_recall_at(self):
        from ml.evaluation.coco_eval import precision_recall_at
        r = precision_recall_at(_gt(), _perfect(score=0.9), conf_threshold=0.5)
        assert (r['true_positives'], r['false_positives'], r['false_negatives']) == (3, 0, 0)
        assert r['f1'] == pytest.approx(1.0)


class TestHighRecall:
    def test_lowest_threshold_meeting_precision(self):
        """3 correct boxes at 0.3, 2 false at 0.2: precision is 1.0 for thresholds 0.25-0.3 and 0.6 at 0.2, so pick 0.25."""
        from ml.evaluation.coco_eval import high_recall_threshold
        dets = _perfect(score=0.3) + [
            {'image_id': 1, 'category_id': 2, 'bbox': [150, 150, 20, 20], 'score': 0.2},
            {'image_id': 2, 'category_id': 0, 'bbox': [0, 150, 20, 20], 'score': 0.2}]
        assert high_recall_threshold(_gt(), dets, min_precision=0.7) == pytest.approx(0.25)
        assert high_recall_threshold(_gt(), dets, min_precision=0.5) == pytest.approx(0.05)

    def test_dedupe_detections_per_page(self):
        from ml.evaluation.coco_eval import dedupe_detections
        dets = [{'image_id': 1, 'category_id': 26, 'bbox': [0, 0, 100, 20], 'score': 0.4},
                {'image_id': 1, 'category_id': 30, 'bbox': [0, 0, 100, 20], 'score': 0.3},
                {'image_id': 2, 'category_id': 30, 'bbox': [0, 0, 100, 20], 'score': 0.3}]   # other page: kept
        out = dedupe_detections(dets)
        assert sorted((d['image_id'], d['category_id']) for d in out) == [(1, 26), (2, 30)]
