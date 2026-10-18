"""
IndicDoc AI - Test Suite
Comprehensive tests for the ML pipeline.
"""
import pytest
import sys
import os
import json
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConfig:
    def test_config_imports(self):
        from ml.config import PROJECT_ROOT, NUM_CLASSES, CLASS_NAMES, IMG_SIZE
        assert NUM_CLASSES == 42
        assert len(CLASS_NAMES) == 42
        assert IMG_SIZE == 640
        assert PROJECT_ROOT.exists()

    def test_paths_exist(self):
        from ml.config import PROJECT_ROOT
        assert (PROJECT_ROOT / 'ml').exists()
        assert (PROJECT_ROOT / 'app').exists()
        assert (PROJECT_ROOT / 'docs').exists()


class TestPreprocessing:
    def test_resize(self):
        from ml.preprocessing.preprocess import resize_image
        img = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        resized = resize_image(img, (64, 64))
        assert resized.shape == (64, 64, 3)

    def test_normalize(self):
        from ml.preprocessing.preprocess import normalize_image
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        normed = normalize_image(img)
        assert normed.max() <= 1.0
        assert normed.min() >= 0.0

    def test_enhance_contrast(self):
        from ml.preprocessing.preprocess import enhance_contrast
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        enhanced = enhance_contrast(img)
        assert enhanced.shape == img.shape

    def test_deskew(self):
        from ml.preprocessing.preprocess import deskew_image
        img = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        deskewed = deskew_image(img)
        assert deskewed.shape == img.shape

    @staticmethod
    def _tilted_page(angle: float) -> np.ndarray:
        """White page with dark text lines, rotated by `angle` degrees (cv2.getRotationMatrix2D convention)."""
        import cv2
        img = np.full((1000, 800, 3), 255, np.uint8)
        for y in range(120, 880, 28):
            cv2.line(img, (100, y), (700, y), (0, 0, 0), 8)
        M = cv2.getRotationMatrix2D((400, 500), angle, 1.0)
        return cv2.warpAffine(img, M, (800, 1000), borderValue=(255, 255, 255))

    @pytest.mark.parametrize("angle", [-10, -5, -2, 0, 3, 8])
    def test_estimate_skew_angle(self, angle):
        from ml.preprocessing.preprocess import estimate_skew_angle
        assert estimate_skew_angle(self._tilted_page(angle)) == pytest.approx(angle, abs=0.5)

    @pytest.mark.parametrize("angle", [-10, -5, 3, 8])
    def test_deskew_straightens_page(self, angle):
        from ml.preprocessing.preprocess import deskew_image, estimate_skew_angle
        straightened = deskew_image(self._tilted_page(angle))
        assert estimate_skew_angle(straightened) == pytest.approx(0, abs=0.5)

    def test_deskew_blank_page_unchanged(self):
        from ml.preprocessing.preprocess import deskew_image
        blank = np.full((100, 100, 3), 255, np.uint8)
        assert np.array_equal(deskew_image(blank), blank)

    def test_preprocess_document(self):
        from ml.preprocessing.preprocess import preprocess_document
        img = np.random.randint(50, 200, (200, 300, 3), dtype=np.uint8)
        processed = preprocess_document(img)
        assert processed.shape[:2] == img.shape[:2]


class TestTransforms:
    def test_train_transforms(self):
        from ml.preprocessing.transforms import get_train_transforms
        t = get_train_transforms(640)
        assert t is not None

    def test_val_transforms(self):
        from ml.preprocessing.transforms import get_val_transforms
        t = get_val_transforms(640)
        assert t is not None

    def test_train_transforms_keep_pixels_in_unit_range(self):
        """Faster R-CNN normalizes internally, so transforms must output plain [0, 1] floats."""
        import torch
        from ml.preprocessing.transforms import get_train_transforms, get_val_transforms
        img = np.random.randint(0, 256, (100, 80, 3), dtype=np.uint8)
        for tf in (get_train_transforms(64), get_val_transforms(64)):
            out = tf(image=img, bboxes=[[10, 10, 40, 50]], category_ids=[1])['image']
            assert out.dtype == torch.float32
            assert out.min() >= 0.0 and out.max() <= 1.0

    def test_transforms_keep_page_aspect_ratio(self):
        """Inference sees pages at their natural shape, so training must not squash them into squares."""
        from ml.preprocessing.transforms import get_train_transforms, get_val_transforms
        img = np.full((1000, 700, 3), 255, np.uint8)  # portrait page
        for tf in (get_train_transforms(640), get_val_transforms(640)):
            out = tf(image=img, bboxes=[[70, 100, 350, 300]], category_ids=[1])
            assert tuple(out['image'].shape) == (3, 640, 448)
            assert list(out['bboxes'][0]) == pytest.approx([44.8, 64, 224, 192])

    def test_train_transforms_never_mirror_pages(self):
        """Mirrored text is not a realistic page, so the box must never move to the other side."""
        from ml.preprocessing.transforms import get_train_transforms
        img = np.full((100, 100, 3), 255, np.uint8)
        tf = get_train_transforms(100)  # page is 100x100, so no resizing either
        for _ in range(30):
            box = tf(image=img, bboxes=[[5, 10, 30, 40]], category_ids=[1])['bboxes'][0]
            assert list(box) == pytest.approx([5, 10, 30, 40])

    def test_inference_transforms(self):
        from ml.preprocessing.transforms import get_inference_transforms
        t = get_inference_transforms(640)
        assert t is not None


class TestSampleDataCreation:
    def test_create_sample_data(self, tmp_path):
        """Test sample data creation by directly calling the generation logic."""
        import random
        from PIL import Image, ImageDraw
        from ml.config import CLASS_NAMES

        output_dir = tmp_path / 'sample'
        img_dir = output_dir / 'images'
        img_dir.mkdir(parents=True)

        num_images = 5
        coco_data = {
            "info": {"description": "Test", "version": "1.0", "year": 2024},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [{"id": i, "name": name, "supercategory": "document"} for i, name in enumerate(CLASS_NAMES)]
        }

        ann_id = 1
        random.seed(42)
        for img_id in range(1, num_images + 1):
            width, height = 800, 1000
            img = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(img)
            file_name = f"sample_{img_id:04d}.jpg"
            coco_data["images"].append({
                "id": img_id, "width": width, "height": height, "file_name": file_name
            })
            num_boxes = random.randint(3, 10)
            for _ in range(num_boxes):
                cat_id = random.randint(0, len(CLASS_NAMES) - 1)
                x = random.randint(50, width - 200)
                y = random.randint(50, height - 200)
                w = random.randint(50, 150)
                h = random.randint(20, 100)
                draw.rectangle([x, y, x + w, y + h], outline=(100, 100, 100), width=2)
                coco_data["annotations"].append({
                    "id": ann_id, "image_id": img_id, "category_id": cat_id,
                    "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0, "segmentation": []
                })
                ann_id += 1
            img.save(img_dir / file_name)

        with open(output_dir / "annotations.json", "w") as f:
            json.dump(coco_data, f, indent=4)

        assert (output_dir / 'annotations.json').exists()
        assert len(list((output_dir / 'images').glob('*.jpg'))) == 5

        with open(output_dir / 'annotations.json') as f:
            data = json.load(f)
        assert 'images' in data
        assert 'annotations' in data
        assert 'categories' in data
        assert len(data['images']) == 5
        assert len(data['categories']) == 42


    def test_create_sample_dataset_is_consistent(self, tmp_path):
        """The real generator: every image has an entry, boxes stay inside the page, and a seed reproduces it."""
        from ml.datasets.create_sample_data import create_sample_dataset

        create_sample_dataset(num_images=6, output_dir=tmp_path / 'a', seed=7)
        create_sample_dataset(num_images=6, output_dir=tmp_path / 'b', seed=7)

        data = json.loads((tmp_path / 'a' / 'annotations.json').read_text())
        files = sorted(p.name for p in (tmp_path / 'a' / 'images').glob('*.jpg'))
        assert sorted(img['file_name'] for img in data['images']) == files and len(files) == 6
        assert {a['image_id'] for a in data['annotations']} == {img['id'] for img in data['images']}
        for a in data['annotations']:
            x, y, w, h = a['bbox']
            assert 0 <= x and 0 <= y and x + w <= 800 and y + h <= 1000
        assert data == json.loads((tmp_path / 'b' / 'annotations.json').read_text())


class TestDataset:
    @pytest.fixture
    def sample_data(self, tmp_path):
        """Create minimal sample data for testing."""
        img_dir = tmp_path / 'images'
        img_dir.mkdir()
        from PIL import Image
        for i in range(3):
            img = Image.new('RGB', (100, 100), color='white')
            img.save(img_dir / f'img_{i}.jpg')

        annotations = {
            'images': [
                {'id': i, 'width': 100, 'height': 100, 'file_name': f'img_{i}.jpg'}
                for i in range(3)
            ],
            'annotations': [
                {'id': j, 'image_id': i, 'category_id': j % 5, 'bbox': [10, 10, 30, 30],
                 'area': 900, 'iscrowd': 0, 'segmentation': []}
                for i in range(3) for j in range(i * 2, i * 2 + 2)
            ],
            'categories': [
                {'id': i, 'name': f'class_{i}', 'supercategory': 'doc'}
                for i in range(5)
            ]
        }
        ann_path = tmp_path / 'annotations.json'
        with open(ann_path, 'w') as f:
            json.dump(annotations, f)
        return str(ann_path), str(img_dir)

    def test_dataset_loading(self, sample_data):
        from ml.datasets.coco_dataset import COCODataset
        ann_path, img_dir = sample_data
        ds = COCODataset(ann_path, img_dir)
        assert len(ds) == 3

    def test_dataset_label_offset(self, sample_data):
        """Faster R-CNN reserves label 0 for background, so class ids are shifted by +1."""
        from ml.datasets.coco_dataset import COCODataset
        ann_path, img_dir = sample_data
        raw = COCODataset(ann_path, img_dir)[0][1]['labels'].tolist()
        shifted = COCODataset(ann_path, img_dir, label_offset=1)[0][1]['labels'].tolist()
        assert raw == [0, 1]
        assert shifted == [1, 2]

    def test_dataset_item(self, sample_data):
        from ml.datasets.coco_dataset import COCODataset
        ann_path, img_dir = sample_data
        ds = COCODataset(ann_path, img_dir)
        img, target = ds[0]
        assert 'boxes' in target
        assert 'labels' in target


class TestMetrics:
    def test_calculate_iou_perfect(self):
        from ml.evaluation.metrics import calculate_iou
        box1 = [0, 0, 10, 10]
        box2 = [0, 0, 10, 10]
        assert calculate_iou(box1, box2) == pytest.approx(1.0)

    def test_calculate_iou_no_overlap(self):
        from ml.evaluation.metrics import calculate_iou
        box1 = [0, 0, 10, 10]
        box3 = [20, 20, 30, 30]
        assert calculate_iou(box1, box3) == pytest.approx(0.0)

    def test_calculate_iou_partial(self):
        from ml.evaluation.metrics import calculate_iou
        box1 = [0, 0, 10, 10]
        box2 = [5, 5, 15, 15]
        iou = calculate_iou(box1, box2)
        assert 0.0 < iou < 1.0
        # Overlap area: 5*5=25, Union: 100+100-25=175
        assert iou == pytest.approx(25 / 175, abs=0.01)

    def test_calculate_ap(self):
        from ml.evaluation.metrics import calculate_ap
        precision = np.array([1.0, 0.8, 0.6, 0.4])
        recall = np.array([0.2, 0.4, 0.6, 0.8])
        ap = calculate_ap(precision, recall)
        assert 0.0 <= ap <= 1.0

    def test_save_metrics_report(self, tmp_path):
        from ml.evaluation.metrics import save_metrics_report
        metrics = {"precision": 0.85, "recall": 0.78, "f1": 0.81}
        filepath = str(tmp_path / "metrics" / "test_metrics.json")
        save_metrics_report(metrics, filepath)
        assert Path(filepath).exists()
        with open(filepath) as f:
            loaded = json.load(f)
        assert loaded["precision"] == 0.85

    def test_generate_confusion_matrix(self):
        from ml.evaluation.metrics import generate_confusion_matrix
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 1, 0, 2, 2]
        cm = generate_confusion_matrix(y_true, y_pred)
        assert cm.shape == (3, 3)
        assert cm[0, 0] == 2  # Class 0 correct


class TestModels:
    def test_baseline_model_creation(self):
        from ml.models.baseline_frcnn import get_baseline_model
        model = get_baseline_model(num_classes=43, pretrained=False)
        assert model is not None

    def test_baseline_model_forward(self):
        import torch
        from ml.models.baseline_frcnn import get_baseline_model
        model = get_baseline_model(num_classes=43, pretrained=False)
        model.eval()
        dummy = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            output = model([dummy[0]])
        assert 'boxes' in output[0]
        assert 'labels' in output[0]
        assert 'scores' in output[0]


class TestVisualization:
    def test_draw_predictions(self):
        from ml.evaluation.visualize import draw_predictions
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        # Updated signature: labels is List[str], no class_names parameter
        fig = draw_predictions(img, [[10, 10, 50, 50]], ['TestClass'], [0.9])
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_draw_ground_truth(self):
        from ml.evaluation.visualize import draw_ground_truth
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        fig = draw_ground_truth(img, [[10, 10, 50, 50]], ['TestClass'])
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_training_curves(self):
        from ml.evaluation.visualize import plot_training_curves
        fig = plot_training_curves([0.5, 0.4, 0.3], [0.6, 0.5, 0.4])
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_class_distribution(self, tmp_path):
        from ml.evaluation.visualize import plot_class_distribution
        ann = {
            "annotations": [{"category_id": 0}, {"category_id": 0}, {"category_id": 1}],
            "categories": [{"id": 0, "name": "A"}, {"id": 1, "name": "B"}]
        }
        ann_path = str(tmp_path / "ann.json")
        with open(ann_path, 'w') as f:
            json.dump(ann, f)
        fig = plot_class_distribution(ann_path)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestEndToEnd:
    def test_full_pipeline(self):
        """Test: config -> preprocess -> model creation -> inference."""
        import torch
        from ml.config import NUM_CLASSES, IMG_SIZE
        from ml.preprocessing.preprocess import resize_image, normalize_image
        from ml.models.baseline_frcnn import get_baseline_model

        # Create dummy image
        img = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)

        # Preprocess
        resized = resize_image(img, (IMG_SIZE, IMG_SIZE))
        assert resized.shape == (IMG_SIZE, IMG_SIZE, 3)

        # Model
        model = get_baseline_model(NUM_CLASSES + 1, pretrained=False)
        model.eval()

        # Inference
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            results = model([tensor])

        assert 'boxes' in results[0]
        assert 'labels' in results[0]
        assert 'scores' in results[0]


class TestCOCOToYOLO:
    def test_coco_to_yolo_conversion(self, tmp_path):
        from ml.models.proposed_yolov8 import coco_to_yolo

        coco_ann = {
            "images": [{"id": 1, "width": 100, "height": 100, "file_name": "img.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 20, 30, 40]}
            ],
            "categories": [{"id": 0, "name": "test"}]
        }
        ann_path = str(tmp_path / "ann.json")
        with open(ann_path, 'w') as f:
            json.dump(coco_ann, f)

        output_dir = str(tmp_path / "labels")
        coco_to_yolo(ann_path, output_dir)

        label_file = tmp_path / "labels" / "img.txt"
        assert label_file.exists()
        content = label_file.read_text().strip()
        parts = content.split()
        assert len(parts) == 5
        assert parts[0] == "0"  # category_id

    def test_coco_to_yolo_is_idempotent(self, tmp_path):
        """Re-running the conversion must not append duplicate labels."""
        from ml.models.proposed_yolov8 import coco_to_yolo

        coco_ann = {
            "images": [{"id": 1, "width": 200, "height": 100, "file_name": "img.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 3, "bbox": [20, 10, 40, 20]},
                {"id": 2, "image_id": 1, "category_id": 0, "bbox": [0, 0, 200, 100]},
            ],
            "categories": [{"id": i, "name": f"c{i}"} for i in range(4)]
        }
        ann_path = str(tmp_path / "ann.json")
        with open(ann_path, 'w') as f:
            json.dump(coco_ann, f)

        output_dir = str(tmp_path / "labels")
        coco_to_yolo(ann_path, output_dir)
        coco_to_yolo(ann_path, output_dir)

        lines = (tmp_path / "labels" / "img.txt").read_text().splitlines()
        assert len(lines) == 2
        # [x, y, w, h] = [20, 10, 40, 20] on a 200x100 image -> centre (0.2, 0.2), size (0.2, 0.2)
        assert [float(v) for v in lines[0].split()] == pytest.approx([3, 0.2, 0.2, 0.2, 0.2])
        assert [float(v) for v in lines[1].split()] == pytest.approx([0, 0.5, 0.5, 1.0, 1.0])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


class TestInference:
    def test_load_model_from_state_dict(self, tmp_path):
        """Training scripts save state_dicts; load_model must accept them."""
        import torch
        from ml.config import NUM_CLASSES
        from ml.models.baseline_frcnn import get_baseline_model
        from ml.inference.predict import load_model

        trained = get_baseline_model(NUM_CLASSES + 1, pretrained=False)
        path = tmp_path / 'baseline.pth'
        torch.save(trained.state_dict(), path)

        model = load_model(str(path), 'fasterrcnn', 'cpu')
        assert not model.training
        loaded = model.state_dict()
        for k, v in trained.state_dict().items():
            assert torch.equal(loaded[k], v)

    def test_frcnn_labels_are_zero_indexed(self, tmp_path):
        """predict_single returns 0-indexed class ids (index into CLASS_NAMES) for every model."""
        import torch
        from ml.inference.predict import predict_single

        class FakeFRCNN(torch.nn.Module):
            def forward(self, images):
                # Faster R-CNN reserves label 0 for background, so class 0 comes out as 1
                return [{'boxes': torch.tensor([[0., 0., 10., 10.]]),
                         'labels': torch.tensor([1]), 'scores': torch.tensor([0.9])}]

        out = predict_single(FakeFRCNN(), torch.zeros(3, 20, 20), 'fasterrcnn', 'cpu')
        assert out['labels'].tolist() == [0]


class TestTrainBaseline:
    def test_one_epoch_saves_best_checkpoint(self, tmp_path):
        """Train 1 epoch on 2 tiny images; the best checkpoint must load back into the model."""
        import argparse
        import torch
        from PIL import Image, ImageDraw
        from ml.config import NUM_CLASSES
        from ml.models.baseline_frcnn import get_baseline_model
        from ml.training.train_baseline import main

        img_dir = tmp_path / 'images'
        img_dir.mkdir()
        images, anns = [], []
        for i in range(2):
            img = Image.new('RGB', (200, 260), 'white')
            ImageDraw.Draw(img).rectangle([20, 30, 150, 90], fill='black')
            img.save(img_dir / f'p{i}.jpg')
            images.append({'id': i, 'width': 200, 'height': 260, 'file_name': f'p{i}.jpg'})
            anns.append({'id': i, 'image_id': i, 'category_id': 0, 'bbox': [20, 30, 130, 60], 'area': 7800, 'iscrowd': 0})
        ann_path = tmp_path / 'ann.json'
        ann_path.write_text(json.dumps({'images': images, 'annotations': anns,
                                        'categories': [{'id': 0, 'name': 'c0'}]}))

        output = tmp_path / 'best.pth'
        history = main(argparse.Namespace(
            train_ann=str(ann_path), train_img=str(img_dir), val_ann=str(ann_path), val_img=str(img_dir),
            epochs=1, lr=0.005, batch_size=2, img_size=128, num_workers=0,
            pretrained=False, output=str(output), use_mlflow=False, history=str(tmp_path / 'hist.json')))

        assert len(history['train_losses']) == len(history['val_losses']) == 1
        assert json.loads((tmp_path / 'hist.json').read_text()) == history
        assert history['device'] in ('cuda', 'mps', 'cpu') and history['batch_size'] == 2
        assert len(history['epoch_end_seconds']) == 1 and history['epoch_end_seconds'][0] > 0
        assert np.isfinite(history['train_losses'][0]) and np.isfinite(history['val_losses'][0])
        model = get_baseline_model(NUM_CLASSES + 1, pretrained=False)
        model.load_state_dict(torch.load(output, weights_only=True))


class TestTrainBudget:
    def test_epochs_fitting_budget(self):
        from ml.training.train_baseline import epochs_within_budget
        assert epochs_within_budget(max_epochs=20, budget_s=3600, first_epoch_s=1000) == 3
        assert epochs_within_budget(max_epochs=2, budget_s=3600, first_epoch_s=100) == 2   # capped by max_epochs
        assert epochs_within_budget(max_epochs=20, budget_s=50, first_epoch_s=100) == 1    # at least the epoch already done
        assert epochs_within_budget(max_epochs=20, budget_s=None, first_epoch_s=100) == 20  # no budget


class TestTrainYolo:
    def test_one_epoch_from_relative_paths(self, tmp_path, monkeypatch):
        """README usage passes relative paths; training must still find the images and save best weights."""
        import argparse
        import shutil
        from ml.config import PROCESSED_DATA_DIR
        from ml.inference.predict import load_model
        from ml.training.train_yolo import main

        # Copy 4 sample images + annotations into tmp and run from tmp with relative paths
        src = PROCESSED_DATA_DIR / 'sample'
        data = json.loads((src / 'annotations.json').read_text())
        keep = data['images'][:4]
        ids = {img['id'] for img in keep}
        (tmp_path / 'ds' / 'images').mkdir(parents=True)
        for img in keep:
            shutil.copy(src / 'images' / img['file_name'], tmp_path / 'ds' / 'images' / img['file_name'])
        data['images'] = keep
        data['annotations'] = [a for a in data['annotations'] if a['image_id'] in ids]
        (tmp_path / 'ds' / 'annotations.json').write_text(json.dumps(data))
        monkeypatch.chdir(tmp_path)

        output = tmp_path / 'best.pt'
        main(argparse.Namespace(
            train_ann='ds/annotations.json', train_img='ds/images', val_ann='ds/annotations.json', val_img='ds/images',
            convert_annotations=True, epochs=1, batch=2, imgsz=64, model_size='n', device='cpu',
            weights_dir=str(tmp_path / 'weights'), project=str(tmp_path / 'runs'), output=str(output),
            history=str(tmp_path / 'hist.csv')))

        assert output.exists()
        assert 'train/box_loss' in (tmp_path / 'hist.csv').read_text().splitlines()[0].replace(' ', '')
        model = load_model(str(output), 'yolov8', 'cpu')
        assert len(model.names) == 42
        assert not list(tmp_path.glob('yolov8*.pt'))  # pretrained weights must not land in the working directory


class TestDevice:
    def test_get_device_returns_usable_device(self):
        import torch
        from ml.config import get_device
        device = get_device()
        assert device in ('cuda', 'mps', 'cpu')
        torch.zeros(1, device=device)


class TestPdfPages:
    @staticmethod
    def _pdf(n_pages: int) -> bytes:
        import pymupdf
        doc = pymupdf.open()
        for i in range(n_pages):
            page = doc.new_page(width=595, height=842)  # A4 in points
            page.insert_text((72, 72 + 20 * i), f"Page {i + 1}", fontsize=24)
        return doc.tobytes()

    def test_page_count(self):
        from ml.preprocessing.preprocess import pdf_page_count
        assert pdf_page_count(self._pdf(3)) == 3

    def test_renders_requested_page_at_150_dpi(self):
        from ml.preprocessing.preprocess import render_pdf_page
        data = self._pdf(3)
        img = render_pdf_page(data, page_index=2)
        assert img.shape == (1755, 1240, 3)  # A4 (595 x 842 pt) at 150 dpi, rounded up by PyMuPDF
        first = render_pdf_page(data, page_index=0)
        assert not np.array_equal(img, first)  # different text position -> different page

    def test_huge_page_is_rendered_at_lower_dpi(self):
        """A 100 x 100 inch page at 150 dpi would be 225 megapixels; the renderer lowers the dpi to fit the cap."""
        import pymupdf
        from ml.preprocessing.preprocess import render_pdf_page
        doc = pymupdf.open()
        doc.new_page(width=7200, height=7200)
        img = render_pdf_page(doc.tobytes(), page_index=0, max_pixels=20_000_000)
        assert img.shape[0] * img.shape[1] <= 20_000_000 * 1.01

    def test_out_of_range_page_raises(self):
        from ml.preprocessing.preprocess import render_pdf_page
        with pytest.raises(IndexError):
            render_pdf_page(self._pdf(1), page_index=1)


class TestDecodeImage:
    @staticmethod
    def _png(size, mode='RGB') -> bytes:
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.new(mode, size, 255 if mode != 'RGB' else 'white').save(buf, format='PNG')
        return buf.getvalue()

    def test_decodes_normal_image_to_bgr(self):
        from ml.preprocessing.preprocess import decode_image
        img = decode_image(self._png((60, 40)))
        assert img.shape == (40, 60, 3)

    def test_rejects_too_many_pixels_before_decoding(self):
        """A tiny file can declare a huge image (decompression bomb); the header check must catch it."""
        from ml.preprocessing.preprocess import decode_image
        bomb = self._png((12000, 12000), mode='1')  # 144 megapixels, a few KB on disk
        assert len(bomb) < 100_000
        with pytest.raises(ValueError, match='megapixels'):
            decode_image(bomb, max_pixels=40_000_000)

    def test_rejects_non_image(self):
        from ml.preprocessing.preprocess import decode_image
        with pytest.raises(ValueError):
            decode_image(b'not an image')


class TestOfflineModelBuild:
    def test_untrained_build_downloads_nothing(self, monkeypatch):
        """Loading a trained checkpoint must work offline: building the empty architecture may not fetch weights."""
        import torchvision.models._api as tv_api
        from ml.models.baseline_frcnn import get_baseline_model

        def no_network(*args, **kwargs):
            raise AssertionError(f"tried to fetch weights {args[0] if args else kwargs}")

        # torchvision fetches weights through its own reference (the local cache would hide a real download)
        monkeypatch.setattr(tv_api, 'load_state_dict_from_url', no_network)
        model = get_baseline_model(43, pretrained=False)
        kinds = {type(m).__name__ for m in model.modules() if 'BatchNorm' in type(m).__name__}
        assert kinds == {'FrozenBatchNorm2d'}   # same layer types (and state_dict keys) as the trained model

    def test_trained_checkpoint_loads_strictly(self):
        import torch
        from ml.config import BASELINE_CHECKPOINT, NUM_CLASSES
        from ml.models.baseline_frcnn import get_baseline_model
        if not BASELINE_CHECKPOINT.exists():
            pytest.skip('no trained checkpoint in outputs/models')
        model = get_baseline_model(NUM_CLASSES + 1, pretrained=False)
        model.load_state_dict(torch.load(BASELINE_CHECKPOINT, map_location='cpu', weights_only=True), strict=True)


class TestAuthorsModel:
    """The IndicDLP authors' DocLayout-YOLO checkpoint (CC-BY 4.0, Zenodo record 15881917)."""

    def test_loads_and_predicts_zero_indexed_labels(self):
        from PIL import Image
        from ml.config import AUTHORS_CHECKPOINT, CLASS_NAMES
        from ml.inference.predict import load_model, predict_single
        if not AUTHORS_CHECKPOINT.exists():
            pytest.skip('authors checkpoint not downloaded (python scripts/download_models.py)')
        model = load_model(str(AUTHORS_CHECKPOINT), 'doclayout', 'cpu')
        assert [model.names[i] for i in range(len(model.names))] == CLASS_NAMES
        page = Image.open(Path(__file__).parent.parent / 'app' / 'assets' / 'samples' / 'english_research_papers.png').convert('RGB')
        out = predict_single(model, page, 'doclayout', 'cpu', conf=0.25)
        assert len(out['boxes']) > 10                      # a real research-paper page has many elements
        assert 0 <= out['labels'].min() and out['labels'].max() < len(CLASS_NAMES)

    def test_runtime_installs_disabled_for_doclayout_yolo(self):
        import doclayout_yolo.utils.checks as checks
        from ml.inference.predict import disable_runtime_installs
        disable_runtime_installs()
        assert checks.AUTOINSTALL is False


class TestDedupe:
    def test_same_box_with_several_labels_keeps_best(self):
        from ml.inference.predict import dedupe_boxes
        r = {'boxes': np.array([[0, 0, 100, 20], [1, 0, 100, 21], [0, 50, 100, 80]], dtype=float),
             'labels': np.array([26, 30, 21]), 'scores': np.array([0.33, 0.27, 0.9])}
        out = dedupe_boxes(r)
        assert out['labels'].tolist() == [21, 26]          # sorted by score; the 0.27 duplicate is gone
        assert out['scores'].tolist() == pytest.approx([0.9, 0.33])

    def test_nested_but_different_boxes_are_kept(self):
        """A list inside a sidebar overlaps it, but not at IoU > 0.7: both stay."""
        from ml.inference.predict import dedupe_boxes
        r = {'boxes': np.array([[0, 0, 400, 400], [10, 10, 200, 100]], dtype=float),
             'labels': np.array([27, 40]), 'scores': np.array([0.8, 0.7])}
        assert len(dedupe_boxes(r)['boxes']) == 2

    def test_empty(self):
        from ml.inference.predict import dedupe_boxes
        r = {'boxes': np.zeros((0, 4)), 'labels': np.zeros(0), 'scores': np.zeros(0)}
        assert len(dedupe_boxes(r)['boxes']) == 0
