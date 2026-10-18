import torch
import time
import numpy as np
from PIL import Image
from typing import Dict, Any, List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def disable_runtime_installs() -> None:
    """
    Stop Ultralytics from pip-installing packages while serving. It patches PIL.Image.open so that any image that
    fails to open triggers `pip install pi-heif` from the internet, which a bad upload would set off in production.
    The flag is read when Ultralytics is imported, so it is also switched off on an already-imported Ultralytics.
    """
    import os
    import sys
    os.environ["YOLO_AUTOINSTALL"] = "False"
    for name in ("ultralytics.utils", "ultralytics.utils.checks", "doclayout_yolo.utils", "doclayout_yolo.utils.checks"):
        if name in sys.modules:
            sys.modules[name].AUTOINSTALL = False

def load_model(model_path: str, model_type: str, device: str = 'cpu') -> Any:
    """Load Faster R-CNN or YOLOv8 model."""
    if model_type.lower() == 'fasterrcnn':
        # Training scripts save a state_dict, so rebuild the architecture and load the weights into it
        from ml.config import NUM_CLASSES
        from ml.models.baseline_frcnn import get_baseline_model
        model = get_baseline_model(NUM_CLASSES + 1, pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        return model
    elif model_type.lower() == 'yolov8':
        from ultralytics import YOLO
        model = YOLO(model_path)
        model.to(device)
        return model
    elif model_type.lower() == 'doclayout':
        # DocLayout-YOLO (a YOLOv10 fork); the IndicDLP authors' checkpoint uses the same 42 labels in the same order
        from doclayout_yolo import YOLOv10
        model = YOLOv10(model_path)
        model.to(device)
        return model
    else:
        raise ValueError("Unsupported model type")

def predict_single(model: Any, image: Any, model_type: str, device: str = 'cpu', conf: float = None) -> Dict:
    """
    Predict on a single image. Labels are 0-indexed class ids (index into CLASS_NAMES) for both models.
    conf: optional minimum score. Evaluation passes a very low value, since mAP needs low-confidence detections too.
    """
    if model_type.lower() == 'fasterrcnn':
        if isinstance(image, Image.Image):
            import torchvision.transforms as T
            image = T.ToTensor()(image)
        image = image.to(device)
        with torch.no_grad():
            output = model([image])[0]
        keep = output['scores'] >= conf if conf is not None else slice(None)
        return {
            'boxes': output['boxes'][keep].cpu().numpy(),
            # Faster R-CNN reserves 0 for background, so shift back to 0-indexed class ids
            'labels': output['labels'][keep].cpu().numpy() - 1,
            'scores': output['scores'][keep].cpu().numpy()
        }
    elif model_type.lower() in ('yolov8', 'doclayout'):
        kwargs = {'conf': conf} if conf is not None else {}
        if model_type.lower() == 'doclayout':
            from ml.config import AUTHORS_IMG_SIZE
            kwargs['imgsz'] = AUTHORS_IMG_SIZE
        results = model.predict(image, verbose=False, device=device, **kwargs)[0]
        boxes = results.boxes
        return {
            'boxes': boxes.xyxy.cpu().numpy(),
            'labels': boxes.cls.cpu().numpy(),
            'scores': boxes.conf.cpu().numpy()
        }
    return {}

def demo_inference(image: np.ndarray) -> Dict:
    """
    Placeholder detections for when no trained checkpoint exists. The boxes are random and
    flagged with is_demo=True so every UI can label them as synthetic, not model output.
    """
    h, w = image.shape[:2]
    rng = np.random.RandomState(42)  # local RNG: same boxes every call, no global side effects
    n = rng.randint(4, 9)
    boxes, labels, scores = [], [], []
    for _ in range(n):
        bw = rng.randint(w // 6, w // 2)
        bh = rng.randint(h // 8, h // 3)
        x1 = rng.randint(0, max(1, w - bw))
        y1 = rng.randint(0, max(1, h - bh))
        boxes.append([x1, y1, x1 + bw, y1 + bh])
        labels.append(rng.randint(0, 10))
        scores.append(round(rng.uniform(0.55, 0.97), 2))
    return {
        'boxes': np.array(boxes), 'labels': np.array(labels),
        'scores': np.array(scores), 'inference_time': 0.0, 'is_demo': True
    }

def predict_batch(model: Any, images: List, model_type: str, device: str = 'cpu') -> List[Dict]:
    """Predict on a batch of images."""
    return [predict_single(model, img, model_type, device) for img in images]

def measure_inference_time(model: Any, image: Any, model_type: str, num_runs: int = 10, warmup: int = 3, device: str = 'cpu') -> float:
    """Measure average inference time."""
    for _ in range(warmup):
        predict_single(model, image, model_type, device)
        
    start_time = time.time()
    for _ in range(num_runs):
        predict_single(model, image, model_type, device)
    end_time = time.time()
    
    return (end_time - start_time) / num_runs

def save_annotated_image(image: Any, predictions: Dict, save_path: str) -> None:
    """Save image with drawn predictions."""
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
    elif isinstance(image, Image.Image):
        image = np.array(image)
        
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    for box, label, score in zip(predictions.get('boxes', []), predictions.get('labels', []), predictions.get('scores', [])):
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(box[0], box[1] - 5, f"cls {int(label)}: {score:.2f}", color='red', fontsize=8)
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)

def dedupe_boxes(results: Dict, iou_threshold: float = 0.7) -> Dict:
    """
    Class-agnostic duplicate removal: when boxes overlap at IoU > iou_threshold, keep only the most confident one.
    DocLayout-YOLO scores every label separately, so at low thresholds one heading can come back as section-title,
    sub-section-title and subsub-section-title; a page element should be shown, and read, once.
    """
    import torchvision
    boxes = np.asarray(results['boxes'], dtype=np.float32).reshape(-1, 4)
    if len(boxes) == 0:
        return {**results, 'boxes': boxes, 'labels': np.asarray(results['labels']), 'scores': np.asarray(results['scores'])}
    scores = np.asarray(results['scores'], dtype=np.float32)
    keep = torchvision.ops.nms(torch.from_numpy(boxes), torch.from_numpy(scores), iou_threshold).numpy()
    return {**results, 'boxes': boxes[keep], 'labels': np.asarray(results['labels'])[keep], 'scores': scores[keep]}
