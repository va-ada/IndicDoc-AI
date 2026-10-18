import os
import json
from pathlib import Path
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

def get_proposed_model(model_size: str = 'n', weights_dir: str = None):
    """
    Initialize YOLOv8 model for fine-tuning.
    Args:
        model_size (str): 'n', 's', 'm', 'l', or 'x' for YOLOv8 model sizes.
        weights_dir (str, optional): Folder to download/cache the COCO-pretrained weights in.
            Without it, ultralytics downloads into the current working directory.
    """
    if YOLO is None:
        raise ImportError("ultralytics is not installed. Run: pip install ultralytics")

    # Load a pretrained YOLOv8 model
    model_name = f'yolov8{model_size}.pt'
    if weights_dir:
        Path(weights_dir).mkdir(parents=True, exist_ok=True)
        model_name = str(Path(weights_dir) / model_name)
    model = YOLO(model_name)
    return model

def coco_to_yolo(coco_json_path: str, output_dir: str):
    """
    Converts COCO format annotations to YOLO format txt files.
    """
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)
        
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Mapping image id to file
    img_dict = {img['id']: img for img in coco['images']}

    # Collect every label line first, then write each file once with 'w',
    # so re-running the conversion overwrites instead of appending duplicates
    lines_per_file = {}

    for ann in coco['annotations']:
        img_id = ann['image_id']
        cat_id = ann['category_id'] # Note: YOLO classes are 0-indexed. Adjust if necessary.
        img_info = img_dict[img_id]
        
        img_w = img_info['width']
        img_h = img_info['height']
        
        # COCO: [x_min, y_min, width, height]
        x_min, y_min, w, h = ann['bbox']
        
        # YOLO: [x_center, y_center, width, height] (normalized)
        x_center = (x_min + w / 2) / img_w
        y_center = (y_min + h / 2) / img_h
        norm_w = w / img_w
        norm_h = h / img_h
        
        txt_filename = Path(img_info['file_name']).stem + '.txt'
        lines_per_file.setdefault(txt_filename, []).append(
            f"{cat_id} {x_center} {y_center} {norm_w} {norm_h}\n")

    for txt_filename, lines in lines_per_file.items():
        with open(output_dir / txt_filename, 'w', encoding='utf-8') as txt_f:
            txt_f.writelines(lines)
