import os
import json
import random
from pathlib import Path
from PIL import Image, ImageDraw

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from ml.config import PROCESSED_DATA_DIR, CLASS_NAMES, SEED

def create_sample_dataset(num_images: int = 30, output_dir: Path = PROCESSED_DATA_DIR / 'sample', seed: int = SEED):
    """
    Creates a small synthetic sample dataset for testing the pipeline.
    This is strictly for testing code and NOT for real training.
    """
    rng = random.Random(seed)  # seeded so the committed sample data can be regenerated exactly
    output_dir = Path(output_dir)
    img_dir = output_dir / 'images'
    
    img_dir.mkdir(parents=True, exist_ok=True)
    
    coco_data = {
        "info": {
            "description": "Synthetic Sample Dataset for IndicDoc Pipeline Testing",
            "version": "1.0",
            "year": 2024
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name, "supercategory": "document"} for i, name in enumerate(CLASS_NAMES)]
    }
    
    ann_id = 1
    
    for img_id in range(1, num_images + 1):
        width, height = 800, 1000
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        
        file_name = f"sample_{img_id:04d}.jpg"
        
        coco_data["images"].append({
            "id": img_id,
            "width": width,
            "height": height,
            "file_name": file_name
        })
        
        # Draw some random bounding boxes
        num_boxes = rng.randint(3, 10)
        for _ in range(num_boxes):
            cat_id = rng.randint(0, len(CLASS_NAMES) - 1)
            
            x = rng.randint(50, width - 200)
            y = rng.randint(50, height - 200)
            w = rng.randint(50, 150)
            h = rng.randint(20, 100)
            
            # Draw box on image
            color = (rng.randint(0,200), rng.randint(0,200), rng.randint(0,200))
            draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
            
            coco_data["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
                "segmentation": []
            })
            ann_id += 1
            
        img.save(img_dir / file_name)
        
    with open(output_dir / "annotations.json", "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=4)
        
    print(f"Sample dataset created at {output_dir}")
    print(f"Generated {num_images} images with {ann_id - 1} annotations.")

if __name__ == "__main__":
    create_sample_dataset()
