import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class COCODataset(Dataset):
    """
    PyTorch Dataset for COCO-format annotated images.
    """
    
    def __init__(self, annotation_file: str, img_dir: str, transforms=None, label_offset: int = 0):
        """
        Args:
            annotation_file (str): Path to the COCO JSON annotation file.
            img_dir (str): Directory with all the images.
            transforms (callable, optional): Optional transform to be applied on a sample.
            label_offset (int): Added to every category_id. Use 1 for Faster R-CNN, which reserves 0 for background.
        """
        self.img_dir = Path(img_dir)
        self.transforms = transforms
        self.label_offset = label_offset
        
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.coco = json.load(f)
            
        self.images = {img['id']: img for img in self.coco['images']}
        self.categories = {cat['id']: cat for cat in self.coco['categories']}
        
        self.img_to_anns = {}
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
            
        self.img_ids = list(self.images.keys())
        
    def __len__(self) -> int:
        return len(self.img_ids)
        
    def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, torch.Tensor]]:
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        
        img_path = self.img_dir / img_info['file_name']
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        
        # Load annotations
        anns = self.img_to_anns.get(img_id, [])
        
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in anns:
            x, y, w, h = ann['bbox']
            # COCO to Pascal VOC format [xmin, ymin, xmax, ymax]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'] + self.label_offset)
            areas.append(ann.get('area', w * h))
            iscrowd.append(ann.get('iscrowd', 0))
            
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
            areas = torch.empty((0,), dtype=torch.float32)
            iscrowd = torch.empty((0,), dtype=torch.int64)
            
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": iscrowd
        }
        
        if self.transforms:
            img = np.array(img)
            # Albumentations format
            transformed = self.transforms(image=img, bboxes=boxes.numpy(), category_ids=labels.numpy())
            img = transformed['image']
            
            if len(transformed['bboxes']) > 0:
                target['boxes'] = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
                target['labels'] = torch.as_tensor(transformed['category_ids'], dtype=torch.int64)
            else:
                target['boxes'] = torch.empty((0, 4), dtype=torch.float32)
                target['labels'] = torch.empty((0,), dtype=torch.int64)

        return img, target

def collate_fn(batch):
    """
    Collate function to handle variable number of boxes per image.
    """
    return tuple(zip(*batch))
