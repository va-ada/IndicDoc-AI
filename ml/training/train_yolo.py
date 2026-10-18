import argparse
import shutil
import yaml
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from ml.config import *
from ml.models.proposed_yolov8 import get_proposed_model, coco_to_yolo

def create_yolo_yaml(train_img_dir: str, val_img_dir: str, num_classes: int, class_names: list, output_path: str):
    # Absolute paths: ultralytics resolves relative paths against the YAML's own folder, not the working directory
    data = {
        'train': str(Path(train_img_dir).resolve()),
        'val': str(Path(val_img_dir).resolve()),
        'nc': num_classes,
        'names': class_names
    }
    with open(output_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)

def main(args) -> Path:
    project = Path(args.project)
    project.mkdir(parents=True, exist_ok=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # 1. Convert COCO to YOLO if needed. Ultralytics finds labels by swapping 'images' for 'labels' in each image path
    if args.convert_annotations:
        print("Converting COCO annotations to YOLO format...")
        coco_to_yolo(args.train_ann, str(Path(args.train_img).parent / 'labels'))
        coco_to_yolo(args.val_ann, str(Path(args.val_img).parent / 'labels'))

    # 2. Create YAML
    yaml_path = project / 'yolo_dataset.yaml'
    create_yolo_yaml(args.train_img, args.val_img, NUM_CLASSES, CLASS_NAMES, str(yaml_path))

    # 3. Train
    print("Initializing YOLO model...")
    model = get_proposed_model(model_size=args.model_size, weights_dir=args.weights_dir)

    print("Starting training...")
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(project),
        name='indicdoc_yolo',
        exist_ok=True,
        seed=SEED,
        workers=NUM_WORKERS,
        fliplr=0.0,  # no mirror flips: a mirrored page is not a realistic document
        **({'time': args.hours} if getattr(args, 'hours', None) else {}),  # ultralytics fits the schedule to this budget
    )

    # 4. Copy the best weights to the path the app and backend load from
    best = Path(model.trainer.save_dir) / 'weights' / 'best.pt'
    shutil.copy(best, output)
    print(f"Training complete. Best weights copied to {output}")
    history = getattr(args, 'history', None)
    if history:  # ultralytics logs per-epoch losses and val mAP in results.csv
        Path(history).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(model.trainer.save_dir) / 'results.csv', history)
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_ann', type=str, required=True, help="Path to train annotations JSON")
    parser.add_argument('--train_img', type=str, required=True, help="Path to train images directory")
    parser.add_argument('--val_ann', type=str, required=True, help="Path to val annotations JSON")
    parser.add_argument('--val_img', type=str, required=True, help="Path to val images directory")
    parser.add_argument('--convert_annotations', action='store_true', help="Convert COCO to YOLO before training")
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch', type=int, default=BATCH_SIZE)
    parser.add_argument('--imgsz', type=int, default=IMG_SIZE)
    parser.add_argument('--model_size', type=str, default='s', choices=['n', 's', 'm', 'l', 'x'])
    parser.add_argument('--device', type=str, default=get_device())
    parser.add_argument('--weights_dir', type=str, default=str(MODEL_DIR), help="Where pretrained COCO weights are cached")
    parser.add_argument('--project', type=str, default=str(OUTPUT_DIR / 'yolo_runs'))
    parser.add_argument('--output', type=str, default=str(PROPOSED_CHECKPOINT), help="Where to copy the best weights")
    parser.add_argument('--hours', type=float, default=None, help="Time budget in hours (overrides --epochs)")
    parser.add_argument('--history', type=str, default=str(METRICS_DIR / 'proposed_training_history.csv'),
                        help="Where to copy ultralytics' per-epoch results.csv")
    args = parser.parse_args()
    main(args)
