import albumentations as A
from albumentations.pytorch import ToTensorV2

# Faster R-CNN normalizes with ImageNet mean/std inside the model (GeneralizedRCNNTransform),
# so these transforms only scale pixels to [0, 1]. Normalizing here as well would do it twice.

def get_train_transforms(img_size: int = 640):
    """
    Returns Albumentations transforms for training.
    """
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),  # keep the page's aspect ratio, as at inference
        # No flips: a mirrored page is not a realistic document. Simulate scan quality instead.
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.ToFloat(max_value=255.0),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'], min_visibility=0.2))

def get_val_transforms(img_size: int = 640):
    """
    Returns Albumentations transforms for validation.
    """
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),  # keep the page's aspect ratio, as at inference
        A.ToFloat(max_value=255.0),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

def get_inference_transforms(img_size: int = 640):
    """
    Returns Albumentations transforms for inference (without bboxes).
    """
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),  # keep the page's aspect ratio, as at inference
        A.ToFloat(max_value=255.0),
        ToTensorV2()
    ])
