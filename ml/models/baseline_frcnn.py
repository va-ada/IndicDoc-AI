import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import misc as misc_nn_ops
from ml.config import IMG_SIZE

def get_baseline_model(num_classes: int, pretrained: bool = True):
    """
    Returns a Faster R-CNN model modified for the specified number of classes.
    Note: num_classes should include the background class (e.g., 42 IndicDLP classes + 1 bg = 43).

    pretrained=True starts from COCO weights (for training). pretrained=False builds the same architecture
    without fetching any weights, for loading a trained checkpoint (works offline).
    """
    # min_size/max_size: torchvision's 800/1333 default upscales our 640px pages ~1.8x internally;
    # working near the training resolution keeps a CPU/laptop-GPU training budget realistic
    size = dict(min_size=IMG_SIZE, max_size=1024)
    if pretrained:
        # Load a model pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT', **size)
    else:
        # fasterrcnn_resnet50_fpn(weights=None) would still download ImageNet backbone weights, and dropping
        # them would switch to plain BatchNorm and change the state_dict. Build the trained layout directly:
        # ResNet-50 with frozen batch norm + FPN (3 trainable stages, as when training from COCO weights)
        backbone = torchvision.models.resnet50(weights=None, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
        model = FasterRCNN(_resnet_fpn_extractor(backbone, 3), num_classes=91, **size)

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
