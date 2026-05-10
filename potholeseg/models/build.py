from typing import Any, Dict

from torchvision.models.detection import MaskRCNN

from potholeseg.models.backbones import build_backbone
from potholeseg.models.transform import NoResizeGeneralizedRCNNTransform


def build_model(cfg: Dict[str, Any]) -> MaskRCNN:
    """
    Build detection / instance segmentation model from YAML config.

    Currently supported:
        - Mask R-CNN
        - TorchVision backend
        - Registered FPN backbones
    """
    architecture = cfg["model"]["architecture"].lower()

    if architecture != "maskrcnn":
        raise ValueError(f"Unsupported architecture: {architecture}")

    backbone = build_backbone(cfg)

    roi_heads_cfg = cfg["model"]["roi_heads"]

    model = MaskRCNN(
        backbone=backbone,
        num_classes=cfg["data"]["num_classes"],
        box_score_thresh=roi_heads_cfg.get("score_thresh", 0.05),
        box_detections_per_img=roi_heads_cfg.get("detections_per_img", 100),
    )

    transform_cfg = cfg["model"].get("transform", {})

    if transform_cfg.get("no_resize_internal", True):
        model.transform = NoResizeGeneralizedRCNNTransform(
            image_mean=transform_cfg.get("image_mean", [0.485, 0.456, 0.406]),
            image_std=transform_cfg.get("image_std", [0.229, 0.224, 0.225]),
            size_divisible=transform_cfg.get("size_divisible", 32),
        )

    return model


def count_parameters(model) -> Dict[str, int]:
    """
    Count total, trainable, and frozen parameters.
    """
    total = 0
    trainable = 0

    for param in model.parameters():
        n = param.numel()
        total += n

        if param.requires_grad:
            trainable += n

    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def print_model_summary(model) -> None:
    """
    Print model parameter summary.
    """
    params = count_parameters(model)

    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Frozen parameters: {params['frozen']:,}")