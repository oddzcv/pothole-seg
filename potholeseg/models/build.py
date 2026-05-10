from typing import Any, Dict

from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator

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

    rpn_cfg = cfg["model"].get("rpn", {})
    rpn_anchor_generator = build_rpn_anchor_generator(cfg)

    model = MaskRCNN(
        backbone=backbone,
        num_classes=cfg["data"]["num_classes"],
        rpn_anchor_generator=rpn_anchor_generator,
        rpn_pre_nms_top_n_train=rpn_cfg.get("pre_nms_top_n_train", 2000),
        rpn_pre_nms_top_n_test=rpn_cfg.get("pre_nms_top_n_test", 1000),
        rpn_post_nms_top_n_train=rpn_cfg.get("post_nms_top_n_train", 2000),
        rpn_post_nms_top_n_test=rpn_cfg.get("post_nms_top_n_test", 1000),
        rpn_nms_thresh=rpn_cfg.get("nms_thresh", 0.7),
        rpn_fg_iou_thresh=rpn_cfg.get("fg_iou_thresh", 0.7),
        rpn_bg_iou_thresh=rpn_cfg.get("bg_iou_thresh", 0.3),
        rpn_batch_size_per_image=rpn_cfg.get("batch_size_per_image", 256),
        rpn_positive_fraction=rpn_cfg.get("positive_fraction", 0.5),
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

def build_rpn_anchor_generator(cfg):
    rpn_cfg = cfg["model"].get("rpn", {})
    anchor_cfg = rpn_cfg.get("anchor_generator", {})

    if not anchor_cfg.get("enabled", False):
        return None

    sizes = anchor_cfg.get(
        "sizes",
        [[16], [32], [64], [128], [256]],
    )
    aspect_ratios = anchor_cfg.get(
        "aspect_ratios",
        [[0.5, 1.0, 2.0]] * len(sizes),
    )

    sizes = tuple(tuple(int(v) for v in level) for level in sizes)
    aspect_ratios = tuple(tuple(float(v) for v in level) for level in aspect_ratios)

    return AnchorGenerator(
        sizes=sizes,
        aspect_ratios=aspect_ratios,
    )