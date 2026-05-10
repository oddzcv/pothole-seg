from typing import Dict, Any

import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from . import register_backbone


@register_backbone("mobilenetv3_large_fpn")
def build_mobilenetv3_large_fpn(cfg: Dict[str, Any]) -> nn.Module:
    """
    Build MobileNetV3-Large-FPN backbone for TorchVision Mask R-CNN.

    Feature maps:
        features[3]  -> 24 channels
        features[6]  -> 40 channels
        features[12] -> 112 channels
        features[16] -> 960 channels

    FPN output channels:
        default = 256
    """
    backbone_cfg = cfg["model"]["backbone"]

    pretrained = backbone_cfg.get("pretrained", True)
    out_channels = backbone_cfg.get("out_channels", 256)

    weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    body = mobilenet_v3_large(weights=weights).features

    return_layers = {
        "3": "0",
        "6": "1",
        "12": "2",
        "16": "3",
    }

    in_channels_list = [24, 40, 112, 960]

    backbone = BackboneWithFPN(
        body,
        return_layers,
        in_channels_list,
        out_channels,
        extra_blocks=LastLevelMaxPool(),
    )

    backbone.out_channels = out_channels

    return backbone