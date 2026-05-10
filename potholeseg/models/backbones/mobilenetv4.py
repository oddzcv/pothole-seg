from collections import OrderedDict
from typing import Any, Dict

import torch.nn as nn
import timm
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from . import register_backbone


class TimmBackboneWithFPN(nn.Module):
    """
    Generic timm feature extractor + FPN wrapper for TorchVision Mask R-CNN.

    This wrapper uses timm.create_model(..., features_only=True), then feeds
    multi-scale feature maps into TorchVision FeaturePyramidNetwork.
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        out_channels: int = 256,
        out_indices=None,
    ):
        super().__init__()

        available = timm.list_models(model_name)

        if len(available) == 0:
            similar = timm.list_models("*mobilenetv4*")
            raise ValueError(
                f"timm model not found: {model_name}\n"
                f"Available MobileNetV4-like models: {similar}"
            )

        if out_indices is None:
            temp_model = timm.create_model(
                model_name,
                pretrained=False,
                features_only=True,
            )

            num_features = len(temp_model.feature_info)

            if num_features >= 4:
                out_indices = tuple(range(num_features - 4, num_features))
            else:
                out_indices = tuple(range(num_features))

            del temp_model

        self.body = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )

        in_channels_list = self.body.feature_info.channels()
        reductions = self.body.feature_info.reduction()

        print("timm backbone:", model_name)
        print("out_indices:", out_indices)
        print("in_channels_list:", in_channels_list)
        print("feature reductions:", reductions)

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )

        self.out_channels = out_channels

    def forward(self, x):
        features = self.body(x)

        features = OrderedDict(
            (str(i), feature)
            for i, feature in enumerate(features)
        )

        return self.fpn(features)


def build_timm_mobilenetv4_fpn(cfg: Dict[str, Any]) -> nn.Module:
    backbone_cfg = cfg["model"]["backbone"]

    model_name = backbone_cfg.get(
        "timm_model_name",
        "mobilenetv4_conv_medium.e500_r256_in1k",
    )

    pretrained = backbone_cfg.get("pretrained", True)
    out_channels = backbone_cfg.get("out_channels", 256)
    out_indices = backbone_cfg.get("out_indices", None)

    if out_indices is not None:
        out_indices = tuple(out_indices)

    return TimmBackboneWithFPN(
        model_name=model_name,
        pretrained=pretrained,
        out_channels=out_channels,
        out_indices=out_indices,
    )


@register_backbone("mobilenetv4_conv_medium_fpn")
def build_mobilenetv4_conv_medium_fpn(cfg: Dict[str, Any]) -> nn.Module:
    return build_timm_mobilenetv4_fpn(cfg)


@register_backbone("mobilenetv4_conv_large_fpn")
def build_mobilenetv4_conv_large_fpn(cfg: Dict[str, Any]) -> nn.Module:
    return build_timm_mobilenetv4_fpn(cfg)