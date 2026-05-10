from typing import Callable, Dict, Any

import torch.nn as nn


BACKBONE_REGISTRY: Dict[str, Callable[[Dict[str, Any]], nn.Module]] = {}


def register_backbone(name: str):
    """
    Register a backbone builder function.

    Example:
        @register_backbone("mobilenetv3_large_fpn")
        def build_backbone(cfg):
            ...
    """
    def wrapper(fn: Callable[[Dict[str, Any]], nn.Module]):
        if name in BACKBONE_REGISTRY:
            raise ValueError(f"Backbone already registered: {name}")

        BACKBONE_REGISTRY[name] = fn
        return fn

    return wrapper


def build_backbone(cfg: Dict[str, Any]) -> nn.Module:
    """
    Build backbone from config.
    """
    backbone_name = cfg["model"]["backbone"]["name"]

    if backbone_name not in BACKBONE_REGISTRY:
        available = ", ".join(sorted(BACKBONE_REGISTRY.keys()))
        raise ValueError(
            f"Backbone not registered: {backbone_name}. "
            f"Available backbones: {available}"
        )

    return BACKBONE_REGISTRY[backbone_name](cfg)


def available_backbones():
    """
    Return list of registered backbone names.
    """
    return sorted(BACKBONE_REGISTRY.keys())


# Import registered backbones here.
# Do not remove this import because it triggers registration.
from .mobilenetv3 import build_mobilenetv3_large_fpn  # noqa: E402,F401
from .mobilenetv4 import (  # noqa: E402,F401
    build_mobilenetv4_conv_medium_fpn,
    build_mobilenetv4_conv_large_fpn,
)