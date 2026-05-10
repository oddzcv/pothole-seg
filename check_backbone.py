import torch

from potholeseg.config import load_config
from potholeseg.models.backbones import (
    build_backbone,
    available_backbones,
)


def main():
    cfg = load_config("configs/maskrcnn_mobilenetv3_large_fpn.yaml")

    print("Available backbones:", available_backbones())
    print("Backbone from config:", cfg["model"]["backbone"]["name"])

    backbone = build_backbone(cfg)
    backbone.eval()

    # Ukuran kecil untuk test lokal agar tidak berat.
    x = torch.rand(1, 3, 416, 416)

    with torch.no_grad():
        features = backbone(x)

    print("\nBackbone output feature maps:")
    for name, feat in features.items():
        print(f"  {name}: {tuple(feat.shape)}")

    print("\nBackbone out_channels:", backbone.out_channels)
    print("Backbone check passed.")


if __name__ == "__main__":
    main()