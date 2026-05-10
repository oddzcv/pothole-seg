import torch

from potholeseg.config import load_config
from potholeseg.models.backbones import build_backbone, available_backbones


CONFIGS = [
    "configs/maskrcnn_mobilenetv3_large_fpn.yaml",
    "configs/maskrcnn_mobilenetv4_conv_medium_fpn.yaml",
    "configs/maskrcnn_mobilenetv4_conv_large_fpn.yaml",
]


def check_one_config(config_path: str):
    print("=" * 80)
    print("Checking:", config_path)
    print("=" * 80)

    cfg = load_config(config_path)

    print("Project:", cfg["project"]["name"])
    print("Backbone:", cfg["model"]["backbone"]["name"])
    print("Available backbones:", available_backbones())

    backbone = build_backbone(cfg)
    backbone.eval()

    x = torch.rand(1, 3, 416, 416)

    with torch.no_grad():
        features = backbone(x)

    print("\nFeature maps:")
    for name, feat in features.items():
        print(f"  {name}: {tuple(feat.shape)}")

    print("out_channels:", backbone.out_channels)

    if backbone.out_channels != 256:
        raise RuntimeError(f"Expected out_channels=256, got {backbone.out_channels}")

    print("Passed:", config_path)


def main():
    for config_path in CONFIGS:
        check_one_config(config_path)

    print("\nAll backbone configs passed.")


if __name__ == "__main__":
    main()