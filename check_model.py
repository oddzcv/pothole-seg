import torch

from potholeseg.config import load_config
from potholeseg.models import build_model, print_model_summary


def make_dummy_batch(device):
    """
    Create dummy image and target to test Mask R-CNN forward loss.

    Image size is small for local test.
    """
    image = torch.rand(3, 416, 416, dtype=torch.float32)

    boxes = torch.tensor(
        [
            [60.0, 80.0, 180.0, 220.0],
            [220.0, 200.0, 340.0, 360.0],
        ],
        dtype=torch.float32,
    )

    labels = torch.tensor([1, 1], dtype=torch.int64)

    masks = torch.zeros((2, 416, 416), dtype=torch.uint8)
    masks[0, 80:220, 60:180] = 1
    masks[1, 200:360, 220:340] = 1

    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    iscrowd = torch.zeros((2,), dtype=torch.int64)

    target = {
        "boxes": boxes,
        "labels": labels,
        "masks": masks,
        "image_id": torch.tensor([0], dtype=torch.int64),
        "area": area,
        "iscrowd": iscrowd,
    }

    image = image.to(device)
    target = {
        k: v.to(device)
        for k, v in target.items()
    }

    return [image], [target]


def main():
    cfg = load_config("configs/maskrcnn_mobilenetv3_large_fpn.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = build_model(cfg)
    model.to(device)

    print_model_summary(model)

    images, targets = make_dummy_batch(device)

    model.train()

    with torch.amp.autocast(
        device_type=device.type,
        enabled=(device.type == "cuda"),
    ):
        loss_dict = model(images, targets)

    print("\nLoss dict:")
    for name, value in loss_dict.items():
        print(f"  {name}: {float(value.detach().cpu()):.6f}")

    required_losses = {
        "loss_classifier",
        "loss_box_reg",
        "loss_mask",
        "loss_objectness",
        "loss_rpn_box_reg",
    }

    missing = required_losses - set(loss_dict.keys())

    if missing:
        raise RuntimeError(f"Missing losses: {missing}")

    print("\nModel forward loss check passed.")

    model.eval()

    with torch.no_grad():
        outputs = model(images)

    output = outputs[0]

    print("\nInference output keys:", list(output.keys()))
    print("Boxes shape:", tuple(output["boxes"].shape))
    print("Scores shape:", tuple(output["scores"].shape))
    print("Labels shape:", tuple(output["labels"].shape))
    print("Masks shape:", tuple(output["masks"].shape))

    if output["masks"].numel() > 0:
        print("First mask shape:", tuple(output["masks"][0, 0].shape))

    print("\nModel inference check passed.")


if __name__ == "__main__":
    main()