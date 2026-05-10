import torch
from torch.utils.data import DataLoader, Dataset

from potholeseg.config import load_config
from potholeseg.models import build_model
from potholeseg.engine import (
    build_optimizer,
    WarmupCosineScheduler,
    train_one_epoch,
    validate_loss,
)


class TinyDetectionDataset(Dataset):
    """
    Tiny synthetic dataset for checking training engine.
    """

    def __init__(self, length=2, image_size=224):
        self.length = length
        self.image_size = image_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        h = self.image_size
        w = self.image_size

        image = torch.rand(3, h, w, dtype=torch.float32)

        boxes = torch.tensor(
            [
                [40.0, 50.0, 140.0, 160.0],
            ],
            dtype=torch.float32,
        )

        labels = torch.tensor([1], dtype=torch.int64)

        masks = torch.zeros((1, h, w), dtype=torch.uint8)
        masks[0, 50:160, 40:140] = 1

        area = (boxes[:, 2] - boxes[:, 0]) * (
            boxes[:, 3] - boxes[:, 1]
        )

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((1,), dtype=torch.int64),

            # Metadata should be ignored during loss computation.
            "original_size": torch.tensor([h, w], dtype=torch.int64),
            "processed_size": torch.tensor([h, w], dtype=torch.int64),
            "scale_factor": torch.tensor([1.0, 1.0], dtype=torch.float32),
        }

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    cfg = load_config("configs/maskrcnn_mobilenetv3_large_fpn.yaml")

    # Keep local test lightweight.
    cfg["train"]["batch_size"] = 1
    cfg["train"]["val_batch_size"] = 1
    cfg["train"]["num_workers"] = 0
    cfg["train"]["amp"] = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = build_model(cfg)
    model.to(device)

    optimizer = build_optimizer(model, cfg)
    scheduler = WarmupCosineScheduler(optimizer, cfg)

    train_ds = TinyDetectionDataset(length=2, image_size=224)
    val_ds = TinyDetectionDataset(length=1, image_size=224)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["val_batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    scaler = torch.amp.GradScaler(
        device="cuda",
        enabled=False,
    )

    train_metrics, global_step = train_one_epoch(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loader=train_loader,
        device=device,
        epoch=1,
        cfg=cfg,
        scaler=scaler,
        start_global_step=0,
    )

    print("\nTrain metrics:")
    for key, value in train_metrics.items():
        print(f"  {key}: {value:.6f}")

    print("Global step:", global_step)

    val_metrics = validate_loss(
        model=model,
        loader=val_loader,
        device=device,
    )

    print("\nValidation loss metrics:")
    for key, value in val_metrics.items():
        print(f"  {key}: {value:.6f}")

    required_train = {
        "loss",
        "loss_classifier",
        "loss_box_reg",
        "loss_mask",
        "loss_objectness",
        "loss_rpn_box_reg",
    }

    missing_train = required_train - set(train_metrics.keys())

    if missing_train:
        raise RuntimeError(f"Missing train metrics: {missing_train}")

    required_val = {
        "val_loss",
        "val_loss_classifier",
        "val_loss_box_reg",
        "val_loss_mask",
        "val_loss_objectness",
        "val_loss_rpn_box_reg",
    }

    missing_val = required_val - set(val_metrics.keys())

    if missing_val:
        raise RuntimeError(f"Missing val metrics: {missing_val}")

    print("\nTraining engine check passed.")


if __name__ == "__main__":
    main()