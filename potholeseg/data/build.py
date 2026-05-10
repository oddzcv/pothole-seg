from pathlib import Path
from typing import Any, Dict

from pycocotools.coco import COCO
from torch.utils.data import DataLoader

from potholeseg.data.coco_dataset import CocoInstanceSegDatasetMmdetLike
from potholeseg.data.collate import collate_fn


def build_cat_id_to_label(ann_file: str | Path) -> Dict[int, int]:
    """
    Build mapping from COCO category_id to TorchVision foreground label.

    COCO category_id can be 0.
    TorchVision foreground labels must start from 1.
    """
    coco = COCO(str(ann_file))
    cat_ids = sorted(coco.getCatIds())

    return {
        cat_id: i + 1
        for i, cat_id in enumerate(cat_ids)
    }


def build_datasets(
    cfg: Dict[str, Any],
    prepared_splits: Dict[str, Dict[str, Path]],
):
    """
    Build train, valid, and test datasets.
    """
    cat_id_to_label = build_cat_id_to_label(
        prepared_splits["train"]["ann_file"]
    )

    train_ds = CocoInstanceSegDatasetMmdetLike(
        img_dir=prepared_splits["train"]["img_dir"],
        ann_file=prepared_splits["train"]["ann_file"],
        split="train",
        cfg=cfg,
        cat_id_to_label=cat_id_to_label,
        skip_empty=True,
    )

    val_ds = CocoInstanceSegDatasetMmdetLike(
        img_dir=prepared_splits["valid"]["img_dir"],
        ann_file=prepared_splits["valid"]["ann_file"],
        split="valid",
        cfg=cfg,
        cat_id_to_label=cat_id_to_label,
        skip_empty=False,
    )

    test_ds = CocoInstanceSegDatasetMmdetLike(
        img_dir=prepared_splits["test"]["img_dir"],
        ann_file=prepared_splits["test"]["ann_file"],
        split="test",
        cfg=cfg,
        cat_id_to_label=cat_id_to_label,
        skip_empty=False,
    )

    return train_ds, val_ds, test_ds, cat_id_to_label


def build_dataloaders(
    cfg: Dict[str, Any],
    train_ds,
    val_ds,
    test_ds,
):
    """
    Build train, validation, and test dataloaders.
    """
    train_cfg = cfg["train"]

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=train_cfg["pin_memory"],
        persistent_workers=train_cfg["num_workers"] > 0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["val_batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=train_cfg["pin_memory"],
        persistent_workers=train_cfg["num_workers"] > 0,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=train_cfg["test_batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=train_cfg["pin_memory"],
        persistent_workers=train_cfg["num_workers"] > 0,
    )

    return train_loader, val_loader, test_loader