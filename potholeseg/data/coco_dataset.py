import random
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from potholeseg.data.transforms import (
    build_train_albu_like_mmdet,
    get_eval_scale,
    get_train_scales,
    resize_keep_ratio_image_boxes_masks,
)


class CocoInstanceSegDatasetMmdetLike(Dataset):
    """
    COCO instance segmentation dataset with MMDetection-like pipeline.

    Training pipeline:
        LoadImageFromFile
        LoadAnnotations(with_bbox=True, with_mask=True)
        RandomFlip(prob=0.5)
        Albu(...)
        RandomChoiceResize(scales=[...], keep_ratio=True)
        PackDetInputs

    Validation / test pipeline:
        LoadImageFromFile
        LoadAnnotations(with_bbox=True, with_mask=True)
        Resize(scale=(1600, 900), keep_ratio=True)
        PackDetInputs

    Notes:
        COCO category_id can be 0.
        TorchVision foreground labels must start from 1.
    """

    def __init__(
        self,
        img_dir: str | Path,
        ann_file: str | Path,
        split: str,
        cfg: Dict[str, Any],
        cat_id_to_label: Dict[int, int] | None = None,
        skip_empty: bool | None = None,
    ) -> None:
        self.img_dir = Path(img_dir)
        self.ann_file = Path(ann_file)
        self.split = split
        self.cfg = cfg

        self.coco = COCO(str(self.ann_file))
        self.cat_ids = sorted(self.coco.getCatIds())

        if cat_id_to_label is None:
            self.cat_id_to_label = {
                cat_id: i + 1 for i, cat_id in enumerate(self.cat_ids)
            }
        else:
            self.cat_id_to_label = dict(cat_id_to_label)

        self.label_to_cat_id = {
            label: cat_id for cat_id, label in self.cat_id_to_label.items()
        }

        self.class_names = ["__background__"]

        for cat_id in self.cat_ids:
            self.class_names.append(self.coco.cats[cat_id]["name"])

        self.train_scales = get_train_scales(cfg)
        self.test_scale = get_eval_scale(cfg, "test")
        self.val_scale = get_eval_scale(cfg, "val")

        self.train_albu = build_train_albu_like_mmdet(cfg) if split == "train" else None

        data_cfg = cfg["data"]

        if skip_empty is None:
            skip_empty = bool(data_cfg.get("filter_empty_gt", True)) and split == "train"

        self.skip_empty = skip_empty
        self.min_size = int(data_cfg.get("min_size", 32))

        ids = sorted(self.coco.imgs.keys())

        if self.skip_empty:
            ids = self._filter_valid_ids(ids)

        self.ids = ids

    def _filter_valid_ids(self, ids):
        filtered = []

        for img_id in ids:
            img_info = self.coco.loadImgs([img_id])[0]

            if min(img_info["width"], img_info["height"]) < self.min_size:
                continue

            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            anns = self.coco.loadAnns(ann_ids)

            has_valid = False

            for ann in anns:
                if ann.get("iscrowd", 0) == 1:
                    continue

                x, y, w, h = ann.get("bbox", [0, 0, 0, 0])

                if w >= 1 and h >= 1:
                    has_valid = True
                    break

            if has_valid:
                filtered.append(img_id)

        return filtered

    def __len__(self) -> int:
        return len(self.ids)

    def _resolve_image_path(self, file_name: str) -> Path:
        path = self.img_dir / file_name

        if path.exists():
            return path

        matches = list(self.img_dir.rglob(Path(file_name).name))

        if matches:
            return matches[0]

        raise FileNotFoundError(f"Image not found: {file_name}")

    def _load_one(self, index: int):
        img_id = self.ids[index]
        img_info = self.coco.loadImgs([img_id])[0]
        img_path = self._resolve_image_path(img_info["file_name"])

        image = cv2.imread(str(img_path))

        if image is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = image.shape[:2]

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        masks = []
        iscrowd = []

        for ann in anns:
            x, y, bw, bh = ann.get("bbox", [0, 0, 0, 0])

            if bw <= 1 or bh <= 1:
                continue

            mask = self.coco.annToMask(ann).astype(np.uint8)

            if mask.sum() == 0:
                continue

            x1 = max(0.0, x)
            y1 = max(0.0, y)
            x2 = min(float(orig_w), x + bw)
            y2 = min(float(orig_h), y + bh)

            if x2 <= x1 or y2 <= y1:
                continue

            cat_id = ann["category_id"]

            if cat_id not in self.cat_id_to_label:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat_id_to_label[cat_id])
            masks.append(mask)
            iscrowd.append(ann.get("iscrowd", 0))

        if self.split == "train":
            if len(boxes) > 0:
                transformed = self.train_albu(
                    image=image,
                    masks=masks,
                    bboxes=boxes,
                    labels=labels,
                    iscrowd=iscrowd,
                )

                image = transformed["image"]
                masks = transformed["masks"]
                boxes = transformed["bboxes"]
                labels = transformed["labels"]
                iscrowd = transformed["iscrowd"]

            scale = random.choice(self.train_scales)

        elif self.split in ["valid", "val"]:
            scale = self.val_scale

        elif self.split == "test":
            scale = self.test_scale

        else:
            raise ValueError(f"Unsupported split: {self.split}")

        image, boxes, masks, scale_factor_xy = resize_keep_ratio_image_boxes_masks(
            image=image,
            boxes=boxes,
            masks=masks,
            scale=scale,
        )

        proc_h, proc_w = image.shape[:2]

        if len(boxes) > 0:
            boxes = np.array(boxes, dtype=np.float32)
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, proc_w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, proc_h)

            labels = np.array(labels, dtype=np.int64)
            iscrowd = np.array(iscrowd, dtype=np.int64)

            if isinstance(masks, list):
                masks = np.stack(masks).astype(np.uint8)

            keep = (
                (boxes[:, 2] > boxes[:, 0])
                & (boxes[:, 3] > boxes[:, 1])
                & (masks.reshape(masks.shape[0], -1).sum(axis=1) > 0)
            )

            boxes = boxes[keep]
            masks = masks[keep]
            labels = labels[keep]
            iscrowd = iscrowd[keep]

        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            masks = np.zeros((0, proc_h, proc_w), dtype=np.uint8)
            labels = np.zeros((0,), dtype=np.int64)
            iscrowd = np.zeros((0,), dtype=np.int64)

        if self.split == "train" and len(boxes) == 0:
            return None

        image_tensor = torch.as_tensor(
            image.transpose(2, 0, 1),
            dtype=torch.float32,
        ) / 255.0

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        masks_tensor = torch.as_tensor(masks, dtype=torch.uint8)
        iscrowd_tensor = torch.as_tensor(iscrowd, dtype=torch.int64)

        if boxes_tensor.numel() > 0:
            area = (
                (boxes_tensor[:, 3] - boxes_tensor[:, 1])
                * (boxes_tensor[:, 2] - boxes_tensor[:, 0])
            )
        else:
            area = torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "masks": masks_tensor,
            "image_id": torch.tensor([img_id], dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd_tensor,
            "original_size": torch.tensor([orig_h, orig_w], dtype=torch.int64),
            "processed_size": torch.tensor([proc_h, proc_w], dtype=torch.int64),
            "scale_factor": torch.tensor(scale_factor_xy, dtype=torch.float32),
        }

        return image_tensor, target

    def __getitem__(self, index: int):
        if self.split == "train":
            for offset in range(10):
                item = self._load_one((index + offset) % len(self.ids))

                if item is not None:
                    return item

        item = self._load_one(index)

        if item is None:
            return self._load_one((index + 1) % len(self.ids))

        return item