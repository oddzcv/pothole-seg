import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from potholeseg.config import load_config
from potholeseg.data import (
    CocoInstanceSegDatasetMmdetLike,
    collate_fn,
)
from torch.utils.data import DataLoader


def create_synthetic_coco_dataset(root: Path):
    img_dir = root / "train"
    img_dir.mkdir(parents=True, exist_ok=True)

    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:] = (80, 80, 80)

    # Draw a white rectangle as pothole-like object.
    cv2.rectangle(image, (80, 90), (180, 170), (255, 255, 255), -1)

    image_path = img_dir / "sample.jpg"
    cv2.imwrite(str(image_path), image)

    coco = {
        "images": [
            {
                "id": 0,
                "file_name": "sample.jpg",
                "width": 320,
                "height": 240,
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 0,
                "category_id": 0,
                "bbox": [80, 90, 100, 80],
                "area": 8000,
                "iscrowd": 0,
                "segmentation": [
                    [
                        80, 90,
                        180, 90,
                        180, 170,
                        80, 170,
                    ]
                ],
            }
        ],
        "categories": [
            {
                "id": 0,
                "name": "Lubang",
                "supercategory": "Lubang",
            }
        ],
    }

    ann_file = img_dir / "_annotations.coco.json"

    with open(ann_file, "w", encoding="utf-8") as f:
        json.dump(coco, f)

    return img_dir, ann_file


def main():
    cfg = load_config("configs/maskrcnn_mobilenetv3_large_fpn.yaml")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        img_dir, ann_file = create_synthetic_coco_dataset(tmp)

        cat_id_to_label = {0: 1}

        dataset = CocoInstanceSegDatasetMmdetLike(
            img_dir=img_dir,
            ann_file=ann_file,
            split="test",
            cfg=cfg,
            cat_id_to_label=cat_id_to_label,
            skip_empty=False,
        )

        print("Dataset length:", len(dataset))
        print("Class names:", dataset.class_names)
        print("cat_id_to_label:", dataset.cat_id_to_label)
        print("label_to_cat_id:", dataset.label_to_cat_id)

        image, target = dataset[0]

        print("\nSingle sample:")
        print("  image:", tuple(image.shape))
        print("  boxes:", target["boxes"])
        print("  labels:", target["labels"])
        print("  masks:", tuple(target["masks"].shape))
        print("  original_size:", target["original_size"])
        print("  processed_size:", target["processed_size"])
        print("  scale_factor:", target["scale_factor"])

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
        )

        images, targets = next(iter(loader))

        print("\nDataloader batch:")
        print("  images:", len(images))
        print("  targets:", len(targets))
        print("  first image:", tuple(images[0].shape))
        print("  first target keys:", list(targets[0].keys()))

        print("\nData module check passed.")


if __name__ == "__main__":
    main()