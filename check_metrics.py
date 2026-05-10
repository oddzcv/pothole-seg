import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from potholeseg.config import load_config
from potholeseg.data import CocoInstanceSegDatasetMmdetLike, collate_fn
from potholeseg.metrics import evaluate_model


def create_synthetic_coco_dataset(root: Path):
    img_dir = root / "test"
    img_dir.mkdir(parents=True, exist_ok=True)

    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:] = (80, 80, 80)
    cv2.rectangle(image, (80, 90), (180, 170), (255, 255, 255), -1)

    image_path = img_dir / "sample.jpg"
    cv2.imwrite(str(image_path), image)

    coco = {
        "info": {},
        "licenses": [],
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


class DummyPerfectModel:
    """
    Dummy model that returns near-perfect prediction for the synthetic dataset.
    """

    def __init__(self):
        self.roi_heads = SimpleNamespace(score_thresh=0.05)

    def eval(self):
        return self

    def __call__(self, images):
        outputs = []

        for image in images:
            _, h, w = image.shape

            # Synthetic GT original:
            # bbox original = [80, 90, 100, 80]
            # processed scale factor for test = 3.75
            # bbox processed xyxy:
            # [80,90,180,170] * 3.75
            box = torch.tensor(
                [[300.0, 337.5, 675.0, 637.5]],
                dtype=torch.float32,
                device=image.device,
            )

            mask = torch.zeros((1, 1, h, w), dtype=torch.float32, device=image.device)
            mask[:, :, 337:638, 300:675] = 1.0

            outputs.append(
                {
                    "boxes": box,
                    "labels": torch.tensor([1], dtype=torch.int64, device=image.device),
                    "scores": torch.tensor([0.99], dtype=torch.float32, device=image.device),
                    "masks": mask,
                }
            )

        return outputs


def main():
    cfg = load_config("configs/maskrcnn_mobilenetv3_large_fpn.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        img_dir, ann_file = create_synthetic_coco_dataset(tmp)

        dataset = CocoInstanceSegDatasetMmdetLike(
            img_dir=img_dir,
            ann_file=ann_file,
            split="test",
            cfg=cfg,
            cat_id_to_label={0: 1},
            skip_empty=False,
        )

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
        )

        model = DummyPerfectModel()

        metrics = evaluate_model(
            model=model,
            loader=loader,
            dataset=dataset,
            device=device,
            cfg=cfg,
        )

        print("\nMetrics:")
        print(json.dumps(metrics, indent=2))

        required_keys = [
            "bbox_mAP",
            "bbox_mAP_50",
            "segm_mAP",
            "segm_mAP_50",
            "pothole/mIoU",
            "pothole/Dice",
        ]

        for key in required_keys:
            if key not in metrics:
                raise RuntimeError(f"Missing metric key: {key}")

        print("\nMetrics module check passed.")


if __name__ == "__main__":
    main()