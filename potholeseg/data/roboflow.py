import json
import os
from pathlib import Path
from typing import Any, Dict


def download_roboflow_dataset(
    cfg: Dict[str, Any],
    api_key: str | None = None,
) -> Path:
    """
    Download Roboflow dataset using configuration.

    Args:
        cfg: YAML config.
        api_key: Roboflow API key. If None, uses ROBOFLOW_API_KEY env variable.

    Returns:
        Dataset root path.
    """
    from roboflow import Roboflow

    api_key = api_key or os.getenv("ROBOFLOW_API_KEY")

    if not api_key:
        raise ValueError(
            "Roboflow API key not found. "
            "Set ROBOFLOW_API_KEY environment variable or pass api_key explicitly."
        )

    data_cfg = cfg["data"]

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(data_cfg["workspace"]).project(data_cfg["project"])
    version = project.version(data_cfg["version"])
    dataset = version.download(data_cfg["format"])

    return Path(dataset.location)


def fix_coco_categories_like_mmdet(
    ann_path: str | Path,
    output_path: str | Path,
    class_name: str = "Lubang",
    category_id: int = 0,
) -> Path:
    """
    Fix COCO categories to match the MMDetection notebook scenario:

        categories = [{"id": 0, "name": "Lubang", "supercategory": "Lubang"}]

    This function maps all existing annotation category_id values to category_id.
    """
    ann_path = Path(ann_path)
    output_path = Path(output_path)

    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    old_categories = data.get("categories", [])
    old_cat_ids = {cat["id"] for cat in old_categories}

    for ann in data.get("annotations", []):
        if ann.get("category_id") in old_cat_ids:
            ann["category_id"] = category_id

    data["categories"] = [
        {
            "id": category_id,
            "name": class_name,
            "supercategory": class_name,
        }
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    return output_path


def prepare_roboflow_coco_splits(
    cfg: Dict[str, Any],
    data_root: str | Path,
) -> Dict[str, Dict[str, Path]]:
    """
    Prepare Roboflow COCO-segmentation splits.

    Expected Roboflow structure:
        data_root/
            train/
                _annotations.coco.json
            valid/
                _annotations.coco.json
            test/
                _annotations.coco.json

    Returns:
        {
            "train": {"img_dir": ..., "ann_file": ...},
            "valid": {"img_dir": ..., "ann_file": ...},
            "test": {"img_dir": ..., "ann_file": ...},
        }
    """
    data_root = Path(data_root)
    data_cfg = cfg["data"]

    split_names = {
        "train": data_cfg.get("train_split", "train"),
        "valid": data_cfg.get("val_split", "valid"),
        "test": data_cfg.get("test_split", "test"),
    }

    prepared = {}

    for canonical_split, folder_name in split_names.items():
        split_dir = data_root / folder_name
        raw_ann = split_dir / "_annotations.coco.json"

        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        if not raw_ann.exists():
            raise FileNotFoundError(f"COCO annotation not found: {raw_ann}")

        if data_cfg.get("fix_categories", True):
            fixed_ann = split_dir / f"{canonical_split}_annotations.fixed.json"

            fixed_ann = fix_coco_categories_like_mmdet(
                ann_path=raw_ann,
                output_path=fixed_ann,
                class_name=data_cfg["class_name"],
                category_id=data_cfg["coco_category_id"],
            )
        else:
            fixed_ann = raw_ann

        prepared[canonical_split] = {
            "img_dir": split_dir,
            "ann_file": fixed_ann,
        }

    return prepared