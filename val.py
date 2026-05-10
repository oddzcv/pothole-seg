import argparse
import json
from pathlib import Path

import torch

from potholeseg.config import load_config, print_config_summary
from potholeseg.utils import (
    set_seed,
    get_device,
    print_device_info,
    setup_project_dirs,
)
from potholeseg.data import (
    prepare_roboflow_coco_splits,
    download_roboflow_dataset,
    build_datasets,
    build_dataloaders,
)
from potholeseg.models import build_model, print_model_summary
from potholeseg.engine import load_checkpoint
from potholeseg.metrics import evaluate_model, save_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Mask R-CNN checkpoint for pothole instance segmentation."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to checkpoint .pth file.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "val", "test"],
        help="Dataset split to evaluate.",
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help=(
            "Optional existing Roboflow dataset root. "
            "If not provided, dataset will be downloaded from Roboflow."
        ),
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=(
            "Optional Roboflow API key. "
            "If not provided, uses ROBOFLOW_API_KEY environment variable."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cuda", "cpu"],
        help="Override device from config.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output JSON path for metrics.",
    )

    return parser.parse_args()


def normalize_split(split: str) -> str:
    if split == "val":
        return "valid"
    return split


def update_config_from_args(cfg, args):
    if args.device is not None:
        cfg["project"]["device"] = args.device

    return cfg


def select_dataset_and_loader(split, train_ds, val_ds, test_ds, train_loader, val_loader, test_loader):
    split = normalize_split(split)

    if split == "train":
        return train_ds, train_loader

    if split == "valid":
        return val_ds, val_loader

    if split == "test":
        return test_ds, test_loader

    raise ValueError(f"Unsupported split: {split}")


def main():
    args = parse_args()

    cfg = load_config(args.config)
    cfg = update_config_from_args(cfg, args)

    split = normalize_split(args.split)

    print_config_summary(cfg)

    set_seed(cfg["project"]["seed"])

    device = get_device(cfg["project"]["device"])
    print_device_info(device)

    dirs = setup_project_dirs(cfg)

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------
    if args.data_root is not None:
        data_root = Path(args.data_root)
        print("\nUsing existing data root:", data_root)
    else:
        print("\nDownloading dataset from Roboflow...")
        data_root = download_roboflow_dataset(
            cfg=cfg,
            api_key=args.api_key,
        )
        print("Downloaded data root:", data_root)

    prepared_splits = prepare_roboflow_coco_splits(
        cfg=cfg,
        data_root=data_root,
    )

    train_ds, val_ds, test_ds, cat_id_to_label = build_datasets(
        cfg=cfg,
        prepared_splits=prepared_splits,
    )

    train_loader, val_loader, test_loader = build_dataloaders(
        cfg=cfg,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
    )

    dataset, loader = select_dataset_and_loader(
        split=split,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    print("\nDataset summary:")
    print("  Split:", split)
    print("  Class names:", dataset.class_names)
    print("  cat_id_to_label:", cat_id_to_label)
    print("  label_to_cat_id:", dataset.label_to_cat_id)
    print("  Dataset length:", len(dataset))

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------
    model = build_model(cfg)
    model.to(device)

    print("\nModel summary:")
    print_model_summary(model)

    print("\nLoading checkpoint:", args.weights)

    ckpt = load_checkpoint(
        path=args.weights,
        model=model,
        optimizer=None,
        map_location=device,
        strict=True,
    )

    print("Loaded checkpoint epoch:", ckpt.get("epoch", "unknown"))

    if "best_metric_name" in ckpt:
        print("Best metric:", ckpt.get("best_metric_name"), ckpt.get("best_metric_value"))

    model.eval()

    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------
    print(f"\nEvaluating split: {split}")

    metrics = evaluate_model(
        model=model,
        loader=loader,
        dataset=dataset,
        device=device,
        cfg=cfg,
    )

    print("\nMetrics:")
    print(json.dumps(metrics, indent=2))

    if args.output is not None:
        output_path = Path(args.output)
    else:
        output_path = dirs["test_eval_dir"] / f"metrics_{split}.json"

    save_metrics(metrics, output_path)

    print("\nSaved metrics:", output_path)


if __name__ == "__main__":
    main()