import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
import matplotlib.pyplot as plt

from potholeseg.config import load_config, print_config_summary
from potholeseg.utils import (
    set_seed,
    get_device,
    print_device_info,
    setup_project_dirs,
)
from potholeseg.data import (
    download_roboflow_dataset,
    prepare_roboflow_coco_splits,
    build_datasets,
    build_dataloaders,
)
from potholeseg.models import build_model, print_model_summary
from potholeseg.engine import (
    build_optimizer,
    WarmupCosineScheduler,
    train_one_epoch,
    validate_loss,
    save_checkpoint,
)
from potholeseg.metrics import evaluate_model, save_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Mask R-CNN for pothole instance segmentation."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
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
        "--epochs",
        type=int,
        default=None,
        help="Override max epochs from config.",
    )

    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Run validation evaluation every N epochs.",
    )

    parser.add_argument(
        "--skip-val-eval",
        action="store_true",
        help="Skip COCO/Pothole validation evaluation during training.",
    )

    return parser.parse_args()


def update_config_from_args(cfg, args):
    if args.device is not None:
        cfg["project"]["device"] = args.device

    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs

    return cfg


def save_training_log(history, logs_dir):
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    csv_path = logs_dir / "training_log.csv"
    df = pd.DataFrame(history)
    df.to_csv(csv_path, index=False)

    if "loss" in df.columns and "val_loss" in df.columns:
        plot_path = logs_dir / "training_validation_loss.png"

        plt.figure(figsize=(8, 5))
        plt.plot(df["epoch"], df["loss"], label="train_loss")
        plt.plot(df["epoch"], df["val_loss"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.grid(True)
        plt.legend()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

    return csv_path


def get_monitor_value(metrics, monitor_name):
    if monitor_name not in metrics:
        available = ", ".join(metrics.keys())
        raise KeyError(
            f"Monitor metric '{monitor_name}' not found. "
            f"Available metrics: {available}"
        )

    return float(metrics[monitor_name])


def is_improved(current, best, mode, min_delta):
    if mode == "max":
        return current > best + min_delta

    if mode == "min":
        return current < best - min_delta

    raise ValueError(f"Unsupported mode: {mode}")


def main():
    args = parse_args()

    cfg = load_config(args.config)
    cfg = update_config_from_args(cfg, args)

    print_config_summary(cfg)

    set_seed(cfg["project"]["seed"])

    device = get_device(cfg["project"]["device"])
    print_device_info(device)

    dirs = setup_project_dirs(cfg)

    print("\nProject directories:")
    for name, path in dirs.items():
        print(f"  {name}: {path}")

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

    print("\nPrepared splits:")
    for split, paths in prepared_splits.items():
        print(f"  {split}:")
        print(f"    img_dir : {paths['img_dir']}")
        print(f"    ann_file: {paths['ann_file']}")

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

    print("\nDataset summary:")
    print("  Class names:", train_ds.class_names)
    print("  cat_id_to_label:", cat_id_to_label)
    print("  label_to_cat_id:", train_ds.label_to_cat_id)
    print("  Train:", len(train_ds))
    print("  Valid:", len(val_ds))
    print("  Test :", len(test_ds))

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------
    model = build_model(cfg)
    model.to(device)

    print("\nModel summary:")
    print_model_summary(model)

    optimizer = build_optimizer(model, cfg)
    scheduler = WarmupCosineScheduler(optimizer, cfg)

    use_amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"

    scaler = torch.amp.GradScaler(
        device="cuda",
        enabled=use_amp,
    )

    print("\nTraining settings:")
    print("  AMP:", use_amp)
    print("  Epochs:", cfg["train"]["epochs"])
    print("  Batch size:", cfg["train"]["batch_size"])
    print("  Optimizer:", cfg["train"]["optimizer"]["name"])
    print("  LR:", cfg["train"]["optimizer"]["lr"])

    # ------------------------------------------------------------
    # Checkpoint / early stopping
    # ------------------------------------------------------------
    checkpoint_cfg = cfg["train"]["checkpoint"]
    early_cfg = cfg["train"]["early_stopping"]

    monitor = checkpoint_cfg.get("monitor", "segm_mAP")
    mode = checkpoint_cfg.get("mode", "max")

    min_delta = float(early_cfg.get("min_delta", 0.0))
    patience = int(early_cfg.get("patience", 15))
    early_enabled = bool(early_cfg.get("enabled", True))

    if mode == "max":
        best_value = -float("inf")
    elif mode == "min":
        best_value = float("inf")
    else:
        raise ValueError(f"Unsupported checkpoint mode: {mode}")

    epochs_without_improve = 0
    global_step = 0
    history = []

    best_path = dirs["checkpoints_dir"] / checkpoint_cfg["best_model_name"]
    last_path = dirs["checkpoints_dir"] / checkpoint_cfg["last_model_name"]

    # ------------------------------------------------------------
    # Train loop
    # ------------------------------------------------------------
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch}/{cfg['train']['epochs']}")
        print(f"{'=' * 80}")

        train_metrics, global_step = train_one_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loader=train_loader,
            device=device,
            epoch=epoch,
            cfg=cfg,
            scaler=scaler,
            start_global_step=global_step,
        )

        val_loss_metrics = validate_loss(
            model=model,
            loader=val_loader,
            device=device,
        )

        val_eval_metrics = {}

        should_eval = (
            not args.skip_val_eval
            and epoch % int(args.eval_every) == 0
        )

        if should_eval:
            print("\nRunning validation evaluation...")
            val_eval_metrics = evaluate_model(
                model=model,
                loader=val_loader,
                dataset=val_ds,
                device=device,
                cfg=cfg,
            )

            val_metrics_path = (
                dirs["test_eval_dir"] / f"metrics_valid_epoch_{epoch}.json"
            )
            save_metrics(val_eval_metrics, val_metrics_path)
            print("Saved validation metrics:", val_metrics_path)

        row = {
            "epoch": epoch,
            "lr": scheduler.get_lr(),
            **train_metrics,
            **val_loss_metrics,
            **{f"val_{k}": v for k, v in val_eval_metrics.items()},
        }

        history.append(row)

        print("\nEpoch summary:")
        print(json.dumps(row, indent=2))

        save_training_log(history, dirs["logs_dir"])

        if checkpoint_cfg.get("save_last", True):
            save_checkpoint(
                path=last_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                cfg=cfg,
                extra={
                    "history": history,
                    "cat_id_to_label": cat_id_to_label,
                    "global_step": global_step,
                },
            )
            print("Saved last checkpoint:", last_path)

        # If validation evaluation is skipped, monitor val_loss as fallback.
        if should_eval:
            monitor_metrics = val_eval_metrics
        else:
            monitor_metrics = {
                "val_loss": val_loss_metrics["val_loss"]
            }

            if monitor not in monitor_metrics:
                monitor = "val_loss"
                mode = "min"

        current_value = get_monitor_value(monitor_metrics, monitor)

        improved = is_improved(
            current=current_value,
            best=best_value,
            mode=mode,
            min_delta=min_delta,
        )

        if improved:
            best_value = current_value
            epochs_without_improve = 0

            if checkpoint_cfg.get("save_best", True):
                save_checkpoint(
                    path=best_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    cfg=cfg,
                    extra={
                        "history": history,
                        "cat_id_to_label": cat_id_to_label,
                        "global_step": global_step,
                        "best_metric_name": monitor,
                        "best_metric_value": best_value,
                    },
                )
                print(f"Saved best checkpoint: {best_path}")
                print(f"Best {monitor}: {best_value:.6f}")
        else:
            epochs_without_improve += 1
            print(
                f"No improvement on {monitor}. "
                f"Current={current_value:.6f}, Best={best_value:.6f}. "
                f"Patience={epochs_without_improve}/{patience}"
            )

        if early_enabled and epochs_without_improve >= patience:
            print("\nEarly stopping triggered.")
            break

    # ------------------------------------------------------------
    # Final test evaluation with best checkpoint if available
    # ------------------------------------------------------------
    if best_path.exists():
        print("\nLoading best checkpoint for final test evaluation...")
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
        model.to(device)
        model.eval()

    print("\nRunning final test evaluation...")
    test_metrics = evaluate_model(
        model=model,
        loader=test_loader,
        dataset=test_ds,
        device=device,
        cfg=cfg,
    )

    test_metrics_path = dirs["test_eval_dir"] / "metrics_test.json"
    save_metrics(test_metrics, test_metrics_path)

    print("\nFinal test metrics:")
    print(json.dumps(test_metrics, indent=2))
    print("Saved test metrics:", test_metrics_path)

    print("\nTraining finished.")


if __name__ == "__main__":
    main()