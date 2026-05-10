import argparse
from pathlib import Path

import cv2
import torch

from potholeseg.config import load_config, print_config_summary
from potholeseg.utils import (
    set_seed,
    get_device,
    print_device_info,
    setup_project_dirs,
    read_rgb_image,
    image_to_tensor,
    draw_prediction,
    save_rgb_image,
)
from potholeseg.models import build_model, print_model_summary
from potholeseg.engine import load_checkpoint


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run image prediction using trained Mask R-CNN checkpoint."
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
        "--source",
        type=str,
        required=True,
        help="Image path or folder path.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save prediction images.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cuda", "cpu"],
        help="Override device from config.",
    )

    parser.add_argument(
        "--score-thr",
        type=float,
        default=None,
        help="Override prediction score threshold.",
    )

    parser.add_argument(
        "--mask-thr",
        type=float,
        default=None,
        help="Override mask threshold.",
    )

    parser.add_argument(
        "--mask-alpha",
        type=float,
        default=None,
        help="Override mask overlay alpha.",
    )

    return parser.parse_args()


def update_config_from_args(cfg, args):
    if args.device is not None:
        cfg["project"]["device"] = args.device

    if args.score_thr is not None:
        cfg["predict"]["score_thr"] = args.score_thr

    if args.mask_thr is not None:
        cfg["predict"]["mask_thr"] = args.mask_thr

    if args.mask_alpha is not None:
        cfg["predict"]["mask_alpha"] = args.mask_alpha

    return cfg


def collect_images(source: str | Path):
    source = Path(source)

    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {source.suffix}")
        return [source]

    image_paths = []

    for path in sorted(source.rglob("*")):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(path)

    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in folder: {source}")

    return image_paths


@torch.no_grad()
def predict_one_image(
    model,
    image_path: Path,
    output_dir: Path,
    device,
    class_names,
    score_thr: float,
    mask_thr: float,
    mask_alpha: float,
):
    image_rgb = read_rgb_image(image_path)
    image_tensor = image_to_tensor(image_rgb).to(device)

    output = model([image_tensor])[0]

    vis_rgb = draw_prediction(
        image_rgb=image_rgb,
        output=output,
        class_names=class_names,
        score_thr=score_thr,
        mask_thr=mask_thr,
        mask_alpha=mask_alpha,
    )

    output_path = output_dir / f"{image_path.stem}_prediction.jpg"
    save_rgb_image(vis_rgb, output_path)

    num_predictions = int((output["scores"].detach().cpu() >= score_thr).sum().item())

    return output_path, num_predictions


def main():
    args = parse_args()

    cfg = load_config(args.config)
    cfg = update_config_from_args(cfg, args)

    print_config_summary(cfg)

    set_seed(cfg["project"]["seed"])

    device = get_device(cfg["project"]["device"])
    print_device_info(device)

    dirs = setup_project_dirs(cfg)

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = dirs["predictions_dir"]

    output_dir.mkdir(parents=True, exist_ok=True)

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
    # Class names
    # ------------------------------------------------------------
    class_names = [
        "__background__",
        cfg["data"]["class_name"],
    ]

    score_thr = float(cfg["predict"]["score_thr"])
    mask_thr = float(cfg["predict"]["mask_thr"])
    mask_alpha = float(cfg["predict"]["mask_alpha"])

    print("\nPrediction settings:")
    print("  Score threshold:", score_thr)
    print("  Mask threshold :", mask_thr)
    print("  Mask alpha     :", mask_alpha)
    print("  Output dir     :", output_dir)

    image_paths = collect_images(args.source)

    print("\nImages found:", len(image_paths))

    for image_path in image_paths:
        output_path, num_predictions = predict_one_image(
            model=model,
            image_path=image_path,
            output_dir=output_dir,
            device=device,
            class_names=class_names,
            score_thr=score_thr,
            mask_thr=mask_thr,
            mask_alpha=mask_alpha,
        )

        print(
            f"{image_path.name} -> {output_path} "
            f"({num_predictions} predictions above threshold)"
        )

    print("\nPrediction finished.")


if __name__ == "__main__":
    main()