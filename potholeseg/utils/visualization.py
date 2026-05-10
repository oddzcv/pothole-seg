from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

from potholeseg.metrics.mask_utils import ensure_full_size_masks


def draw_prediction(
    image_rgb: np.ndarray,
    output: dict,
    class_names: Sequence[str],
    score_thr: float = 0.05,
    mask_thr: float = 0.5,
    mask_alpha: float = 0.45,
) -> np.ndarray:
    """
    Draw boxes, labels, scores, and instance masks on RGB image.

    Args:
        image_rgb: RGB image, uint8, shape [H, W, 3].
        output: TorchVision Mask R-CNN output.
        class_names: List of class names. Index 0 should be background.
        score_thr: Detection confidence threshold.
        mask_thr: Mask probability threshold.
        mask_alpha: Mask overlay opacity.

    Returns:
        RGB visualization image.
    """
    image = image_rgb.copy()
    h, w = image.shape[:2]

    boxes = output["boxes"].detach().cpu().numpy()
    labels = output["labels"].detach().cpu().numpy()
    scores = output["scores"].detach().cpu().numpy()

    masks_full = ensure_full_size_masks(
        output,
        image_hw=(h, w),
    ).numpy()

    vis = image.copy()
    overlay = image.copy()

    for box, label, score, mask in zip(boxes, labels, scores, masks_full):
        if score < score_thr:
            continue

        binary_mask = (mask[0] >= mask_thr).astype(np.uint8)

        if binary_mask.sum() == 0:
            continue

        overlay[binary_mask == 1] = np.array([255, 0, 0], dtype=np.uint8)

        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        cv2.drawContours(
            vis,
            contours,
            -1,
            (255, 255, 0),
            2,
        )

        x1, y1, x2, y2 = box.astype(int)

        cv2.rectangle(
            vis,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        if 0 <= label < len(class_names):
            name = class_names[label]
        else:
            name = str(label)

        text = f"{name}: {score:.2f}"

        cv2.putText(
            vis,
            text,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

    vis = cv2.addWeighted(
        vis,
        1.0 - mask_alpha,
        overlay,
        mask_alpha,
        0,
    )

    return vis


def save_rgb_image(image_rgb: np.ndarray, output_path: str | Path) -> Path:
    """
    Save RGB image to disk using OpenCV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), image_bgr)

    return output_path


def read_rgb_image(path: str | Path) -> np.ndarray:
    """
    Read image from disk and return RGB uint8 image.
    """
    path = Path(path)

    image_bgr = cv2.imread(str(path))

    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    return image_rgb


def image_to_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    """
    Convert RGB uint8 image to float tensor [C, H, W] in range [0, 1].
    """
    image_tensor = torch.as_tensor(
        image_rgb.transpose(2, 0, 1),
        dtype=torch.float32,
    ) / 255.0

    return image_tensor