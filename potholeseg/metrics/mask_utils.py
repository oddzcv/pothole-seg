from typing import Tuple

import cv2
import numpy as np
import torch
import torchvision.models.detection.roi_heads as roi_heads


PASTE_MASKS_IN_IMAGE = getattr(roi_heads, "paste_masks_in_image", None)

if PASTE_MASKS_IN_IMAGE is None:
    PASTE_MASKS_IN_IMAGE = getattr(roi_heads, "_paste_masks_in_image")


def ensure_full_size_masks(output: dict, image_hw: Tuple[int, int]) -> torch.Tensor:
    """
    Ensure output['masks'] has full image size.

    TorchVision Mask R-CNN may output:
        [N, 1, 28, 28]

    For visualization/evaluation, it must be pasted into:
        [N, 1, H, W]

    Args:
        output: Model prediction dict.
        image_hw: Processed image size as (H, W).

    Returns:
        Full-size masks tensor on CPU, shape [N, 1, H, W].
    """
    masks = output["masks"].detach().cpu()
    boxes = output["boxes"].detach().cpu()

    h, w = image_hw

    if masks.numel() == 0:
        return masks

    if masks.ndim == 3:
        masks = masks[:, None, :, :]

    if tuple(masks.shape[-2:]) == (int(h), int(w)):
        return masks

    full_masks = PASTE_MASKS_IN_IMAGE(
        masks,
        boxes,
        (int(h), int(w)),
    )

    return full_masks


def rescale_box_to_original(box, target: dict):
    """
    Rescale xyxy box from processed image coordinate to original COCO coordinate.
    """
    sx, sy = target["scale_factor"].detach().cpu().numpy().tolist()

    x1, y1, x2, y2 = box

    x1 = x1 / sx
    x2 = x2 / sx
    y1 = y1 / sy
    y2 = y2 / sy

    orig_h, orig_w = target["original_size"].detach().cpu().numpy().tolist()

    x1 = max(0.0, min(float(x1), float(orig_w)))
    x2 = max(0.0, min(float(x2), float(orig_w)))
    y1 = max(0.0, min(float(y1), float(orig_h)))
    y2 = max(0.0, min(float(y2), float(orig_h)))

    return [x1, y1, x2, y2]


def resize_mask_to_original(binary_mask: np.ndarray, target: dict) -> np.ndarray:
    """
    Resize processed binary mask to original COCO image size.
    """
    orig_h, orig_w = target["original_size"].detach().cpu().numpy().tolist()

    if binary_mask.shape != (orig_h, orig_w):
        binary_mask = cv2.resize(
            binary_mask.astype(np.uint8),
            (int(orig_w), int(orig_h)),
            interpolation=cv2.INTER_NEAREST,
        )

    return binary_mask.astype(np.uint8)