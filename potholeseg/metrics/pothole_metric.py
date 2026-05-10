from typing import Dict

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from potholeseg.metrics.mask_utils import ensure_full_size_masks


@torch.no_grad()
def evaluate_pothole_metric(
    model,
    loader,
    device,
    class_label: int = 1,
    prefix: str = "pothole",
    score_thr: float = 0.05,
    mask_thr: float = 0.5,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """
    Custom pothole metric equivalent to the MMDetection notebook scenario.

    Logic:
        1. Merge all GT masks in an image into one binary mask.
        2. Keep predicted masks with score > score_thr and label == class_label.
        3. Merge predicted masks into one binary mask.
        4. Resize prediction mask to GT shape if needed.
        5. Compute global IoU and Dice.
    """
    model.eval()

    old_score_thresh = None

    if hasattr(model, "roi_heads") and hasattr(model.roi_heads, "score_thresh"):
        old_score_thresh = model.roi_heads.score_thresh
        model.roi_heads.score_thresh = score_thr

    total_intersection = 0.0
    total_union = 0.0
    total_area_pred = 0.0
    total_area_gt = 0.0

    try:
        for images, targets in tqdm(loader, desc="[PotholeMetric]"):
            images_device = [img.to(device) for img in images]
            outputs = model(images_device)

            for image_tensor, output, target in zip(images, outputs, targets):
                proc_h, proc_w = image_tensor.shape[-2:]

                gt_masks = target["masks"].detach().cpu()
                gt_labels = target["labels"].detach().cpu()

                if gt_masks.numel() > 0:
                    gt_keep = gt_labels == class_label

                    if gt_keep.any():
                        gt_mask_merged = (
                            gt_masks[gt_keep]
                            .bool()
                            .any(dim=0)
                            .numpy()
                            .astype(np.uint8)
                        )
                    else:
                        gt_mask_merged = np.zeros((proc_h, proc_w), dtype=np.uint8)
                else:
                    gt_mask_merged = np.zeros((proc_h, proc_w), dtype=np.uint8)

                pred_scores = output["scores"].detach().cpu()
                pred_labels = output["labels"].detach().cpu()

                pred_masks = ensure_full_size_masks(
                    output,
                    image_hw=(int(proc_h), int(proc_w)),
                )

                pred_keep = (
                    (pred_scores > score_thr)
                    & (pred_labels == class_label)
                )

                if pred_keep.any():
                    pred_mask_array = pred_masks[pred_keep, 0].numpy()
                    pred_mask_merged = (
                        pred_mask_array >= mask_thr
                    ).any(axis=0).astype(np.uint8)
                else:
                    pred_mask_merged = np.zeros_like(gt_mask_merged, dtype=np.uint8)

                if pred_mask_merged.shape != gt_mask_merged.shape:
                    h, w = gt_mask_merged.shape
                    pred_mask_merged = cv2.resize(
                        pred_mask_merged.astype(np.uint8),
                        (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    )

                intersection = np.logical_and(
                    pred_mask_merged,
                    gt_mask_merged,
                ).sum()

                area_pred = pred_mask_merged.sum()
                area_gt = gt_mask_merged.sum()
                union = area_pred + area_gt - intersection

                total_intersection += float(intersection)
                total_union += float(union)
                total_area_pred += float(area_pred)
                total_area_gt += float(area_gt)

    finally:
        if old_score_thresh is not None:
            model.roi_heads.score_thresh = old_score_thresh

    iou = total_intersection / (total_union + eps)
    dice = (2.0 * total_intersection) / (
        total_area_pred + total_area_gt + eps
    )

    return {
        f"{prefix}/mIoU": float(iou),
        f"{prefix}/Dice": float(dice),
    }