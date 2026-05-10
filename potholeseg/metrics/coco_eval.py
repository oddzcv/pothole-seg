import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from tqdm.auto import tqdm

from potholeseg.metrics.mask_utils import (
    ensure_full_size_masks,
    rescale_box_to_original,
    resize_mask_to_original,
)


def _set_model_score_thresh(model, score_thr: float):
    """
    Temporarily set TorchVision ROI score threshold if available.
    """
    if hasattr(model, "roi_heads") and hasattr(model.roi_heads, "score_thresh"):
        old_score_thresh = model.roi_heads.score_thresh
        model.roi_heads.score_thresh = score_thr
        return old_score_thresh

    return None


def _restore_model_score_thresh(model, old_score_thresh):
    if old_score_thresh is not None:
        model.roi_heads.score_thresh = old_score_thresh


@torch.no_grad()
def collect_coco_predictions(
    model,
    loader,
    dataset,
    device,
    score_thr: float = 0.05,
    mask_thr: float = 0.5,
) -> Tuple[List[dict], List[int]]:
    """
    Collect predictions in COCO result format.

    Important:
        - model predicts on processed image size.
        - bbox is rescaled to original COCO size.
        - mask is pasted to processed image size, then resized to original COCO size.
    """
    model.eval()

    old_score_thresh = _set_model_score_thresh(model, score_thr)

    results = []
    used_img_ids = []

    label_to_cat_id = dataset.label_to_cat_id

    try:
        for images, targets in tqdm(loader, desc="[collect COCO predictions]"):
            images_device = [img.to(device) for img in images]
            outputs = model(images_device)

            for image_tensor, output, target in zip(images, outputs, targets):
                image_id = int(target["image_id"].item())
                used_img_ids.append(image_id)

                boxes = output["boxes"].detach().cpu().numpy()
                labels = output["labels"].detach().cpu().numpy()
                scores = output["scores"].detach().cpu().numpy()

                proc_h, proc_w = image_tensor.shape[-2:]

                masks = ensure_full_size_masks(
                    output,
                    image_hw=(int(proc_h), int(proc_w)),
                ).numpy()

                for i in range(len(scores)):
                    score = float(scores[i])

                    if score < score_thr:
                        continue

                    label = int(labels[i])

                    if label not in label_to_cat_id:
                        continue

                    cat_id = int(label_to_cat_id[label])

                    x1, y1, x2, y2 = rescale_box_to_original(
                        boxes[i].tolist(),
                        target,
                    )

                    bw = max(0.0, x2 - x1)
                    bh = max(0.0, y2 - y1)

                    if bw <= 0 or bh <= 0:
                        continue

                    binary_mask_processed = (masks[i, 0] >= mask_thr).astype(np.uint8)

                    if binary_mask_processed.sum() == 0:
                        continue

                    binary_mask_original = resize_mask_to_original(
                        binary_mask_processed,
                        target,
                    )

                    rle = maskUtils.encode(np.asfortranarray(binary_mask_original))
                    rle["counts"] = rle["counts"].decode("ascii")

                    results.append(
                        {
                            "image_id": image_id,
                            "category_id": cat_id,
                            "bbox": [
                                float(x1),
                                float(y1),
                                float(bw),
                                float(bh),
                            ],
                            "score": score,
                            "segmentation": rle,
                        }
                    )

    finally:
        _restore_model_score_thresh(model, old_score_thresh)

    used_img_ids = sorted(list(set(used_img_ids)))

    return results, used_img_ids


def summarize_coco_stats(coco_eval: COCOeval, prefix: str) -> Dict[str, float]:
    stats = coco_eval.stats

    return {
        f"{prefix}_mAP": float(stats[0]),
        f"{prefix}_mAP_50": float(stats[1]),
        f"{prefix}_mAP_75": float(stats[2]),
        f"{prefix}_mAP_s": float(stats[3]),
        f"{prefix}_mAP_m": float(stats[4]),
        f"{prefix}_mAP_l": float(stats[5]),
    }


def evaluate_coco_bbox_segm(
    model,
    loader,
    dataset,
    device,
    score_thr: float = 0.05,
    mask_thr: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate bbox and segmentation metrics using COCO protocol.
    """
    results, img_ids = collect_coco_predictions(
        model=model,
        loader=loader,
        dataset=dataset,
        device=device,
        score_thr=score_thr,
        mask_thr=mask_thr,
    )

    print("Total predictions for COCOeval:", len(results))
    print("Total evaluated images:", len(img_ids))

    if len(results) > 0:
        print("Example COCO result:")
        print(results[0])

    if len(results) == 0:
        return {
            "bbox_mAP": 0.0,
            "bbox_mAP_50": 0.0,
            "bbox_mAP_75": 0.0,
            "bbox_mAP_s": 0.0,
            "bbox_mAP_m": 0.0,
            "bbox_mAP_l": 0.0,
            "segm_mAP": 0.0,
            "segm_mAP_50": 0.0,
            "segm_mAP_75": 0.0,
            "segm_mAP_s": 0.0,
            "segm_mAP_m": 0.0,
            "segm_mAP_l": 0.0,
        }

    coco_gt = dataset.coco
    coco_dt = coco_gt.loadRes(results)

    metrics = {}

    for iou_type, prefix in [("bbox", "bbox"), ("segm", "segm")]:
        print(f"\nCOCO Evaluation: {iou_type}")

        coco_eval = COCOeval(
            cocoGt=coco_gt,
            cocoDt=coco_dt,
            iouType=iou_type,
        )

        coco_eval.params.imgIds = img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        metrics.update(
            summarize_coco_stats(coco_eval, prefix)
        )

    return metrics


def save_metrics(metrics: Dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return output_path