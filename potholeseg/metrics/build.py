from typing import Any, Dict

from potholeseg.metrics.coco_eval import evaluate_coco_bbox_segm, save_metrics
from potholeseg.metrics.pothole_metric import evaluate_pothole_metric


def evaluate_model(
    model,
    loader,
    dataset,
    device,
    cfg: Dict[str, Any],
) -> Dict[str, float]:
    """
    Run all configured metrics.
    """
    metrics = {}

    eval_cfg = cfg["evaluation"]

    if "coco_bbox" in eval_cfg["metrics"] or "coco_segm" in eval_cfg["metrics"]:
        coco_cfg = eval_cfg["coco"]

        coco_metrics = evaluate_coco_bbox_segm(
            model=model,
            loader=loader,
            dataset=dataset,
            device=device,
            score_thr=coco_cfg.get("score_thr", 0.05),
            mask_thr=coco_cfg.get("mask_thr", 0.5),
        )

        metrics.update(coco_metrics)

    if "pothole_metric" in eval_cfg["metrics"]:
        pm_cfg = eval_cfg["pothole_metric"]

        pothole_metrics = evaluate_pothole_metric(
            model=model,
            loader=loader,
            device=device,
            class_label=cfg["data"]["object_label"],
            prefix=pm_cfg.get("prefix", "pothole"),
            score_thr=pm_cfg.get("score_thr", 0.05),
            mask_thr=pm_cfg.get("mask_thr", 0.5),
        )

        metrics.update(pothole_metrics)

    return metrics


__all__ = [
    "evaluate_model",
    "save_metrics",
]