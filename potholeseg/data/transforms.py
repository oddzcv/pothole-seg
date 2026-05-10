from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import albumentations as A


def resize_keep_ratio_image_boxes_masks(
    image: np.ndarray,
    boxes: Sequence[Sequence[float]],
    masks: Sequence[np.ndarray],
    scale: Tuple[int, int],
):
    """
    Emulate MMDetection Resize(scale=(W, H), keep_ratio=True).

    Args:
        image: RGB image, shape (H, W, 3), uint8.
        boxes: Bounding boxes in xyxy / Pascal VOC format.
        masks: Instance masks, each shape (H, W).
        scale: Target scale in (width, height).

    Returns:
        resized_image, resized_boxes, resized_masks, scale_factor_xy
    """
    target_w, target_h = scale
    orig_h, orig_w = image.shape[:2]

    scale_factor = min(target_w / orig_w, target_h / orig_h)

    new_w = int(round(orig_w * scale_factor))
    new_h = int(round(orig_h * scale_factor))

    resized_image = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR,
    )

    if len(boxes) > 0:
        boxes = np.array(boxes, dtype=np.float32)
        boxes[:, [0, 2]] *= new_w / orig_w
        boxes[:, [1, 3]] *= new_h / orig_h
    else:
        boxes = np.zeros((0, 4), dtype=np.float32)

    if len(masks) > 0:
        resized_masks = []

        for mask in masks:
            resized_mask = cv2.resize(
                mask.astype(np.uint8),
                (new_w, new_h),
                interpolation=cv2.INTER_NEAREST,
            )
            resized_masks.append(resized_mask)

        masks = np.stack(resized_masks).astype(np.uint8)
    else:
        masks = np.zeros((0, new_h, new_w), dtype=np.uint8)

    scale_factor_xy = np.array(
        [
            new_w / orig_w,
            new_h / orig_h,
        ],
        dtype=np.float32,
    )

    return resized_image, boxes, masks, scale_factor_xy


def get_train_scales(cfg: Dict[str, Any]) -> List[Tuple[int, int]]:
    """
    Get RandomChoiceResize scales from config.

    YAML stores scales as list of lists, e.g. [[1333, 800], [1600, 900]].
    This function converts them to list of tuples.
    """
    scales = cfg["augmentation"]["train"]["random_choice_resize"]["scales"]
    return [tuple(map(int, scale)) for scale in scales]


def get_eval_scale(cfg: Dict[str, Any], split: str) -> Tuple[int, int]:
    """
    Get validation or test resize scale from config.
    """
    split = "val" if split in ["valid", "val"] else split

    if split not in ["val", "test"]:
        raise ValueError(f"Unsupported eval split: {split}")

    scale = cfg["augmentation"][split]["resize"]["scale"]
    return tuple(map(int, scale))


def build_train_albu_like_mmdet(cfg: Dict[str, Any]):
    """
    Build Albumentations pipeline equivalent to the MMDetection config:

    RandomFlip(prob=0.5)
    Albu([
        RandomBrightnessContrast,
        HueSaturationValue,
        MotionBlur,
        Sharpen
    ])
    """
    train_aug = cfg["augmentation"]["train"]

    transforms = []

    flip_cfg = train_aug["random_flip"]

    if flip_cfg.get("enabled", True):
        transforms.append(
            A.HorizontalFlip(
                p=flip_cfg.get("p", 0.5),
            )
        )

    albu_cfg = train_aug["albumentations"]

    rbc_cfg = albu_cfg["random_brightness_contrast"]
    if rbc_cfg.get("enabled", True):
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=rbc_cfg.get("brightness_limit", 0.2),
                contrast_limit=rbc_cfg.get("contrast_limit", 0.2),
                p=rbc_cfg.get("p", 0.7),
            )
        )

    hsv_cfg = albu_cfg["hue_saturation_value"]
    if hsv_cfg.get("enabled", True):
        transforms.append(
            A.HueSaturationValue(
                hue_shift_limit=hsv_cfg.get("hue_shift_limit", 8),
                sat_shift_limit=hsv_cfg.get("sat_shift_limit", 15),
                val_shift_limit=hsv_cfg.get("val_shift_limit", 10),
                p=hsv_cfg.get("p", 0.4),
            )
        )

    blur_cfg = albu_cfg["motion_blur"]
    if blur_cfg.get("enabled", True):
        transforms.append(
            A.MotionBlur(
                blur_limit=blur_cfg.get("blur_limit", 4),
                p=blur_cfg.get("p", 0.2),
            )
        )

    sharpen_cfg = albu_cfg["sharpen"]
    if sharpen_cfg.get("enabled", True):
        transforms.append(
            A.Sharpen(
                alpha=tuple(sharpen_cfg.get("alpha", [0.1, 0.25])),
                lightness=tuple(sharpen_cfg.get("lightness", [0.9, 1.1])),
                p=sharpen_cfg.get("p", 0.15),
            )
        )

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels", "iscrowd"],
            min_visibility=0.0,
        ),
    )