# potholeseg/data/enhancement.py

from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np


def _clahe_rgb(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_grid_size), int(tile_grid_size)),
    )
    l = clahe.apply(l)

    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)


def _gray_world_white_balance(image: np.ndarray) -> np.ndarray:
    image_f = image.astype(np.float32)
    means = image_f.reshape(-1, 3).mean(axis=0)
    gray = means.mean()
    scale = gray / (means + 1e-6)
    out = image_f * scale[None, None, :]
    return np.clip(out, 0, 255).astype(np.uint8)


def _gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    gamma = max(float(gamma), 1e-6)
    inv_gamma = 1.0 / gamma
    table = ((np.arange(256) / 255.0) ** inv_gamma * 255.0).astype(np.uint8)
    return cv2.LUT(image, table)


def _guided_like_filter(image: np.ndarray, radius: int = 5) -> np.ndarray:
    # Lightweight approximation using bilateral filtering.
    # This is practical for dataset preprocessing and inference.
    radius = int(radius)
    if radius <= 0:
        return image
    return cv2.bilateralFilter(image, d=radius, sigmaColor=50, sigmaSpace=50)


def enhance_image(image: np.ndarray, cfg: Dict[str, Any] | None = None) -> np.ndarray:
    """
    Physics-inspired underwater/low-contrast image enhancement.

    Input:
        image: RGB uint8 image.

    Output:
        RGB uint8 enhanced image.

    Notes:
        This implements the reproducible part of the paper's enhancement stage:
        color compensation, contrast enhancement, gamma correction, and filtering.
        It is intentionally non-learned because the paper does not provide
        training code or pretrained weights for its CNN color-correction module.
    """
    if cfg is None:
        cfg = {}

    if not cfg.get("enabled", False):
        return image

    out = image

    if cfg.get("white_balance", True):
        out = _gray_world_white_balance(out)

    if cfg.get("clahe", True):
        out = _clahe_rgb(
            out,
            clip_limit=cfg.get("clahe_clip_limit", 2.0),
            tile_grid_size=cfg.get("clahe_tile_grid_size", 8),
        )

    gamma = cfg.get("gamma", None)
    if gamma is not None:
        out = _gamma_correction(out, gamma=float(gamma))

    if cfg.get("guided_filter", True):
        out = _guided_like_filter(
            out,
            radius=cfg.get("guided_filter_radius", 5),
        )

    return out