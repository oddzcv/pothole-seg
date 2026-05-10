import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.transform import resize_boxes
import torchvision.models.detection.roi_heads as roi_heads


# TorchVision version compatibility:
# Some versions expose paste_masks_in_image, others use _paste_masks_in_image.
PASTE_MASKS_IN_IMAGE = getattr(roi_heads, "paste_masks_in_image", None)

if PASTE_MASKS_IN_IMAGE is None:
    PASTE_MASKS_IN_IMAGE = getattr(roi_heads, "_paste_masks_in_image")


class NoResizeGeneralizedRCNNTransform(nn.Module):
    """
    TorchVision detection transform without resizing.

    This transform is used because the dataset pipeline already applies
    MMDetection-like resizing:
        - train: RandomChoiceResize
        - val/test: Resize(scale=(1600, 900), keep_ratio=True)

    Responsibilities:
        - normalize images
        - batch images with padding
        - avoid internal resizing
        - paste 28x28 Mask R-CNN masks into full image size during inference
    """

    def __init__(
        self,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        size_divisible: int = 32,
    ) -> None:
        super().__init__()

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]

        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        self.image_mean = image_mean
        self.image_std = image_std
        self.size_divisible = size_divisible

    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        if not image.is_floating_point():
            raise TypeError(
                f"Expected input image to be float tensor, got {image.dtype}."
            )

        dtype = image.dtype
        device = image.device

        mean = torch.as_tensor(
            self.image_mean,
            dtype=dtype,
            device=device,
        )[:, None, None]

        std = torch.as_tensor(
            self.image_std,
            dtype=dtype,
            device=device,
        )[:, None, None]

        return (image - mean) / std

    def batch_images(self, images: List[torch.Tensor]) -> torch.Tensor:
        max_size = list(images[0].shape)

        for image in images[1:]:
            for dim in range(len(max_size)):
                max_size[dim] = max(max_size[dim], image.shape[dim])

        stride = self.size_divisible
        max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
        max_size[2] = int(math.ceil(max_size[2] / stride) * stride)

        batch_shape = [len(images)] + max_size
        batched_images = images[0].new_full(batch_shape, 0)

        for image, padded_image in zip(images, batched_images):
            padded_image[
                : image.shape[0],
                : image.shape[1],
                : image.shape[2],
            ].copy_(image)

        return batched_images

    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[dict]] = None,
    ) -> Tuple[ImageList, Optional[List[dict]]]:
        images = [self.normalize(image) for image in images]

        image_sizes = [
            image.shape[-2:]
            for image in images
        ]

        batched_images = self.batch_images(images)
        image_list = ImageList(batched_images, image_sizes)

        return image_list, targets

    def postprocess(
        self,
        result: List[dict],
        image_shapes: List[Tuple[int, int]],
        original_image_sizes: List[Tuple[int, int]],
    ) -> List[dict]:
        """
        Paste Mask R-CNN 28x28 masks into full image size.

        TorchVision Mask R-CNN mask head returns per-instance masks of size 28x28.
        The default GeneralizedRCNNTransform normally pastes those masks to image
        coordinates. Since we replaced the transform to avoid resizing, we must
        still perform this mask postprocessing step.
        """
        if self.training:
            return result

        for pred, image_shape, original_image_size in zip(
            result,
            image_shapes,
            original_image_sizes,
        ):
            boxes = pred["boxes"]

            if image_shape != original_image_size:
                boxes = resize_boxes(
                    boxes,
                    image_shape,
                    original_image_size,
                )
                pred["boxes"] = boxes

            if "masks" in pred and pred["masks"].numel() > 0:
                pred["masks"] = PASTE_MASKS_IN_IMAGE(
                    pred["masks"],
                    boxes,
                    original_image_size,
                )

        return result