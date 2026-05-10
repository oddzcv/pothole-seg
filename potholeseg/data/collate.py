def collate_fn(batch):
    """
    Collate function for TorchVision detection models.

    TorchVision Mask R-CNN expects:
        images: list[Tensor]
        targets: list[dict]
    """
    return tuple(zip(*batch))