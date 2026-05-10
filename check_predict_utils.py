import numpy as np
import torch

from potholeseg.utils import draw_prediction


def main():
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    image[:] = 80

    output = {
        "boxes": torch.tensor([[40.0, 50.0, 150.0, 170.0]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
        "scores": torch.tensor([0.95], dtype=torch.float32),
        "masks": torch.zeros((1, 1, 224, 224), dtype=torch.float32),
    }

    output["masks"][0, 0, 50:170, 40:150] = 1.0

    vis = draw_prediction(
        image_rgb=image,
        output=output,
        class_names=["__background__", "Lubang"],
        score_thr=0.05,
        mask_thr=0.5,
        mask_alpha=0.45,
    )

    print("Visualization shape:", vis.shape)
    print("Visualization dtype:", vis.dtype)
    print("Predict utility check passed.")


if __name__ == "__main__":
    main()