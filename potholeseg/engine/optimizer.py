from typing import Any, Dict

import torch


def build_optimizer(model, cfg: Dict[str, Any]):
    """
    Build optimizer from config.
    """
    opt_cfg = cfg["train"]["optimizer"]
    name = opt_cfg["name"].lower()

    params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    if name == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=opt_cfg["lr"],
            momentum=opt_cfg.get("momentum", 0.9),
            weight_decay=opt_cfg.get("weight_decay", 0.0001),
        )

    elif name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg.get("weight_decay", 0.0001),
        )

    else:
        raise ValueError(f"Unsupported optimizer: {name}")

    return optimizer