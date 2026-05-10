from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(
    path: str | Path,
    model,
    optimizer,
    epoch: int,
    cfg: Dict[str, Any],
    extra: Dict[str, Any] | None = None,
) -> Path:
    """
    Save training checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "cfg": cfg,
    }

    if extra:
        payload.update(extra)

    torch.save(payload, path)

    return path


def load_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    map_location="cpu",
    strict: bool = True,
):
    """
    Load checkpoint into model and optionally optimizer.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=map_location)

    model.load_state_dict(ckpt["model"], strict=strict)

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])

    return ckpt