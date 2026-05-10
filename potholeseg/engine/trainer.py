from typing import Dict, List, Tuple

import torch
from tqdm.auto import tqdm


def move_targets_to_device(targets, device):
    """
    Move tensor values in target dictionaries to device.

    Non-tensor values are preserved.
    """
    moved = []

    for target in targets:
        moved_target = {}

        for key, value in target.items():
            if torch.is_tensor(value):
                moved_target[key] = value.to(device)
            else:
                moved_target[key] = value

        moved.append(moved_target)

    return moved


def set_batchnorm_eval(module):
    """
    Put BatchNorm layers into eval mode.
    Useful when computing validation loss with model.train().
    """
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()


def filter_detection_target_keys(target):
    """
    TorchVision detection models only need these keys for loss computation.
    Metadata such as original_size, processed_size, and scale_factor should not
    be passed into the model during training/validation loss.
    """
    allowed = {
        "boxes",
        "labels",
        "masks",
        "image_id",
        "area",
        "iscrowd",
    }

    return {
        key: value
        for key, value in target.items()
        if key in allowed
    }


def prepare_targets_for_loss(targets, device):
    targets = [
        filter_detection_target_keys(target)
        for target in targets
    ]

    return move_targets_to_device(targets, device)


def train_one_epoch(
    model,
    optimizer,
    scheduler,
    loader,
    device,
    epoch: int,
    cfg,
    scaler=None,
    start_global_step: int = 0,
) -> Tuple[Dict[str, float], int]:
    """
    Train one epoch.

    Args:
        epoch: one-based epoch number.
        scheduler: WarmupCosineScheduler or None.
        start_global_step: global iteration before this epoch.

    Returns:
        metrics, new_global_step
    """
    model.train()

    running = {
        "loss": 0.0,
        "loss_classifier": 0.0,
        "loss_box_reg": 0.0,
        "loss_mask": 0.0,
        "loss_objectness": 0.0,
        "loss_rpn_box_reg": 0.0,
    }

    train_cfg = cfg["train"]
    use_amp = (
        bool(train_cfg.get("amp", False))
        and device.type == "cuda"
        and scaler is not None
    )

    grad_clip_cfg = train_cfg.get("gradient_clipping", {})
    grad_clip_enabled = bool(grad_clip_cfg.get("enabled", False))
    grad_clip_max_norm = float(grad_clip_cfg.get("max_norm", 5.0))

    global_step = start_global_step

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]")

    for images, targets in pbar:
        if scheduler is not None:
            lr = scheduler.step(
                epoch_idx=epoch - 1,
                global_step=global_step,
            )
        else:
            lr = optimizer.param_groups[0]["lr"]

        images = [
            image.to(device)
            for image in images
        ]

        targets = prepare_targets_for_loss(targets, device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        if use_amp:
            scaler.scale(losses).backward()

            if grad_clip_enabled:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip_max_norm,
                )

            scaler.step(optimizer)
            scaler.update()

        else:
            losses.backward()

            if grad_clip_enabled:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip_max_norm,
                )

            optimizer.step()

        running["loss"] += float(losses.detach().cpu())
        running["loss_classifier"] += float(
            loss_dict["loss_classifier"].detach().cpu()
        )
        running["loss_box_reg"] += float(
            loss_dict["loss_box_reg"].detach().cpu()
        )
        running["loss_mask"] += float(
            loss_dict["loss_mask"].detach().cpu()
        )
        running["loss_objectness"] += float(
            loss_dict["loss_objectness"].detach().cpu()
        )
        running["loss_rpn_box_reg"] += float(
            loss_dict["loss_rpn_box_reg"].detach().cpu()
        )

        global_step += 1

        pbar.set_postfix(
            loss=round(float(losses.detach().cpu()), 4),
            mask=round(float(loss_dict["loss_mask"].detach().cpu()), 4),
            lr=f"{lr:.6f}",
        )

    n = max(1, len(loader))

    for key in running:
        running[key] /= n

    return running, global_step


@torch.no_grad()
def validate_loss(model, loader, device) -> Dict[str, float]:
    """
    Compute validation loss for TorchVision detection model.

    TorchVision detection models return loss only in train mode.
    Therefore:
        model.train()
        no_grad()
        BatchNorm layers set to eval()
    """
    model.train()
    model.apply(set_batchnorm_eval)

    running = {
        "val_loss": 0.0,
        "val_loss_classifier": 0.0,
        "val_loss_box_reg": 0.0,
        "val_loss_mask": 0.0,
        "val_loss_objectness": 0.0,
        "val_loss_rpn_box_reg": 0.0,
    }

    pbar = tqdm(loader, desc="[valid loss]")

    for images, targets in pbar:
        images = [
            image.to(device)
            for image in images
        ]

        targets = prepare_targets_for_loss(targets, device)

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        running["val_loss"] += float(losses.detach().cpu())
        running["val_loss_classifier"] += float(
            loss_dict["loss_classifier"].detach().cpu()
        )
        running["val_loss_box_reg"] += float(
            loss_dict["loss_box_reg"].detach().cpu()
        )
        running["val_loss_mask"] += float(
            loss_dict["loss_mask"].detach().cpu()
        )
        running["val_loss_objectness"] += float(
            loss_dict["loss_objectness"].detach().cpu()
        )
        running["val_loss_rpn_box_reg"] += float(
            loss_dict["loss_rpn_box_reg"].detach().cpu()
        )

        pbar.set_postfix(
            val_loss=round(float(losses.detach().cpu()), 4)
        )

    n = max(1, len(loader))

    for key in running:
        running[key] /= n

    return running