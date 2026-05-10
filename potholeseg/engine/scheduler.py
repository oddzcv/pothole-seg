import math
from typing import Any, Dict


class WarmupCosineScheduler:
    """
    MMDetection-like scheduler:
        LinearLR warmup for N iterations
        CosineAnnealingLR after warmup

    This scheduler is stepped every training iteration.
    """

    def __init__(self, optimizer, cfg: Dict[str, Any]):
        self.optimizer = optimizer

        train_cfg = cfg["train"]
        opt_cfg = train_cfg["optimizer"]
        sch_cfg = train_cfg["scheduler"]

        self.base_lr = float(opt_cfg["lr"])
        self.eta_min = float(sch_cfg.get("eta_min", 1e-5))
        self.t_max = int(sch_cfg.get("t_max", train_cfg["epochs"]))

        self.warmup_iters = int(sch_cfg.get("warmup_iters", 500))
        self.warmup_start_factor = float(
            sch_cfg.get("warmup_start_factor", 0.001)
        )

    def cosine_lr_for_epoch(self, epoch_idx: int) -> float:
        return self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
            1.0 + math.cos(math.pi * epoch_idx / self.t_max)
        )

    def set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def get_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def step(self, epoch_idx: int, global_step: int) -> float:
        """
        Args:
            epoch_idx: zero-based epoch index.
            global_step: zero-based global training iteration.

        Returns:
            Updated learning rate.
        """
        cosine_lr = self.cosine_lr_for_epoch(epoch_idx)

        if global_step < self.warmup_iters:
            alpha = global_step / max(1, self.warmup_iters)
            factor = self.warmup_start_factor + alpha * (
                1.0 - self.warmup_start_factor
            )
            lr = cosine_lr * factor
        else:
            lr = cosine_lr

        self.set_lr(lr)
        return lr