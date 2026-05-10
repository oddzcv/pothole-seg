from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to YAML config.

    Returns:
        Dictionary configuration.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"Config file is empty: {config_path}")

    return cfg


def print_config_summary(cfg: Dict[str, Any]) -> None:
    """
    Print important configuration summary.
    """
    print("Project:", cfg["project"]["name"])
    print("Device:", cfg["project"]["device"])
    print("Work dir:", cfg["project"]["work_dir"])

    print("\nDataset:")
    print("  Source:", cfg["data"]["source"])
    print("  Roboflow workspace:", cfg["data"]["workspace"])
    print("  Roboflow project:", cfg["data"]["project"])
    print("  Version:", cfg["data"]["version"])
    print("  Class:", cfg["data"]["class_name"])
    print("  Num classes:", cfg["data"]["num_classes"])

    print("\nModel:")
    print("  Architecture:", cfg["model"]["architecture"])
    print("  Backbone:", cfg["model"]["backbone"]["name"])
    print("  Pretrained:", cfg["model"]["backbone"]["pretrained"])

    print("\nTraining:")
    print("  Epochs:", cfg["train"]["epochs"])
    print("  Batch size:", cfg["train"]["batch_size"])
    print("  Optimizer:", cfg["train"]["optimizer"]["name"])
    print("  LR:", cfg["train"]["optimizer"]["lr"])

    print("\nEvaluation:")
    print("  Metrics:", ", ".join(cfg["evaluation"]["metrics"]))
    print("  COCO score thr:", cfg["evaluation"]["coco"]["score_thr"])
    print("  Mask thr:", cfg["evaluation"]["coco"]["mask_thr"])