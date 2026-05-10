from pathlib import Path
from typing import Dict, Any


def ensure_dir(path: str | Path) -> Path:
    """
    Create directory if it does not exist.

    Args:
        path: Directory path.

    Returns:
        Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_project_dirs(cfg: Dict[str, Any]) -> Dict[str, Path]:
    """
    Create and return important project directories.

    Args:
        cfg: Loaded YAML config.

    Returns:
        Dictionary of Path objects.
    """
    work_dir = ensure_dir(cfg["project"]["work_dir"])
    test_eval_dir = ensure_dir(cfg["project"]["test_eval_dir"])

    checkpoints_dir = ensure_dir(work_dir / "checkpoints")
    logs_dir = ensure_dir(work_dir / "logs")
    predictions_dir = ensure_dir(work_dir / "predictions")
    exports_dir = ensure_dir(work_dir / "exports")

    return {
        "work_dir": work_dir,
        "test_eval_dir": test_eval_dir,
        "checkpoints_dir": checkpoints_dir,
        "logs_dir": logs_dir,
        "predictions_dir": predictions_dir,
        "exports_dir": exports_dir,
    }