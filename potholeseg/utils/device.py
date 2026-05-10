import torch


def get_device(device: str = "auto") -> torch.device:
    """
    Select device for training or inference.

    Args:
        device: "auto", "cuda", or "cpu".

    Returns:
        torch.device object.
    """
    device = str(device).lower()

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")

    if device == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device option: {device}")


def print_device_info(device: torch.device) -> None:
    """
    Print device information.
    """
    print("Device:", device)

    if device.type == "cuda":
        print("CUDA available:", torch.cuda.is_available())
        print("CUDA version:", torch.version.cuda)
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("CUDA available:", torch.cuda.is_available())