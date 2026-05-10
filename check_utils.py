from potholeseg.config import load_config
from potholeseg.utils import (
    set_seed,
    get_device,
    print_device_info,
    setup_project_dirs,
)


def main():
    cfg = load_config("configs/maskrcnn_mobilenetv3_large_fpn.yaml")

    set_seed(cfg["project"]["seed"])

    device = get_device(cfg["project"]["device"])
    print_device_info(device)

    dirs = setup_project_dirs(cfg)

    print("\nProject directories:")
    for name, path in dirs.items():
        print(f"  {name}: {path}")

    print("\nUtility check passed.")


if __name__ == "__main__":
    main()