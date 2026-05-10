from potholeseg.config import load_config, print_config_summary


def main():
    cfg = load_config("configs/maskrcnn_mobilenetv3_large_fpn.yaml")
    print_config_summary(cfg)


if __name__ == "__main__":
    main()