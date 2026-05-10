import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Kaggle training launcher for pothole instance segmentation."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional epoch override.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device override.",
    )

    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Run validation evaluation every N epochs.",
    )

    parser.add_argument(
        "--skip-val-eval",
        action="store_true",
        help="Skip validation COCO/Pothole evaluation during training.",
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Optional existing dataset root. If omitted, Roboflow download is used.",
    )

    return parser.parse_args()


def set_roboflow_key_from_kaggle_secret():
    """
    Load ROBOFLOW_API_KEY from Kaggle Secrets when running inside Kaggle.

    In Kaggle Notebook:
        Add secret named ROBOFLOW_API_KEY.
    """
    if os.getenv("ROBOFLOW_API_KEY"):
        print("ROBOFLOW_API_KEY already exists in environment.")
        return

    try:
        from kaggle_secrets import UserSecretsClient

        user_secrets = UserSecretsClient()
        os.environ["ROBOFLOW_API_KEY"] = user_secrets.get_secret("ROBOFLOW_API_KEY")
        print("Loaded ROBOFLOW_API_KEY from Kaggle Secrets.")

    except Exception as exc:
        print("Could not load ROBOFLOW_API_KEY from Kaggle Secrets.")
        print("Reason:", repr(exc))
        print(
            "If dataset download is needed, set ROBOFLOW_API_KEY manually "
            "or pass --data-root to use an existing dataset."
        )


def run_command(command):
    print("\nRunning command:")
    print(" ".join(command))
    print()

    result = subprocess.run(command)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed with return code {result.returncode}")


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    print("Repository root:", repo_root)
    print("Config:", args.config)

    set_roboflow_key_from_kaggle_secret()

    command = [
        sys.executable,
        "train.py",
        "--config",
        args.config,
        "--device",
        args.device,
        "--eval-every",
        str(args.eval_every),
    ]

    if args.epochs is not None:
        command.extend(["--epochs", str(args.epochs)])

    if args.skip_val_eval:
        command.append("--skip-val-eval")

    if args.data_root is not None:
        command.extend(["--data-root", args.data_root])

    run_command(command)


if __name__ == "__main__":
    main()