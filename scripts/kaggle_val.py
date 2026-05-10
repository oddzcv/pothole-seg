import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Kaggle validation launcher for trained pothole segmentation model."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to checkpoint .pth file.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "val", "test"],
        help="Dataset split to evaluate.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device override.",
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Optional existing dataset root. If omitted, Roboflow download is used.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output JSON path.",
    )

    return parser.parse_args()


def set_roboflow_key_from_kaggle_secret():
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

    set_roboflow_key_from_kaggle_secret()

    command = [
        sys.executable,
        "val.py",
        "--config",
        args.config,
        "--weights",
        args.weights,
        "--split",
        args.split,
        "--device",
        args.device,
    ]

    if args.data_root is not None:
        command.extend(["--data-root", args.data_root])

    if args.output is not None:
        command.extend(["--output", args.output])

    run_command(command)


if __name__ == "__main__":
    main()