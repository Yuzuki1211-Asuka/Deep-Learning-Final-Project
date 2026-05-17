import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--title", default="Training and Validation Loss")
    args = parser.parse_args()

    if not os.path.exists(args.log_file):
        raise FileNotFoundError(f"Log file not found: {args.log_file}")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    df = pd.read_csv(args.log_file)

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], marker="o", label="Train Loss")
    plt.plot(df["epoch"], df["valid_loss"], marker="s", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(args.title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(args.save_path, dpi=300)
    print(f"Loss curve saved to: {args.save_path}")
    print(df.tail(1).to_string(index=False))


if __name__ == "__main__":
    main()
