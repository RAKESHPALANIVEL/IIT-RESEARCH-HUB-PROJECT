"""Create training_loss.png and validation_loss.png from the CSV logs written by trainer.py.

    python scripts/plot_losses.py English
    python scripts/plot_losses.py Telugu
"""
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
COLS = ["total", "mel", "mel_postnet", "pitch", "energy", "duration"]


def read(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [int(r["step"]) for r in rows], {c: [float(r[c]) for r in rows] for c in COLS}


def plot(lang, split, csv_path, out_path):
    steps, d = read(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(steps, d["total"], color="k", lw=1.6)
    if split == "validation":
        best = min(range(len(steps)), key=lambda i: d["total"][i])
        axes[0].axvline(steps[best], color="r", ls="--", lw=0.9,
                        label="min @ step {} ({:.3f})".format(steps[best], d["total"][best]))
        axes[0].legend()
    axes[0].set_title("{} {} - total loss".format(lang, split))
    for c in COLS[1:]:
        axes[1].plot(steps, d[c], label=c, lw=1.2)
    axes[1].set_yscale("log")
    axes[1].set_title("{} {} - components".format(lang, split))
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("wrote", out_path)


if __name__ == "__main__":
    lang = sys.argv[1]
    log_dir = os.path.join(ROOT, lang, "output", "log")
    out_dir = os.path.join(ROOT, "Results", lang)
    os.makedirs(out_dir, exist_ok=True)
    plot(lang, "training", os.path.join(log_dir, "train", "train_loss.csv"), os.path.join(out_dir, "training_loss.png"))
    plot(lang, "validation", os.path.join(log_dir, "val", "val_loss.csv"), os.path.join(out_dir, "validation_loss.png"))
