from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_gesture_file(filepath: Path):
    samples = []
    current = None
    with filepath.open("r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("sample"):
                if current:
                    samples.append(current)
                current = []
            elif line == "ax,ay,az":
                continue
            elif line and "," in line:
                try:
                    ax, ay, az = map(float, line.split(","))
                except ValueError:
                    continue
                current.append([ax, ay, az])
        if current:
            samples.append(current)
    return samples


def load_class_samples(root: Path, cls: str):
    samples = []
    for path in sorted((root / cls).glob("*.txt")):
        for s in parse_gesture_file(path):
            arr = np.asarray(s, dtype=float)
            if arr.size == 0:
                continue
            samples.append(arr)
    return samples


def plot_class_axis(samples, cls, axis_idx, out_dir, show):
    fig, ax = plt.subplots(figsize=(10, 4))
    for sample in samples:
        if sample.shape[1] <= axis_idx:
            continue
        ax.plot(sample[:, axis_idx], alpha=0.2, linewidth=1)
    ax.set_title(f"{cls} axis {axis_idx} (all samples)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Acceleration")
    ax.grid(True, alpha=0.3)
    out_path = out_dir / f"{cls}_axis_{axis_idx}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot acceleration traces for non-negative classes.")
    parser.add_argument("--root", default="dataset_good", help="Dataset root.")
    parser.add_argument("--out-dir", default="good_plots", help="Output directory for plots.")
    parser.add_argument("--show", action="store_true", help="Show plots interactively.")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = ["ring", "slope", "wave"]
    for cls in classes:
        samples = load_class_samples(root, cls)
        if not samples:
            print(f"[WARN] No samples found for class {cls}")
            continue
        for axis_idx in range(3):
            plot_class_axis(samples, cls, axis_idx, out_dir, args.show)
    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
