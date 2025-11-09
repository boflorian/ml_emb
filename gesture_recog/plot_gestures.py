#!/usr/bin/env python3
import argparse
from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D projection)

SAMPLE_RE = re.compile(r"^sample\d+\s*$", re.IGNORECASE)

def parse_gesture_file(filepath: Path):
    """Return a list of np.ndarray, each shape (T, 3) for ax, ay, az."""
    samples, cur = [], []
    with open(filepath, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if SAMPLE_RE.match(s):
                if cur:
                    samples.append(np.asarray(cur, dtype=np.float32))
                    cur = []
                continue
            if s.lower().startswith("ax,ay,az"):
                continue
            if "," in s:
                try:
                    ax, ay, az = map(float, s.split(","))
                    cur.append([ax, ay, az])
                except ValueError:
                    # skip malformed rows
                    pass
    if cur:
        samples.append(np.asarray(cur, dtype=np.float32))
    return samples

def main():
    p = argparse.ArgumentParser(description="Plot all IMU samples from a personX.txt in 3D (ax, ay, az).")
    p.add_argument("file", type=Path, help="Path to personX.txt")
    p.add_argument("--max-samples", type=int, default=None, help="Plot only the first N samples")
    p.add_argument("--downsample", type=int, default=1, help="Keep every k-th point for plotting speed (default: 1)")
    p.add_argument("--save", type=Path, default=None, help="Save figure to this path instead of showing")
    args = p.parse_args()

    samples = parse_gesture_file(args.file)
    if not samples:
        raise SystemExit(f"No samples found in {args.file}")

    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Auto color cycle; label each sample
    for i, seq in enumerate(samples, start=1):
        seq = seq[:: args.downsample]
        ax.plot(seq[:, 0], seq[:, 1], seq[:, 2], label=f"sample{i}", linewidth=1.2, alpha=0.9)

    ax.set_title(f"3D IMU traces — {args.file.name}")
    ax.set_xlabel("ax (g)")
    ax.set_ylabel("ay (g)")
    ax.set_zlabel("az (g)")

    # Make it easier to view: equal-ish aspect
    all_pts = np.vstack([s[:: args.downsample] for s in samples])
    mins, maxs = all_pts.min(axis=0), all_pts.max(axis=0)
    ranges = maxs - mins
    center = (maxs + mins) / 2.0
    max_range = ranges.max() if ranges.max() > 0 else 1.0
    for axis, c in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], center):
        axis(c - max_range / 2, c + max_range / 2)

    # Put legend outside if many samples
    if len(samples) <= 20:
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0))
        plt.tight_layout(rect=[0, 0, 0.8, 1])
    else:
        plt.tight_layout()

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.save, dpi=150)
        print(f"Saved plot to {args.save}")
    else:
        plt.show()

if __name__ == "__main__":
    main()