#!/usr/bin/env python3
"""
Quick visual comparison between training data and GAN generated sequences.
- Loads IMU gesture sequences from a real dataset and a GAN-synthesized dataset
- Creates side-by-side overlays of random samples
- Plots per-axis value distributions to spot coverage/shift issues
"""

import argparse
import pathlib
import random
import re
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

SAMPLE_RE = re.compile(r"^sample\d+\s*$", re.IGNORECASE)
AXES = ["ax", "ay", "az"]
CATEGORIES = ["ring", "slope", "wing"]
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_REAL_ROOT = (SCRIPT_DIR / ".." / "dataset_magic_wand").resolve()
DEFAULT_FAKE_ROOT = (SCRIPT_DIR / ".." / "dataset_magic_wand_gan").resolve()


def parse_gesture_file(filepath: pathlib.Path) -> List[Dict]:
    """
    Parse a gesture file and return a list of samples.
    Each sample: {'sample_id': int, 'data': np.ndarray [T, 3]}
    """
    samples = []
    current_sample = None
    sample_idx = -1

    with filepath.open("r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if SAMPLE_RE.match(line):
                if current_sample is not None and current_sample["data"]:
                    current_sample["data"] = np.array(
                        current_sample["data"], dtype=np.float32
                    )
                    samples.append(current_sample)

                sample_idx += 1
                current_sample = {"sample_id": sample_idx, "data": []}
                continue

            if line.lower().startswith("ax"):
                continue

            parts = line.split(",")
            if len(parts) != 3:
                continue
            try:
                ax, ay, az = map(float, parts)
            except ValueError:
                continue
            current_sample["data"].append([ax, ay, az])

    if current_sample is not None and current_sample["data"]:
        current_sample["data"] = np.array(current_sample["data"], dtype=np.float32)
        samples.append(current_sample)

    return samples


def load_category_sequences(
    root: pathlib.Path, category: str, limit: Optional[int] = None
) -> List[np.ndarray]:
    """
    Load all sequences for a single category.
    """
    cat_dir = root / category
    if not cat_dir.exists():
        raise FileNotFoundError(f"{cat_dir} does not exist")

    sequences: List[np.ndarray] = []
    for txt_path in sorted(cat_dir.glob("*.txt")):
        samples = parse_gesture_file(txt_path)
        for s in samples:
            sequences.append(s["data"])
            if limit is not None and len(sequences) >= limit:
                return sequences

    return sequences


def pad_or_crop(seq: np.ndarray, win: int) -> np.ndarray:
    """
    Make sequence length exactly win using center crop or reflection padding.
    """
    T = seq.shape[0]
    if T == win:
        return seq
    if T > win:
        start = (T - win) // 2
        return seq[start : start + win]
    pad_len = win - T
    return np.pad(seq, ((0, pad_len), (0, 0)), mode="reflect")


def plot_sample_overlays(
    real_seqs: Sequence[np.ndarray],
    fake_seqs: Sequence[np.ndarray],
    out_path: pathlib.Path,
    labels: Optional[Sequence[str]] = None,
):
    count = min(len(real_seqs), len(fake_seqs))
    if count == 0:
        return

    fig, axs = plt.subplots(count, 3, figsize=(11, 2.4 * count), sharex=True)
    if count == 1:
        axs = np.array([axs])

    for i in range(count):
        for c, comp in enumerate(AXES):
            axs[i, c].plot(real_seqs[i][:, c], label="real", color="steelblue", alpha=0.95)
            axs[i, c].plot(fake_seqs[i][:, c], label="gan", color="darkorange", alpha=0.9)
            if i == 0:
                axs[i, c].set_title(comp)
            if c == 0:
                row_label = labels[i] if labels is not None else f"sample {i+1}"
                axs[i, c].set_ylabel(row_label)
            if i == count - 1:
                axs[i, c].set_xlabel("timestep")

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_value_histograms(
    real_seqs: Sequence[np.ndarray],
    fake_seqs: Sequence[np.ndarray],
    out_path: pathlib.Path,
):
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    for c, comp in enumerate(AXES):
        real_vals = np.concatenate([seq[:, c] for seq in real_seqs], axis=0)
        fake_vals = np.concatenate([seq[:, c] for seq in fake_seqs], axis=0)
        axs[c].hist(real_vals, bins=60, density=True, histtype="step", label="real")
        axs[c].hist(fake_vals, bins=60, density=True, histtype="stepfilled", alpha=0.5, label="gan")
        axs[c].set_title(comp)
    axs[0].set_ylabel("density")
    axs[1].set_xlabel("value")
    fig.legend(loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visual comparison between GAN-generated data and the training set."
    )
    parser.add_argument("--real-root", type=pathlib.Path, default=DEFAULT_REAL_ROOT)
    parser.add_argument("--fake-root", type=pathlib.Path, default=DEFAULT_FAKE_ROOT)
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=CATEGORIES,
        help="Gesture categories to compare; one random sample will be drawn per category",
    )
    parser.add_argument("--win", type=int, default=128, help="Sequence length for alignment")
    parser.add_argument(
        "--examples",
        type=int,
        default=4,
        help="Unused: kept for backward compatibility; per-category sampling is used instead",
    )
    parser.add_argument("--sample-limit", type=int, default=400, help="Max sequences to load from each set")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("gan_comparison_plots"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    real_all: List[np.ndarray] = []
    fake_all: List[np.ndarray] = []
    real_subset: List[np.ndarray] = []
    fake_subset: List[np.ndarray] = []
    labels: List[str] = []

    for cat in args.categories:
        real_cat = load_category_sequences(args.real_root, cat, limit=args.sample_limit)
        fake_cat = load_category_sequences(args.fake_root, cat, limit=args.sample_limit)

        if not real_cat:
            raise SystemExit(f"No real sequences found under {args.real_root}/{cat}")
        if not fake_cat:
            raise SystemExit(f"No GAN sequences found under {args.fake_root}/{cat}")

        real_cat = [pad_or_crop(seq, args.win) for seq in real_cat]
        fake_cat = [pad_or_crop(seq, args.win) for seq in fake_cat]
        real_all.extend(real_cat)
        fake_all.extend(fake_cat)

        real_choice = random.choice(real_cat)
        fake_choice = random.choice(fake_cat)
        real_subset.append(real_choice)
        fake_subset.append(fake_choice)
        labels.append(cat)

    if not real_subset or not fake_subset:
        raise SystemExit("Not enough samples to plot – check the dataset sizes or --categories flag.")

    overlay_path = args.out_dir / "real_vs_gan_overlays.png"
    hist_path = args.out_dir / "value_distributions.png"
    plot_sample_overlays(real_subset, fake_subset, overlay_path, labels=labels)
    plot_value_histograms(real_all, fake_all, hist_path)

    print(f"Saved overlay plot to {overlay_path}")
    print(f"Saved value distribution plot to {hist_path}")


if __name__ == "__main__":
    main()
