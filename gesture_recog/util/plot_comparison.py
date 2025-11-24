import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pathlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse

from ml_emb.gesture_recog.util.data_loader import (
    CATEGORIES,
    CAT_TO_ID,
    iter_samples,
    lowpass_filter,
    normalize_clip,
)

MAGIC_WAND_ROOT = pathlib.Path("../dataset_magic_wand")
PICO_ROOT = pathlib.Path("../dataset_pico_gestures/processed")

LP_WINDOW = 1  # Set to 1 to disable lowpass for raw comparison

def load_and_preprocess_sample(sample_iter, num_samples=5, selected_classes=None):
    """Load and preprocess a few samples from the iterator, filtered by classes."""
    if selected_classes is None:
        selected_ids = set(range(len(CATEGORIES)))
    else:
        selected_ids = {CAT_TO_ID[c] for c in selected_classes if c in CAT_TO_ID}
    
    samples = []
    labels = []
    for x, y in sample_iter:
        if y in selected_ids:
            # Preprocess
            x = lowpass_filter(x, window=LP_WINDOW)
            x = normalize_clip(x)
            samples.append(x.numpy() if isinstance(x, tf.Tensor) else x)
            labels.append(y)
            if len(samples) >= num_samples:
                break
    return samples, labels

def plot_signals(magic_samples, magic_labels, pico_samples, pico_labels, categories=CATEGORIES):
    """Plot the signals for both datasets in one figure."""
    num_samples = min(len(magic_samples), len(pico_samples), 4)  
    fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8), sharex=True)
    fig.suptitle("Magic Wand vs Pico Dataset Signals Comparison")

    for i in range(num_samples):
        # Magic Wand
        ax = axes[0, i]
        x, y = magic_samples[i], magic_labels[i]
        t = np.arange(x.shape[0])
        ax.plot(t, x[:, 0], label='X', color='r')
        ax.plot(t, x[:, 1], label='Y', color='g')
        ax.plot(t, x[:, 2], label='Z', color='b')
        ax.set_ylabel('Accel')
        ax.set_title(f"Magic Wand: {categories[y]}")
        ax.legend()
        ax.grid(True)

        # Pico
        ax = axes[1, i]
        x, y = pico_samples[i], pico_labels[i]
        t = np.arange(x.shape[0])
        ax.plot(t, x[:, 0], label='X', color='r')
        ax.plot(t, x[:, 1], label='Y', color='g')
        ax.plot(t, x[:, 2], label='Z', color='b')
        ax.set_ylabel('Accel')
        ax.set_title(f"Pico: {categories[y]}")
        ax.legend()
        ax.grid(True)

    plt.xlabel('Time Steps')
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot comparison of signals from Magic Wand and Pico datasets.")
    parser.add_argument("--classes", nargs='*', default=CATEGORIES,
                        help=f"Classes to plot: {CATEGORIES} (default all)")
    parser.add_argument("--num_samples", type=int, default=4,
                        help="Number of samples per dataset/class (default 4)")
    args = parser.parse_args()

    selected_classes = args.classes
    num_samples = args.num_samples

    # Load from Magic Wand
    print(f"Loading Magic Wand samples for classes: {selected_classes}...")
    magic_iter = iter_samples(subjects=None, root_dir=MAGIC_WAND_ROOT)
    magic_samples, magic_labels = load_and_preprocess_sample(magic_iter, num_samples=num_samples, selected_classes=selected_classes)

    # Load from Pico
    print(f"Loading Pico samples for classes: {selected_classes}...")
    pico_iter = iter_samples(subjects=None, root_dir=PICO_ROOT)
    pico_samples, pico_labels = load_and_preprocess_sample(pico_iter, num_samples=num_samples, selected_classes=selected_classes)

    # Plot both in one figure
    if magic_samples and pico_samples:
        plot_signals(magic_samples, magic_labels, pico_samples, pico_labels)
    else:
        print("No samples found for the selected classes.")

if __name__ == "__main__":
    main()