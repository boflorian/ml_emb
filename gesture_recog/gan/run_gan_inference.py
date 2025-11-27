#!/usr/bin/env python3
"""
Run inference on the GAN-generated Magic Wand dataset using the best
validation-accuracy CNN model from the main pipeline.
"""
import json
import os
import pathlib
import sys
from typing import Iterator, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Ensure repository root is on the path when executed as a script
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from util.data_loader import (  # noqa: E402
    CATEGORIES,
    CAT_TO_ID,
    parse_gesture_file,
    lowpass_filter,
    normalize_clip,
)
from main import (  # noqa: E402
    WIN_CNN,
    LP_WINDOW_CNN,
    find_best_averaged_model,
    evaluate_on_dataset,
)

GAN_ROOT = REPO_ROOT / "dataset_magic_wand_gan"
# The GAN set uses "wave" but the trained models expect "wing"
CLASS_ALIAS = {"wave": "wing"}


def iter_gan_samples() -> Iterator[Tuple[np.ndarray, np.int32]]:
    """Yield (sequence[T,3], label) pairs from the GAN dataset."""
    for cls_dir in sorted(GAN_ROOT.iterdir()):
        if not cls_dir.is_dir():
            continue
        raw_label = cls_dir.name
        canonical_label = CLASS_ALIAS.get(raw_label, raw_label)
        if canonical_label not in CATEGORIES:
            print(f"Skipping unknown class folder: {raw_label}")
            continue
        y = np.int32(CAT_TO_ID[canonical_label])
        for txt in sorted(cls_dir.glob("*.txt")):
            for sample in parse_gesture_file(txt):
                x = np.asarray(sample["data"], dtype=np.float32)
                if x.size == 0:
                    continue
                yield x, y


def build_gan_dataset(batch_size: int = 64, win: int = WIN_CNN, lp_window: int = LP_WINDOW_CNN):
    """Create a tf.data.Dataset matching the main inference preprocessing."""
    output_signature = (
        tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    ds = tf.data.Dataset.from_generator(iter_gan_samples, output_signature=output_signature)

    def to_fixed_len(x):
        x = x[:win]
        pad = tf.maximum(0, win - tf.shape(x)[0])
        return tf.pad(x, [[0, pad], [0, 0]])

    def preprocess(x, y):
        x = lowpass_filter(x, window=lp_window)
        x = normalize_clip(x)
        x = to_fixed_len(x)
        return x, y

    return ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def load_best_cnn():
    """Load the CNN averaged model with the highest validation accuracy."""
    model_path = find_best_averaged_model("cnn")
    print(f"Loading CNN model from: {model_path}")
    model = keras.models.load_model(model_path)
    return model, model_path


def main():
    if not GAN_ROOT.exists():
        raise FileNotFoundError(f"GAN dataset not found at {GAN_ROOT}")

    # Ensure relative paths (e.g., benchmark_models/) resolve from repo root
    os.chdir(REPO_ROOT)

    ds = build_gan_dataset()
    model, model_path = load_best_cnn()

    print("\n=== Running inference on GAN dataset ===")
    metrics = evaluate_on_dataset(model, ds, class_names=CATEGORIES, threshold=0.4)

    metrics_path = GAN_ROOT.parent / "gan" / "metrics_gan_cnn.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nMetrics saved to {metrics_path}")

    print(f"\nModel used: {model_path}")


if __name__ == "__main__":
    main()
