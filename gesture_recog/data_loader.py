import pathlib
from ast import parse

import numpy as np
import tensorflow as tf

ROOT = pathlib.Path('dataset_magic_wand')
CATEGORIES = ["negative", "ring", "slope", "wing"]
CAT_TO_ID = {c:i for i,c in enumerate(CATEGORIES)}
BATCH = 64


def parse_gesture_file(filepath):
    """
    Parse a gesture file and return samples with their accelerometer data.

    Returns:
        List of dictionaries, each containing:
        - 'sample_id': Sample identifier
        - 'data': List of [ax, ay, az] accelerometer readings
    """
    samples = []
    current_sample = None

    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()

            if line.startswith('sample'):
                # New sample detected
                if current_sample:
                    samples.append(current_sample)
                current_sample = {
                    'sample_id': line,
                    'data': []
                }
            elif line == 'ax,ay,az':
                # Skip header line
                continue
            elif line and ',' in line:
                # Parse accelerometer data
                try:
                    ax, ay, az = map(float, line.split(','))
                    current_sample['data'].append([ax, ay, az])
                except ValueError:
                    continue

        # Add the last sample
        if current_sample:
            samples.append(current_sample)

    return samples


def iter_samples(root_dir=ROOT):
    root = pathlib.Path(root_dir)
    for cls in CATEGORIES:
        for txt in sorted((root / cls).glob("person*.txt")):
            for rec in parse_gesture_file(txt):  # rec is {'sample_id': ..., 'data': [[...], ...]}
                x = np.asarray(rec["data"], dtype=np.float32)  # (T, 3)
                if x.size == 0:
                    continue  # skip empty samples just in case
                y = np.int32(CAT_TO_ID[cls])
                yield x, y

def normalize_clip(x):
    """Clip and z-score normalize one sample."""
    x = tf.clip_by_value(x, -80.0, 80.0)
    mean = tf.reduce_mean(x, axis=0, keepdims=True)
    std = tf.math.reduce_std(x, axis=0, keepdims=True) + 1e-6
    return (x - mean) / std

def build_dataset(batch_size=64):
    """Return tf.data.Dataset of (seq[T,3], label) pairs, normalized and padded."""
    output_signature = (
        tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    ds = tf.data.Dataset.from_generator(
        lambda: iter_samples(),
        output_signature=output_signature
    )

    ds = ds.map(lambda x, y: (normalize_clip(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.padded_batch(
        batch_size,
        padded_shapes=(tf.TensorShape([None, 3]), tf.TensorShape([])),
        padding_values=(tf.constant(0.0, dtype=tf.float32),
                        tf.constant(0, dtype=tf.int32))
    )

    return ds.prefetch(tf.data.AUTOTUNE)

