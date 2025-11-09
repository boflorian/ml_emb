import pathlib
from ast import parse
import re 

import numpy as np
import tensorflow as tf

ROOT = pathlib.Path('dataset_magic_wand')
CATEGORIES = ["negative", "ring", "slope", "wing"]
CAT_TO_ID = {c:i for i,c in enumerate(CATEGORIES)}
PERSON_RE = re.compile(r"person(\d+)\.txt$", re.IGNORECASE)


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


def _person_id(path: pathlib.Path) -> int:
    m = PERSON_RE.search(path.name)
    return int(m.group(1)) if m else -1

def iter_samples(subjects, root_dir=ROOT):
    """Yield (seq[T,3] float32, label int32) only for the given subject ids."""
    subj_set = set(subjects) if subjects is not None else None
    root = pathlib.Path(root_dir)
    for cls in CATEGORIES:
        for txt in sorted((root / cls).glob("person*.txt")):
            pid = _person_id(txt)
            if subj_set is not None and pid not in subj_set:
                continue
            for rec in parse_gesture_file(txt):
                x = np.asarray(rec["data"], dtype=np.float32)
                if x.size == 0:
                    continue
                y = np.int32(CAT_TO_ID[cls])
                yield x, y

def normalize_clip(x):
    x = tf.clip_by_value(x, -80.0, 80.0)
    mean = tf.reduce_mean(x, axis=0, keepdims=True)
    std  = tf.math.reduce_std(x, axis=0, keepdims=True) + 1e-6
    return (x - mean) / std

def _make_ds(subjects, batch_size):
    """Internal: build one dataset for a subject list."""
    output_signature = (
        tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    ds = tf.data.Dataset.from_generator(
        lambda: iter_samples(subjects),
        output_signature=output_signature
    )
    ds = ds.map(lambda x, y: (normalize_clip(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.padded_batch(
        batch_size,
        padded_shapes=(tf.TensorShape([None, 3]), tf.TensorShape([])),
        padding_values=(tf.constant(0.0, tf.float32), tf.constant(0, tf.int32))
    )
    return ds.prefetch(tf.data.AUTOTUNE)

def build_train_test_datasets(
    train_subjects=(0,1,2,3,4,5,6),  # 6–7 participants
    test_subjects=(7,),              # 1–2 participants
    batch_size=64
):
    """Return (train_ds, test_ds) split by participant IDs."""
    train_ds = _make_ds(train_subjects, batch_size)
    test_ds  = _make_ds(test_subjects,  batch_size)
    return train_ds, test_ds
