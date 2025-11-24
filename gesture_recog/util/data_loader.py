import pathlib
import re

import numpy as np
from ml_emb.gesture_recog.util.augmentation import *
from util.feature_extraction import tf_extract_features

ROOT = pathlib.Path('../dataset_magic_wand')
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


def segment_windows(x, y, win=128, hop=64, drop_short=False):
    """
    Avoids having to pad samples, models prefer same size inputs 
    x: (T,3)  -> frames: (N, win, 3)
    y: scalar -> labels: (N,)
    """
    if win is None or hop is None:
        return tf.data.Dataset.from_tensors((x, y))  # no segmentation

    # Frame along time axis, pad at the end so we don't drop tail segments
    frames = tf.signal.frame(x, frame_length=win, frame_step=hop, axis=0, pad_end=True)  # (N, win, 3)

    if drop_short:
        # Remove frames that were padded entirely (rare). Keep if any real samples exist:
        # Create a mask where any non-zero row exists in the frame before normalization.
        mask = tf.reduce_any(tf.not_equal(frames, 0.0), axis=[1, 2])  # (N,)
        frames = tf.boolean_mask(frames, mask)

    n = tf.shape(frames)[0]
    labels = tf.repeat(y, n)
    return tf.data.Dataset.from_tensor_slices((frames, labels))


def lowpass_filter(x, window=5):
    """ Moving avwerage lowpass filter over time, independent axes"""
    w = tf.ones([window], tf.float32) / tf.cast(window, tf.float32)
    f = tf.reshape(w, [window, 1, 1])

    xN = tf.expand_dims(x, 0)
    outs = []

    for c in range(3):
        xi = xN[:, :, c:c+1]
        yi = tf.nn.conv1d(xi, filters=f, stride=1, padding='SAME')
        outs.append(yi) 
    y = tf.concat(outs, axis=-1)
    return y[0] 


def normalize_clip(x):
    x = tf.clip_by_value(x, -80.0, 80.0)
    mean = tf.reduce_mean(x, axis=0, keepdims=True)
    std  = tf.math.reduce_std(x, axis=0, keepdims=True) + 1e-6
    return (x - mean) / std


def _make_ds(subjects, batch_size, lp_window=5,
             win=None, hop=None, drop_short=False,
             augment=False,
             aug_cfg=None,
             extract_features=False):
    
    output_signature = (
        tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    ds = tf.data.Dataset.from_generator(lambda: iter_samples(subjects),
                                        output_signature=output_signature)

    if win is not None and hop is not None:
        ds = ds.flat_map(lambda x, y: segment_windows(x, y, win=win, hop=hop, drop_short=drop_short))

    aug_cfg = aug_cfg or {}  # default strengths

    def preprocess_train(x, y):
        x = augment_sample(x, **aug_cfg)                    
        x = lowpass_filter(x, window=lp_window)
        x = normalize_clip(x)
        return x, y

    def preprocess_eval(x, y):
        x = lowpass_filter(x, window=lp_window)
        x = normalize_clip(x)
        return x, y

    ds = ds.map(preprocess_train if augment else preprocess_eval,
                num_parallel_calls=tf.data.AUTOTUNE)

    if extract_features:
        ds = ds.map(lambda x, y: (tf_extract_features(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.shuffle(5000, reshuffle_each_iteration=True)

    if win is not None and hop is not None:
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        ds = ds.padded_batch(
            batch_size,
            padded_shapes=(tf.TensorShape([None, 3]), tf.TensorShape([])),
            padding_values=(tf.constant(0.0, tf.float32), tf.constant(0, tf.int32))
        ).prefetch(tf.data.AUTOTUNE)
    return ds


def build_train_test_datasets(
    train_subjects=(0,1,2,3,4,5,6), # fallback, should be overwritten 
    test_subjects=(7,), # fallback, should be overwritten 
    batch_size=64,
    lp_window=5,
    win=None, hop=None, drop_short=False, aug_cfg=None, 
    augment=True,
    extract_features=False
):
    train_ds = _make_ds(train_subjects, batch_size, lp_window,
                        win, hop, drop_short, augment=augment, aug_cfg=aug_cfg, extract_features=extract_features)
    test_ds  = _make_ds(test_subjects,  batch_size, lp_window,
                        win, hop, drop_short, augment=False, aug_cfg=None, extract_features=extract_features)
    return train_ds, test_ds
