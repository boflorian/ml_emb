import numpy as np
import tensorflow as tf

def extract_statistical_features(window):
    """
    Extract statistical features from a time window (T, 3).
    Returns a 1D array of features.
    """
    features = []
    for axis in range(3):  # X, Y, Z
        signal = window[:, axis]
        features.extend([
            np.mean(signal),      # Mean
            np.std(signal),       # Standard deviation
            np.min(signal),       # Minimum
            np.max(signal),       # Maximum
            np.max(signal) - np.min(signal),  # Range
            np.median(signal),    # Median
            np.percentile(signal, 25),  # 25th percentile
            np.percentile(signal, 75),  # 75th percentile
            np.var(signal),       # Variance
            np.sqrt(np.mean(signal**2)),  # RMS
        ])
    # Cross-axis features (optional)
    # e.g., correlation between axes, but keep simple for now
    return np.array(features, dtype=np.float32)

def tf_extract_features(window):
    """
    TensorFlow-compatible version for mapping.
    window: (T, 3) tensor
    """
    features = []
    for axis in range(3):
        signal = window[:, axis]
        features.extend([
            tf.reduce_mean(signal),
            tf.math.reduce_std(signal),
            tf.reduce_min(signal),
            tf.reduce_max(signal),
            tf.reduce_max(signal) - tf.reduce_min(signal),
            tf.reduce_mean(tf.sort(signal)[tf.shape(signal)[0] // 2]),  # Approx median
            tf.reduce_mean(tf.sort(signal)[:tf.shape(signal)[0] // 4]),  # Approx 25th
            tf.reduce_mean(tf.sort(signal)[3 * tf.shape(signal)[0] // 4:]),  # Approx 75th
            tf.math.reduce_variance(signal),
            tf.sqrt(tf.reduce_mean(signal**2)),
        ])
    return tf.stack(features)