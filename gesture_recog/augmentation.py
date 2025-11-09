import tensorflow as tf
import math

def random_rotation_matrix(max_deg=10.0):
    """Small random 3D rotation matrix via axis-angle."""
    max_rad = max_deg * math.pi / 180.0
    theta = tf.random.uniform([], -max_rad, max_rad)  # angle
    # random unit axis
    v = tf.random.normal([3])
    v = v / (tf.norm(v) + 1e-9)
    vx, vy, vz = v[0], v[1], v[2]

    c = tf.cos(theta)
    s = tf.sin(theta)
    C = 1.0 - c

    R = tf.stack([
        tf.stack([c + vx*vx*C,      vx*vy*C - vz*s, vx*vz*C + vy*s]),
        tf.stack([vy*vx*C + vz*s,   c + vy*vy*C,    vy*vz*C - vx*s]),
        tf.stack([vz*vx*C - vy*s,   vz*vy*C + vx*s, c + vz*vz*C   ]),
    ], axis=0)  # [3,3]
    return R

def augment_sample(x,
                   rot_deg=10.0,
                   scale_low=0.95, scale_high=1.05,
                   jitter_std=0.02,         # in raw units (g); tweak to your range
                   shift_max=5,             # samples
                   time_mask_prob=0.2,
                   time_mask_max_ratio=0.1):
    """
    x: (T,3) float32, raw (before LPF/normalize). Returns (T,3).
    Order: rotate -> scale -> jitter -> time-shift -> optional time-mask.
    """
    T = tf.shape(x)[0]

    # 1) small 3D rotation
    R = random_rotation_matrix(rot_deg)            # [3,3]
    x = tf.matmul(x, R)                            # (T,3) * (3,3)

    # 2) per-axis scaling
    s = tf.random.uniform([3], scale_low, scale_high)
    x = x * s

    # 3) jitter (Gaussian)
    if jitter_std and jitter_std > 0:
        x = x + tf.random.normal(tf.shape(x), stddev=jitter_std)

    # 4) time shift (pad + slice, keeps length)
    if shift_max and shift_max > 0:
        shift = tf.random.uniform([], -shift_max, shift_max + 1, dtype=tf.int32)
        x = tf.roll(x, shift=shift, axis=0)

    # 5) time mask (SpecAugment-style on time axis)
    def _apply_mask():
        max_len = tf.cast(tf.maximum(1, tf.cast(tf.round(tf.cast(T, tf.float32) * time_mask_max_ratio), tf.int32)), tf.int32)
        mlen = tf.random.uniform([], 1, tf.maximum(2, max_len), dtype=tf.int32)
        start = tf.random.uniform([], 0, tf.maximum(1, T - mlen + 1), dtype=tf.int32)
        pre  = x[:start]
        mid  = tf.zeros([mlen, 3], dtype=x.dtype)
        post = x[start+mlen:]
        return tf.concat([pre, mid, post], axis=0)
    do_mask = tf.less(tf.random.uniform([]), time_mask_prob)
    x = tf.cond(do_mask, _apply_mask, lambda: x)

    return x