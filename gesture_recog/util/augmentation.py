import tensorflow as tf
import math

# --- random 3D rotation matrix ---
def random_rotation_matrix(max_deg=15.0):
    max_rad = max_deg * math.pi / 180.0
    theta = tf.random.uniform([], -max_rad, max_rad)
    v = tf.random.normal([3]); v = v / (tf.norm(v) + 1e-9)
    vx, vy, vz = v[0], v[1], v[2]
    c, s, C = tf.cos(theta), tf.sin(theta), 1.0 - tf.cos(theta)
    R = tf.stack([
        tf.stack([c + vx*vx*C,      vx*vy*C - vz*s, vx*vz*C + vy*s]),
        tf.stack([vy*vx*C + vz*s,   c + vy*vy*C,    vy*vz*C - vx*s]),
        tf.stack([vz*vx*C - vy*s,   vz*vy*C + vx*s, c + vz*vz*C   ]),
    ], axis=0)  # [3,3]
    return R

# --- random smooth curve (for warps) via piecewise-linear control points ---
def _random_smooth_curve(T, num_knots=4, low=0.9, high=1.1):
    # Control points along time: [0 .. T-1]
    knots_x = tf.linspace(0.0, tf.cast(T-1, tf.float32), num_knots)
    knots_y = tf.random.uniform([num_knots], low, high)  # multiplicative factors
    t = tf.linspace(0.0, tf.cast(T-1, tf.float32), T)    # positions to sample
    # Linear interpolation
    idx = tf.searchsorted(knots_x, t, side="right") - 1
    idx = tf.clip_by_value(idx, 0, num_knots - 2)
    x0 = tf.gather(knots_x, idx)
    x1 = tf.gather(knots_x, idx + 1)
    y0 = tf.gather(knots_y, idx)
    y1 = tf.gather(knots_y, idx + 1)
    w = tf.where(x1 > x0, (t - x0) / (x1 - x0), tf.zeros_like(t))
    return y0 * (1.0 - w) + y1 * w  # [T]

# --- magnitude warp: multiply by smooth envelope ---
def magnitude_warp(x, strength=0.10, num_knots=4):
    # envelope around 1.0 in [1-strength, 1+strength]
    T = tf.shape(x)[0]
    env = _random_smooth_curve(T, num_knots=num_knots,
                               low=1.0 - strength, high=1.0 + strength)  # [T]
    return x * env[:, tf.newaxis]

# --- time warp: resample along a smooth monotonic mapping (piecewise-linear) ---
def time_warp_linear(x, max_warp=0.2, num_knots=4):
    """
    x: (T,3); max_warp is fraction of length (e.g., 0.2 -> ±20% local time stretch).
    Returns (T,3) with linear interpolation.
    """
    T = tf.shape(x)[0]
    T_f = tf.cast(T, tf.float32)
    # Build a cumulative time mapping by integrating a smooth positive curve
    speed = _random_smooth_curve(T, num_knots=num_knots,
                                 low=1.0 - max_warp, high=1.0 + max_warp)  # [T]
    cumsum = tf.cumsum(speed)
    # Normalize mapping to [0, T-1]
    tau = (cumsum - cumsum[0]) / (cumsum[-1] - cumsum[0] + 1e-9) * (T_f - 1.0)  # [T]

    # Linear resample x at positions tau
    tau0 = tf.cast(tf.floor(tau), tf.int32)
    tau1 = tf.clip_by_value(tau0 + 1, 0, T - 1)
    w1 = tau - tf.cast(tau0, tf.float32)
    w0 = 1.0 - w1
    x0 = tf.gather(x, tau0)
    x1 = tf.gather(x, tau1)
    return w0[:, None]*x0 + w1[:, None]*x1

# --- core augmentation (raw domain) ---
def augment_sample(
    x,
    rot_deg=15.0,
    scale_low=0.9, scale_high=1.1,
    jitter_std=0.03,            # adjust to your sensor noise
    shift_max=6,                # samples
    time_mask_prob=0.3, time_mask_max_ratio=0.1,
    mag_warp_strength=0.10,     # 0..0.3
    time_warp_ratio=0.15,       # 0..0.3
):
    T = tf.shape(x)[0]

    # 1) rotation
    R = random_rotation_matrix(rot_deg)
    x = tf.matmul(x, R)

    # 2) per-axis scaling
    s = tf.random.uniform([3], scale_low, scale_high)
    x = x * s

    # 3) magnitude warp (smooth envelope)
    if mag_warp_strength and mag_warp_strength > 0:
        x = magnitude_warp(x, strength=mag_warp_strength, num_knots=4)

    # 4) time warp (smooth resampling)
    if time_warp_ratio and time_warp_ratio > 0:
        x = time_warp_linear(x, max_warp=time_warp_ratio, num_knots=4)

    # 5) jitter
    if jitter_std and jitter_std > 0:
        x = x + tf.random.normal(tf.shape(x), stddev=jitter_std)

    # 6) time shift
    if shift_max and shift_max > 0:
        shift = tf.random.uniform([], -shift_max, shift_max + 1, dtype=tf.int32)
        x = tf.roll(x, shift=shift, axis=0)

    # 7) time mask
    if time_mask_prob and tf.less(tf.random.uniform([]), time_mask_prob):
        max_len = tf.maximum(1, tf.cast(tf.round(tf.cast(T, tf.float32) * time_mask_max_ratio), tf.int32))
        mlen = tf.random.uniform([], 1, tf.maximum(2, max_len + 1), dtype=tf.int32)
        start = tf.random.uniform([], 0, tf.maximum(1, T - mlen + 1), dtype=tf.int32)
        x = tf.concat([x[:start], tf.zeros([mlen, 3], dtype=x.dtype), x[start+mlen:]], axis=0)

    return x