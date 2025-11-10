import tensorflow as tf
import argparse
from pathlib import Path
import re
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D projection)

from data_loader import parse_gesture_file, iter_samples


SAMPLE_RE = re.compile(r"^sample\d+\s*$", re.IGNORECASE)


def iir_alpha_from_fc(fc_hz: float, fs_hz: float) -> float:
    """
    First-order low-pass IIR (one-pole) with given cutoff fc (Hz) at sampling fs (Hz).
    y[n] = alpha * y[n-1] + (1-alpha) * x[n], with alpha = exp(-2*pi*fc/fs).
    """
    if fc_hz <= 0:
        # degenerate -> pure integrator; clamp
        return 0.0
    return float(np.exp(-2.0 * np.pi * fc_hz / fs_hz))


def lowpass_gravity(acc: np.ndarray, fs: float, fc: float) -> np.ndarray:
    """
    Estimate gravity via one-pole low-pass. acc: [T,3] (m/s^2).
    Returns g_est [T,3].
    """
    alpha = iir_alpha_from_fc(fc, fs)
    g = np.zeros_like(acc)
    # initialize with first sample scaled to ~|g|=9.81 (helps convergence if start is not static)
    g0 = acc[0].copy()
    mag = np.linalg.norm(g0) + 1e-9
    g[0] = g0 * (9.81 / mag)

    for i in range(1, acc.shape[0]):
        g[i] = alpha * g[i - 1] + (1.0 - alpha) * acc[i]
    return g


def remove_gravity(acc: np.ndarray, fs: float, fc: float = 0.7) -> np.ndarray:
    """
    Complementary split: linear_acc = acc - lowpass(acc).
    """
    g_est = lowpass_gravity(acc, fs, fc)
    return acc - g_est, g_est


def trim_stillness(acc_lin: np.ndarray, thresh: float = 0.15):
    """
    Trim leading/trailing near-still segments by thresholding the norm.
    thresh in m/s^2. Returns sliced acc_lin and the applied indices.
    """
    norms = np.linalg.norm(acc_lin, axis=1)
    T = len(norms)
    left = 0
    while left < T and norms[left] < thresh:
        left += 1
    right = T - 1
    while right > left and norms[right] < thresh:
        right -= 1
    return acc_lin[left : right + 1], left, right


def integrate_twice(acc_lin: np.ndarray, fs: float, method: str='euler',
                    zero_vel_ends: bool = False):
    """
    Integrate linear acceleration to velocity and position.
    method: 'euler' or 'trapz'
    zero_vel_ends: if True, remove a linear trend from v so v[0]~v[-1]~0.
    Returns: v [T,3], p [T,3]
    """
    dt = 1.0 / fs
    T = acc_lin.shape[0]
    v = np.zeros_like(acc_lin)
    p = np.zeros_like(acc_lin)

    if method == "euler":
        for i in range(1, T):
            v[i] = v[i - 1] + acc_lin[i - 1] * dt
            p[i] = p[i - 1] + v[i - 1] * dt + 0.5 * acc_lin[i - 1] * dt * dt
    else:  # trapezoidal
        for i in range(1, T):
            a_avg = 0.5 * (acc_lin[i] + acc_lin[i - 1])
            v[i] = v[i - 1] + a_avg * dt
            p[i] = p[i - 1] + 0.5 * (v[i] + v[i - 1]) * dt

    if zero_vel_ends and T > 1:
        # subtract a per-axis linear trend from v so v[0] and v[-1] are ~0
        t = np.arange(T, dtype=float)
        for ax in range(3):
            # fit line to v[:,ax]
            A = np.vstack([t, np.ones_like(t)]).T
            m, b = np.linalg.lstsq(A, v[:, ax], rcond=None)[0]
            v[:, ax] = v[:, ax] - (m * t + b)
        # re-integrate position from detrended velocity to keep consistency
        p = np.zeros_like(p)
        for i in range(1, T):
            p[i] = p[i - 1] + 0.5 * (v[i] + v[i - 1]) * dt

    return v, p


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------
def plot_trajectory(p: np.ndarray, title: str):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(p[:, 0], p[:, 1], p[:, 2])
    ax.scatter(p[0, 0], p[0, 1], p[0, 2], s=40, marker="o", label="start")
    ax.scatter(p[-1, 0], p[-1, 1], p[-1, 2], s=40, marker="^", label="end")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(title)
    ax.legend(loc="best")
    plt.tight_layout()


def plot_signals(t: np.ndarray, acc: np.ndarray, g_est: np.ndarray,
                 acc_lin: np.ndarray, v: np.ndarray, p: np.ndarray, name: str):
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labs = ["x", "y", "z"]
    for d in range(3):
        axs[0].plot(t, acc[:, d], label=f"a{labs[d]} (raw)")
        axs[0].plot(t, g_est[:, d], linestyle="--", label=f"g{labs[d]} (LPF)")
        axs[0].plot(t, acc_lin[:, d], label=f"a{labs[d]}-lin")
        axs[1].plot(t, v[:, d], label=f"v{labs[d]}")
        axs[2].plot(t, p[:, d], label=f"p{labs[d]}")
    axs[0].set_ylabel("acc (m/s²)")
    axs[1].set_ylabel("vel (m/s)")
    axs[2].set_ylabel("pos (m)")
    axs[2].set_xlabel("time (s)")
    axs[0].legend(ncol=3, fontsize=8)
    axs[1].legend(ncol=3, fontsize=8)
    axs[2].legend(ncol=3, fontsize=8)
    fig.suptitle(f"Signals: {name}")
    plt.tight_layout(rect=[0, 0, 1, 0.97])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", type=Path, help='Path to file')
    ap.add_argument("--fc", type=float, default=0.5, help="LP cutoff for gravity removal [Hz]")

    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise FileNotFoundError(path)


    output_signature = (
        tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    samples = parse_gesture_file(path)

    blocks = {}
    
    for i, s in enumerate(samples):
        name = s.get("sample_id", f"traj_{i:02d}")
        acc  = np.asarray(s["data"], dtype=float)
        fs   = float(s.get("fs", 30.0))
        blocks[name] = {"acc": acc, "fs": fs}
    
    # Now process and plot each trajectory per label
    for name, item in blocks.items():
        acc = item["acc"]
        fs  = item["fs"]
    
        t = np.arange(len(acc)) / fs
    
        # Remove gravity
        acc_lin, g_est = remove_gravity(acc, fs, fc=args.fc)

        # Integrate
        v, p = integrate_twice(acc_lin, fs)
    
        # Plots
        plot_trajectory(p, title=f"Trajectory ({name})")

    plt.show()


if __name__ == "__main__":
    main()