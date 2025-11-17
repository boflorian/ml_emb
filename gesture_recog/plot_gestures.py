import argparse
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D projection)

from data_loader import parse_gesture_file, iter_samples  # your existing helpers


SAMPLE_RE = re.compile(r"^sample\d+\s*$", re.IGNORECASE)


# ------------------------------------------------------------
# Filters & helpers
# ------------------------------------------------------------
def iir_alpha_from_fc(fc_hz: float, fs_hz: float) -> float:
    """
    First-order low-pass IIR (one-pole) with given cutoff fc (Hz) at sampling fs (Hz).
    y[n] = alpha * y[n-1] + (1-alpha) * x[n], with alpha = exp(-2*pi*fc/fs).
    """
    if fc_hz <= 0:
        return 0.0
    return float(np.exp(-2.0 * np.pi * fc_hz / fs_hz))


def estimate_g0(acc: np.ndarray, fs: float) -> np.ndarray:
    """
    Estimate initial gravity vector from the first ~0.3 s (or fewer if short).
    Scales its magnitude to 9.81 m/s^2 to aid convergence.
    """
    n = max(1, int(min(0.3 * fs, len(acc))))
    g0 = np.mean(acc[:n], axis=0)
    mag = np.linalg.norm(g0) + 1e-9
    return g0 * (9.81 / mag)


def lowpass_gravity(acc: np.ndarray, fs: float, fc: float) -> np.ndarray:
    """
    One-pole low-pass estimate of gravity (sensor frame).
    """
    alpha = iir_alpha_from_fc(fc, fs)
    g = np.zeros_like(acc)
    g[0] = estimate_g0(acc, fs)
    for i in range(1, acc.shape[0]):
        g[i] = alpha * g[i - 1] + (1.0 - alpha) * acc[i]
    return g


def lowpass_gravity_zero_phase(acc: np.ndarray, fs: float, fc: float) -> np.ndarray:
    """
    Zero-phase version: run the same one-pole forward, then backward.
    Removes phase lag that otherwise biases a_lin.
    """
    g_fwd = lowpass_gravity(acc, fs, fc)
    g_bwd = lowpass_gravity(g_fwd[::-1], fs, fc)[::-1]
    return g_bwd


def remove_gravity(acc: np.ndarray, fs: float, fc: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    """
    Complementary split: linear_acc = acc - lowpass(acc) (zero-phase to avoid lag).
    """
    g_est = lowpass_gravity_zero_phase(acc, fs, fc)
    return acc - g_est, g_est


def trim_stillness(acc_lin: np.ndarray, thresh: float = 0.15):
    """
    Trim leading/trailing near-still segments by thresholding the norm.
    thresh in m/s^2. Returns sliced acc_lin and the indices [left, right].
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


def integrate_twice(acc_lin: np.ndarray, fs: float, method: str = "trapz",
                    zero_vel_ends: bool = True):
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
        A = np.vstack([t, np.ones_like(t)]).T
        for ax in range(3):
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


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", type=Path, help="Path to file")
    ap.add_argument("--fc", type=float, default=0.7, help="LP cutoff for gravity removal [Hz]")
    ap.add_argument("--thresh", type=float, default=0.15, help="Stillness threshold [m/s^2]")
    ap.add_argument("--method", type=str, default="trapz", choices=["euler", "trapz"],
                    help="Integration method")
    ap.add_argument("--no-zupt", action="store_true",
                    help="Disable end-to-end zero-velocity pinning")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise FileNotFoundError(path)

    samples = parse_gesture_file(path)

    # Build simple dict of blocks {name: {"acc":..., "fs":...}}
    blocks = {}
    for i, s in enumerate(samples):
        name = s.get("sample_id", f"traj_{i:02d}")
        acc = np.asarray(s["data"], dtype=float)
        fs = float(s.get("fs", 30.0))

        # Heuristic unit fix: if |acc| median < 3, assume "g" units -> scale to m/s^2
        if np.nanmedian(np.linalg.norm(acc, axis=1)) < 3.0:
            acc = acc * 9.81

        blocks[name] = {"acc": acc, "fs": fs}

    # Process and plot each trajectory
    for name, item in blocks.items():
        acc = item["acc"]
        fs = item["fs"]

        # Gravity removal (zero-phase, avoids lag)
        acc_lin, g_est = remove_gravity(acc, fs, fc=args.fc)

        # Work only on active window, de-bias per axis
        acc_act, left, right = trim_stillness(acc_lin, thresh=args.thresh)
        acc_act = acc_act - np.mean(acc_act, axis=0)

        # Integrate with mild physical constraints
        v_act, p_act = integrate_twice(
            acc_act, fs,
            method=args.method,
            zero_vel_ends=(not args.no_zupt)
        )

        plot_trajectory(p_act, title=f"Trajectory ({name})")

    plt.show()


if __name__ == "__main__":
    main()