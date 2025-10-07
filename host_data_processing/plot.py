import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional but recommended
from scipy.signal import butter, filtfilt

# --- Files ---
raw_file = "imu_log.csv"      # expects columns: timestamp_us, ax, ay, az, gx, gy, gz
fft_file = "imu_spectra.csv"  # optional (kept for your existing workflow)

# --- Constants ---
G = 9.80665  # m/s^2

# ===== Helpers: quaternions & rotation =====
def quat_mul(q, r):
    # Hamilton product q⊗r, both [w, x, y, z]
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = r
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_exp(omega_dt):
    # omega_dt is a 3-vector = angular rate * dt (rad)
    theta = np.linalg.norm(omega_dt)
    if theta < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = omega_dt / theta
    half = 0.5 * theta
    return np.hstack([np.cos(half), axis*np.sin(half)])

def quat_to_R(q):
    # Convert unit quaternion [w,x,y,z] to 3x3 rotation (world_from_body)
    w,x,y,z = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w),   1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w), 1-2*(x*x+y*y)]
    ])
    return R

def butter_highpass(cut, fs, order=2):
    b,a = butter(order, cut/(0.5*fs), btype='highpass')
    return b,a

# ===== Load data =====
raw = pd.read_csv(raw_file)

# Require these columns
needed = {"timestamp_us", "ax_raw","ay_raw","az_raw","gx_raw","gy_raw","gz_raw"}
missing = needed - set(raw.columns)
if missing:
    raise ValueError(f"Missing columns in {raw_file}: {missing}")

# Time base in seconds starting at 0
t = (raw["timestamp_us"].values - raw["timestamp_us"].iloc[0]) / 1e6
dt = np.diff(t, prepend=t[0])
# Replace first dt=0 with median of the rest to avoid NaNs
if len(dt) > 1:
    dt[0] = np.median(dt[1:])

fs = 1.0 / np.median(dt)  # sampling rate (Hz), used for filters

# Body-frame IMU
acc_b = raw[["ax_raw","ay_raw","az_raw"]].to_numpy().astype(float)
gyro = raw[["gx_raw","gy_raw","gz_raw"]].to_numpy().astype(float)  # rad/s

# ===== Quick bias estimation from first second (assume stationary) =====
t_stationary = 1.0  # seconds
mask0 = t <= t_stationary
if not np.any(mask0):
    mask0 = t <= (t[0] + (t[-1]-t[0])*0.1)  # fallback: first 10%

acc_bias_b = acc_b[mask0].mean(axis=0)     # accelerometer bias (incl. gravity component in body frame initially)
gyro_bias  = gyro[mask0].mean(axis=0)      # gyro bias

# Remove biases
acc_b_unbiased = acc_b - acc_bias_b
gyro_unbiased  = gyro - gyro_bias

# ===== Integrate gyro to orientation (world_from_body) =====
# Initialize assuming world z ≈ body z at start (i.e., q = identity).
# For better init, you could align measured gravity at t=0 to world z.
q = np.zeros((len(t), 4))
q[0] = np.array([1.0, 0.0, 0.0, 0.0])  # w,x,y,z

for k in range(1, len(t)):
    omega = gyro_unbiased[k]           # rad/s in body frame
    dq = quat_exp(omega * dt[k])       # small rotation over dt
    q[k] = quat_mul(q[k-1], dq)
    q[k] = q[k] / np.linalg.norm(q[k]) # normalize

# ===== Rotate body acceleration to world & subtract gravity =====
a_world = np.zeros_like(acc_b_unbiased)
for k in range(len(t)):
    R_wb = quat_to_R(q[k])             # world_from_body
    a_world[k] = R_wb @ acc_b_unbiased[k]

# Now remove gravity (pointing +z in world)
a_nav = a_world.copy()
a_nav[:, 2] = a_nav[:, 2] - G

# ===== Integrate to velocity and position =====
v = np.zeros_like(a_nav)
p = np.zeros_like(a_nav)

# Gentle high-pass on acceleration can help, but we’ll high-pass velocity instead
# Integrate with trapezoid
for k in range(1, len(t)):
    v[k] = v[k-1] + 0.5*(a_nav[k] + a_nav[k-1]) * dt[k]
# High-pass filter velocity to fight drift (cut at ~0.05–0.2 Hz; tune!)
cut_hz = max(0.05, 2.0/ (t[-1]-t[0] + 1e-9))   # at least one full cycle over the recording
b,a = butter_highpass(cut_hz, fs, order=2)
v_hp = np.vstack([filtfilt(b,a, v[:,i]) for i in range(3)]).T

for k in range(1, len(t)):
    p[k] = p[k-1] + 0.5*(v_hp[k] + v_hp[k-1]) * dt[k]

# --- Optional: simple ZUPT when "probably stationary"
# A crude detector: small gyro and accel close to -g
gyro_norm = np.linalg.norm(gyro_unbiased, axis=1)
acc_err   = np.linalg.norm(a_world - np.array([0,0,G]), axis=1)  # magnitude error from gravity
maybe_still = (gyro_norm < 0.05) & (acc_err < 0.15*G)            # tune thresholds

# Zero velocity during stationary to kill drift
v_zupt = v_hp.copy()
v_zupt[maybe_still] = 0.0

# Reintegrate position with ZUPT velocity
p_zupt = np.zeros_like(p)
for k in range(1, len(t)):
    p_zupt[k] = p_zupt[k-1] + 0.5*(v_zupt[k] + v_zupt[k-1]) * dt[k]

# ===== Plots =====

# 1) Time-domain raw signals (fixed: no FFT overlaid)
plt.figure(figsize=(10,5))
for col in ["ax_raw","ay_raw","az_raw","gx_raw","gy_raw","gz_raw"]:
    if col in raw.columns:
        plt.plot(t, raw[col], label=col)
plt.xlabel("Time [s]")
plt.ylabel("Sensor Value (SI)")
plt.title("IMU Raw Signals")
plt.legend()
plt.tight_layout()
plt.show()

# 2) Velocity over time (after gravity comp + HP + ZUPT)
plt.figure(figsize=(10,4))
plt.plot(t, v_zupt[:,0], label="vx")
plt.plot(t, v_zupt[:,1], label="vy")
plt.plot(t, v_zupt[:,2], label="vz")
plt.xlabel("Time [s]")
plt.ylabel("Velocity [m/s]")
plt.title("Velocity (world frame; HP + ZUPT)")
plt.legend()
plt.tight_layout()
plt.show()

# 3) Position over time
plt.figure(figsize=(10,4))
plt.plot(t, p_zupt[:,0], label="x")
plt.plot(t, p_zupt[:,1], label="y")
plt.plot(t, p_zupt[:,2], label="z")
plt.xlabel("Time [s]")
plt.ylabel("Position [m]")
plt.title("Position (world frame; integrated)")
plt.legend()
plt.tight_layout()
plt.show()

# 4) 3D trajectory (if you want a quick look)
try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(6,6))
    ax3d = fig.add_subplot(111, projection='3d')
    ax3d.plot(p_zupt[:,0], p_zupt[:,1], p_zupt[:,2])
    ax3d.set_xlabel("x [m]"); ax3d.set_ylabel("y [m]"); ax3d.set_zlabel("z [m]")
    ax3d.set_title("IMU Trajectory (world frame)")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("3D plot skipped:", e)

# (Optional) keep your FFT plotting separate so axes make sense
# fft = pd.read_csv(fft_file)
# plt.figure(figsize=(10,4))
# for axis in fft['axis'].unique():
#     subset = fft[fft['axis'] == axis].sort_values('freq_hz')
#     plt.plot(subset['freq_hz'], subset['amp'], label=axis)
# plt.xlabel("Frequency [Hz]"); plt.ylabel("Amplitude"); plt.title("IMU FFT")
# plt.legend(); plt.tight_layout(); plt.show()