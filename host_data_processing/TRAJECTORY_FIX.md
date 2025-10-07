# Trajectory Plotting Fix

## Problem Identified

Your `plot.py` script was using **raw sensor values** (`ax_raw`, `gx_raw`, etc.) which are in **LSB (Least Significant Bit) units** from the IMU hardware. These are NOT in physical units!

### What Was Wrong

```python
# BEFORE (WRONG):
acc_b = raw[["ax_raw","ay_raw","az_raw"]].to_numpy()  # ❌ LSB values!
gyro = raw[["gx_raw","gy_raw","gz_raw"]].to_numpy()   # ❌ LSB values!

# Then the script treated them as if they were:
# - acc_b in m/s² (but they were actually in LSB: 16384 LSB/g)
# - gyro in rad/s (but they were actually in LSB: 131 LSB/dps)
```

### Why This Caused Wrong Trajectories

When you integrate acceleration to get velocity and position:

```
position = ∫∫ acceleration dt²
```

If your acceleration is **off by a factor of 16,384**, your trajectory will be:
- **16,384× too large** in acceleration
- **Wildly incorrect** position after double integration

Similarly for gyroscope rotation calculations.

---

## Solution Implemented

### Use `*_f32` Columns (Physical Units)

Your CSV contains properly converted values in the `*_f32` columns:

```python
# AFTER (CORRECT):
# ax_f32, ay_f32, az_f32 are in g (gravity units)
acc_b = raw[["ax_f32","ay_f32","az_f32"]].to_numpy() * G  # Convert g → m/s²

# gx_f32, gy_f32, gz_f32 are in dps (degrees per second)
gyro = raw[["gx_f32","gy_f32","gz_f32"]].to_numpy() * (np.pi/180.0)  # Convert dps → rad/s
```

### Unit Conversions Applied

| Sensor | CSV Column | CSV Units | Script Needs | Conversion |
|--------|-----------|-----------|--------------|------------|
| Accelerometer | `ax_f32` | g | m/s² | × 9.80665 |
| Gyroscope | `gx_f32` | dps | rad/s | × π/180 |

---

## What Changed in the Code

### 1. **Data Loading** (Lines 49-68)

**Before:**
```python
needed = {"timestamp_us", "ax_raw","ay_raw","az_raw","gx_raw","gy_raw","gz_raw"}
acc_b = raw[["ax_raw","ay_raw","az_raw"]].to_numpy().astype(float)
gyro = raw[["gx_raw","gy_raw","gz_raw"]].to_numpy().astype(float)
```

**After:**
```python
needed = {"timestamp_us", "ax_f32","ay_f32","az_f32","gx_f32","gy_f32","gz_f32"}
acc_b = raw[["ax_f32","ay_f32","az_f32"]].to_numpy() * G  # g → m/s²
gyro = raw[["gx_f32","gy_f32","gz_f32"]].to_numpy() * (np.pi/180.0)  # dps → rad/s
```

### 2. **Plotting Section** (Lines 143-161)

**Before:**
```python
for col in ["ax_raw","ay_raw","az_raw","gx_raw","gy_raw","gz_raw"]:
    plt.plot(t, raw[col], label=col)
```

**After:**
```python
# Separate plots for accelerometer and gyroscope with proper units
plt.subplot(2, 1, 1)
for col in ["ax_f32","ay_f32","az_f32"]:
    plt.plot(t, raw[col], label=col.replace('_f32', ''))
plt.ylabel("Acceleration [g]")

plt.subplot(2, 1, 2)
for col in ["gx_f32","gy_f32","gz_f32"]:
    plt.plot(t, raw[col], label=col.replace('_f32', ''))
plt.ylabel("Angular Rate [dps]")
```

---

## Should You Use Quantized Values (`*_q15`)?

### **NO - Use `*_f32` for Trajectory Estimation**

Here's why:

| Data Type | When to Use | Why |
|-----------|-------------|-----|
| `*_raw` | ❌ Never for analysis | Raw LSB values, not physical units |
| `*_f32` | ✅ **Trajectory, plotting, analysis** | **Proper physical units (g, dps, µT)** |
| `*_q15` | ✅ ML model input only | Normalized for neural network training |

### Why Not Quantized?

**Quantized values are normalized to [-1, 1):**
- `ax_q15 = 32` represents `32/32768 = 0.000976` (normalized)
- To use for trajectory, you'd need to:
  1. Dequantize: `32/32768 = 0.000976`
  2. Denormalize: `0.000976 × 2.0 = 0.001953` g
  3. Convert to m/s²: `0.001953 × 9.80665 = 0.01914` m/s²

**That's exactly what `ax_f32` already is!**

So `*_f32` values are:
- ✅ Already in physical units
- ✅ Full precision (float32)
- ✅ No quantization error
- ✅ Ready to use

---

## Expected Improvements

After this fix, your trajectory should:

1. **Be at realistic scale**
   - Position in meters (not thousands of meters)
   - Velocity in m/s (not absurd values)

2. **Show correct motion**
   - Rotations will be accurate (gyro in correct rad/s)
   - Accelerations will be physically meaningful

3. **Still have IMU drift** (this is normal!)
   - Double integration of accelerometer always drifts
   - High-pass filter helps but doesn't eliminate it
   - ZUPT (Zero Velocity Update) helps during stationary periods
   - For accurate trajectory, you need additional sensors (GPS, vision, etc.)

---

## Additional Tips for Better Trajectory

### 1. **Calibrate Your IMU**
```python
# The script already does basic bias estimation:
acc_bias_b = acc_b[mask0].mean(axis=0)  # First 1 second
gyro_bias = gyro[mask0].mean(axis=0)
```

Make sure the sensor is **completely stationary** for the first 1-2 seconds!

### 2. **Tune the High-Pass Filter**
```python
cut_hz = max(0.05, 2.0/ (t[-1]-t[0] + 1e-9))  # Current: ~0.05 Hz
```

- Lower cutoff (0.01-0.05 Hz): Preserves more motion but more drift
- Higher cutoff (0.1-0.2 Hz): Removes more drift but loses slow motion

### 3. **Tune ZUPT Thresholds**
```python
maybe_still = (gyro_norm < 0.05) & (acc_err < 0.15*G)
```

- Too tight: Misses stationary periods, more drift
- Too loose: Zeros velocity during actual motion

### 4. **Consider Magnetometer**
Your CSV has magnetometer data (`mx_f32`, `my_f32`, `mz_f32`) which can help with:
- Heading (yaw) correction
- Reducing orientation drift
- Better world-frame alignment

---

## Summary

✅ **Changed from `*_raw` to `*_f32` columns**  
✅ **Added proper unit conversions (g→m/s², dps→rad/s)**  
✅ **Updated plotting to show physical units**  
✅ **Your trajectory should now be at the correct scale!**

The quantized values (`*_q15`) are specifically for ML model training and should **NOT** be used for trajectory estimation or data analysis. Stick with `*_f32` for all physics-based calculations!
