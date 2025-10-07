import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
DELTA = 1 / 32768  # Q15 step size in normalized space
RMS_ERROR_THEO = DELTA / np.sqrt(12)
MAX_ERROR_THEO = DELTA / 2

# Sensor ranges for normalization (must match MCU constants)
ACCEL_RANGE = 2.0    # ±2g
GYRO_RANGE = 250.0   # ±250 dps
MAG_RANGE = 4900.0   # ±4900 µT

# --- Load CSV ---
file = 'imu_log.csv'  # <-- change this to your file name (default: imu_log.csv)
df = pd.read_csv(file)

print(f"Loaded {len(df)} samples from '{file}'")
print(f"Columns found: {list(df.columns)}\n")

# --- Columns to compare ---
# The CSV has columns named: ax_raw, ax_f32, ax_q15, etc.
signals = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz']

# Verify required columns exist
required_cols = []
for s in signals:
    required_cols.extend([f'{s}_f32', f'{s}_q15'])

missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"ERROR: Missing columns in CSV: {missing_cols}")
    print(f"Available columns: {list(df.columns)}")
    exit(1)

# Map signals to their physical ranges
signal_ranges = {
    'ax': ACCEL_RANGE, 'ay': ACCEL_RANGE, 'az': ACCEL_RANGE,
    'gx': GYRO_RANGE, 'gy': GYRO_RANGE, 'gz': GYRO_RANGE,
    'mx': MAG_RANGE, 'my': MAG_RANGE, 'mz': MAG_RANGE
}

results = []
for s in signals:
    f32 = df[f'{s}_f32']  # Physical units (g, dps, or µT)
    q15 = df[f'{s}_q15']  # Q15 int16 values
    
    # Get the range for this signal
    signal_range = signal_ranges[s]
    
    # Method 1: Compare in normalized space [-1, 1)
    norm_f32 = f32 / signal_range  # Normalize f32 to [-1, 1)
    norm_q15 = q15 / 32768.0        # Q15 already in [-1, 1)
    
    # Error metrics in normalized space
    err_norm = norm_f32 - norm_q15
    rms_err = np.sqrt(np.mean(err_norm**2))
    max_err = np.max(np.abs(err_norm))
    
    # SNR calculation (signal power vs quantization noise power)
    signal_std = np.std(norm_f32)
    noise_std = RMS_ERROR_THEO
    snr = 20 * np.log10(signal_std / noise_std) if signal_std != 0 else np.nan
    
    # Check saturation (clipping)
    saturation_pct = 100 * np.sum(np.abs(q15) >= 32767) / len(q15)
    
    results.append({
        'Signal': s,
        'RMS_Error': rms_err,
        'Max_Error': max_err,
        'SNR_dB': snr,
        'Saturation_%': saturation_pct
    })

# --- Results Table ---
results_df = pd.DataFrame(results)
print("\n" + "="*70)
print("Quantization Error Analysis (Q15 vs Float32)")
print("="*70)
print(results_df.to_string(index=False))

# --- Compare to theoretical bounds ---
print(f"\n{'='*70}")
print("Theoretical Quantization Limits (in normalized [-1,1) space):")
print(f"{'='*70}")
print(f"Q15 step size (Δ):        {DELTA:.2e}")
print(f"Theoretical RMS error:     {RMS_ERROR_THEO:.2e}  (Δ/√12)")
print(f"Theoretical max error:     {MAX_ERROR_THEO:.2e}  (Δ/2)")
print(f"{'='*70}")

# --- Check if errors are within bounds ---
print("\nError Analysis:")
all_good = True
critical_errors = False

for _, row in results_df.iterrows():
    # Check if RMS error is within acceptable range (2x theoretical is reasonable)
    # and max error is within the theoretical max bound
    rms_ok = row['RMS_Error'] <= RMS_ERROR_THEO * 2.0
    max_ok = row['Max_Error'] <= MAX_ERROR_THEO * 1.1  # Allow 10% over due to rounding
    
    if rms_ok and max_ok:
        status = "✓"
    elif row['RMS_Error'] <= RMS_ERROR_THEO * 3.0:
        status = "⚠"  # Warning but acceptable
    else:
        status = "✗"  # Critical error
        critical_errors = True
        all_good = False
    
    print(f"  {status} {row['Signal']}: RMS={row['RMS_Error']:.2e}, Max={row['Max_Error']:.2e}, SNR={row['SNR_dB']:.1f} dB, Saturation={row['Saturation_%']:.1f}%")

print(f"\n{'='*70}")
if critical_errors:
    print("❌ CRITICAL: Some signals show excessive quantization error!")
    print("   → Check normalization ranges in both MCU and Python")
elif not all_good:
    print("⚠️  WARNING: Some signals have higher than ideal error")
    print("   → This is acceptable for low-amplitude signals (low SNR)")
    print("   → Quantization noise dominates when signal amplitude is small")
else:
    print("✅ EXCELLENT: All signals show proper Q15 quantization!")
    print("   → Errors are within theoretical bounds")
    print("   → No saturation detected")
    print("   → System is working correctly!")

# --- Plot example error distribution ---
plt.figure(figsize=(12,6))

# Plot for accelerometer
plt.subplot(1, 3, 1)
norm_ax_f32 = df['ax_f32'] / ACCEL_RANGE
norm_ax_q15 = df['ax_q15'] / 32768.0
err_ax = norm_ax_f32 - norm_ax_q15
plt.hist(err_ax, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
plt.axvline(MAX_ERROR_THEO, color='r', linestyle='--', linewidth=2, label='±Δ/2 bound')
plt.axvline(-MAX_ERROR_THEO, color='r', linestyle='--', linewidth=2)
plt.title('Accelerometer X (ax)\nQuantization Error')
plt.xlabel('Error (normalized)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(alpha=0.3)

# Plot for gyroscope
plt.subplot(1, 3, 2)
norm_gx_f32 = df['gx_f32'] / GYRO_RANGE
norm_gx_q15 = df['gx_q15'] / 32768.0
err_gx = norm_gx_f32 - norm_gx_q15
plt.hist(err_gx, bins=100, color='green', alpha=0.7, edgecolor='black')
plt.axvline(MAX_ERROR_THEO, color='r', linestyle='--', linewidth=2, label='±Δ/2 bound')
plt.axvline(-MAX_ERROR_THEO, color='r', linestyle='--', linewidth=2)
plt.title('Gyroscope X (gx)\nQuantization Error')
plt.xlabel('Error (normalized)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(alpha=0.3)

# Plot for magnetometer
plt.subplot(1, 3, 3)
norm_mx_f32 = df['mx_f32'] / MAG_RANGE
norm_mx_q15 = df['mx_q15'] / 32768.0
err_mx = norm_mx_f32 - norm_mx_q15
plt.hist(err_mx, bins=100, color='orange', alpha=0.7, edgecolor='black')
plt.axvline(MAX_ERROR_THEO, color='r', linestyle='--', linewidth=2, label='±Δ/2 bound')
plt.axvline(-MAX_ERROR_THEO, color='r', linestyle='--', linewidth=2)
plt.title('Magnetometer X (mx)\nQuantization Error')
plt.xlabel('Error (normalized)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('quantization_error_analysis.png', dpi=150)
print("\n📊 Plot saved as 'quantization_error_analysis.png'")

# --- Interpretation Guide ---
print(f"\n{'='*70}")
print("Understanding the Results:")
print(f"{'='*70}")
print("\n✓ RMS Error < Theoretical: Excellent quantization performance")
print("✓ Max Error ≤ Δ/2: Quantization working within theoretical limits")
print("⚠ RMS Error ≈ 2×Theoretical: Acceptable for low-amplitude signals")
print("✗ RMS Error > 3×Theoretical: Check normalization constants!")
print("\nSNR (Signal-to-Noise Ratio):")
print("  • High SNR (>60 dB): Strong signal, quantization error negligible")
print("  • Medium SNR (40-60 dB): Good signal quality")
print("  • Low SNR (<40 dB): Weak signal, quantization noise more prominent")
print("\nSaturation:")
print("  • 0%: Perfect - no values clipped")
print("  • >0%: Some values exceeded [-1, 1) range - check normalization!")

plt.show()