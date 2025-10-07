import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
DELTA = 1 / 32768  # Q15 step size
RMS_ERROR_THEO = DELTA / np.sqrt(12)
MAX_ERROR_THEO = DELTA / 2

# --- Load CSV ---
file = 'imu_data.csv'  # <-- change this to your file name
df = pd.read_csv(file)

# --- Columns to compare ---
signals = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz']

results = []
for s in signals:
    f32 = df[f'{s}_f32']
    q15 = df[f'{s}_q15']
    
    # Convert Q15 to float in [-1, 1)
    q15_float = q15 / 32768.0
    
    # Error metrics
    err = f32 - q15_float
    rms_err = np.sqrt(np.mean(err**2))
    max_err = np.max(np.abs(err))
    snr = 20 * np.log10(np.std(f32) / (DELTA / np.sqrt(12))) if np.std(f32) != 0 else np.nan
    
    results.append({
        'Signal': s,
        'RMS_Error': rms_err,
        'Max_Error': max_err,
        'SNR_dB': snr
    })

# --- Results Table ---
results_df = pd.DataFrame(results)
print("\nQuantization Error Analysis (Q15 vs Float32):")
print(results_df.to_string(index=False))

# --- Compare to theoretical bounds ---
print(f"\nTheoretical RMS error: {RMS_ERROR_THEO:.2e}")
print(f"Theoretical max |error| bound: {MAX_ERROR_THEO:.2e}")

# --- Plot example error distribution ---
plt.figure(figsize=(10,5))
err = df['ax_f32'] - df['ax_q15'] / 32768.0
plt.hist(err, bins=100, color='steelblue', alpha=0.7)
plt.axvline(MAX_ERROR_THEO, color='r', linestyle='--', label='±Δ/2 bound')
plt.axvline(-MAX_ERROR_THEO, color='r', linestyle='--')
plt.title('Quantization Error Distribution (ax)')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()