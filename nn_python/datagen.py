import numpy as np
import pandas as pd

# Config
N = 20_000          # number of samples
seed = 42
rng = np.random.default_rng(seed)

# Ranges from your slide
T_MIN, T_MAX   = -70.0, 60.0      # °C
H_MIN, H_MAX   = 0.0, 100.0       # % RH
AQI_MIN, AQI_MAX = 0.0, 500.0     # AQI

# Sample features uniformly
T   = rng.uniform(T_MIN, T_MAX,   size=N)
RH  = rng.uniform(H_MIN, H_MAX,   size=N)
AQI = rng.uniform(AQI_MIN, AQI_MAX, size=N)

# ---------------------------
# Option A: Hard-rule labels
# (transparent, deterministic)
# ---------------------------
# Example comfort region (tweak to taste):
#   Temperature 18–26 °C
#   Humidity    30–60 %
#   AQI         <= 100
hard_label = (
    (T >= 18) & (T <= 26) &
    (RH >= 30) & (RH <= 60) &
    (AQI <= 100)
).astype(int)

# ---------------------------
# Option B: Soft/probabilistic labels
# (more realistic)
# ---------------------------
# Define a "comfort score" in [0,1] using smooth preferences:
# - Temperature preference around 22 °C (sigma controls tolerance)
# - Humidity around 45 %
# - AQI lower is better (exponential decay)
def soft_comfort_prob(T, RH, AQI):
    # Gaussian preferences
    temp_pref = np.exp(-0.5 * ((T - 22.0) / 8.0)**2)      # broader if sigma=8
    hum_pref  = np.exp(-0.5 * ((RH - 45.0) / 20.0)**2)
    aqi_pref  = np.exp(-AQI / 100.0)                      # fast drop with AQI

    # Combine and squish to [0,1]; weight if desired
    raw = 0.5*temp_pref + 0.3*hum_pref + 0.2*aqi_pref
    # Optional small randomness to avoid perfectly separable data
    noise = rng.normal(0, 0.03, size=raw.shape)
    prob = np.clip(raw + noise, 0, 1)
    return prob

prob = soft_comfort_prob(T, RH, AQI)
soft_label = rng.binomial(1, prob)

# Choose which label to use
label = soft_label     # or: hard_label

# Build DataFrame
df = pd.DataFrame({
    "temperature_c": T,
    "humidity_pct": RH,
    "aqi": AQI,
    "comfortable": label
})

# Optional: keep both for comparison
#df["comfortable_hard"] = hard_label
#df["comfortable_soft"] = soft_label

# Train/valid/test split indices (simple)
idx = rng.permutation(N)
train_idx = idx[:int(0.7*N)]
valid_idx = idx[int(0.7*N):int(0.85*N)]
test_idx  = idx[int(0.85*N):]

# splits = np.zeros(N, dtype="U5")
# splits[train_idx] = "train"
# splits[valid_idx] = "valid"
# splits[test_idx]  = "test"
# df["split"] = splits

# Save if you want
df.to_csv("comfort_dataset.csv", index=False)

print(df.head())
print(df["comfortable"].value_counts(normalize=True))