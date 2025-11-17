#!/usr/bin/env python

import pathlib
import re
from typing import List, Dict


import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ========= CONFIG =========

DATA_ROOT = pathlib.Path("dataset_magic_wand") # original dataset
OUT_ROOT = pathlib.Path("dataset_magic_wand_gan") # synthetic dataset root

CATEGORIES = ["negative", "ring", "slope", "wave"]  # which categories to use / generate for

WIN = 128 # sequence length (time steps)
N_CHANNELS = 3 # ax, ay, az
BATCH_SIZE = 64
LATENT_DIM = 64
LR_G = 2e-4
LR_D = 1e-4
N_STEPS = 2500 
N_FAKE_SAVE = 50 

DEVICE = "mps" if torch.backends.mps.is_available() else \
         "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ========= DATA LOADING =========

SAMPLE_RE = re.compile(r"^sample\d+\s*$", re.IGNORECASE)


def parse_gesture_file(filepath: pathlib.Path) -> List[Dict]:
    """
    Parse a gesture file and return a list of samples.
    Each sample: {'sample_id': int, 'data': np.ndarray [T, 3]}
    """
    samples = []
    current_sample = None
    sample_idx = -1

    with filepath.open("r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            # New sample marker: "sampleX"
            if SAMPLE_RE.match(line):
                if current_sample is not None and current_sample["data"]:
                    current_sample["data"] = np.array(
                        current_sample["data"], dtype=np.float32
                    )
                    samples.append(current_sample)

                sample_idx += 1
                current_sample = {"sample_id": sample_idx, "data": []}
                continue

            # Skip header line "ax,ay,az"
            if line.lower().startswith("ax"):
                continue

            # Data line: "ax,ay,az"
            parts = line.split(",")
            if len(parts) != 3:
                continue
            try:
                ax, ay, az = map(float, parts)
            except ValueError:
                continue
            current_sample["data"].append([ax, ay, az])

    if current_sample is not None and current_sample["data"]:
        current_sample["data"] = np.array(current_sample["data"], dtype=np.float32)
        samples.append(current_sample)

    return samples


def load_all_samples(data_root: pathlib.Path, categories: List[str]) -> List[np.ndarray]:
    """
    Load all samples from given categories under data_root.
    Returns list of sequences [T, 3].
    """
    all_samples = []
    for cat in categories:
        cat_dir = data_root / cat
        if not cat_dir.exists():
            print(f"Warning: {cat_dir} does not exist, skipping.")
            continue

        for txt_path in sorted(cat_dir.glob("person*.txt")):
            print(f"Loading {txt_path} ...")
            samples = parse_gesture_file(txt_path)
            for s in samples:
                all_samples.append(s["data"])

    print(f"Total raw samples: {len(all_samples)}")
    return all_samples


def pad_or_crop(seq: np.ndarray, win: int) -> np.ndarray:
    """Make sequence length exactly win."""
    T = seq.shape[0]
    if T == win:
        return seq
    if T > win:
        start = (T - win) // 2
        return seq[start : start + win]
    # T < win: pad at end
    pad_len = win - T
    pad = np.zeros((pad_len, seq.shape[1]), dtype=seq.dtype)
    return np.concatenate([seq, pad], axis=0)


class IMUDataset(Dataset):
    def __init__(self, sequences: List[np.ndarray], win: int):
        self.data = [pad_or_crop(seq, win) for seq in sequences]
        self.data = np.stack(self.data, axis=0)  # [N, win, 3]

        # Per-dataset normalization (store mean/std for later denorm)
        mean = self.data.mean(axis=(0, 1), keepdims=True)
        std = self.data.std(axis=(0, 1), keepdims=True) + 1e-6
        self.data = (self.data - mean) / std

        self.mean = mean.astype(np.float32)  # shape [1,1,3]
        self.std = std.astype(np.float32)

        print(f"IMUDataset: {self.data.shape}, mean ~0, std ~1")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]
        return torch.from_numpy(x)  # [win, 3]


# ========= GAN MODELS =========

class Generator(nn.Module):
    def __init__(self, latent_dim: int, win: int, n_channels: int):
        super().__init__()
        self.win = win
        self.n_channels = n_channels

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128 * (win // 4)),
            nn.ReLU(),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, n_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z)                            # [B, 128*(win//4)]
        x = x.view(z.size(0), 128, self.win // 4) # [B, 128, win//4]
        x = self.net(x)                           # [B, C, win]
        return x.permute(0, 2, 1)                 # [B, win, C]


class Discriminator(nn.Module):
    def __init__(self, win: int, n_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear((win // 4) * 128, 1),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)   # [B, C, win]
        return self.net(x)       # [B, 1]


# ========= TRAIN + GENERATE =========

def train_gan_one_run(dataloader: DataLoader):
    G = Generator(LATENT_DIM, WIN, N_CHANNELS).to(DEVICE)
    D = Discriminator(WIN, N_CHANNELS).to(DEVICE)

    opt_G = torch.optim.Adam(G.parameters(), lr=LR_G, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=LR_D, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    step = 0
    data_iter = iter(dataloader)

    while step < N_STEPS:
        try:
            real = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            real = next(data_iter)

        real = real.to(DEVICE)
        B = real.size(0)
        valid = torch.ones(B, 1, device=DEVICE)
        fake = torch.zeros(B, 1, device=DEVICE)

        # --- Train D ---
        z = torch.randn(B, LATENT_DIM, device=DEVICE)
        gen = G(z).detach()
        logits_real = D(real)
        logits_fake = D(gen)
        loss_D = bce(logits_real, valid) + bce(logits_fake, fake)

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # --- Train G ---
        z = torch.randn(B, LATENT_DIM, device=DEVICE)
        gen = G(z)
        logits_fake = D(gen)
        loss_G = bce(logits_fake, valid)

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        if step % 20 == 0:
            print(f"Step {step:04d} | loss_D={loss_D.item():.4f} | loss_G={loss_G.item():.4f}")

        step += 1

    return G, D


def generate_synthetic_sequences(
    G: Generator,
    n_samples: int,
    win: int,
    latent_dim: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    Generate n_samples fake sequences, denormalized to original scale.
    Returns array [n_samples, win, 3].
    """
    G.eval()
    all_fake = []

    with torch.no_grad():
        remaining = n_samples
        while remaining > 0:
            b = min(remaining, 64)
            z = torch.randn(b, latent_dim, device=DEVICE)
            fake = G(z).cpu().numpy()  # normalized, roughly [-1,1]
            all_fake.append(fake)
            remaining -= b

    fake = np.concatenate(all_fake, axis=0)[:n_samples]  # [N, win, 3]

    # mean/std are [1,1,3]; broadcast to [N,win,3]
    fake_denorm = fake * std + mean
    return fake_denorm


def save_sequences_as_txt(
    sequences: np.ndarray,
    out_dir: pathlib.Path,
    base_name: str = "gan_person",
    start_idx: int = 0,
    decimals: int = 2,
):
    """
    Save each sequence as its own file:
      sample1
      ax,ay,az
      ...
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = "{:." + str(decimals) + "f}"

    for i, seq in enumerate(sequences):
        person_id = start_idx + i
        fname = f"{base_name}{person_id}.txt"
        path = out_dir / fname

        with path.open("w") as f:
            f.write("sample1\n")
            f.write("ax,ay,az\n")
            for t in range(seq.shape[0]):
                ax, ay, az = seq[t]
                f.write(f"{fmt.format(ax)},{fmt.format(ay)},{fmt.format(az)}\n")

        print(f"Saved {path}")


# ========= MAIN =========
def plot_fake_samples(fake, count=5):
    fig, axs = plt.subplots(count, 3, figsize=(12, 2 * count), sharex=True)
    for i in range(count):
        for c, comp in enumerate(["ax", "ay", "az"]):
            axs[i, c].plot(fake[i, :, c])
            axs[i, c].set_title(f"Sample {i+1} – {comp}")
    plt.tight_layout()
    plt.show()


def plot_real_vs_fake(real_array, fake_array, count=3):
    """
    real_array: list/array of real sequences, each [WIN,3]
    fake_array: numpy array [N,WIN,3]
    count: how many sequences to compare
    """

    count = min(count, len(real_array), len(fake_array))

    # Smaller figure: each row ~2 inches tall
    fig, axs = plt.subplots(count, 3, figsize=(10, 2 * count), sharex=True)
    if count == 1:
        axs = np.array([axs])

    for i in range(count):
        real_seq = real_array[i]
        fake_seq = fake_array[i]

        L = min(real_seq.shape[0], fake_seq.shape[0])
        real_seq = real_seq[:L]
        fake_seq = fake_seq[:L]

        for c, comp in enumerate(["ax", "ay", "az"]):
            axs[i, c].plot(real_seq[:, c], label="real", alpha=0.9)
            axs[i, c].plot(fake_seq[:, c], label="fake", alpha=0.9)
            axs[i, c].set_title(comp)
            axs[i, c].legend(fontsize=8)

    plt.tight_layout(pad=0.5)
    plt.show()




def main():
    if not DATA_ROOT.exists():
        raise SystemExit(f"DATA_ROOT {DATA_ROOT} does not exist.")

    sequences = load_all_samples(DATA_ROOT, CATEGORIES)
    if not sequences:
        raise SystemExit("No sequences loaded – check categories and paths.")

    dataset = IMUDataset(sequences, WIN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            shuffle=True, drop_last=True)

    print("Starting GAN smoke-test training...")
    G, _ = train_gan_one_run(dataloader)

    # Generate synthetic samples (denormalized)
    print(f"Generating {N_FAKE_SAVE} synthetic sequences...")
    fake_denorm = generate_synthetic_sequences(
        G,
        n_samples=N_FAKE_SAVE,
        win=WIN,
        latent_dim=LATENT_DIM,
        mean=dataset.mean,   # [1,1,3]
        std=dataset.std,     # [1,1,3]
    )

    # plot_fake_samples(fake_denorm, count=5)
    real_prepped = [pad_or_crop(seq, WIN) for seq in sequences]
    plot_real_vs_fake(real_prepped, fake_denorm, count=3)

    # Save with same structure as original (one sample per file)
    for cat in CATEGORIES:
        out_dir = OUT_ROOT / cat
        save_sequences_as_txt(fake_denorm, out_dir, base_name="gan_person", start_idx=0)
        # if you later want different fakes per category, you can re-run per cat

    print("Done. Synthetic dataset is under:", OUT_ROOT)


if __name__ == "__main__":
    main()