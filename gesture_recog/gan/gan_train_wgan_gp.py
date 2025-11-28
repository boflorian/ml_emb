#!/usr/bin/env python
"""
Longer, stabler GAN training for IMU gesture sequences using WGAN-GP.
- WGAN-GP objective with spectral norm on the discriminator
- Configurable long training (steps, n_critic, checkpoint cadence)
- Checkpoint/resume support and optional fake sample export
"""

import argparse
import json
import pathlib
import random
import re
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# ========= CONFIG =========

DATA_ROOT = pathlib.Path("../dataset_magic_wand")
OUT_ROOT = pathlib.Path("../dataset_magic_wand_gan")
DEFAULT_CATEGORIES = ["slope"]  # can be overridden via CLI

WIN = 128
N_CHANNELS = 3
BATCH_SIZE = 128
LATENT_DIM = 64
LR_G = 2e-4
LR_D = 5e-4
TOTAL_STEPS = 20000
N_CRITIC = 5
GP_LAMBDA = 15.0
MOMENT_LAMBDA = 2.0
LOG_EVERY = 100
CKPT_EVERY = 1000
N_FAKE_SAVE = 1000

DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {DEVICE}")


# ========= DATA LOADING =========

SAMPLE_RE = re.compile(r"^sample\d+\s*$", re.IGNORECASE)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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

            if SAMPLE_RE.match(line):
                if current_sample is not None and current_sample["data"]:
                    current_sample["data"] = np.array(
                        current_sample["data"], dtype=np.float32
                    )
                    samples.append(current_sample)

                sample_idx += 1
                current_sample = {"sample_id": sample_idx, "data": []}
                continue

            if line.lower().startswith("ax"):
                continue

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


def load_all_samples(
    data_root: pathlib.Path, categories: List[str]
) -> List[Tuple[str, np.ndarray]]:
    """
    Load all samples from given categories under data_root.
    Returns list of (category, sequence [T, 3]).
    """
    all_samples = []
    for cat in categories:
        cat_dir = data_root / cat
        if not cat_dir.exists():
            print(f"Warning: {cat_dir} does not exist, skipping.")
            continue

        for txt_path in sorted(cat_dir.glob("person*.txt")):
            samples = parse_gesture_file(txt_path)
            for s in samples:
                all_samples.append((cat, s["data"]))

    print(f"Total raw samples: {len(all_samples)}")
    return all_samples


def pad_or_crop(seq: np.ndarray, win: int, random_crop: bool = False) -> np.ndarray:
    """
    Make sequence length exactly win.
    - Longer: center crop or random crop if random_crop=True
    - Shorter: reflection pad to avoid flat tails
    """
    T = seq.shape[0]
    if T == win:
        return seq
    if T > win:
        if random_crop:
            start = np.random.randint(0, T - win + 1)
        else:
            start = (T - win) // 2
        return seq[start : start + win]

    pad_len = win - T
    # Reflection pad on the time axis
    return np.pad(seq, ((0, pad_len), (0, 0)), mode="reflect")


class IMUDataset(Dataset):
    """
    Stores raw sequences and normalizes on-the-fly using dataset statistics.
    """

    def __init__(self, sequences: List[np.ndarray], win: int, random_crop: bool = True):
        self.raw = [np.array(seq, dtype=np.float32) for seq in sequences]
        self.win = win
        self.random_crop = random_crop

        # Compute stats on deterministic center-cropped sequences
        processed = [pad_or_crop(seq, win, random_crop=False) for seq in self.raw]
        stacked = np.stack(processed, axis=0)  # [N, win, 3]
        mean = stacked.mean(axis=(0, 1), keepdims=True)
        std = stacked.std(axis=(0, 1), keepdims=True) + 1e-6

        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

        print(f"IMUDataset: {stacked.shape}, mean ~0, std ~1")

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        seq = self.raw[idx]
        seq = pad_or_crop(seq, self.win, random_crop=self.random_crop)
        seq = (seq - self.mean) / self.std
        return torch.from_numpy(seq)


# ========= GAN MODELS =========


def spectral_norm(module: nn.Module) -> nn.Module:
    return nn.utils.spectral_norm(module)


class Generator(nn.Module):
    def __init__(self, latent_dim: int, win: int, n_channels: int):
        super().__init__()
        self.win = win

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * (win // 8)),
            nn.ReLU(inplace=True),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(64, n_channels, kernel_size=4, stride=2, padding=1),
        )

        # Learned positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(win, n_channels))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.1)

    def forward(self, z):
        x = self.fc(z)  # [B, 256 * win/8]
        x = x.view(z.size(0), 256, self.win // 8)
        x = self.deconv(x)[:, :, : self.win]  # [B, C, win]
        x = x.permute(0, 2, 1)  # [B, win, C]
        x = x + self.pos_embed.unsqueeze(0)
        return x


class MinibatchStdDev(nn.Module):
    """
    Adds a channel containing per-batch standard deviation to encourage diversity.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        batch_std = torch.sqrt(x.var(dim=0, unbiased=False) + 1e-8)  # [C, T]
        mean_std = batch_std.mean().view(1, 1, 1).expand(x.size(0), 1, x.size(2))
        return torch.cat([x, mean_std], dim=1)


class Discriminator(nn.Module):
    def __init__(self, win: int, n_channels: int):
        super().__init__()
        self.mbstdev = MinibatchStdDev()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv1d(n_channels, 64, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(0.2, inplace=True),
            self.mbstdev,
            nn.Flatten(),
            spectral_norm(nn.Linear((win // 8) * 257, 1)),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, C, win]
        return self.net(x)


# ========= TRAIN + GENERATE =========


def gradient_penalty(D: Discriminator, real: torch.Tensor, fake: torch.Tensor):
    B = real.size(0)
    alpha = torch.rand(B, 1, 1, device=real.device)
    interpolated = real * alpha + fake * (1 - alpha)
    interpolated.requires_grad_(True)

    d_interpolated = D(interpolated)
    grad_outputs = torch.ones_like(d_interpolated)

    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.reshape(B, -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def save_checkpoint(
    path: pathlib.Path,
    step: int,
    G: Generator,
    D: Discriminator,
    opt_G: torch.optim.Optimizer,
    opt_D: torch.optim.Optimizer,
    mean: np.ndarray,
    std: np.ndarray,
    config: Dict,
):
    payload = {
        "step": step,
        "G": G.state_dict(),
        "D": D.state_dict(),
        "opt_G": opt_G.state_dict(),
        "opt_D": opt_D.state_dict(),
        "mean": mean,
        "std": std,
        "config": config,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"[ckpt] Saved checkpoint to {path}")


def load_checkpoint(
    path: pathlib.Path,
    G: Generator,
    D: Discriminator,
    opt_G: torch.optim.Optimizer,
    opt_D: torch.optim.Optimizer,
):
    ckpt = torch.load(path, map_location=DEVICE)
    G.load_state_dict(ckpt["G"])
    D.load_state_dict(ckpt["D"])
    opt_G.load_state_dict(ckpt["opt_G"])
    opt_D.load_state_dict(ckpt["opt_D"])
    start_step = ckpt.get("step", 0)
    mean = ckpt.get("mean")
    std = ckpt.get("std")
    print(f"[ckpt] Loaded checkpoint from {path} at step {start_step}")
    return start_step, mean, std, ckpt.get("config", {})


def train_wgan_gp(
    dataloader: DataLoader,
    total_steps: int,
    n_critic: int,
    gp_lambda: float,
    moment_lambda: float,
    log_every: int,
    ckpt_every: int,
    ckpt_dir: pathlib.Path,
    resume_path: pathlib.Path = None,
    latent_dim: int = LATENT_DIM,
    win: int = WIN,
    n_channels: int = N_CHANNELS,
):
    G = Generator(latent_dim, win, n_channels).to(DEVICE)
    D = Discriminator(win, n_channels).to(DEVICE)

    opt_G = torch.optim.Adam(G.parameters(), lr=LR_G, betas=(0.0, 0.99))
    opt_D = torch.optim.Adam(D.parameters(), lr=LR_D, betas=(0.0, 0.99))

    start_step = 0
    dataset_mean = getattr(dataloader.dataset, "mean", None)
    dataset_std = getattr(dataloader.dataset, "std", None)
    if resume_path is not None and resume_path.exists():
        start_step, dataset_mean, dataset_std, _ = load_checkpoint(
            resume_path, G, D, opt_G, opt_D
        )

    data_iter = iter(dataloader)

    def normalize_batch(real_batch: torch.Tensor) -> torch.Tensor:
        """
        Ensure the shape is [B, T, C]. Handles a few common edge cases when the
        DataLoader yields a squeezed batch.
        """
        if real_batch.dim() == 3:
            return real_batch
        if real_batch.dim() == 2:
            return real_batch.unsqueeze(0)
        if real_batch.dim() == 4 and real_batch.size(1) == 1:
            return real_batch.squeeze(1)
        raise RuntimeError(f"Unexpected batch shape from DataLoader: {tuple(real_batch.shape)}")

    for step in range(start_step, total_steps):
        # If dataloader is exhausted, re-init iterator
        if data_iter is None:
            data_iter = iter(dataloader)
        # ----- Train D for n_critic steps -----
        real_for_g = None
        for _ in range(n_critic):
            try:
                real = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                real = next(data_iter)

            real = normalize_batch(real).to(DEVICE).float()
            B = real.size(0)
            real_for_g = real

            z = torch.randn(B, latent_dim, device=DEVICE)
            fake = G(z)

            d_real = D(real).mean()
            d_fake = D(fake.detach()).mean()
            gp = gradient_penalty(D, real, fake.detach()) * gp_lambda
            drift = 0.001 * (d_real.pow(2) + d_fake.pow(2))

            loss_D = d_fake - d_real + gp + drift

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

        # ----- Train G -----
        if real_for_g is None:
            raise RuntimeError("No real batch available for generator step.")

        z = torch.randn(B, latent_dim, device=DEVICE)
        fake = G(z)
        adv_loss = -D(fake).mean()
        moment_loss = torch.tensor(0.0, device=DEVICE)
        if moment_lambda > 0:
            fake_mean = fake.mean(dim=(0, 1))
            fake_std = fake.std(dim=(0, 1))
            real_mean = real_for_g.mean(dim=(0, 1))
            real_std = real_for_g.std(dim=(0, 1))
            moment_loss = (fake_mean - real_mean).abs().mean() + (fake_std - real_std).abs().mean()
        loss_G = adv_loss + moment_lambda * moment_loss

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        if step % log_every == 0:
            print(
                f"Step {step:05d} | loss_D={loss_D.item():.4f} "
                f"(real={d_real.item():.3f}, fake={d_fake.item():.3f}, gp={gp.item():.3f}) "
                f"| loss_G={loss_G.item():.4f} (adv={adv_loss.item():.4f}, moment={moment_loss.item():.4f})"
            )

        if step > 0 and step % ckpt_every == 0:
            ckpt_path = ckpt_dir / f"ckpt_step{step:06d}.pt"
            save_checkpoint(
                ckpt_path,
                step,
                G,
                D,
                opt_G,
                opt_D,
                dataset_mean,
                dataset_std,
                config={
                    "total_steps": total_steps,
                    "n_critic": n_critic,
                    "gp_lambda": gp_lambda,
                    "moment_lambda": moment_lambda,
                    "win": win,
                    "latent_dim": latent_dim,
                },
            )

    return G, D, dataset_mean, dataset_std


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
    if out_dir.exists():
        # Remove old .txt files
        for old_file in out_dir.glob("*.txt"):
            old_file.unlink()

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


def main():
    parser = argparse.ArgumentParser(description="Train WGAN-GP on IMU gestures.")
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS, help="Total train steps")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        help="Categories to train on (space separated)",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--n-critic", type=int, default=N_CRITIC, help="D steps per G step")
    parser.add_argument("--gp", type=float, default=GP_LAMBDA, help="Gradient penalty weight")
    parser.add_argument(
        "--moment-lambda",
        type=float,
        default=MOMENT_LAMBDA,
        help="Weight for matching per-axis mean/std between real and fake batches",
    )
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    parser.add_argument("--ckpt-every", type=int, default=CKPT_EVERY)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--save-fakes", action="store_true", help="Export fake samples after train")
    parser.add_argument("--num-fakes", type=int, default=N_FAKE_SAVE)
    parser.add_argument("--out-root", type=str, default=str(OUT_ROOT))
    parser.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    args = parser.parse_args()

    set_seed(42)

    data_root = pathlib.Path(args.data_root)
    out_root = pathlib.Path(args.out_root)
    if not data_root.exists():
        raise SystemExit(f"DATA_ROOT {data_root} does not exist.")

    categories = [c.strip() for c in args.categories if c.strip()]
    if not categories:
        raise SystemExit("No categories provided.")

    cat_sequences = load_all_samples(data_root, categories)
    if not cat_sequences:
        raise SystemExit("No sequences loaded – check categories and paths.")

    sequences = [seq for _, seq in cat_sequences]
    dataset = IMUDataset(sequences, WIN, random_crop=True)
    effective_batch = min(args.batch_size, len(dataset))
    if effective_batch < 1:
        raise SystemExit("Dataset is empty; cannot start training.")
    if effective_batch < args.batch_size:
        print(f"Batch size clipped to dataset size: {effective_batch}")

    dataloader = DataLoader(
        dataset,
        batch_size=effective_batch,
        shuffle=True,
        drop_last=False,
    )

    print("Starting WGAN-GP training...")
    ckpt_dir = pathlib.Path("checkpoints_wgan_gp")
    resume_path = pathlib.Path(args.resume) if args.resume else None
    G, _, mean, std = train_wgan_gp(
        dataloader,
        total_steps=args.steps,
        n_critic=args.n_critic,
        gp_lambda=args.gp,
        moment_lambda=args.moment_lambda,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        ckpt_dir=ckpt_dir,
        resume_path=resume_path,
        latent_dim=LATENT_DIM,
        win=WIN,
        n_channels=N_CHANNELS,
    )

    if args.save_fakes:
        print(f"Generating {args.num_fakes} synthetic sequences...")
        fake_denorm = generate_synthetic_sequences(
            G,
            n_samples=args.num_fakes,
            win=WIN,
            latent_dim=LATENT_DIM,
            mean=mean,
            std=std,
        )

        stats_path = out_root / "norm_stats.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("w") as f:
            json.dump({"mean": mean.tolist(), "std": std.tolist()}, f, indent=2)
        print(f"Saved normalization stats to {stats_path}")

        for cat in categories:
            out_dir = out_root / cat
            save_sequences_as_txt(
                fake_denorm, out_dir, base_name="gan_person", start_idx=0
            )

    print("Done.")


if __name__ == "__main__":
    main()
