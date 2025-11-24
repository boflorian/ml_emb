import argparse
from pathlib import Path
import numpy as np
from scipy.signal import butter, filtfilt


def preprocess_session1(x_raw, swap_axes=True, flip_y=True,
                        fs_raw=1000, fs_target=100, scale=16384.0):
    """
    scale -> swap/flip -> (optional anti-alias LPF) -> (optional decimate) -> round to 2 decimals.
    """
    if fs_target <= 0 or fs_raw <= 0:
        raise ValueError("fs_raw and fs_target must be positive.")
    fs_target = min(fs_target, fs_raw)

    # scale to g
    g0 = 9.80665  # m/s^2 per g
    x = (x_raw / scale) * g0

    # axis handling
    if swap_axes:
        x = x[:, [0, 2, 1]]   # swap Y and Z
    if flip_y:
        x[:, 1] *= -1         # gravity positive on +Y

    # decimation plan
    decim = max(1, int(round(fs_raw / fs_target)))
    fs_after = fs_raw / decim

    # Only design anti-alias filter if we actually decimate
    if decim > 1:
        cutoff_hz = min(20.0, 0.45 * fs_after)        # safe cutoff for downsampling
        Wn = cutoff_hz / (fs_raw / 2.0)               # normalize by RAW Nyquist
        Wn = min(max(Wn, 1e-6), 0.999)                # keep in (0,1)
        b, a = butter(4, Wn, btype="low")

        padlen = 3 * (max(len(a), len(b)) - 1)
        if x.shape[0] > padlen:
            x = filtfilt(b, a, x, axis=0)

        x = x[::decim]

    # → No normalization. Just round/clip to six decimals for more precision.
    x = np.round(x, 6)
    return x
def parse_gesture_file(path: Path):
    """
    Read one sessionX.txt and return list of np.ndarrays for each sample.
    """
    samples, cur = [], []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.lower().startswith("sample"):
                if cur:
                    samples.append(np.asarray(cur, dtype=np.float32))
                    cur = []
                continue
            if s.lower().startswith("ax,ay,az"):
                continue
            if "," in s:
                try:
                    ax, ay, az = map(float, s.split(","))
                    cur.append([ax, ay, az])
                except ValueError:
                    pass
    if cur:
        samples.append(np.asarray(cur, dtype=np.float32))
    return samples

def write_processed_file(out_path: Path, samples):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for i, s in enumerate(samples, start=1):
            f.write(f"sample{i}\n")
            f.write("ax,ay,az\n")
            np.savetxt(f, s, fmt="%.6f", delimiter=",")  # six decimals
            f.write("\n")

def main():
    p = argparse.ArgumentParser(
        description="Process all IMU sessions from raw/<class> to processed/<class>."
    )
    p.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Dataset root containing 'raw' and 'processed' (default: CWD)",
    )
    p.add_argument("--fs_raw", type=float, default=25)
    p.add_argument("--fs_target", type=float, default=25)
    # Defaults ON; you can disable with --no-swap / --no-flip if ever needed
    p.add_argument("--no-swap", dest="swap_axes", action="store_false")
    p.add_argument("--no-flip", dest="flip_y", action="store_false")
    p.set_defaults(swap_axes=True, flip_y=True)
    args = p.parse_args()

    raw_dir = args.root / "raw"
    out_root = args.root / "processed2"
    if not raw_dir.is_dir():
        raise SystemExit(f"Missing folder: {raw_dir}")

    print(f"Processing {raw_dir}  →  {out_root}")
    total_files = 0
    for class_dir in sorted([d for d in raw_dir.iterdir() if d.is_dir()]):
        txts = sorted(class_dir.glob("*.txt"))
        if not txts:
            continue
        print(f"\n[{class_dir.name}]")
        for file in txts:
            total_files += 1
            print(f"  {file.name}")
            samples = parse_gesture_file(file)
            processed = [
                preprocess_session1(
                    s,
                    swap_axes=args.swap_axes,
                    flip_y=args.flip_y,
                    fs_raw=args.fs_raw,
                    fs_target=args.fs_target,
                )
                for s in samples
            ]
            out_path = out_root / class_dir.name / file.name
            write_processed_file(out_path, processed)

    if total_files == 0:
        print("No .txt session files found under raw/<class>/")
    else:
        print(f"\nDone. Processed {total_files} file(s).")

if __name__ == "__main__":
    main()