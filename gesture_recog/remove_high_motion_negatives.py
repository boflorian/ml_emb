from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


def parse_gesture_file(filepath: Path) -> List[List[float]]:
    samples = []
    current = None
    with filepath.open("r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("sample"):
                if current:
                    samples.append(current)
                current = []
            elif line == "ax,ay,az":
                continue
            elif line and "," in line:
                try:
                    ax, ay, az = map(float, line.split(","))
                except ValueError:
                    continue
                current.append([ax, ay, az])
        if current:
            samples.append(current)
    return samples


def motion_energy(sample: Iterable[Iterable[float]]) -> float:
    arr = np.asarray(sample, dtype=float)
    if arr.size == 0:
        return 0.0
    arr = arr - arr.mean(axis=0, keepdims=True)
    return float(np.linalg.norm(arr, axis=1).mean())


def filter_negative_files(root: Path, threshold: float, move_dir: Path, delete: bool) -> Tuple[int, int]:
    neg_dir = root / "negative"
    if not neg_dir.exists():
        raise FileNotFoundError(f"Negative folder not found: {neg_dir}")

    move_dir.mkdir(parents=True, exist_ok=True)
    removed = 0
    kept = 0

    for path in sorted(neg_dir.glob("*.txt")):
        samples = parse_gesture_file(path)
        if not samples:
            kept += 1
            continue
        # All files are single-sample in dataset_good, but handle multi-sample safely.
        energies = [motion_energy(s) for s in samples]
        energy = float(np.mean(energies))

        if energy > threshold:
            removed += 1
            if delete:
                path.unlink()
            else:
                dest = move_dir / path.name
                shutil.move(path.as_posix(), dest.as_posix())
        else:
            kept += 1

    return removed, kept


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove high-motion negatives from dataset_good.")
    parser.add_argument("--root", default="dataset_good", help="Dataset root containing the negative folder.")
    parser.add_argument("--threshold", type=float, required=True, help="Motion-energy threshold for removal.")
    parser.add_argument("--move-dir", default="dataset_good/negative_filtered",
                        help="Destination for removed negatives (ignored if --delete).")
    parser.add_argument("--delete", action="store_true",
                        help="Delete removed negatives instead of moving them.")
    args = parser.parse_args()

    root = Path(args.root)
    move_dir = Path(args.move_dir)
    removed, kept = filter_negative_files(root, args.threshold, move_dir, args.delete)
    action = "deleted" if args.delete else f"moved to {move_dir}"
    print(f"Removed {removed} negatives ({action}); kept {kept}.")


if __name__ == "__main__":
    main()
