#!/usr/bin/env python3
"""
Launch multiple GAN training runs in parallel for different categories.
Useful for running separate WGAN-GP jobs for slope/ring/wing concurrently
while keeping logs readable with per-category prefixes.
"""

import argparse
import asyncio
import pathlib
from typing import List

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "gan_train_wgan_gp.py"


async def stream_process(cmd: List[str], prefix: str) -> int:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(SCRIPT_DIR),
    )

    assert proc.stdout is not None
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        print(f"[{prefix}] {line.decode().rstrip()}")

    return await proc.wait()


async def main():
    parser = argparse.ArgumentParser(description="Run category-specific GAN jobs in parallel.")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["slope", "ring", "wing"],
        help="Categories to launch (space separated).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=12000,
        help="Train steps per category (set higher for full runs).",
    )
    parser.add_argument(
        "--save-fakes",
        action="store_true",
        help="Pass through to export samples after each run.",
    )
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        help="Additional args forwarded to gan_train_wgan_gp.py after '--'.",
    )
    args = parser.parse_args()

    cmds = []
    for cat in args.categories:
        cmd = [
            "python",
            str(TRAIN_SCRIPT),
            "--categories",
            cat,
            "--steps",
            str(args.steps),
        ]
        if args.save_fakes:
            cmd.append("--save-fakes")
        if args.extra:
            # Support additional args like --batch-size or --ckpt-every after --
            cmd.extend(args.extra)
        cmds.append((cmd, cat))

    print("Launching category jobs:")
    for cmd, cat in cmds:
        print(f"- {cat}: {' '.join(cmd)}")

    tasks = [stream_process(cmd, prefix=cat) for cmd, cat in cmds]
    print("All processes started; streaming logs...\n")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for cat, res in zip(args.categories, results):
        if isinstance(res, Exception):
            print(f"[{cat}] failed with exception: {res}")
        else:
            print(f"[{cat}] exited with code {res}")


if __name__ == "__main__":
    asyncio.run(main())
