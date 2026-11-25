#!/usr/bin/env python3
"""Assign whole ARKitScenes videos to train/val/test and write splits/<name>.txt.

Splitting by video keeps near-duplicate frames of the same room out of both
training and testing. Each file lists one video ID per line. Rerunning with the
same seed and data reproduces the same split.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]


def sample_videos(raw):
    """Map each raw sample folder name to its video ID."""
    return {f.parent.name: json.loads(f.read_text())["video_id"] for f in raw.glob("*/sample.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output", type=Path, default=ROOT / "splits")
    parser.add_argument("--val", type=float, default=0.15, help="Target fraction of samples")
    parser.add_argument("--test", type=float, default=0.15, help="Target fraction of samples")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    counts = Counter(sample_videos(args.raw).values())
    if len(counts) < 3:
        parser.exit(1, f"Need at least 3 videos in {args.raw}, found {len(counts)}\n")
    videos = sorted(counts)
    random.Random(args.seed).shuffle(videos)

    # Fill test, then val, up to their target share of samples; the rest train.
    total = sum(counts.values())
    splits = {"test": [], "val": [], "train": []}
    for name, fraction in (("test", args.test), ("val", args.val)):
        while videos and sum(counts[v] for v in splits[name]) < fraction * total:
            splits[name].append(videos.pop())
    splits["train"] = videos

    args.output.mkdir(exist_ok=True)
    for name, members in splits.items():
        (args.output / f"{name}.txt").write_text("".join(f"{v}\n" for v in sorted(members)))
        samples = sum(counts[v] for v in members)
        print(f"{name:<6}{len(members):>3} videos {samples:>5} samples ({samples / total:.0%})")


if __name__ == "__main__":
    main()
