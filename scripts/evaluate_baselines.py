#!/usr/bin/env python3
"""Score simple depth baselines against FARO ground truth on processed samples.

Baselines: upsampled LiDAR, stereo, raw Depth Anything V2, and Depth Anything
aligned to the LiDAR with a per-image scale (and scale + shift). Alignment is fit
to LiDAR, never to ground truth, since ground truth is unavailable at test time.
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

DATA = Path(__file__).resolve().parents[1] / "data" / "processed_data"
INPUTS = ("lidar.png", "stereo_depth.png", "depthanything_depth.png")
METHODS = ("lidar", "stereo", "depth_anything", "da_scale", "da_scale_shift")


def load_meters(path):
    """Read a millimeter uint16 PNG as float meters, with 0 meaning no measurement."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    return image.astype(np.float32) / 1000


def upsample(depth, shape):
    """Bilinearly resize depth, keeping only pixels whose neighbors were all valid."""
    size = (shape[1], shape[0])
    valid = (depth > 0).astype(np.float32)
    resized = cv2.resize(depth, size, interpolation=cv2.INTER_LINEAR)
    support = cv2.resize(valid, size, interpolation=cv2.INTER_LINEAR)
    return np.where(support > 0.999, resized, 0)


def align_to_lidar(mono, lidar, shift):
    """Fit lidar ~= s * mono (+ t) by least squares at LiDAR resolution."""
    small = cv2.resize(mono, lidar.shape[::-1], interpolation=cv2.INTER_AREA)
    use = (lidar > 0) & (small > 0)
    x, y = small[use], lidar[use]
    if x.size < 10:
        return mono
    if shift:
        s, t = np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]
    else:
        s, t = float(x @ y / (x @ x)), 0.0
    return np.maximum(s * mono + t, 0)


def metrics(pred, truth):
    """Standard depth metrics on pixels where both prediction and ground truth exist."""
    gt_valid = truth > 0
    use = gt_valid & (pred > 0)
    p, g = pred[use], truth[use]
    if p.size == 0:
        return dict(abs_rel=np.nan, rmse=np.nan, delta1=np.nan, coverage=0.0)
    ratio = np.maximum(p / g, g / p)
    return dict(abs_rel=float(np.mean(np.abs(p - g) / g)),
                rmse=float(np.sqrt(np.mean((p - g) ** 2))),
                delta1=float(np.mean(ratio < 1.25)),
                coverage=float(use.sum() / gt_valid.sum()))


def evaluate_sample(images, annotations):
    truth = load_meters(annotations / "ground_truth.png")
    lidar, stereo, mono = (load_meters(images / name) for name in INPUTS)
    if stereo.shape != truth.shape or mono.shape != truth.shape:
        raise ValueError(f"{images.name}: stereo/Depth Anything not on the ground-truth grid")
    predictions = dict(lidar=upsample(lidar, truth.shape), stereo=stereo, depth_anything=mono,
                       da_scale=align_to_lidar(mono, lidar, shift=False),
                       da_scale_shift=align_to_lidar(mono, lidar, shift=True))
    return {method: metrics(predictions[method], truth) for method in METHODS}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--limit", type=int, help="Evaluate at most this many samples")
    parser.add_argument("--csv", type=Path, help="Also write per-sample metrics to this CSV")
    args = parser.parse_args()

    # Only score samples that processing has fully finished.
    names = sorted(p.name for p in (args.data / "annotations").glob("*")
                   if (p / "ground_truth.png").is_file()
                   and all((args.data / "images" / p.name / n).is_file() for n in INPUTS))
    if args.limit:
        names = names[:args.limit]
    if not names:
        parser.exit(1, f"No fully processed samples in {args.data}\n")

    rows = []
    for name in tqdm(names, desc="Evaluating", unit="sample"):
        result = evaluate_sample(args.data / "images" / name, args.data / "annotations" / name)
        rows += [dict(sample=name, method=method, **values) for method, values in result.items()]

    if args.csv:
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    # Average each metric over samples (each image counts equally).
    print(f"\n{len(names)} samples\n")
    print(f"{'method':<18}{'AbsRel':>9}{'RMSE(m)':>10}{'δ<1.25':>9}{'coverage':>10}")
    for method in METHODS:
        values = [r for r in rows if r["method"] == method]
        mean = {k: np.nanmean([r[k] for r in values]) for k in ("abs_rel", "rmse", "delta1", "coverage")}
        print(f"{method:<18}{mean['abs_rel']:>9.3f}{mean['rmse']:>10.3f}"
              f"{mean['delta1']:>9.3f}{mean['coverage']:>9.1%}")


if __name__ == "__main__":
    main()
