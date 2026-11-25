"""PyTorch dataset over processed samples for one video-level split.

Each item is a dict of float32 tensors at the training resolution (depths in meters,
0 where missing, each with a matching validity mask):

    rgb (3,H,W) in [0, 1]      depth_anything (1,H,W)
    lidar, lidar_valid         stereo, stereo_valid
    ground_truth, gt_valid     name (str)

Example:
    from dataset import DepthFusionDataset
    train = DepthFusionDataset("train", augment=True)
    loader = torch.utils.data.DataLoader(train, batch_size=8, shuffle=True)
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def read(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    return image


def resize_depth(depth, size):
    """Resize a depth map with holes, averaging only valid pixels.

    Shrinking keeps an output pixel when at least half its area was measured.
    Enlarging keeps it only when every interpolated neighbor was measured.
    """
    shrinking = size[0] < depth.shape[1]
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    valid = (depth > 0).astype(np.float32)
    total = cv2.resize(depth * valid, size, interpolation=interpolation)
    support = cv2.resize(valid, size, interpolation=interpolation)
    keep = support > (0.5 if shrinking else 0.999)
    return np.where(keep, total / np.maximum(support, 1e-6), 0).astype(np.float32)


def sparsify(lidar, points, size, rng):
    """Keep `points` random LiDAR measurements, placed on the output grid."""
    v, u = np.nonzero(lidar)
    pick = rng.choice(len(u), size=min(points, len(u)), replace=False)
    out = np.zeros(size[::-1], np.float32)
    x = ((u[pick] + 0.5) * size[0] / lidar.shape[1]).astype(int)
    y = ((v[pick] + 0.5) * size[1] / lidar.shape[0]).astype(int)
    out[y, x] = lidar[v[pick], u[pick]]
    return out


class DepthFusionDataset(Dataset):
    """Processed samples whose video is listed in splits/<split>.txt.

    size: output (width, height); 4:3 like the source images.
    lidar_points: None feeds the full ARKit depth; an int keeps only that many
        random points to simulate a sparse sensor (fixed per sample unless augmenting).
    augment: random horizontal flips and, with lidar_points, new points each epoch.
    cache: keep resized samples in memory after the first read (~3 MB each at 320x240).
    """

    def __init__(self, split, size=(320, 240), lidar_points=None, augment=False,
                 cache=True, data=DATA, splits=ROOT / "splits"):
        videos = set((Path(splits) / f"{split}.txt").read_text().split())
        processed = Path(data) / "processed_data"
        self.names = sorted(
            p.name for p in (processed / "annotations").glob("*")
            if (p / "ground_truth.png").is_file()
            and (processed / "images" / p.name / "depthanything_depth.png").is_file()
            and json.loads((Path(data) / "raw" / p.name / "sample.json").read_text())["video_id"] in videos)
        if not self.names:
            raise ValueError(f"No processed samples for split {split!r}")
        self.processed, self.size = processed, tuple(size)
        self.lidar_points, self.augment = lidar_points, augment
        self.cache = {} if cache else None

    def __len__(self):
        return len(self.names)

    def load(self, name):
        """Read one sample at training resolution, with LiDAR at its native resolution."""
        if self.cache is not None and name in self.cache:
            return self.cache[name]
        images, notes = self.processed / "images" / name, self.processed / "annotations" / name
        meters = lambda path: read(path).astype(np.float32) / 1000
        rgb = cv2.resize(read(notes / "main.png"), self.size, interpolation=cv2.INTER_AREA)
        sample = dict(rgb=rgb[:, :, ::-1].astype(np.float32) / 255,
                      depth_anything=resize_depth(meters(images / "depthanything_depth.png"), self.size),
                      stereo=resize_depth(meters(images / "stereo_depth.png"), self.size),
                      ground_truth=resize_depth(meters(notes / "ground_truth.png"), self.size),
                      lidar=meters(images / "lidar.png"))
        if self.cache is not None:
            self.cache[name] = sample
        return sample

    def __getitem__(self, index):
        name = self.names[index]
        sample = dict(self.load(name))
        rng = np.random.default_rng(None if self.augment else index)
        if self.lidar_points is None:
            sample["lidar"] = resize_depth(sample["lidar"], self.size)
        else:
            sample["lidar"] = sparsify(sample["lidar"], self.lidar_points, self.size, rng)
        if self.augment and rng.random() < 0.5:
            sample = {k: v[:, ::-1] for k, v in sample.items()}

        out = {"name": name}
        for key, value in sample.items():
            array = np.ascontiguousarray(value)
            out[key] = torch.from_numpy(array.transpose(2, 0, 1) if array.ndim == 3 else array[None])
        for key, valid in (("lidar", "lidar_valid"), ("stereo", "stereo_valid"),
                           ("ground_truth", "gt_valid")):
            out[valid] = (out[key] > 0).float()
        return out
