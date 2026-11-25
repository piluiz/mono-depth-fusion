# Multi-Sensor and Monocular Depth Fusion for Accurate Depth Estimation

Each depth source has its own weak spot. Low-cost LiDAR is low-resolution, stereo depth has holes and noise, and monocular models like Depth Anything give dense, detailed depth that is only relative, not true metric distance. This project trains a lightweight U-Net that fuses a frozen Depth Anything V2 prediction with LiDAR and stereo depth, plus a confidence map for each, to produce dense metric depth. Training and evaluation use [ARKitScenes](https://rerun.io/examples/spatial-computing/arkit_scenes), with high-resolution FARO laser-scan depth as ground truth. We compare against the raw sensor depth, a scale-aligned Depth Anything baseline, and Prior Depth Anything. Our goal is a model that is more accurate than both the raw sensor depth and the scale-aligned baseline, and ideally one that matches Prior Depth Anything on standard benchmarks while being more robust to realistic sensor errors like misalignment and edge noise.

## ARKitScenes data preparation

Run these commands to download 1,000 samples into `data/raw` and process them into `data/processed_data`.

```bash
python3 -m pip install -r requirements.txt
python3 scripts/download_data.py --count 1000
python3 scripts/process_data.py
```

## Train/val/test split

`scripts/make_split.py` assigns whole videos to train, val, or test (about 70/15/15 by sample count), so frames of the same room never appear in more than one split. It writes one video ID per line to `splits/train.txt`, `splits/val.txt`, and `splits/test.txt`. The split is committed so everyone uses the same one; rerun only if the downloaded videos change.

```bash
python3 scripts/make_split.py
```

## Baselines

`scripts/evaluate_baselines.py` scores each depth source against the FARO ground truth: upsampled LiDAR, stereo, raw Depth Anything, and Depth Anything aligned to the LiDAR with a per-image scale (and scale + shift). It prints AbsRel, RMSE, δ<1.25, and coverage (the share of ground-truth pixels a method fills), averaged over samples.

```bash
python3 scripts/evaluate_baselines.py --split test            # test videos only
python3 scripts/evaluate_baselines.py --csv results.csv       # also save per-sample metrics
```

## Loading data for training

`scripts/dataset.py` is imported by training code, not run directly. It returns each processed sample in a split as tensors at 320×240: RGB, Depth Anything, LiDAR, stereo, and ground truth (depths in meters, 0 where missing), plus a validity mask for LiDAR, stereo, and ground truth.

```python
from dataset import DepthFusionDataset
train = DepthFusionDataset("train", augment=True)        # random flips
test = DepthFusionDataset("test", lidar_points=500)      # simulate a sparse LiDAR
loader = torch.utils.data.DataLoader(train, batch_size=8, shuffle=True)
```