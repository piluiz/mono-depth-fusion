# Multi-Sensor and Monocular Depth Fusion for Accurate Depth Estimation

Each depth source has its own weak spot. Low-cost LiDAR is low-resolution, stereo depth has holes and noise, and monocular models like Depth Anything give dense, detailed depth that is only relative, not true metric distance. This project trains a lightweight U-Net that fuses a frozen Depth Anything V2 prediction with LiDAR and stereo depth, plus a confidence map for each, to produce dense metric depth. Training and evaluation use [ARKitScenes](https://rerun.io/examples/spatial-computing/arkit_scenes), with high-resolution FARO laser-scan depth as ground truth. We compare against the raw sensor depth, a scale-aligned Depth Anything baseline, and Prior Depth Anything. Our goal is a model that is more accurate than both the raw sensor depth and the scale-aligned baseline, and ideally one that matches Prior Depth Anything on standard benchmarks while being more robust to realistic sensor errors like misalignment and edge noise.

## ARKitScenes data preparation

Run these commands to download 1,000 samples into `data/raw` and process them into `data/processed_data`.

```bash
python3 -m pip install -r requirements.txt
python3 scripts/download_data.py --count 1000
python3 scripts/process_data.py
```