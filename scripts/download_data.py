#!/usr/bin/env python3
"""Download matched native ARKitScenes frames and camera intrinsics to data/raw."""
import argparse
from bisect import bisect_left
from contextlib import ExitStack
import csv
from io import TextIOWrapper
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZipFile

BASE = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1"
SPLITS = "https://raw.githubusercontent.com/apple/ARKitScenes/main/depth_upsampling/upsampling_train_val_splits.csv"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "raw"


def download(url, path):
    # curl streams large ZIPs to disk and retries transient transport failures.
    subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                    "--retry", "3", "--output", str(path), url], check=True)


def calibration_files(raw_root, video, sample=None):
    """Reuse local layers or fetch each Rerun layer once per video."""
    cache = raw_root / "calibration" / video
    files = []
    for layer, name in (("calibration", "calibration.rrd"), ("base", "recording_info.rrd")):
        path = cache / name
        if sample is not None and (sample / name).is_file():
            path = sample / name
        if not path.is_file():
            cache.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=cache) as tmp:
                pending = Path(tmp) / name
                url = f"https://huggingface.co/datasets/rerun/arkitscenes-rrd/resolve/main/{layer}/{video}.rrd"
                download(url, pending)
                pending.replace(path)
        files.append(path)
    return files


def frames(archive, folder, suffix):
    index = {}
    for name in archive.namelist():
        path = Path(name)
        if path.parent.name != folder or path.suffix != suffix or path.name.startswith("."):
            continue
        stamp = float(path.stem.rsplit("_", 1)[-1])
        if stamp in index:
            raise ValueError(f"Duplicate {folder} timestamp {stamp}")
        index[stamp] = name
    if not index:
        raise ValueError(f"Missing {folder} entries in {archive.filename}")
    return index


def nearest(times, target):
    i = bisect_left(times, target)
    return min(times[max(0, i - 1):i + 1], key=lambda t: abs(t - target))


def calibration(archive, member):
    return tuple(map(float, archive.read(member).split()))


def download_video(row, root, needed, completed):
    video, split = row["video_id"], row["fold"]
    print(f"Downloading {split}/{video}", flush=True)
    calibration_files(root, video)
    with TemporaryDirectory(dir=root) as temporary:
        tmp = Path(temporary)
        raw = f"{BASE}/raw/{split}/{video}"
        assets = {"ultrawide_intrinsics": raw + "/ultrawide_intrinsics.zip",
                  "lowres_wide_intrinsics": raw + "/lowres_wide_intrinsics.zip",
                  "ultrawide": raw + "/ultrawide.zip",
                  "paired": f"{BASE}/upsampling/{split}/{video}.zip"}
        with ExitStack() as stack:
            archives = {}
            for asset, url in assets.items():
                path = tmp / (asset + ".zip")
                download(url, path)
                archives[asset] = stack.enter_context(ZipFile(path))
            paired = archives["paired"]
            rgb, low, truth = [frames(paired, key, ".png")
                               for key in ("wide", "lowres_depth", "highres_depth")]
            ultra = frames(archives["ultrawide"], "ultrawide", ".png")
            uk = frames(archives["ultrawide_intrinsics"], "ultrawide_intrinsics", ".pincam")
            mk = frames(archives["lowres_wide_intrinsics"], "lowres_wide_intrinsics", ".pincam")
            times, main_times = sorted(ultra.keys() & uk.keys()), sorted(mk)
            if not times:
                return
            for t in sorted(rgb.keys() & low.keys() & truth.keys()):
                u, m = nearest(times, t), nearest(main_times, t)
                if abs(u - t) > 0.050001 or abs(m - t) > 0.017001:
                    continue
                key = video + "_" + Path(rgb[t]).stem.rsplit("_", 1)[-1]
                if key in completed:
                    continue
                main = calibration(archives["lowres_wide_intrinsics"], mk[m])
                wide = calibration(archives["ultrawide_intrinsics"], uk[u])
                stage = tmp / "sample"
                stage.mkdir()
                for index, filename in ((rgb, "main.png"), (low, "lidar.png"), (truth, "ground_truth.png")):
                    with paired.open(index[t]) as src, (stage / filename).open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                with archives["ultrawide"].open(ultra[u]) as src, (stage / "ultrawide.png").open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                meta = dict(video_id=video, visit_id=row["visit_id"], split=split,
                            timestamp_s=t, ultrawide_timestamp_s=u, offset_ms=round((u-t)*1000, 3),
                            main_lowres_intrinsics=main,
                            main_intrinsics_timestamp_s=m, ultrawide_intrinsics=wide)
                (stage / "sample.json").write_text(json.dumps(meta) + "\n")
                stage.rename(root / key)
                completed.add(key)
                if len(completed) == needed:
                    return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--split", choices=["Training", "Validation"], default="Training")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    root = args.output
    root.mkdir(exist_ok=True, parents=True)
    completed = {f.parent.name for f in root.glob("*/sample.json")
                 if json.loads(f.read_text())["split"] == args.split}
    if len(completed) >= args.count:
        print(f"Already have {len(completed)} samples in {root}")
        return
    with urlopen(SPLITS, timeout=60) as response:
        rows = list(csv.DictReader(TextIOWrapper(response, encoding="utf-8-sig")))
    for row in rows:
        if row["fold"] != args.split:
            continue
        try:
            download_video(row, root, args.count, completed)
        except subprocess.CalledProcessError as exc:
            if exc.returncode != 22:
                raise
            print(f"Skipping unavailable archive for {row['video_id']}", flush=True)
        print(f"{len(completed)}/{args.count} samples", flush=True)
        if len(completed) == args.count:
            return
    raise SystemExit(f"Only {len(completed)} complete samples available in {root}")


if __name__ == "__main__":
    main()
