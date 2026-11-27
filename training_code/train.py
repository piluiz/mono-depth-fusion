"""Train the fusion U-Net with python training_code/train.py."""

EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 2e-4
LR_PATIENCE = 2
LR_FACTOR = 0.5
MODEL_KWARGS = {"deep_channels": 56, "bottleneck_channels": 80}
IMAGE_SIZE = (320, 240)  # width, height
DEVICE = "auto"

import argparse
from datetime import datetime
from pathlib import Path
import sys

import torch
import yaml
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.dataset import DepthFusionDataset
from training_code.Unet import FusionUNet


INPUT_KEYS = FusionUNet.input_keys


def select_device(requested):
    available = {"mps": torch.backends.mps.is_available, "cuda": torch.cuda.is_available}
    if requested == "auto":
        for name, is_available in available.items():
            if is_available():
                return torch.device(name)
        return torch.device("cpu")
    if requested in available and not available[requested]():
        raise ValueError(f"Requested device {requested} is unavailable. Use --device auto for fallback.")
    return torch.device(requested)


def run_epoch(model, loader, device, optimizer=None):
    """Return mean absolute error in meters over valid FARO pixels."""
    training = optimizer is not None
    model.train(training)
    total_error = 0.0
    total_pixels = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            mask = batch["gt_valid"]
            count = int(mask.sum().item())
            if count == 0:
                continue

            inputs = torch.cat([batch[key] for key in INPUT_KEYS], dim=1).to(device)
            target = batch["ground_truth"].to(device)
            prediction = model(inputs)
            error = ((prediction - target).abs() * mask.to(device)).sum()
            if training:
                optimizer.zero_grad(set_to_none=True)
                (error / count).backward()
                optimizer.step()
            total_error += error.item()
            total_pixels += count

    if total_pixels == 0:
        raise ValueError("Split contains no valid ground-truth pixels")
    return total_error / total_pixels


def create_run_folder(parent):
    """Reserve a folder named for the local start time of this run."""
    parent.mkdir(parents=True, exist_ok=True)
    while True:
        started = datetime.now().astimezone()
        folder = parent / started.strftime("%Y-%m-%d_%H-%M-%S.%f%z")
        try:
            folder.mkdir()
        except FileExistsError:
            continue
        return folder


def save_loss_curve(run_dir, train_losses, val_losses):
    """Refresh the run's loss plot after each completed epoch."""
    epochs = range(1, len(train_losses) + 1)
    figure = Figure(figsize=(7, 4), layout="constrained")
    ax = figure.subplots()
    marker = "o" if len(train_losses) == 1 else None
    ax.plot(epochs, train_losses, marker=marker, label="Training")
    ax.plot(epochs, val_losses, marker=marker, label="Validation")
    ax.set(xlabel="Epoch", ylabel="Masked mean absolute error (m)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if len(train_losses) == 1:
        ax.set_xticks([1])
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    pending = run_dir / "loss_curve.png.tmp"
    figure.savefig(pending, format="png", dpi=140)
    pending.replace(run_dir / "loss_curve.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default=DEVICE,
                        help="auto selects MPS, then CUDA, then CPU.")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "training_code" / "checkpoints" / "best_unet.pt",
                        help="Parent directory and filename; a timestamped run folder is inserted between them.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch size, and learning rate must be positive")
    if args.checkpoint.name.casefold() in ("", "..", "params.yaml", "loss_curve.png", "loss_curve.png.tmp"):
        parser.error("--checkpoint requires a model filename distinct from the run configuration and loss plot")

    try:
        device = select_device(args.device)
    except ValueError as exc:
        parser.error(str(exc))
    train_data = DepthFusionDataset("train", size=IMAGE_SIZE, augment=True, cache=False)
    val_data = DepthFusionDataset("val", size=IMAGE_SIZE, cache=False)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    model = FusionUNet(**MODEL_KWARGS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE, threshold=0.0)
    run_dir = create_run_folder(args.checkpoint.parent)
    checkpoint = run_dir / args.checkpoint.name
    params = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "lr_scheduler": type(scheduler).__name__,
        "lr_factor": LR_FACTOR,
        "lr_patience": LR_PATIENCE,
        "device": str(device),
        "model": type(model).__name__,
        "model_kwargs": MODEL_KWARGS,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "optimizer": type(optimizer).__name__,
        "loss": "masked_mae_m",
        "input_keys": list(INPUT_KEYS),
        "image_size_wh": list(train_data.size),
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "checkpoint": checkpoint.name,
    }
    with (run_dir / "params.yaml").open("w", encoding="utf-8") as config:
        yaml.safe_dump(params, config, sort_keys=False)
    print(f"Run folder: {run_dir}", flush=True)
    print(f"Training on {device} with {len(train_data)} training and "
          f"{len(val_data)} validation samples", flush=True)

    train_losses = []
    val_losses = []
    best_val = float("inf")
    best_epoch = 0
    with tqdm(total=args.epochs, desc="Training", unit="epoch", dynamic_ncols=True) as progress:
        for epoch in range(1, args.epochs + 1):
            epoch_lr = optimizer.param_groups[0]["lr"]
            train_mae = run_epoch(model, train_loader, device, optimizer)
            val_mae = run_epoch(model, val_loader, device)
            scheduler.step(val_mae)
            if val_mae < best_val:
                best_val = val_mae
                best_epoch = epoch
                pending = checkpoint.with_name(checkpoint.name + ".tmp")
                torch.save({"model": type(model).__name__, "model_kwargs": MODEL_KWARGS,
                            "model_state_dict": model.state_dict(), "epoch": epoch,
                            "val_mae_m": val_mae, "input_keys": INPUT_KEYS}, pending)
                pending.replace(checkpoint)
                progress.write(f"Saved {checkpoint}")
            train_losses.append(train_mae)
            val_losses.append(val_mae)
            save_loss_curve(run_dir, train_losses, val_losses)
            progress.set_postfix_str(
                f"train MAE {train_mae:.5f} m, val MAE {val_mae:.5f} m, lr {epoch_lr:.1e}", refresh=False)
            progress.update(1)
    if best_epoch:
        print(f"Best validation MAE {best_val:.5f} m at epoch {best_epoch}: {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
