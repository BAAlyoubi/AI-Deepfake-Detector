"""Train the MobileNetV2-CBAM-LSTM deepfake detector."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

from .data import make_loader, validate_class_names
from .metrics import compute_metrics
from .model import build_model
from .utils import count_parameters, load_config, resolve_device, save_json, set_seed


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_targets = []
    all_preds = []

    for images, targets in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=scaler is not None):
                logits = model(images)
                loss = criterion(logits, targets)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss += loss.item() * images.size(0)
        all_targets.append(targets.detach().cpu())
        all_preds.append(torch.argmax(logits.detach().cpu(), dim=1))

    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()
    metrics = compute_metrics(y_true, y_pred)
    return total_loss / len(loader.dataset), metrics


def train(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(int(cfg["seed"]))
    device = resolve_device(cfg["device"])
    output_dir = Path(cfg["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = make_loader(
        cfg["data"]["train_dir"],
        cfg["data"]["image_size"],
        cfg["data"]["batch_size"],
        cfg["data"]["num_workers"],
        train=True,
        class_names=cfg["data"]["class_names"],
    )
    val_loader = make_loader(
        cfg["data"]["val_dir"],
        cfg["data"]["image_size"],
        cfg["data"]["batch_size"],
        cfg["data"]["num_workers"],
        train=False,
        class_names=cfg["data"]["class_names"],
    )
    validate_class_names(train_loader.dataset.classes, cfg["data"]["class_names"])
    validate_class_names(val_loader.dataset.classes, cfg["data"]["class_names"])

    model = build_model(**cfg["model"]).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg["train"]["label_smoothing"]))
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    use_amp = bool(cfg["train"]["mixed_precision"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=True) if use_amp else None

    best_f1 = -1.0
    best_epoch = 0
    history = []
    patience = int(cfg["train"]["patience"])

    print(f"Device: {device}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train": train_metrics.__dict__,
            "val": val_metrics.__dict__,
        }
        history.append(row)
        save_json({"history": history}, output_dir / "history.json")

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_metrics.accuracy:.4f} val_f1={val_metrics.f1:.4f}"
        )

        if val_metrics.f1 > best_f1:
            best_f1 = val_metrics.f1
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "class_names": train_loader.dataset.classes,
                    "best_epoch": best_epoch,
                    "best_f1": best_f1,
                },
                output_dir / "best_model.pt",
            )

        if epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.config)
