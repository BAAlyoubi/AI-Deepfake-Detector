"""Evaluate a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from .data import make_loader
from .metrics import compute_metrics
from .model import build_model
from .utils import resolve_device, save_json


@torch.inference_mode()
def evaluate(checkpoint_path: str, data_dir: str, output_path: str | None = None) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = checkpoint["config"]
    device = resolve_device(cfg["device"])

    loader = make_loader(
        data_dir,
        cfg["data"]["image_size"],
        cfg["data"]["batch_size"],
        cfg["data"]["num_workers"],
        train=False,
        class_names=cfg["data"]["class_names"],
    )
    model = build_model(**cfg["model"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    criterion = nn.CrossEntropyLoss()
    all_targets = []
    all_preds = []
    total_loss = 0.0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += loss.item() * images.size(0)
        all_targets.append(targets.cpu())
        all_preds.append(torch.argmax(logits.cpu(), dim=1))

    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()
    metrics = compute_metrics(y_true, y_pred)
    payload = {"loss": total_loss / len(loader.dataset), "metrics": metrics.__dict__}
    print(payload)

    if output_path:
        save_json(payload, Path(output_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt.")
    parser.add_argument("--data-dir", required=True, help="ImageFolder directory to evaluate.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.checkpoint, args.data_dir, args.output)
