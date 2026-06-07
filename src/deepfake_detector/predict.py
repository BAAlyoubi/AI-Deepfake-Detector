"""Run inference on one image."""

from __future__ import annotations

import argparse

import torch
from PIL import Image

from .data import build_transforms
from .model import build_model
from .utils import resolve_device


@torch.inference_mode()
def predict(checkpoint_path: str, image_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = checkpoint["config"]
    class_names = checkpoint.get("class_names", cfg["data"]["class_names"])
    device = resolve_device(cfg["device"])

    model = build_model(**cfg["model"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    transform = build_transforms(cfg["data"]["image_size"], train=False)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    probabilities = torch.softmax(model(tensor), dim=1).squeeze(0).cpu()
    predicted_index = int(torch.argmax(probabilities))

    print(
        {
            "image": image_path,
            "prediction": class_names[predicted_index],
            "confidence": float(probabilities[predicted_index]),
            "probabilities": {
                class_name: float(probabilities[index]) for index, class_name in enumerate(class_names)
            },
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt.")
    parser.add_argument("--image", required=True, help="Image path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(args.checkpoint, args.image)
