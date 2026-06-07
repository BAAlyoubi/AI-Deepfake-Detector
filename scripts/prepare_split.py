"""Create an 80/20 train/validation split from class folders.

Expected input:
    source_dir/real/*.jpg
    source_dir/fake/*.jpg

Output:
    output_dir/train/real
    output_dir/train/fake
    output_dir/val/real
    output_dir/val/fake
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def split_dataset(source_dir: str, output_dir: str, val_ratio: float, seed: int, copy: bool) -> None:
    source = Path(source_dir)
    output = Path(output_dir)
    rng = random.Random(seed)

    for class_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        images = [path for path in class_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
        rng.shuffle(images)
        val_count = int(round(len(images) * val_ratio))
        splits = {"val": images[:val_count], "train": images[val_count:]}

        for split, paths in splits.items():
            target_dir = output / split / class_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            for image_path in paths:
                target_path = target_dir / image_path.name
                if copy:
                    shutil.copy2(image_path, target_path)
                else:
                    shutil.move(str(image_path), str(target_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of moving them.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split_dataset(args.source_dir, args.output_dir, args.val_ratio, args.seed, args.copy)
