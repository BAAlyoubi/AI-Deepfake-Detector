"""Dataset and dataloader helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class OrderedImageFolder(datasets.ImageFolder):
    """ImageFolder variant that keeps class indices aligned with the config."""

    def __init__(self, root: str | Path, class_names: Sequence[str], transform=None) -> None:
        self.expected_class_names = list(class_names)
        super().__init__(root=root, transform=transform)

    def find_classes(self, directory: str):
        available = {path.name for path in Path(directory).iterdir() if path.is_dir()}
        missing = [name for name in self.expected_class_names if name not in available]
        if missing:
            raise FileNotFoundError(f"Missing class folders in {directory}: {missing}")
        return self.expected_class_names, {name: index for index, name in enumerate(self.expected_class_names)}


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def make_image_folder(
    root: str | Path,
    image_size: int,
    train: bool,
    class_names: Sequence[str],
) -> datasets.ImageFolder:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    return OrderedImageFolder(root=root, class_names=class_names, transform=build_transforms(image_size, train))


def make_loader(
    root: str | Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    train: bool,
    class_names: Sequence[str],
) -> DataLoader:
    dataset = make_image_folder(root, image_size=image_size, train=train, class_names=class_names)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
    )


def validate_class_names(found: Iterable[str], expected: Iterable[str]) -> None:
    found_list = list(found)
    expected_list = list(expected)
    if found_list != expected_list:
        raise ValueError(
            "Dataset class folders must match config class_names. "
            f"Found {found_list}, expected {expected_list}."
        )
