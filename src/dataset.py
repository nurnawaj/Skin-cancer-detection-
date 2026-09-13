"""
dataset.py
----------
PyTorch Dataset definition for HAM10000-style skin-lesion data, plus the
image transform pipelines used for training and evaluation.

Classes are NEVER hard-coded here: they are detected dynamically from the
metadata CSV's label column, so this code also works on a differently
structured (but similarly formatted) future dataset.
"""

import os
from typing import Callable, Dict, Optional, Tuple

import pandas as pd
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config as cfg


def build_class_mapping(metadata: pd.DataFrame) -> Dict[str, int]:
    """
    Detect the set of classes present in the metadata automatically and
    build a deterministic (alphabetically sorted) class_to_idx mapping.
    """
    classes = sorted(metadata[cfg.LABEL_COLUMN].dropna().unique().tolist())
    if len(classes) == 0:
        raise ValueError(
            f"No classes found in column '{cfg.LABEL_COLUMN}' of the metadata file."
        )
    return {cls_name: idx for idx, cls_name in enumerate(classes)}


def get_transforms(train: bool, image_size: int = cfg.IMAGE_SIZE) -> transforms.Compose:
    """
    Build the torchvision transform pipeline.

    train=True  -> resize + augmentation + normalize
    train=False -> resize + center-crop + normalize (for val/test/inference)
    """
    if train and cfg.USE_AUGMENTATION:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(
                    brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=cfg.IMAGENET_MEAN, std=cfg.IMAGENET_STD),
            ]
        )
    else:
        # Slightly upscale then center-crop, matching the requested image_size,
        # which is the standard approach for validation/test/inference.
        resize_to = int(image_size * 1.15)
        return transforms.Compose(
            [
                transforms.Resize((resize_to, resize_to)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=cfg.IMAGENET_MEAN, std=cfg.IMAGENET_STD),
            ]
        )


class HAM10000Dataset(Dataset):
    """
    Loads (image, label) pairs from a metadata CSV (with at least
    `image_id` and `dx` columns) and an image directory.
    """

    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        class_to_idx: Dict[str, int],
        transform: Optional[Callable] = None,
    ):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Split file not found: '{csv_path}'. "
                f"Run 'python src/prepare_data.py' first."
            )
        self.metadata = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.metadata)

    def _resolve_image_path(self, image_id: str) -> str:
        # Support both flat layout and the ISIC "part_1/part_2" layout.
        candidate = os.path.join(self.images_dir, image_id + cfg.IMAGE_EXTENSION)
        if os.path.exists(candidate):
            return candidate
        # Fallback: search subdirectories (helps with alternate HAM10000 zip layouts).
        for root, _, files in os.walk(self.images_dir):
            fname = image_id + cfg.IMAGE_EXTENSION
            if fname in files:
                return os.path.join(root, fname)
        raise FileNotFoundError(f"Image file for id '{image_id}' not found under {self.images_dir}")

    def __getitem__(self, idx: int) -> Tuple:
        row = self.metadata.iloc[idx]
        image_id = str(row[cfg.IMAGE_ID_COLUMN])
        label_name = row[cfg.LABEL_COLUMN]
        label = self.class_to_idx[label_name]

        image_path = self._resolve_image_path(image_id)
        try:
            image = Image.open(image_path).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(f"Could not read image '{image_path}': {exc}")

        if self.transform is not None:
            image = self.transform(image)

        return image, label
