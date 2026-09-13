"""
utils.py
--------
Shared helper utilities used across the training, evaluation and
prediction pipelines: reproducibility seeding, early stopping, and
checkpoint save/load helpers.
"""

import os
import random
import json
from typing import Any, Dict, Optional

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """
    Set random seeds across all libraries used in this project so that
    results are as reproducible as possible.

    Note: full bit-for-bit determinism on GPU is not guaranteed because
    some CUDA/cuDNN kernels are inherently non-deterministic for speed
    reasons. This function minimizes randomness but does not eliminate
    all GPU-related variance.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def print_device_info(device: torch.device) -> None:
    """Print a friendly message describing which compute device is in use."""
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        print(f"Using device: CUDA ({name})")
    else:
        print("Using device: CPU")


class EarlyStopping:
    """
    Stops training when a monitored metric has not improved for a given
    number of epochs ("patience").

    By default assumes a LOWER value is better (e.g. validation loss).
    Set `mode="max"` for metrics where higher is better (e.g. macro-F1).
    """

    def __init__(self, patience: int = 5, mode: str = "min", min_delta: float = 0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, current_value: float) -> bool:
        """
        Update the early stopping state with the latest metric value.
        Returns True if training should stop.
        """
        if self.best_score is None:
            self.best_score = current_value
            return False

        improved = (
            (self.mode == "min" and current_value < self.best_score - self.min_delta)
            or (self.mode == "max" and current_value > self.best_score + self.min_delta)
        )

        if improved:
            self.best_score = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def save_checkpoint(
    path: str,
    model_state_dict: Dict[str, Any],
    class_to_idx: Dict[str, int],
    model_name: str,
    image_size: int,
    mean: list,
    std: list,
    config: Dict[str, Any],
    best_metric_name: str,
    best_metric_value: float,
) -> None:
    """
    Save a full, self-describing checkpoint (not just raw weights) so that
    inference can be performed later without needing the original training
    configuration in memory.
    """
    checkpoint = {
        "model_state_dict": model_state_dict,
        "class_to_idx": class_to_idx,
        "idx_to_class": {v: k for k, v in class_to_idx.items()},
        "model_name": model_name,
        "image_size": image_size,
        "mean": mean,
        "std": std,
        "config": config,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: str, map_location=None) -> Dict[str, Any]:
    """Load a checkpoint dictionary saved by `save_checkpoint`."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trained model found at '{path}'. "
            f"Please run 'python src/train.py' first."
        )
    return torch.load(path, map_location=map_location, weights_only=False)


def save_json(data: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


FULL_CLASS_NAMES = {
    "akiec": "Actinic Keratoses / Intraepithelial Carcinoma",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesion",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesion",
}


def get_full_class_name(short_name: str) -> str:
    """Return a human-readable class name, falling back to the raw label."""
    return FULL_CLASS_NAMES.get(short_name, short_name)
