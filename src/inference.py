"""
inference.py
------------
Shared, reusable inference logic used by both `src/predict.py` (CLI) and
`app.py` (Streamlit). Keeping this in one place guarantees the CLI and
the web app always produce identical predictions.
"""

import os
import sys
from typing import Dict, List, Tuple

import torch
from PIL import Image, UnidentifiedImageError

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config as cfg
from src.dataset import get_transforms
from src.model import build_model
from src.utils import load_checkpoint

MAX_UPLOAD_SIZE_MB = 15
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class ModelBundle:
    """Holds everything needed to run inference: model + metadata."""

    def __init__(self, model, class_to_idx, idx_to_class, image_size, mean, std,
                 model_name, best_metric_name, best_metric_value, device):
        self.model = model
        self.class_to_idx = class_to_idx
        self.idx_to_class = idx_to_class
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.model_name = model_name
        self.best_metric_name = best_metric_name
        self.best_metric_value = best_metric_value
        self.device = device


def load_model_bundle(checkpoint_path: str = cfg.MODEL_SAVE_PATH, device: torch.device = None) -> ModelBundle:
    """Load the trained checkpoint and rebuild the model, ready for inference."""
    if device is None:
        device = cfg.DEVICE

    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = checkpoint["idx_to_class"]

    model = build_model(checkpoint["model_name"], num_classes=len(class_to_idx), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return ModelBundle(
        model=model,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        image_size=checkpoint["image_size"],
        mean=checkpoint["mean"],
        std=checkpoint["std"],
        model_name=checkpoint["model_name"],
        best_metric_name=checkpoint.get("best_metric_name", "n/a"),
        best_metric_value=checkpoint.get("best_metric_value", None),
        device=device,
    )


def validate_image_file(file_path: str) -> None:
    """Raise a friendly, descriptive error if the file is not a usable image."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{ext}'. Please upload a JPG, JPEG, or PNG image."
        )

    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        raise ValueError(
            f"Image is too large ({size_mb:.1f} MB). Please upload a file under {MAX_UPLOAD_SIZE_MB} MB."
        )

    try:
        with Image.open(file_path) as img:
            img.verify()
    except (UnidentifiedImageError, OSError):
        raise ValueError("The uploaded file could not be read as a valid image. It may be corrupted.")


def preprocess_image(image: Image.Image, bundle: ModelBundle):
    image = image.convert("RGB")
    transform = get_transforms(train=False, image_size=bundle.image_size)
    tensor = transform(image).unsqueeze(0)  # add batch dimension
    return tensor


def predict_image(image: Image.Image, bundle: ModelBundle, top_k: int = 5) -> Tuple[str, float, List[Dict]]:
    """
    Run inference on a single PIL image.

    Returns:
        predicted_class (str), confidence (float, 0-1), top_predictions (list of
        {"class": str, "full_name": str, "confidence": float}, sorted descending)
    """
    from src.utils import get_full_class_name

    tensor = preprocess_image(image, bundle).to(bundle.device)

    with torch.no_grad():
        outputs = bundle.model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

    top_k = min(top_k, len(probs))
    top_indices = probs.argsort()[::-1][:top_k]

    top_predictions = [
        {
            "class": bundle.idx_to_class[int(i)],
            "full_name": get_full_class_name(bundle.idx_to_class[int(i)]),
            "confidence": float(probs[i]),
        }
        for i in top_indices
    ]

    predicted_class = top_predictions[0]["class"]
    confidence = top_predictions[0]["confidence"]

    return predicted_class, confidence, top_predictions
