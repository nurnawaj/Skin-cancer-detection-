"""
model.py
--------
Modular model-building code. Currently implements ResNet18 transfer
learning, with the architecture kept modular so that ResNet50 or
EfficientNet-B0 can be dropped in later just by adding a branch to
`build_model` — no other file needs to change.
"""

from typing import Dict

import torch.nn as nn
from torchvision import models


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Build a classification model for `num_classes` output classes.

    Currently supported:
      - "resnet18"

    Reserved for future use (not yet implemented, kept here so the
    project structure documents the intended extension points):
      - "resnet50"
      - "efficientnet_b0"
    """
    model_name = model_name.lower()

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    else:
        raise ValueError(
            f"Unsupported MODEL_NAME '{model_name}'. "
            f"Supported options: resnet18, resnet50, efficientnet_b0."
        )


def get_model_summary(model_name: str, class_to_idx: Dict[str, int]) -> Dict:
    """Small helper used by the Streamlit app to display model info."""
    return {
        "model_name": model_name,
        "num_classes": len(class_to_idx),
        "classes": sorted(class_to_idx.keys()),
    }
