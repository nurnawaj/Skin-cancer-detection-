"""
gradcam.py
----------
Optional Grad-CAM interpretability module for the ResNet-family models
used in this project. Highlights which regions of an image most
influenced the model's prediction.

This is an INTERPRETABILITY VISUALIZATION ONLY. It is not proof of
correct medical reasoning and must not be treated as diagnostic evidence.

Kept fully separate/optional from the main app and prediction pipeline
so that if it fails for any reason, the rest of DermaAI keeps working.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """
    Minimal Grad-CAM implementation targeting the last convolutional
    block of a ResNet-style model (`model.layer4`).
    """

    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer if target_layer is not None else self._find_default_layer()

        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _find_default_layer(self):
        if hasattr(self.model, "layer4"):
            return self.model.layer4
        raise ValueError(
            "Could not auto-detect a target layer for Grad-CAM. "
            "Pass `target_layer` explicitly for non-ResNet architectures."
        )

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for `input_tensor` (shape [1, C, H, W]).
        Returns a 2D numpy array normalized to [0, 1].
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(torch.argmax(output, dim=1).item())

        score = output[0, class_idx]
        score.backward()

        # Global-average-pool the gradients to get per-channel importance weights.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


def overlay_heatmap(original_image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """
    Resize the Grad-CAM heatmap to match the original image and blend it
    on top using a simple color map, returning a new PIL image.
    """
    import matplotlib.cm as cm

    original_image = original_image.convert("RGB")
    width, height = original_image.size

    cam_img = Image.fromarray(np.uint8(cam * 255)).resize((width, height), resample=Image.BILINEAR)
    cam_array = np.array(cam_img) / 255.0

    colormap = cm.get_cmap("jet")
    heatmap_rgba = colormap(cam_array)
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap_rgb).convert("RGB")

    blended = Image.blend(original_image, heatmap_img, alpha=alpha)
    return blended
