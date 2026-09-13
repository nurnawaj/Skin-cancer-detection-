"""
predict.py
----------
Command-line single-image prediction script.

Usage:
    python src/predict.py --image path/to/image.jpg

This is a research/educational tool. It does NOT provide a medical
diagnosis.
"""

import argparse
import os
import sys

from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config as cfg
from src.inference import load_model_bundle, predict_image, validate_image_file
from src.utils import print_device_info


def main():
    parser = argparse.ArgumentParser(description="Run skin-lesion prediction on a single image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image (jpg/jpeg/png).")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top predictions to show.")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: file not found: '{args.image}'")
        sys.exit(1)

    try:
        validate_image_file(args.image)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    print_device_info(cfg.DEVICE)

    try:
        bundle = load_model_bundle()
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    image = Image.open(args.image)
    predicted_class, confidence, top_predictions = predict_image(image, bundle, top_k=args.top_k)

    print(f"\nPrediction:\n{predicted_class}")
    print(f"\nConfidence:\n{confidence * 100:.2f}%")
    print(f"\nTop predictions:")
    for rank, pred in enumerate(top_predictions, start=1):
        print(f"{rank}. {pred['class']} ({pred['full_name']}) — {pred['confidence'] * 100:.2f}%")

    print(
        "\n⚠️  This is a research/educational tool and NOT a medical diagnosis. "
        "Consult a qualified dermatologist for any health concerns."
    )


if __name__ == "__main__":
    main()
