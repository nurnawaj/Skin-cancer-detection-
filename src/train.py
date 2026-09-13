"""
train.py
--------
End-to-end training pipeline for the skin-lesion classifier.

Usage:
    python src/train.py
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config as cfg
from src.dataset import HAM10000Dataset, get_transforms, build_class_mapping
from src.model import build_model
from src.utils import seed_everything, print_device_info, EarlyStopping, save_checkpoint

import pandas as pd


def compute_class_weights(train_csv: str, class_to_idx: dict) -> torch.Tensor:
    """
    Compute class weights from the TRAINING split only (never val/test),
    so that under-represented diagnostic classes are not ignored by the
    loss function.
    """
    df = pd.read_csv(train_csv)
    labels = df[cfg.LABEL_COLUMN].map(class_to_idx).values
    classes = np.array(sorted(class_to_idx.values()))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    """Run one training or validation epoch. Returns (loss, accuracy, macro_f1)."""
    model.train() if train else model.eval()

    running_loss = 0.0
    all_preds, all_labels = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        loop = tqdm(loader, desc="Train" if train else "Val", leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    epoch_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return epoch_loss, epoch_acc, epoch_f1


def plot_training_history(history: dict, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history["train_loss"], label="Train Loss", color="#3B82F6")
    axes[0].plot(history["val_loss"], label="Validation Loss", color="#F97316")
    axes[0].set_title("Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train Accuracy", color="#3B82F6")
    axes[1].plot(history["val_acc"], label="Validation Accuracy", color="#F97316")
    axes[1].set_title("Accuracy over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    seed_everything(cfg.RANDOM_SEED)
    device = cfg.DEVICE
    print_device_info(device)

    for path in [cfg.TRAIN_CSV, cfg.VAL_CSV]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"'{path}' not found. Run 'python src/prepare_data.py' first."
            )

    # Build class mapping from the combined train+val+test metadata so it
    # is consistent regardless of which split a class happens to appear in.
    all_rows = pd.concat(
        [pd.read_csv(cfg.TRAIN_CSV), pd.read_csv(cfg.VAL_CSV), pd.read_csv(cfg.TEST_CSV)],
        ignore_index=True,
    )
    class_to_idx = build_class_mapping(all_rows)
    print(f"Classes ({len(class_to_idx)}): {class_to_idx}")

    train_dataset = HAM10000Dataset(
        cfg.TRAIN_CSV, cfg.IMAGES_PATH, class_to_idx, transform=get_transforms(train=True)
    )
    val_dataset = HAM10000Dataset(
        cfg.VAL_CSV, cfg.IMAGES_PATH, class_to_idx, transform=get_transforms(train=False)
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Training or validation dataset is empty. Check your data split.")

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )

    model = build_model(cfg.MODEL_NAME, num_classes=len(class_to_idx), pretrained=cfg.PRETRAINED)
    model.to(device)

    if cfg.USE_CLASS_WEIGHTS:
        class_weights = compute_class_weights(cfg.TRAIN_CSV, class_to_idx).to(device)
        print(f"Using class weights: {class_weights.tolist()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.LR_SCHEDULER_FACTOR, patience=cfg.LR_SCHEDULER_PATIENCE
    )

    early_stopping = EarlyStopping(
        patience=cfg.PATIENCE,
        mode="min" if cfg.BEST_METRIC == "val_loss" else "max",
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
    best_metric_value = float("inf") if cfg.BEST_METRIC == "val_loss" else -float("inf")

    print(f"\nStarting training for up to {cfg.EPOCHS} epochs "
          f"(early stopping patience={cfg.PATIENCE}, best metric={cfg.BEST_METRIC})\n")

    for epoch in range(1, cfg.EPOCHS + 1):
        start = time.time()

        train_loss, train_acc, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        elapsed = time.time() - start
        print(
            f"Epoch {epoch:02d}/{cfg.EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} | "
            f"lr={current_lr:.2e} | {elapsed:.1f}s"
        )

        current_metric = val_loss if cfg.BEST_METRIC == "val_loss" else val_f1
        is_better = (
            (cfg.BEST_METRIC == "val_loss" and current_metric < best_metric_value)
            or (cfg.BEST_METRIC == "val_f1" and current_metric > best_metric_value)
        )

        if is_better:
            best_metric_value = current_metric
            save_checkpoint(
                path=cfg.MODEL_SAVE_PATH,
                model_state_dict=model.state_dict(),
                class_to_idx=class_to_idx,
                model_name=cfg.MODEL_NAME,
                image_size=cfg.IMAGE_SIZE,
                mean=cfg.IMAGENET_MEAN,
                std=cfg.IMAGENET_STD,
                config={
                    "batch_size": cfg.BATCH_SIZE,
                    "epochs": cfg.EPOCHS,
                    "learning_rate": cfg.LEARNING_RATE,
                    "random_seed": cfg.RANDOM_SEED,
                },
                best_metric_name=cfg.BEST_METRIC,
                best_metric_value=float(best_metric_value),
            )
            print(f"  -> New best model saved ({cfg.BEST_METRIC}={best_metric_value:.4f})")

        stop_metric = val_loss if cfg.BEST_METRIC == "val_loss" else val_f1
        if early_stopping.step(stop_metric):
            print(f"\nEarly stopping triggered after epoch {epoch} (no improvement for {cfg.PATIENCE} epochs).")
            break

    plot_training_history(history, cfg.TRAINING_HISTORY_PLOT)
    print(f"\nTraining complete. Best {cfg.BEST_METRIC}: {best_metric_value:.4f}")
    print(f"Best model saved to: {cfg.MODEL_SAVE_PATH}")
    print(f"Training curves saved to: {cfg.TRAINING_HISTORY_PLOT}")


if __name__ == "__main__":
    main()
