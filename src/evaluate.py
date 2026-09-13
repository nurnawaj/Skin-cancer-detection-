"""
evaluate.py
-----------
Loads the best saved checkpoint and evaluates it on the held-out TEST
set only. All metrics reported here are computed directly from actual
model predictions — nothing is fabricated or hard-coded.

Usage:
    python src/evaluate.py
"""

import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config as cfg
from src.dataset import HAM10000Dataset, get_transforms
from src.model import build_model
from src.utils import load_checkpoint, print_device_info, save_json


def evaluate():
    device = cfg.DEVICE
    print_device_info(device)

    if not os.path.exists(cfg.TEST_CSV):
        raise FileNotFoundError(
            f"'{cfg.TEST_CSV}' not found. Run 'python src/prepare_data.py' first."
        )

    checkpoint = load_checkpoint(cfg.MODEL_SAVE_PATH, map_location=device)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = checkpoint["idx_to_class"]
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    model = build_model(checkpoint["model_name"], num_classes=len(class_to_idx), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    test_dataset = HAM10000Dataset(
        cfg.TEST_CSV, cfg.IMAGES_PATH, class_to_idx,
        transform=get_transforms(train=False, image_size=checkpoint["image_size"]),
    )
    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty. Check your data split.")

    test_loader = DataLoader(
        test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS
    )

    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # ---------------- Core metrics ----------------
    accuracy = accuracy_score(all_labels, all_preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    report_text = classification_report(
        all_labels, all_preds, target_names=class_names, zero_division=0
    )
    os.makedirs(cfg.RESULTS_PATH, exist_ok=True)
    with open(cfg.CLASSIFICATION_REPORT_FILE, "w") as f:
        f.write(report_text)

    # ---------------- Confusion matrix ----------------
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig(cfg.CONFUSION_MATRIX_PLOT, dpi=150)
    plt.close()

    # ---------------- ROC-AUC (one-vs-rest) ----------------
    roc_auc_macro = None
    try:
        n_classes = len(class_names)
        labels_binarized = label_binarize(all_labels, classes=list(range(n_classes)))
        # Only compute if every class has at least one positive AND one negative sample.
        valid_classes = [
            i for i in range(n_classes)
            if 0 < labels_binarized[:, i].sum() < len(labels_binarized)
        ]
        if len(valid_classes) >= 2:
            roc_auc_macro = roc_auc_score(
                labels_binarized[:, valid_classes], all_probs[:, valid_classes],
                average="macro", multi_class="ovr",
            )

            plt.figure(figsize=(7, 6))
            for i in valid_classes:
                fpr, tpr, _ = roc_curve(labels_binarized[:, i], all_probs[:, i])
                roc_auc_i = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc_i:.2f})")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curves (One-vs-Rest) — Test Set")
            plt.legend(loc="lower right", fontsize=8)
            plt.tight_layout()
            plt.savefig(cfg.ROC_CURVE_PLOT, dpi=150)
            plt.close()
        else:
            print("Skipping ROC-AUC: not enough classes with both positive and negative samples.")
    except Exception as exc:
        print(f"Could not compute ROC-AUC: {exc}")

    # ---------------- Save metrics.json ----------------
    metrics = {
        "test_accuracy": float(accuracy),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "roc_auc_macro_ovr": float(roc_auc_macro) if roc_auc_macro is not None else None,
        "num_test_samples": int(len(all_labels)),
        "class_names": class_names,
    }
    save_json(metrics, cfg.METRICS_FILE)

    print("\n===== Test Set Evaluation =====")
    print(f"Accuracy:            {accuracy:.4f}")
    print(f"Macro Precision:     {macro_p:.4f}")
    print(f"Macro Recall:        {macro_r:.4f}")
    print(f"Macro F1:            {macro_f1:.4f}")
    print(f"Weighted Precision:  {weighted_p:.4f}")
    print(f"Weighted Recall:     {weighted_r:.4f}")
    print(f"Weighted F1:         {weighted_f1:.4f}")
    if roc_auc_macro is not None:
        print(f"Macro ROC-AUC (OvR): {roc_auc_macro:.4f}")
    print(f"\nClassification report saved to: {cfg.CLASSIFICATION_REPORT_FILE}")
    print(f"Confusion matrix saved to:      {cfg.CONFUSION_MATRIX_PLOT}")
    print(f"Metrics saved to:               {cfg.METRICS_FILE}")


if __name__ == "__main__":
    evaluate()
