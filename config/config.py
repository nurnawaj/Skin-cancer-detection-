"""
config.py
---------
Single source of truth for all configurable values in the DermaAI project.
Every other module imports its settings from here so that changing a
hyperparameter or a path only requires editing this one file.
"""

import os
import torch

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(BASE_DIR, "dataset")
METADATA_PATH = os.path.join(DATASET_PATH, "HAM10000_metadata.csv")
IMAGES_PATH = os.path.join(DATASET_PATH, "HAM10000_images")

TRAIN_CSV = os.path.join(DATASET_PATH, "train.csv")
VAL_CSV = os.path.join(DATASET_PATH, "val.csv")
TEST_CSV = os.path.join(DATASET_PATH, "test.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "skin_cancer_resnet18.pth")

RESULTS_PATH = os.path.join(BASE_DIR, "results")
TRAINING_HISTORY_PLOT = os.path.join(RESULTS_PATH, "training_history.png")
CONFUSION_MATRIX_PLOT = os.path.join(RESULTS_PATH, "confusion_matrix.png")
ROC_CURVE_PLOT = os.path.join(RESULTS_PATH, "roc_curve.png")
CLASSIFICATION_REPORT_FILE = os.path.join(RESULTS_PATH, "classification_report.txt")
METRICS_FILE = os.path.join(RESULTS_PATH, "metrics.json")

# Make sure key output directories always exist.
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(DATASET_PATH, exist_ok=True)

# --------------------------------------------------------------------------
# Image / data settings
# --------------------------------------------------------------------------
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

USE_AUGMENTATION = True  # toggle training-time augmentation on/off

# Column names expected in HAM10000_metadata.csv
IMAGE_ID_COLUMN = "image_id"
LABEL_COLUMN = "dx"
LESION_ID_COLUMN = "lesion_id"  # used for grouped (leakage-free) splitting
IMAGE_EXTENSION = ".jpg"

# --------------------------------------------------------------------------
# Split settings
# --------------------------------------------------------------------------
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Model settings
# --------------------------------------------------------------------------
# Supported now: "resnet18". Architecture is modular so "resnet50" /
# "efficientnet_b0" can be added later in src/model.py without touching
# any other file.
MODEL_NAME = "resnet18"
PRETRAINED = True

# --------------------------------------------------------------------------
# Training hyperparameters
# --------------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
PATIENCE = 5  # early stopping patience (epochs without improvement)

USE_CLASS_WEIGHTS = True  # class-weighted CrossEntropyLoss for imbalance

LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 2

# Metric used to decide which checkpoint is "best": "val_loss" or "val_f1"
BEST_METRIC = "val_f1"

# --------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
