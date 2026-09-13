# DermaAI — Skin Lesion Classification System

DermaAI is an educational, research-oriented deep-learning project that
classifies dermoscopic skin-lesion images into one of the diagnostic
categories found in the **HAM10000** dataset, using transfer learning
(ResNet18) and a modern Streamlit web interface.

> ⚠️ **Medical Disclaimer**
> This project is for **educational and research purposes only**. It is
> **not** a medical device and must **not** be used for diagnosis or
> treatment decisions. Predictions can be wrong. Always consult a
> qualified dermatologist or healthcare professional.

---

## 1. Features

- End-to-end pipeline: data prep → training → evaluation → prediction → web app
- Automatic class detection from metadata (no hard-coded label list)
- Leakage-aware, reproducible train/val/test split (grouped by lesion when possible)
- Class-weighted loss to handle HAM10000's class imbalance
- ResNet18 transfer learning, modular so ResNet50 / EfficientNet can be added later
- AdamW + ReduceLROnPlateau + early stopping
- Full checkpoint (weights + class mapping + config), not just raw weights
- Real metrics only: accuracy, precision, recall, F1 (macro & weighted), confusion matrix, ROC-AUC
- Optional Grad-CAM interpretability visualization
- Modern dark-themed Streamlit dashboard with Home / Prediction / Model Analytics / About pages
- Friendly error handling for missing data, corrupted images, missing model, unsupported formats

---

## 2. Technology Stack

**ML/DL:** Python 3.10+, PyTorch, Torchvision, NumPy, Pandas, Scikit-learn,
Matplotlib, Seaborn, Pillow, OpenCV, tqdm

**Web app:** Streamlit

---

## 3. Project Structure

```
derma-ai/
│
├── app.py                     # Streamlit web application
├── requirements.txt
├── README.md
├── .gitignore
│
├── config/
│   └── config.py               # All configurable paths & hyperparameters
│
├── dataset/
│   ├── HAM10000_metadata.csv   # (you provide this)
│   └── HAM10000_images/        # (you provide this)
│
├── models/
│   └── skin_cancer_resnet18.pth  # created by train.py
│
├── src/
│   ├── prepare_data.py         # validates data, creates train/val/test splits
│   ├── dataset.py              # Dataset class + transforms
│   ├── model.py                # model architecture (modular)
│   ├── train.py                # training pipeline
│   ├── evaluate.py             # test-set evaluation pipeline
│   ├── predict.py              # CLI single-image prediction
│   ├── inference.py            # shared inference logic (CLI + app)
│   ├── gradcam.py              # optional interpretability module
│   └── utils.py                # seeding, early stopping, checkpoint I/O
│
├── results/                    # generated plots & metrics (after training/eval)
└── notebooks/
    └── experimentation.ipynb
```

---

## 4. Dataset Setup

1. Download **HAM10000** (e.g. from the ISIC archive / Kaggle "Skin Cancer MNIST: HAM10000").
2. Place the files so the project looks like this:

```
dataset/
├── HAM10000_metadata.csv
└── HAM10000_images/
    ├── ISIC_0024306.jpg
    ├── ISIC_0024307.jpg
    └── ...
```

The metadata CSV must contain at least an `image_id` column and a `dx`
column (the diagnosis label). If it also contains `lesion_id`, the data
preparation script will use it to perform a **leakage-free, lesion-grouped
split** — all images belonging to the same lesion stay in the same split.
If `lesion_id` is absent, a plain stratified image-level split is used
instead, and a warning is printed documenting this limitation.

---

## 5. Installation

```bash
python -m venv venv
```

Activate the environment:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 6. Running the Pipeline

**Step 1 — Prepare data** (validates dataset, creates splits):
```bash
python src/prepare_data.py
```
Generates `dataset/train.csv`, `dataset/val.csv`, `dataset/test.csv`, and
`dataset/class_to_idx.json`.

**Step 2 — Train the model:**
```bash
python src/train.py
```
Trains ResNet18 with transfer learning, class-weighted loss, augmentation,
AdamW, ReduceLROnPlateau, and early stopping. Saves the best checkpoint to
`models/skin_cancer_resnet18.pth` and training curves to
`results/training_history.png`.

**Step 3 — Evaluate on the test set:**
```bash
python src/evaluate.py
```
Computes accuracy, macro/weighted precision-recall-F1, and (when
mathematically valid) one-vs-rest ROC-AUC — all from real model
predictions. Saves `results/confusion_matrix.png`,
`results/roc_curve.png`, `results/classification_report.txt`, and
`results/metrics.json`.

**Step 4 — Predict a single image (CLI):**
```bash
python src/predict.py --image path/to/image.jpg
```

**Step 5 — Launch the web app:**
```bash
streamlit run app.py
```

---

## 7. Model Architecture

- Backbone: `torchvision.models.resnet18`, pretrained on ImageNet
- Final fully-connected layer replaced to match the number of detected classes
- Architecture is modular (`src/model.py`) — `resnet50` and
  `efficientnet_b0` are already wired in as options; switch by changing
  `MODEL_NAME` in `config/config.py`

---

## 8. Data Split & Reproducibility

- 80% train / 10% validation / 10% test
- Stratified by diagnosis (`dx`), grouped by `lesion_id` when available to prevent leakage
- `RANDOM_SEED = 42` used throughout (Python `random`, NumPy, PyTorch, CUDA)
- Full determinism on GPU is not guaranteed — some cuDNN kernels are
  inherently non-deterministic for performance reasons, so minor run-to-run
  variance on CUDA is expected even with a fixed seed

---

## 9. Handling Class Imbalance

HAM10000 is heavily imbalanced (e.g. `nv` dominates the dataset). This
project computes `class_weight="balanced"` weights from the **training
split only** and applies them to `CrossEntropyLoss`. This is configurable
via `USE_CLASS_WEIGHTS` in `config/config.py`.

---

## 10. Metrics

All metrics are computed directly from the trained model's predictions on
the held-out test set — nothing is fabricated. If you have not yet run
`train.py` and `evaluate.py`, the app and scripts will clearly say:

> "Metrics unavailable — train and evaluate the model first."

---

## 11. Tips for Improving Accuracy

Transfer learning + class weighting + augmentation gets you a solid
baseline, but real-world usefulness depends heavily on training budget and
data quality. To push accuracy further:

- Train for more epochs and/or unfreeze more of the backbone gradually
  (progressive fine-tuning) once the new head has stabilized
- Try `resnet50` or `efficientnet_b0` (already wired into `src/model.py`)
- Use stronger/more targeted augmentation (RandomResizedCrop, mixup/cutmix)
- Try focal loss instead of (or alongside) class weighting for severe imbalance
- Tune the learning rate and batch size; try a cosine LR schedule
- Use test-time augmentation (average predictions over several augmented views)
- Ensemble two or more trained models
- Use a grouped, patient/lesion-level split (already implemented) so your
  validation accuracy reflects real generalization rather than leakage
- Collect or incorporate more data for minority classes (`df`, `vasc`, `akiec`)

None of this changes the fact that this remains a research/educational
tool, not a clinical-grade diagnostic system.

---

## 12. Limitations

- HAM10000 is not representative of all skin tones, lesion types, or imaging conditions
- The model can be confidently wrong, especially on out-of-distribution images
- Grad-CAM visualizations show correlation with model attention, not proof of correct medical reasoning
- This system has not been clinically validated and is not a substitute for professional dermatological evaluation

---

## 13. Future Improvements

- Vision Transformer / EfficientNet backbones
- Focal loss for extreme class imbalance
- Patient-level external validation dataset
- FastAPI backend + React/Next.js frontend
- Docker containerization and cloud deployment
- Model versioning and experiment tracking (e.g. MLflow, Weights & Biases)

---

## 14. Medical Disclaimer

This system is **not a medical device** and should **not** be used for
diagnosis or treatment decisions. It is intended solely for educational
and research demonstration of a deep-learning image-classification
pipeline.
#   S k i n - c a n c e r - d e t e c t i o n -  
 #   S k i n - c a n c e r - d e t e c t i o n -  
 