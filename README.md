
# DermaAI — Skin Lesion Classification System

DermaAI is an educational, research-oriented deep-learning project that classifies dermoscopic skin-lesion images into diagnostic categories from the **HAM10000** dataset using transfer learning (ResNet18) and a modern Streamlit web interface.

> ⚠️ **Medical Disclaimer**
>
> This project is for **educational and research purposes only**. It is not a medical device and must not be used for diagnosis or treatment decisions. Predictions can be wrong. Always consult a qualified dermatologist or healthcare professional.

---

## 1. Features

- End-to-end pipeline: data preparation → training → evaluation → prediction → web app
- Automatic class detection from metadata (no hard-coded label list)
- Leakage-aware, reproducible train/validation/test split (grouped by lesion when possible)
- Class-weighted loss to handle HAM10000 class imbalance
- ResNet18 transfer learning architecture
- Modular support for ResNet50 and EfficientNet
- AdamW optimizer, ReduceLROnPlateau scheduler, and early stopping
- Full model checkpoint (weights + class mapping + configuration)
- Real evaluation metrics: accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC
- Optional Grad-CAM interpretability visualization
- Modern dark-themed Streamlit dashboard with Home, Prediction, Model Analytics, and About pages
- Error handling for missing data, corrupted images, missing models, and unsupported formats

---

## 2. Technology Stack

**Machine Learning / Deep Learning:**
- Python 3.10+
- PyTorch
- Torchvision
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Pillow
- OpenCV
- tqdm

**Web Application:**
- Streamlit

---

## 3. Project Structure

```text
derma-ai/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── config/
│   └── config.py
│
├── dataset/
│   ├── HAM10000_metadata.csv
│   └── HAM10000_images/
│
├── models/
│   └── skin_cancer_resnet18.pth
│
├── src/
│   ├── prepare_data.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── inference.py
│   ├── gradcam.py
│   └── utils.py
│
├── results/
│
└── notebooks/
    └── experimentation.ipynb
```

---

## 4. Dataset Setup

1. Download the **HAM10000 dataset** from the ISIC Archive or Kaggle (Skin Cancer MNIST: HAM10000).
2. Extract the downloaded files.
3. Place the dataset inside the `dataset/` directory.

Expected structure:

```text
dataset/
├── HAM10000_metadata.csv
└── HAM10000_images/
    ├── ISIC_0024306.jpg
    ├── ISIC_0024307.jpg
    └── ...
```

The metadata CSV must contain at least:

- `image_id` — Image identifier
- `dx` — Diagnosis label

If the CSV also contains `lesion_id`, the data preparation script uses it to create lesion-grouped splits, ensuring images belonging to the same lesion do not appear in different subsets.

If `lesion_id` is unavailable, a stratified image-level split is used instead.

---

## 5. Installation

**Step 1 — Clone the repository**

```bash
git clone https://github.com/your-username/derma-ai.git
cd derma-ai
```

**Step 2 — Create a virtual environment**

```bash
python -m venv venv
```

**Step 3 — Activate the environment**

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

**Step 4 — Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 6. Running the Pipeline

### Step 1 — Prepare Dataset

Validate the dataset and generate train, validation, and test splits.

```bash
python src/prepare_data.py
```

Generated files:

- `dataset/train.csv`
- `dataset/val.csv`
- `dataset/test.csv`
- `dataset/class_to_idx.json`

### Step 2 — Train the Model

```bash
python src/train.py
```

The training pipeline uses:

- ResNet18 transfer learning
- Class-weighted CrossEntropyLoss
- Data augmentation
- AdamW optimizer
- ReduceLROnPlateau learning-rate scheduler
- Early stopping

The best model checkpoint is saved to:

```text
models/skin_cancer_resnet18.pth
```

Training curves are saved to:

```text
results/training_history.png
```

### Step 3 — Evaluate the Model

```bash
python src/evaluate.py
```

Evaluation metrics include:

- Accuracy
- Macro precision, recall, and F1-score
- Weighted precision, recall, and F1-score
- Confusion matrix
- One-vs-rest ROC-AUC (when mathematically valid)

Generated evaluation files:

```text
results/
├── confusion_matrix.png
├── roc_curve.png
├── classification_report.txt
└── metrics.json
```

### Step 4 — Predict a Single Image

```bash
python src/predict.py --image path/to/image.jpg
```

### Step 5 — Launch the Web Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 7. Model Architecture

DermaAI uses **ResNet18**, a convolutional neural network pretrained on ImageNet.

Key components:

- Pretrained ResNet18 backbone
- Transfer learning for skin-lesion classification
- Modified fully connected layer for HAM10000 diagnostic classes
- Configurable architecture through `config/config.py`

The following architectures are also wired into the modular model implementation:

- ResNet50
- EfficientNet-B0

To change the architecture, update `MODEL_NAME` in:

```text
config/config.py
```

---

## 8. Data Split & Reproducibility

The dataset is divided into:

| Dataset | Percentage |
|---|---:|
| Training | 80% |
| Validation | 10% |
| Testing | 10% |

Additional measures:

- Stratified splitting based on diagnosis (`dx`)
- Lesion-grouped splitting when `lesion_id` is available
- Fixed random seed (`RANDOM_SEED = 42`)
- Reproducible Python, NumPy, and PyTorch random seeds

Full GPU determinism is not guaranteed because some CUDA/cuDNN operations are inherently non-deterministic.

---

## 9. Handling Class Imbalance

HAM10000 contains a significant class imbalance, with the melanocytic nevus (`nv`) category dominating the dataset.

To address this issue, DermaAI uses:

- Class weights calculated from the training split only
- Weighted CrossEntropyLoss
- Data augmentation during training

Class weighting can be enabled or disabled using:

```python
USE_CLASS_WEIGHTS
```

in `config/config.py`.

---

## 10. Evaluation Metrics

DermaAI evaluates the trained model using actual predictions on the held-out test dataset.

The evaluation pipeline reports:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC-AUC

Metrics are generated only after training and evaluation.

If the model has not been trained or evaluated, the application displays:

```text
Metrics unavailable — train and evaluate the model first.
```

No evaluation metrics are fabricated or hard-coded.

---

## 11. Tips for Improving Accuracy

Potential improvements include:

- Increasing the number of training epochs
- Using progressive fine-tuning
- Experimenting with ResNet50 or EfficientNet-B0
- Applying stronger data augmentation
- Experimenting with Focal Loss
- Tuning learning rates and batch sizes
- Using cosine learning-rate scheduling
- Applying test-time augmentation
- Experimenting with model ensembles
- Incorporating additional minority-class samples

These techniques may improve performance but do not guarantee better real-world clinical outcomes.

---

## 12. Limitations

- HAM10000 may not represent all skin tones, lesion types, or imaging conditions.
- Predictions may be incorrect, including highly confident predictions.
- Performance on external datasets may differ significantly.
- Grad-CAM visualizations indicate model attention, not proof of correct medical reasoning.
- The system has not undergone clinical validation.
- It is not suitable for independent medical diagnosis or treatment decisions.

---

## 13. Future Improvements

- Vision Transformer (ViT) architecture
- Advanced EfficientNet backbones
- Focal Loss for severe class imbalance
- External patient-level validation
- FastAPI backend with React/Next.js frontend
- Docker containerization
- Cloud deployment
- Experiment tracking using MLflow or Weights & Biases

---

## 14. Medical Disclaimer

DermaAI is strictly an **educational and research-oriented project** demonstrating deep-learning-based skin-lesion image classification.

It is not a certified medical device, has not been clinically validated, and must not be used for diagnosis, treatment recommendations, or medical decision-making.

Always consult a qualified dermatologist or healthcare professional for medical evaluation.
