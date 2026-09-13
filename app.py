"""
app.py
------
DermaAI Streamlit web application.

Run with:
    streamlit run app.py
"""

import os
import sys
import io

import streamlit as st
from PIL import Image, UnidentifiedImageError
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import config as cfg
from src.inference import load_model_bundle, predict_image, ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE_MB
from src.utils import load_json, get_full_class_name

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="DermaAI — Skin Lesion Classification",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Styling — dark navy theme with glass-style cards
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at 10% 0%, #16213e 0%, #0d1526 45%, #0a0f1e 100%);
        color: #E6E9F0;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12182c 0%, #0d1220 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 18px;
        padding: 1.5rem 1.75rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
        margin-bottom: 1.2rem;
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60A5FA, #A78BFA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #9CA6C0;
        margin-top: 0.2rem;
    }
    .disclaimer-box {
        background: rgba(249, 115, 22, 0.12);
        border: 1px solid rgba(249, 115, 22, 0.4);
        border-radius: 14px;
        padding: 1rem 1.25rem;
        color: #FDBA74;
        font-size: 0.92rem;
    }
    .pred-badge {
        display: inline-block;
        background: linear-gradient(90deg, #3B82F6, #8B5CF6);
        color: white;
        padding: 0.35rem 1rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.05rem;
    }
    div[data-testid="stMetricValue"] {
        color: #60A5FA;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DISCLAIMER_TEXT = (
    "⚠️ This application is for educational and research purposes only. "
    "It is not a medical diagnostic tool. Do not use the prediction to make "
    "medical decisions. Consult a qualified dermatologist or healthcare "
    "professional for diagnosis."
)


# --------------------------------------------------------------------------
# Cached resource loading
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model_bundle():
    return load_model_bundle()


def model_is_available() -> bool:
    return os.path.exists(cfg.MODEL_SAVE_PATH)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🩺 DermaAI")
    st.caption("AI-Powered Skin Lesion Classification")
    st.divider()

    page = st.radio(
        "Navigate",
        ["Home", "Prediction", "Model Analytics", "About"],
        key="nav_radio",
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Model status**")
    if model_is_available():
        st.success("Trained model found ✅")
    else:
        st.warning("No trained model found. Run the training pipeline first.")

    st.markdown("**Dataset**")
    if os.path.exists(cfg.METADATA_PATH):
        st.success("HAM10000 metadata found ✅")
    else:
        st.info("Dataset not detected.")

    device_label = "CUDA (GPU)" if cfg.DEVICE.type == "cuda" else "CPU"
    st.markdown(f"**Device:** {device_label}")


# --------------------------------------------------------------------------
# HOME PAGE
# --------------------------------------------------------------------------
if page == "Home":
    st.markdown('<p class="hero-title">DermaAI</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">AI-Powered Skin Lesion Classification</p>', unsafe_allow_html=True)
    st.write("")

    st.markdown(
        """
        <div class="glass-card">
        Upload a skin-lesion image and analyze it using a deep-learning model
        trained on the HAM10000 dataset with transfer learning (ResNet18).
        Get a predicted diagnostic category, confidence score, and top
        alternative predictions in seconds.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="glass-card"><b>🧠 Transfer Learning</b><br>'
            'ResNet18 pretrained on ImageNet, fine-tuned on HAM10000.</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="glass-card"><b>📊 Real Metrics</b><br>'
            'Accuracy, F1, ROC-AUC computed from actual test predictions.</div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="glass-card"><b>🔍 Interpretability</b><br>'
            'Optional Grad-CAM heatmaps show which regions influenced the result.</div>',
            unsafe_allow_html=True,
        )

    st.write("")

    def _go_to_prediction():
        st.session_state["nav_radio"] = "Prediction"

    st.button("🖼️ Upload Image →", type="primary", on_click=_go_to_prediction)

    st.caption("Supported formats: JPG / JPEG / PNG")
    st.markdown(f'<div class="disclaimer-box">{DISCLAIMER_TEXT}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# PREDICTION PAGE
# --------------------------------------------------------------------------
elif page == "Prediction":
    st.markdown('<p class="hero-title">Prediction</p>', unsafe_allow_html=True)
    st.markdown(f'<div class="disclaimer-box">{DISCLAIMER_TEXT}</div>', unsafe_allow_html=True)
    st.write("")

    if not model_is_available():
        st.error(
            "No trained model was found. Please run the training pipeline "
            "(`python src/prepare_data.py` then `python src/train.py`) before using prediction."
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload a skin-lesion image (JPG, JPEG, or PNG)", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            size_mb = uploaded_file.size / (1024 * 1024)
            if size_mb > MAX_UPLOAD_SIZE_MB:
                st.error(f"File is too large ({size_mb:.1f} MB). Please upload a file under {MAX_UPLOAD_SIZE_MB} MB.")
            else:
                try:
                    image_bytes = uploaded_file.read()
                    image = Image.open(io.BytesIO(image_bytes))
                    image.verify()
                    # Re-open after verify() since verify() invalidates the file handle.
                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                except (UnidentifiedImageError, OSError):
                    st.error("The uploaded file could not be read as a valid image. It may be corrupted.")
                    image = None

                if image is not None:
                    try:
                        bundle = get_model_bundle()
                    except Exception as exc:
                        st.error(f"Could not load the trained model: {exc}")
                        bundle = None

                    if bundle is not None:
                        with st.spinner("Analyzing image..."):
                            try:
                                predicted_class, confidence, top_predictions = predict_image(
                                    image, bundle, top_k=5
                                )
                            except Exception as exc:
                                st.error(f"Prediction failed: {exc}")
                                predicted_class = None

                        if predicted_class is not None:
                            left, right = st.columns([1, 1.2])

                            with left:
                                st.markdown("#### Uploaded Image")
                                st.image(image, use_container_width=True)

                            with right:
                                st.markdown("#### Prediction Result")
                                full_name = get_full_class_name(predicted_class)
                                st.markdown(
                                    f'<span class="pred-badge">{full_name} ({predicted_class})</span>',
                                    unsafe_allow_html=True,
                                )
                                st.metric("Confidence", f"{confidence * 100:.2f}%")

                                st.markdown("##### Top Predictions")
                                for pred in top_predictions:
                                    label = f"{pred['full_name']} ({pred['class']})"
                                    st.write(label)
                                    st.progress(min(pred["confidence"], 1.0))
                                    st.caption(f"{pred['confidence'] * 100:.2f}%")

                            st.write("")
                            with st.expander("🔍 Grad-CAM interpretability (optional)"):
                                st.caption(
                                    "Shows which image regions most influenced the prediction. "
                                    "This is an interpretability visualization only — not proof of "
                                    "correct medical reasoning."
                                )
                                if st.button("Generate Grad-CAM heatmap"):
                                    try:
                                        from src.gradcam import GradCAM, overlay_heatmap
                                        from src.inference import preprocess_image

                                        cam_model = bundle.model
                                        gradcam = GradCAM(cam_model)
                                        input_tensor = preprocess_image(image, bundle).to(bundle.device)
                                        input_tensor.requires_grad_(True)
                                        cam = gradcam.generate(input_tensor)
                                        overlay = overlay_heatmap(image, cam)
                                        st.image(overlay, caption="Grad-CAM heatmap overlay", use_container_width=True)
                                    except Exception as exc:
                                        st.warning(f"Grad-CAM could not be generated: {exc}")

                            st.markdown(f'<div class="disclaimer-box">{DISCLAIMER_TEXT}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# MODEL ANALYTICS PAGE
# --------------------------------------------------------------------------
elif page == "Model Analytics":
    st.markdown('<p class="hero-title">Model Analytics</p>', unsafe_allow_html=True)
    st.write("")

    metrics = load_json(cfg.METRICS_FILE)

    if metrics is None:
        st.info("Train the model first to view analytics.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test Accuracy", f"{metrics['test_accuracy'] * 100:.2f}%")
        c2.metric("Macro F1", f"{metrics['macro_f1']:.4f}")
        c3.metric("Macro Precision", f"{metrics['macro_precision']:.4f}")
        c4.metric("Macro Recall", f"{metrics['macro_recall']:.4f}")

        if metrics.get("roc_auc_macro_ovr") is not None:
            st.metric("Macro ROC-AUC (OvR)", f"{metrics['roc_auc_macro_ovr']:.4f}")

        st.caption(f"Evaluated on {metrics['num_test_samples']} held-out test images.")
        st.write("")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Confusion Matrix")
            if os.path.exists(cfg.CONFUSION_MATRIX_PLOT):
                st.image(cfg.CONFUSION_MATRIX_PLOT, use_container_width=True)
            else:
                st.info("Confusion matrix not yet generated.")
        with col2:
            st.markdown("#### Training Curves")
            if os.path.exists(cfg.TRAINING_HISTORY_PLOT):
                st.image(cfg.TRAINING_HISTORY_PLOT, use_container_width=True)
            else:
                st.info("Training history not yet generated.")

        if os.path.exists(cfg.ROC_CURVE_PLOT):
            st.markdown("#### ROC Curves (One-vs-Rest)")
            st.image(cfg.ROC_CURVE_PLOT, use_container_width=True)

        if os.path.exists(cfg.CLASSIFICATION_REPORT_FILE):
            with st.expander("Full classification report"):
                with open(cfg.CLASSIFICATION_REPORT_FILE) as f:
                    st.code(f.read())


# --------------------------------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------------------------------
elif page == "About":
    st.markdown('<p class="hero-title">About DermaAI</p>', unsafe_allow_html=True)
    st.write("")

    st.markdown(
        """
        <div class="glass-card">
        DermaAI is an educational deep-learning project for skin-lesion image
        classification using transfer learning. It is built to demonstrate a
        complete, reproducible ML pipeline — from data preparation through
        training, evaluation, and deployment in an interactive web app.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Technology")
    st.markdown("- Python\n- PyTorch / Torchvision\n- ResNet18 (transfer learning)\n- HAM10000 dataset\n- Streamlit")

    st.markdown("#### Model Information")
    if model_is_available():
        try:
            bundle = get_model_bundle()
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.write(f"**Model:** {bundle.model_name}")
                st.write("**Framework:** PyTorch")
                st.write(f"**Input:** {bundle.image_size} × {bundle.image_size} RGB")
                st.write("**Training:** Transfer Learning")
            with info_col2:
                st.write("**Dataset:** HAM10000")
                st.write(f"**Classes:** {len(bundle.class_to_idx)}")
                st.write(f"**Best {bundle.best_metric_name}:** "
                         f"{bundle.best_metric_value:.4f}" if bundle.best_metric_value else "n/a")
            st.write("**Class list:** " + ", ".join(sorted(bundle.class_to_idx.keys())))
        except Exception as exc:
            st.warning(f"Could not load model info: {exc}")
    else:
        st.info("Train the model to see detailed model information here.")

    st.write("")
    st.markdown(f'<div class="disclaimer-box"><b>Disclaimer</b><br>{DISCLAIMER_TEXT}</div>', unsafe_allow_html=True)