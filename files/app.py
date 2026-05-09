"""
=============================================================================
OIL SPILL DETECTION — Streamlit App
=============================================================================
Folder structure required:
    oil_spill_app/
    ├── app.py
    ├── preprocessing.py
    └── model/
        ├── model.py
        ├── inference.py
        └── neuralnet.pth     ← your trained weights file

Run with:
    streamlit run app.py
=============================================================================
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))   # make local imports work

import streamlit as st
from PIL import Image
from model.inference import load_model, predict
from utils.preprocessing import preprocess

# ── Class Names ───────────────────────────────────────────────────────────
CLASS_NAMES = ["No Oil Spill ✅", "Oil Spill Detected ⚠️"]

# ── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Oil Spill Detector", page_icon="🛢️", layout="centered")

st.title("🛢️ Oil Spill Detection")
st.markdown("Upload a **SAR satellite image** and the model will classify it.")
st.markdown("---")

# ── Load Model (cached — only runs once) ──────────────────────────────────
@st.cache_resource
def get_model():
    return load_model()

with st.spinner("Loading model..."):
    model = get_model()
st.success("✅ Model loaded and ready!")

st.markdown("---")

# ── File Uploader ─────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a SAR image",
    type=["png", "jpg", "jpeg", "tif"],
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.markdown("---")

    with st.spinner("Analysing image..."):
        tensor = preprocess(image)
        label, confidence = predict(model, tensor, CLASS_NAMES)

    # Result
    if "Oil Spill Detected" in label:
        st.error(f"### Prediction: {label}")
    else:
        st.success(f"### Prediction: {label}")

    st.metric("Confidence", f"{confidence * 100:.1f}%")
    st.progress(confidence, text=f"{label} — {confidence*100:.1f}%")

else:
    st.info("👆 Upload an image above to get started.")

st.markdown("---")
st.caption("Oil Spill Detection | SAR Satellite Image Classifier")
