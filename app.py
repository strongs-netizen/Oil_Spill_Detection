"""
=============================================================================
OIL SPILL DETECTION — Streamlit Web App
=============================================================================

Run with:
    streamlit run app.py

Make sure your trained model weights file is in the same folder:
    oil_spill_model_weights.pth
=============================================================================
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import streamlit as st
from PIL import Image
from torchvision import transforms
streamlit run app.pyimport streamlit as st

# Only run Streamlit code if in Streamlit context
if __name__ == "__main__" or hasattr(st, 'runtime') and st.runtime.exists():
    st.set_page_config(page_title="Oil Spill Detector")
    # ... rest of your Streamlit app
# ── Page Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Oil Spill Detector",
    page_icon="🛢️",
    layout="centered",
)

# ── Constants ─────────────────────────────────────────────────────────────
MODEL_PATH   = "oil_spill_model_weights.pth"   # path to your saved weights
MODEL_NAME   = "resnet50"                       # must match what you trained
NUM_CLASSES  = 2
IMG_SIZE     = 224
CLASS_NAMES  = ["No Oil Spill ✅", "Oil Spill Detected ⚠️"]
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ── Image Transform (same as val_transform during training) ───────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ── Load Model (cached so it only loads once) ─────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained model weights. Cached so it runs only once."""
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

# ── Prediction Function ───────────────────────────────────────────────────
def predict(image: Image.Image, model: nn.Module):
    """
    Takes a PIL image, runs it through the model,
    and returns (predicted_class_index, probabilities_list).
    """
    # Convert to RGB (handles grayscale SAR images, same as training)
    image = image.convert("L").convert("RGB")

    # Apply transforms and add batch dimension: [1, 3, 224, 224]
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)[0]   # shape: [2]

    predicted_idx = probs.argmax().item()
    return predicted_idx, probs.cpu().tolist()


# ── UI ────────────────────────────────────────────────────────────────────

st.title("🛢️ Oil Spill Detection")
st.markdown(
    "Upload a **SAR satellite image** and the model will classify it as "
    "**Oil Spill** or **No Oil Spill**."
)
st.markdown("---")

# Check that model weights file exists before trying to load
if not os.path.exists(MODEL_PATH):
    st.error(
        f"⚠️ Model weights not found at `{MODEL_PATH}`. "
        "Please place your trained `oil_spill_model_weights.pth` file "
        "in the same folder as this script."
    )
    st.stop()

# Load the model
with st.spinner("Loading model..."):
    model = load_model()
st.success("Model loaded and ready!")

st.markdown("---")

# File uploader
uploaded_file = st.file_uploader(
    "Upload a SAR image",
    type=["png", "jpg", "jpeg", "tif"],
    help="Supported formats: PNG, JPG, JPEG, TIF"
)

if uploaded_file is not None:
    # Show the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.markdown("---")

    # Run prediction
    with st.spinner("Analysing image..."):
        predicted_idx, probs = predict(image, model)

    predicted_label = CLASS_NAMES[predicted_idx]
    confidence      = probs[predicted_idx] * 100

    # Result display
    if predicted_idx == 1:
        st.error(f"### Prediction: {predicted_label}")
    else:
        st.success(f"### Prediction: {predicted_label}")

    st.metric(label="Confidence", value=f"{confidence:.1f}%")

    # Probability breakdown
    st.markdown("#### Probability Breakdown")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("No Oil Spill ✅", f"{probs[0]*100:.1f}%")
    with col2:
        st.metric("Oil Spill ⚠️", f"{probs[1]*100:.1f}%")

    # Progress bars for visual intuition
    st.progress(probs[0], text="No Oil Spill")
    st.progress(probs[1], text="Oil Spill")

else:
    st.info("👆 Upload an image above to get started.")

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(f"Running on: **{DEVICE.upper()}**  |  Model: **{MODEL_NAME}**")
