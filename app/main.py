# app/main.py

import sys
import os
import json

# Ensure project root is on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pipelines.inference_pipeline import InferencePipeline
from core.detector import PersonDetector

DETECTION_CONFIDENCE = 0.35

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Unbias",
    page_icon="⚖️",
    layout="wide"
)

# ---------------------------
# Title & Description
# ---------------------------
st.title("⚖️ Unbias")
st.subheader("Ethical, Uncertainty-Aware Participant Analysis")

st.markdown(
    """
**Unbias** estimates participant counts and gender *presentation*  
using computer vision — **without identifying individuals**.

This system is designed to **surface uncertainty**, not hide it.
"""
)

# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.header("⚙️ Controls")

enable_gender = st.sidebar.checkbox(
    "Enable gender presentation estimation",
    value=True
)

show_boxes = st.sidebar.checkbox("Show detected people", value=True)
confidence_note = st.sidebar.checkbox("Show confidence explanation", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
### What Unbias Does NOT Do
- ❌ No facial recognition  
- ❌ No identity inference  
- ❌ No image storage  
- ❌ No binary enforcement
"""
)

# ---------------------------
# Load Pipeline & Detector
# ---------------------------
@st.cache_resource
def load_pipeline():
    return InferencePipeline()

pipeline = load_pipeline()
detector = PersonDetector(conf=DETECTION_CONFIDENCE)

# ---------------------------
# Image Upload
# ---------------------------
uploaded_file = st.file_uploader(
    "Upload an image with people",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("Upload an image to begin analysis.")
    st.stop()

# ---------------------------
# Read Image
# ---------------------------
file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ---------------------------
# Run Detection
# ---------------------------
boxes = detector.detect(image)

annotated = image_rgb.copy()
if show_boxes:
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

# ---------------------------
# Run Inference (Conditional)
# ---------------------------
if enable_gender:
    results = pipeline.run(image)
else:
    results = {
        "female": 0.0,
        "male": 0.0,
        "uncertain": 0.0,
        "total": float(len(boxes))
    }

# ---------------------------
# Confidence Summary
# ---------------------------
if enable_gender and results["total"] > 0:
    avg_conf = max(results["female"], results["male"]) / results["total"]
else:
    avg_conf = 0.0

if avg_conf > 0.45:
    confidence_label = "High"
elif avg_conf > 0.30:
    confidence_label = "Moderate"
else:
    confidence_label = "Low"

# ---------------------------
# Layout: Image + Results
# ---------------------------
col1, col2 = st.columns([1.2, 1])

with col1:
    st.image(
        annotated,
        caption=f"Detected participants: {len(boxes)}",
        width="stretch"
    )

    st.caption(
        f"Detection threshold: {DETECTION_CONFIDENCE} · "
        "Some individuals may be excluded at higher confidence levels."
    )

with col2:
    st.markdown("### 📊 Aggregate Estimates")

    st.markdown(
        f"""
**Model confidence level:** `{confidence_label}`  
**Gender estimation enabled:** `{enable_gender}`
"""
    )

    labels = ["Female-presenting", "Male-presenting", "Uncertain"]
    values = [
        results["female"],
        results["male"],
        results["uncertain"]
    ]

    fig, ax = plt.subplots()
    bars = ax.bar(labels, values)
    ax.set_ylabel("Estimated count")
    ax.set_ylim(0, results["total"] + 1)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom"
        )

    st.pyplot(fig)

    st.markdown(
        f"""
**Total participants detected:** `{int(results["total"])}`  
"""
    )

# ---------------------------
# Confidence & Ethics Panel
# ---------------------------
if confidence_note:
    st.markdown("---")
    st.markdown("### 🧠 How to Read These Results")

    st.markdown(
        """
- Values are **probabilistic estimates**, not exact counts  
- “Uncertain” reflects visual ambiguity or low confidence  
- The model may hesitate rather than guess  
- Results describe **appearance**, not identity  

> Uncertainty is treated as a sign of integrity,
> not model failure.
"""
    )

# ---------------------------
# Downloadable Report
# ---------------------------
report = {
    "participants_detected": int(results["total"]),
    "distribution": {
        "female": results["female"],
        "male": results["male"],
        "uncertain": results["uncertain"]
    },
    "detection_confidence_threshold": DETECTION_CONFIDENCE,
    "gender_estimation_enabled": enable_gender,
    "model_confidence": confidence_label,
    "notes": "Probabilistic estimate. No identities inferred."
}

st.download_button(
    label="⬇ Download analysis report (JSON)",
    data=json.dumps(report, indent=2),
    file_name="unbias_report.json",
    mime="application/json"
)

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption(
    "Unbias • Ethical Computer Vision • Probabilistic, Privacy-Preserving Analysis"
)
