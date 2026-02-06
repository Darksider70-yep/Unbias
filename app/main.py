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
st.subheader("Uncertainty-Aware Gender Presentation Signal Analysis")

st.markdown(
    """
**Unbias** estimates *gender presentation signals* in groups  
using computer vision — **without identifying individuals**.

This system models **appearance signals**, not identity or self-definition.
"""
)

# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.header("⚙️ Controls")

enable_signals = st.sidebar.checkbox(
    "Enable presentation signal estimation",
    value=True
)

show_boxes = st.sidebar.checkbox("Show detected people", value=True)
confidence_note = st.sidebar.checkbox("Show explanation", value=True)

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
# Run Inference
# ---------------------------
if enable_signals:
    results = pipeline.run(image)
else:
    results = {
        "femininity": 0.0,
        "masculinity": 0.0,
        "ambiguity": 0.0,
        "total": float(len(boxes))
    }

# ---------------------------
# Confidence Summary
# ---------------------------
if enable_signals and results["total"] > 0:
    dominant_signal = max(
        results["femininity"],
        results["masculinity"]
    )
    avg_conf = dominant_signal / results["total"]
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
        f"Detection confidence threshold: {DETECTION_CONFIDENCE}"
    )

with col2:
    st.markdown("### 📊 Aggregate Presentation Signals")

    st.markdown(
        f"""
**Signal confidence level:** `{confidence_label}`  
**Signal estimation enabled:** `{enable_signals}`
"""
    )

    labels = [
        "Femininity signal",
        "Masculinity signal",
        "Ambiguity signal"
    ]
    values = [
        results["femininity"],
        results["masculinity"],
        results["ambiguity"]
    ]

    fig, ax = plt.subplots()
    bars = ax.bar(labels, values)
    ax.set_ylabel("Aggregated signal mass")
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
**Total presentation signal mass:** `{results["total"]:.2f}`
"""
    )

# ---------------------------
# Explanation Panel
# ---------------------------
if confidence_note:
    st.markdown("---")
    st.markdown("### 🧠 How to Interpret These Results")

    st.markdown(
        """
- Signals are **probabilistic**, not categorical  
- “Ambiguity” represents visual uncertainty  
- The system may intentionally hesitate  
- Outputs describe **appearance signals**, not gender identity  

> Uncertainty is a design feature, not a failure.
"""
    )

# ---------------------------
# Downloadable Report
# ---------------------------
report = {
    "participants_detected": int(results["total"]),
    "presentation_signals": {
        "femininity": results["femininity"],
        "masculinity": results["masculinity"],
        "ambiguity": results["ambiguity"]
    },
    "detection_confidence_threshold": DETECTION_CONFIDENCE,
    "signal_estimation_enabled": enable_signals,
    "signal_confidence_level": confidence_label,
    "notes": "Signals estimate visual presentation only. No identities inferred."
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
    "Unbias • Presentation-Signal Estimation • Privacy-Preserving Computer Vision"
)
