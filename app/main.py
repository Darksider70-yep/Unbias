# app/main.py

import sys
import os
import json

# Ensure project root is on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import cv2
import numpy as np

from pipelines.inference_pipeline import InferencePipeline
from core.detector import PersonDetector

from ui_component import (
    render_header,
    render_sidebar_controls,
    render_image_panel,
    render_signal_chart,
    render_signal_summary,
    render_explanation_panel,
    compute_signal_confidence,
    render_footer
)

# ---------------------------
# Configuration
# ---------------------------
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
# Header
# ---------------------------
render_header()

# ---------------------------
# Sidebar Controls
# ---------------------------
enable_signals, show_boxes, show_explanation = render_sidebar_controls()

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
# Confidence Computation
# ---------------------------
confidence_label = compute_signal_confidence(results)

# ---------------------------
# Layout
# ---------------------------
col1, col2 = st.columns([1.2, 1])

with col1:
    render_image_panel(
        annotated,
        boxes,
        DETECTION_CONFIDENCE
    )

with col2:
    render_signal_summary(
        results,
        confidence_label,
        enable_signals
    )
    render_signal_chart(results)

# ---------------------------
# Explanation Panel
# ---------------------------
if show_explanation:
    render_explanation_panel()

# ---------------------------
# Downloadable Report
# ---------------------------
report = {
    "participants_detected": int(len(boxes)),
    "presentation_signals": {
        "femininity": results["femininity"],
        "masculinity": results["masculinity"],
        "ambiguity": results["ambiguity"]
    },
    "signal_confidence_level": confidence_label,
    "signal_estimation_enabled": enable_signals,
    "detection_confidence_threshold": DETECTION_CONFIDENCE,
    "notes": (
        "Signals estimate visual presentation only. "
        "No identity or gender is inferred."
    )
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
render_footer()
