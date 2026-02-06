# app/ui_component.py

import streamlit as st
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


def render_header():
    st.title("⚖️ Unbias")
    st.subheader("Uncertainty-Aware Gender Presentation Signal Analysis")

    st.markdown(
        """
**Unbias** estimates *gender presentation signals* in groups  
using computer vision — **without identifying individuals**.

This system models **appearance signals**, not identity or self-definition.
"""
    )


def render_sidebar_controls() -> Tuple[bool, bool, bool]:
    st.sidebar.header("⚙️ Controls")

    enable_signals = st.sidebar.checkbox(
        "Enable presentation signal estimation",
        value=True
    )

    show_boxes = st.sidebar.checkbox(
        "Show detected people",
        value=True
    )

    show_explanation = st.sidebar.checkbox(
        "Show explanation",
        value=True
    )

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

    return enable_signals, show_boxes, show_explanation


def render_image_panel(image, boxes: List[tuple], detection_confidence: float):
    st.image(
        image,
        caption=f"Detected participants: {len(boxes)}",
        width="stretch"
    )

    st.caption(
        f"Detection confidence threshold: {detection_confidence}"
    )


def render_signal_chart(results: Dict[str, float]):
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


def compute_signal_confidence(results: Dict[str, float]) -> str:
    if results["total"] <= 0:
        return "Low"

    dominant_signal = max(
        results["femininity"],
        results["masculinity"]
    )

    avg_conf = dominant_signal / results["total"]

    if avg_conf > 0.45:
        return "High"
    elif avg_conf > 0.30:
        return "Moderate"
    else:
        return "Low"


def render_signal_summary(results: Dict[str, float], confidence_label: str, enabled: bool):
    st.markdown("### 📊 Aggregate Presentation Signals")

    st.markdown(
        f"""
**Signal confidence level:** `{confidence_label}`  
**Signal estimation enabled:** `{enabled}`
"""
    )

    st.markdown(
        f"""
**Total presentation signal mass:** `{results["total"]:.2f}`
"""
    )


def render_explanation_panel():
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


def render_footer():
    st.markdown("---")
    st.caption(
        "Unbias • Presentation-Signal Estimation • Privacy-Preserving Computer Vision"
    )
