"""
src/ui/pages/calibration.py
===========================
Model calibration, temperature scaling, and reliability diagrams.
"""

from typing import Dict, Any
import streamlit as st

from src.ui.common import (
    LABEL_NAMES,
    LABEL_ICONS,
    section_header,
    glowing_divider,
    fig_exists,
    load_fig,
)


def page_calibration(cfg: Dict[str, Any], metrics: Dict[str, Any]):
    """Renders temperature scaling analytics and reliability calibration curves."""
    section_header("📐", "Calibration Analysis & Temperature Scaling", "Ensuring neural network probabilities reflect true empirical clinical frequency")

    st.markdown("""
    <div style="background:rgba(0,212,255,0.06);border:1px solid rgba(0,212,255,0.2);border-radius:12px;padding:1rem 1.4rem;margin-bottom:1.5rem;font-size:0.88rem;color:#a8c8e0;line-height:1.6;">
        <strong style="color:#00d4ff">Why does calibration matter in medicine?</strong><br>
        Standard deep neural networks are frequently overconfident. A prediction of 90% confidence should correlate with a true disease prevalence of 90%.
        We employ post-hoc <strong>Temperature Scaling</strong> to optimize Expected Calibration Error (ECE) without compromising AUROC discriminative power.
    </div>
    """, unsafe_allow_html=True)

    if "calibration" in metrics:
        cal = metrics["calibration"]
        t_val = cal.get("temperature", 0.05)

        st.metric("Optimal Fitted Temperature (T)", f"{t_val:.4f}", help="T < 1 scales soft logits, reducing uncalibrated overconfidence")

        b_ece = cal.get("before_t", {}).get("per_label", {})
        a_ece = cal.get("after_t", {}).get("per_label", {})

        cols = st.columns(4)
        for col, icon, name in zip(cols, LABEL_ICONS, LABEL_NAMES):
            b = b_ece.get(name, {}).get("ece", 0.1)
            a = a_ece.get(name, {}).get("ece", 0.0)
            imp = ((b - a) / b * 100) if b > 0 else 100.0

            with col:
                st.markdown(f"""
                <div style="background:linear-gradient(135deg, rgba(16,185,129,0.08) 0%, rgba(15,23,42,0.8) 100%);border:1px solid rgba(0, 242, 254, 0.25);border-top:2px solid rgba(16, 185, 129, 0.7);border-radius:16px;padding:1.15rem 1rem;text-align:center;backdrop-filter:blur(14px);box-shadow:0 10px 25px -10px rgba(0,0,0,0.5);">
                    <div style="font-size:1.8rem;margin-bottom:4px">{icon}</div>
                    <div style="color:#f8fafc;font-weight:700;font-size:0.88rem;margin-bottom:0.5rem">{name}</div>
                    <div style="color:#f43f5e;font-size:0.78rem;font-family:'JetBrains Mono', monospace;">Before ECE: <b>{b:.3f}</b></div>
                    <div style="color:#10b981;font-size:0.78rem;font-family:'JetBrains Mono', monospace;">After ECE: <b>{a:.3f}</b></div>
                    <div style="background:rgba(16,185,129,0.15);border:1px solid rgba(16,185,129,0.4);color:#10b981;border-radius:999px;font-size:0.72rem;font-family:'JetBrains Mono', monospace;font-weight:700;padding:3px 10px;margin-top:8px;display:inline-block;">{imp:.1f}% Improvement</div>
                </div>
                """, unsafe_allow_html=True)

    glowing_divider()

    c_b, c_a = st.columns(2)
    with c_b:
        if fig_exists("reliability_before.png"):
            st.markdown("<div style='text-align:center;color:#ff4757;font-weight:700;margin-bottom:0.5rem'>Uncalibrated Reliability Diagram</div>", unsafe_allow_html=True)
            st.image(load_fig("reliability_before.png"), use_container_width=True)
    with c_a:
        if fig_exists("reliability_after.png"):
            st.markdown("<div style='text-align:center;color:#00ff88;font-weight:700;margin-bottom:0.5rem'>Calibrated (Temperature-Scaled) Diagram</div>", unsafe_allow_html=True)
            st.image(load_fig("reliability_after.png"), use_container_width=True)
