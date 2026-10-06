"""
src/ui/pages/home.py
====================
Executive dashboard and workstation quick-launch hub.
"""

from typing import Dict, Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from src.ui.common import (
    ROOT,
    LABEL_NAMES,
    LABEL_ICONS,
    hero_header,
    section_header,
    glowing_divider,
    status_dot,
)


def page_home(cfg: Dict[str, Any], metrics: Dict[str, Any]):
    """Renders the executive home dashboard and rapid launchpad."""
    hero_header()

    # ── Quick Action Launch Hub ──
    st.markdown("""
    <div style="background:linear-gradient(135deg, rgba(0, 242, 254, 0.08) 0%, rgba(99, 102, 241, 0.06) 100%);border:1px solid rgba(0, 242, 254, 0.28);border-radius:18px;padding:1.3rem 1.8rem;margin-bottom:1.5rem;backdrop-filter:blur(16px);box-shadow:0 12px 30px -10px rgba(0, 242, 254, 0.15);">
        <div style="font-size:1.25rem;font-weight:800;color:#f8fafc;letter-spacing:-0.01em;">
            🚀 Launch Radiograph Workstation
        </div>
        <div style="color:#94a3b8;font-size:0.86rem;margin-top:4px;max-width:680px;line-height:1.5;">
            Upload your personal chest X-ray or choose from verified clinical cases to run DenseNet-121 classification, multi-pathology Grad-CAM heatmaps, and automated PDF reporting.
        </div>
    </div>
    """, unsafe_allow_html=True)

    qa_c1, qa_c2, qa_c3 = st.columns(3)
    with qa_c1:
        if st.button("📤 Upload Your Own X-Ray", key="home_btn_upload", use_container_width=True):
            st.session_state["nav_choice"] = "🔬  Analyze X-ray"
            st.session_state["xray_source_radio"] = "📤  Upload Local Radiograph File"
            st.rerun()
    with qa_c2:
        if st.button("🗂️ Explore Demo Clinical Cases", key="home_btn_demo", use_container_width=True):
            st.session_state["nav_choice"] = "🔬  Analyze X-ray"
            st.session_state["xray_source_radio"] = "🗂️  Preloaded Clinical Cases (Instant Demo)"
            st.rerun()
    with qa_c3:
        if st.button("🔄 Comparative Study (Prior vs Current)", key="home_btn_comp", use_container_width=True):
            st.session_state["nav_choice"] = "🔄  Comparative Study"
            st.rerun()

    glowing_divider()

    # System Status Row
    section_header("⚡", "System Readiness & Pipeline Status", "Real-time module availability")
    c1, c2, c3, c4, c5 = st.columns(5)
    checks = {
        "Base Model Weights":   (ROOT / "results" / "checkpoints" / "best_model.pth").exists(),
        "Calibrated Weights":   (ROOT / "results" / "checkpoints" / "best_model_calibrated.pth").exists(),
        "Training History":     "train_history" in metrics,
        "NIH External Eval":    "external_eval" in metrics,
        "Calibration Matrix":   "calibration" in metrics,
    }
    for col, (label, ok) in zip([c1, c2, c3, c4, c5], checks.items()):
        with col:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg, {'rgba(16,185,129,0.12)' if ok else 'rgba(244,63,94,0.12)'} 0%, rgba(15,23,42,0.8) 100%);border:1px solid {'rgba(16,185,129,0.35)' if ok else 'rgba(244,63,94,0.35)'};border-top:2px solid {'rgba(16,185,129,0.7)' if ok else 'rgba(244,63,94,0.7)'};border-radius:16px;padding:1.2rem 1rem;text-align:center;backdrop-filter:blur(14px);box-shadow:0 10px 25px -10px {'rgba(16,185,129,0.2)' if ok else 'rgba(244,63,94,0.2)'};">
                <div style="font-size:1.6rem;margin-bottom:0.3rem">{'✅' if ok else '⏳'}</div>
                <div style="color:#f8fafc;font-size:0.83rem;font-weight:600;margin-bottom:0.4rem">{label}</div>
                {status_dot(ok)}
            </div>
            """, unsafe_allow_html=True)

    glowing_divider()

    # Quick metrics if training history exists
    if "train_history" in metrics:
        section_header("📈", "Empirical Validation Metrics", "CheXpert & NIH ChestX-ray14 benchmark results")
        hist = metrics["train_history"]
        best = max(hist["val_metrics"], key=lambda m: m.get("mean_auroc", 0))

        cols = st.columns(len(LABEL_NAMES) + 1)
        for col, icon, name in zip(cols, LABEL_ICONS, LABEL_NAMES):
            m = best["per_label"].get(name, {})
            auroc = m.get("auroc", None)
            with col:
                st.metric(label=f"{icon} {name}", value=f"{auroc:.4f}" if auroc else "—")
        with cols[-1]:
            st.metric("🎯 Mean AUROC", f"{best['mean_auroc']:.4f}" if best.get('mean_auroc') else "—")

        glowing_divider()

    # Loss curve
    if "train_history" in metrics:
        section_header("📉", "Model Training Curve", "Binary cross-entropy loss convergence")
        hist = metrics["train_history"]
        fig, ax = plt.subplots(figsize=(10, 3.2))
        epochs = list(range(1, len(hist["train_loss"]) + 1))
        ax.plot(epochs, hist["train_loss"], color="#00d4ff", lw=2.5, label="Train Loss", marker="o", ms=4)
        ax.plot(epochs, hist["val_loss"], color="#00ff88", lw=2.5, label="Val Loss", marker="s", ms=4)
        ax.fill_between(epochs, hist["train_loss"], alpha=0.08, color="#00d4ff")
        ax.fill_between(epochs, hist["val_loss"], alpha=0.08, color="#00ff88")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("BCE Loss", fontsize=10)
        ax.set_title("DenseNet-121 Convergence (5 Epochs)", fontsize=12, fontweight="bold", color="#e8f4fd", pad=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.35)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    glowing_divider()

    # Project Overview Cards
    section_header("🗺️", "Architecture & Workflow Pipeline")
    pipeline_cards = [
        ("📥", "Multi-Modal Input", "Upload, Web URL, Camera & Demos"),
        ("🛠️", "PACS Windowing", "CLAHE, Invert, Contrast & Level"),
        ("🧠", "DenseNet-121", "Multi-label CheXpert backbone"),
        ("🌡️", "Temp Calibration", "Post-hoc ECE temperature scaling"),
        ("🔥", "Grad-CAM Matrix", "Simultaneous 4-pathology heatmaps"),
        ("📄", "Structured Report", "RSNA format & printable PDF export"),
    ]
    cards_html = "".join([
        f'<div style="flex:1;min-width:150px;background:rgba(13,21,38,0.9);border:1px solid rgba(0,212,255,0.2);border-radius:14px;padding:1rem;text-align:center;">'
        f'<div style="font-size:1.8rem">{icon}</div>'
        f'<div style="color:#e8f4fd;font-weight:700;font-size:0.88rem;margin:6px 0 3px">{title}</div>'
        f'<div style="color:#7a9ab8;font-size:0.75rem">{desc}</div>'
        f'</div>'
        for icon, title, desc in pipeline_cards
    ])
    st.markdown(f'<div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:1rem;">{cards_html}</div>', unsafe_allow_html=True)
