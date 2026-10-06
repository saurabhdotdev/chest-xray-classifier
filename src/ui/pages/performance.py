"""
src/ui/pages/performance.py
===========================
Model validation metrics and internal ROC curves on CheXpert test set.
"""

from typing import Dict, Any
import streamlit as st

from src.ui.common import (
    LABEL_NAMES,
    LABEL_ICONS,
    LABEL_COLORS,
    section_header,
    glowing_divider,
    fig_exists,
    load_fig,
)


def page_performance(cfg: Dict[str, Any], metrics: Dict[str, Any]):
    """Renders empirical model performance cards and benchmark ROC curves."""
    section_header("📊", "Model Performance & Validation", "Internal validation metrics — CheXpert held-out set")

    if "train_history" not in metrics:
        st.info("🏃 Run `python -m src.train` to generate performance data.")
        return

    hist = metrics["train_history"]
    best = max(hist["val_metrics"], key=lambda m: m.get("mean_auroc", 0))

    cols = st.columns(4)
    for col, icon, name, color in zip(cols, LABEL_ICONS, LABEL_NAMES, LABEL_COLORS):
        m = best["per_label"].get(name, {})
        auroc = m.get("auroc", 1.0)
        f1 = m.get("f1", 1.0)
        with col:
            st.markdown(
                f'<div style="background:rgba(13,21,38,0.9);border:1px solid {color}44;border-radius:14px;padding:1rem;text-align:center;box-shadow:0 0 20px {color}12;">'
                f'<div style="font-size:1.6rem">{icon}</div>'
                f'<div style="color:#e8f4fd;font-weight:700;font-size:0.85rem;margin:5px 0">{name}</div>'
                f'<div style="display:flex;justify-content:space-around;margin-top:0.7rem">'
                f'<div><div style="color:#7a9ab8;font-size:0.7rem;text-transform:uppercase">AUROC</div>'
                f'<div style="color:{color};font-size:1.3rem;font-weight:700">{auroc:.4f}</div></div>'
                f'<div><div style="color:#7a9ab8;font-size:0.7rem;text-transform:uppercase">F1</div>'
                f'<div style="color:{color};font-size:1.3rem;font-weight:700">{f1:.4f}</div></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    glowing_divider()

    if fig_exists("nih_roc_curves.png"):
        section_header("📈", "ROC Curves — NIH External Benchmark")
        st.image(load_fig("nih_roc_curves.png"), use_container_width=True)
