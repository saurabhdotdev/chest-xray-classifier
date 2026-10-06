"""
src/ui/pages/shift.py
=====================
Dataset shift and zero-shot external evaluation on NIH ChestX-ray14.
"""

from typing import Dict, Any
import streamlit as st

from src.ui.common import (
    LABEL_NAMES,
    LABEL_ICONS,
    LABEL_COLORS,
    section_header,
)


def page_shift(cfg: Dict[str, Any], metrics: Dict[str, Any]):
    """Renders out-of-distribution evaluation results comparing Stanford CheXpert to NIH ChestX-ray14."""
    section_header("🌍", "Dataset Shift & External Generalization",
                   "CheXpert (Stanford Medicine) → NIH ChestX-ray14 (Bethesda Clinical Center)")

    st.markdown("""
    <div style="background:rgba(0,212,255,0.06);border:1px solid rgba(0,212,255,0.2);border-radius:12px;padding:1rem 1.4rem;margin-bottom:1.5rem;font-size:0.88rem;color:#a8c8e0;line-height:1.6;">
        <strong style="color:#00d4ff">Dataset Shift Challenge:</strong><br>
        Medical AI often degrades when deployed at external hospital systems due to scanner variations, patient demographics, and label extraction discrepancies.
        We report external zero-shot generalization on NIH ChestX-ray14 with non-parametric 95% bootstrap confidence intervals.
    </div>
    """, unsafe_allow_html=True)

    if "external_eval" in metrics:
        ext = metrics["external_eval"]
        nih = ext.get("nih_results", {})
        int_val = ext.get("internal_val_aurocs", {})
        drop = ext.get("drop", {})

        cols = st.columns(4)
        for col, icon, name, color in zip(cols, LABEL_ICONS, LABEL_NAMES, LABEL_COLORS):
            d = drop.get(name, 0.0)
            ci = nih.get(name, {}).get("ci_95", [1.0, 1.0])

            with col:
                st.markdown(f"""
                <div style="background:linear-gradient(135deg, {color}10 0%, rgba(15,23,42,0.8) 100%);border:1px solid {color}44;border-top:2px solid {color}88;border-radius:16px;padding:1.2rem 1rem;text-align:center;backdrop-filter:blur(14px);box-shadow:0 10px 25px -10px {color}22;">
                    <div style="font-size:1.8rem;margin-bottom:4px">{icon}</div>
                    <div style="color:#f8fafc;font-weight:700;font-size:0.88rem;margin-bottom:0.5rem">{name}</div>
                    <div style="color:#38bdf8;font-size:0.78rem;font-family:'JetBrains Mono', monospace;">Internal CheXpert: <b>{int_val.get(name, 1.0):.4f}</b></div>
                    <div style="color:#10b981;font-size:0.78rem;font-family:'JetBrains Mono', monospace;">NIH Zero-Shot: <b>{nih.get(name, {}).get('auroc', 1.0):.4f}</b></div>
                    <div style="color:#94a3b8;font-size:0.72rem;font-family:'JetBrains Mono', monospace;margin-top:2px">95% CI: [{ci[0]:.3f} – {ci[1]:.3f}]</div>
                    <div style="background:rgba(0,242,254,0.12);border:1px solid rgba(0,242,254,0.35);color:#00f2fe;border-radius:999px;font-size:0.72rem;font-family:'JetBrains Mono', monospace;font-weight:700;padding:3px 10px;margin-top:8px;display:inline-block;">Δ Drop: {d:+.4f}</div>
                </div>
                """, unsafe_allow_html=True)
