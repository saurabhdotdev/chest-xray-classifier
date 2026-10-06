"""
src/ui/components/findings_summary.py
=====================================
Multi-dimensional AI findings summary featuring polar radar analysis,
horizontal calibrated-vs-raw confidence spectrum, and clinical interpretations.
"""

import math
from typing import Sequence
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.ui.common import LABEL_NAMES, LABEL_ICONS, LABEL_COLORS


def render_ai_findings_summary(
    active_probs: Sequence[float],
    raw_probs: Sequence[float],
    threshold: float
):
    """Renders a two-panel AI findings summary: radar chart + horizontal bar chart + clinical interpretation."""
    chart_col, bar_col = st.columns([1, 1])

    # ── Radar / Spider Chart ─────────────────────────────────────────────────
    with chart_col:
        st.markdown(
            "<div style='text-align:center; color:#94a3b8; font-size:0.82rem; "
            "font-weight:600; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.5rem'>"
            "🕸 Pathology Probability Radar"
            "</div>",
            unsafe_allow_html=True,
        )

        N = len(LABEL_NAMES)
        angles = [n * 2 * math.pi / N for n in range(N)]
        angles += angles[:1]  # close polygon

        cal_vals = list(active_probs) + [active_probs[0]]
        raw_vals = list(raw_probs) + [raw_probs[0]]
        thresh_v = [threshold] * (N + 1)

        fig_r, ax_r = plt.subplots(figsize=(4.8, 4.8), subplot_kw=dict(polar=True))
        fig_r.patch.set_facecolor("#070d1b")
        ax_r.set_facecolor("#070d1b")

        # Concentric grid rings
        for ring in [0.25, 0.5, 0.75, 1.0]:
            ring_alpha = 0.07 if ring < 1.0 else 0.15
            ax_r.plot(angles, [ring] * (N + 1),
                      color=(1, 1, 1, ring_alpha),
                      lw=0.7, ls="--")

        # Spoke lines
        for a in angles[:-1]:
            ax_r.plot([0, a], [0, 1], color=(1, 1, 1, 0.1), lw=0.6)

        # Threshold ring (amber)
        ax_r.plot(angles, thresh_v, color="#f59e0b", lw=1.2, ls=":", label=f"Threshold ({threshold:.0%})")
        ax_r.fill(angles, thresh_v, color="#f59e0b", alpha=0.04)

        # Raw probs (muted indigo)
        ax_r.plot(angles, raw_vals, color="#6366f1", lw=1.3, ls="--", alpha=0.65, label="Raw Probs")
        ax_r.fill(angles, raw_vals, color="#6366f1", alpha=0.08)

        # Calibrated probs (cyan glow)
        ax_r.plot(angles, cal_vals, color="#00f2fe", lw=2.2, label="Calibrated Probs")
        ax_r.fill(angles, cal_vals, color="#00f2fe", alpha=0.18)

        # Data point markers
        ax_r.scatter(angles[:-1], list(active_probs),
                     c=["#f43f5e" if p >= threshold else "#10b981" for p in active_probs],
                     s=62, zorder=5, edgecolors="white", linewidths=0.8)

        # Axis labels (plain English labels to avoid unicode warnings in matplotlib)
        ax_r.set_xticks(angles[:-1])
        ax_r.set_xticklabels(LABEL_NAMES, fontsize=8.5, color="#cbd5e1")
        ax_r.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax_r.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=6.5, color="#475569")
        ax_r.set_ylim(0, 1)
        ax_r.spines["polar"].set_color("#1e3a5f")
        ax_r.tick_params(colors="#475569")
        ax_r.grid(False)

        ax_r.legend(
            loc="upper right", bbox_to_anchor=(1.35, 1.18),
            fontsize=7.5, framealpha=0.0,
            labelcolor="#94a3b8"
        )
        plt.tight_layout(pad=1.2)
        st.pyplot(fig_r, use_container_width=True)
        plt.close(fig_r)

    # ── Horizontal Bar Chart + Risk Gauge ────────────────────────────────────
    with bar_col:
        st.markdown(
            "<div style='text-align:center; color:#94a3b8; font-size:0.82rem; "
            "font-weight:600; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.5rem'>"
            "📊 Confidence Spectrum"
            "</div>",
            unsafe_allow_html=True,
        )

        bar_colors = [
            "#f43f5e" if p >= threshold else "#10b981"
            for p in active_probs
        ]
        bar_colors_raw = [c + "55" for c in bar_colors]  # faded for raw

        fig_b, ax_b = plt.subplots(figsize=(4.8, 3.6))
        fig_b.patch.set_facecolor("#070d1b")
        ax_b.set_facecolor("#0a1020")

        y_pos = range(len(LABEL_NAMES))
        bar_h = 0.34

        # Raw bars (faded, behind)
        ax_b.barh(
            [y - bar_h / 2 for y in y_pos], raw_probs,
            height=bar_h, color=bar_colors_raw, label="Raw", zorder=2
        )
        # Calibrated bars (solid, front)
        cal_bars = ax_b.barh(
            [y + bar_h / 2 for y in y_pos], active_probs,
            height=bar_h, color=bar_colors, label="Calibrated", zorder=3
        )

        # Threshold line
        ax_b.axvline(threshold, color="#f59e0b", lw=1.4, ls="--", label=f"Threshold ({threshold:.0%})", zorder=4)

        # Value labels on calibrated bars
        for p, bar in zip(active_probs, cal_bars):
            ax_b.text(
                min(p + 0.015, 0.97), bar.get_y() + bar.get_height() / 2,
                f"{p:.1%}",
                va="center", ha="left",
                color="white", fontsize=8, fontweight="bold"
            )

        ax_b.set_yticks(list(y_pos))
        ax_b.set_yticklabels(LABEL_NAMES, fontsize=9, color="#cbd5e1")
        ax_b.set_xlim(0, 1.08)
        ax_b.set_xlabel("Probability", fontsize=8.5, color="#64748b")
        ax_b.set_title("Calibrated vs Raw Confidence", fontsize=9.5,
                       fontweight="bold", color="#e2e8f0", pad=8)
        ax_b.spines[["top", "right"]].set_visible(False)
        ax_b.spines[["left", "bottom"]].set_color("#1e2d45")
        ax_b.tick_params(axis="x", colors="#475569", labelsize=7.5)
        ax_b.tick_params(axis="y", colors="#94a3b8")
        ax_b.set_facecolor("#0a1020")
        ax_b.grid(axis="x", color=(1, 1, 1, 0.05), lw=0.7, zorder=0)
        ax_b.legend(fontsize=7.5, framealpha=0.0, loc="lower right")
        plt.tight_layout(pad=1.2)
        st.pyplot(fig_b, use_container_width=True)
        plt.close(fig_b)

    # ── Clinical Interpretation Panel ────────────────────────────────────────
    n_pos = sum(1 for p in active_probs if p >= threshold)
    mean_conf = float(np.mean(active_probs))
    max_idx = int(np.argmax(active_probs))
    dom_name = LABEL_NAMES[max_idx]
    dom_prob = float(active_probs[max_idx])
    dom_icon = LABEL_ICONS[max_idx]

    if n_pos == 0:
        interp_icon, interp_color, interp_title, interp_body = (
            "✅", "#10b981",
            "All Pathologies Below Detection Threshold",
            f"The model assigns sub-threshold probability to all four monitored pathologies. "
            f"The dominant signal is <b>{dom_icon} {dom_name}</b> at <b>{dom_prob:.1%}</b>, "
            f"which remains below the operating threshold of <b>{threshold:.0%}</b>. "
            "These findings are consistent with a normal or near-normal chest radiograph under the current sensitivity setting."
        )
    elif n_pos == 1:
        interp_icon, interp_color, interp_title, interp_body = (
            "⚠️", "#f59e0b",
            f"Single Pathology Flagged: {dom_icon} {dom_name}",
            f"The model detects <b>{dom_icon} {dom_name}</b> with <b>{dom_prob:.1%}</b> confidence. "
            f"The remaining three pathologies remain below the <b>{threshold:.0%}</b> operating threshold. "
            "Recommend focused radiological review of the anatomical region highlighted by the Grad-CAM heatmap."
        )
    else:
        pos_names = ", ".join(
            f"{LABEL_ICONS[i]} {LABEL_NAMES[i]}"
            for i, p in enumerate(active_probs) if p >= threshold
        )
        interp_icon, interp_color, interp_title, interp_body = (
            "🚨", "#f43f5e",
            f"{n_pos} Co-occurring Pathologies Detected",
            f"The model flags <b>{pos_names}</b> simultaneously above the <b>{threshold:.0%}</b> threshold. "
            f"Multi-pathology co-occurrence may indicate a complex clinical syndrome (e.g. CHF decompensation "
            f"with cardiomegaly, pulmonary edema, and bilateral effusions). Mean confidence: <b>{mean_conf:.1%}</b>. "
            "Immediate specialist review is recommended."
        )

    st.markdown(
        f'<div style="background:linear-gradient(135deg, {interp_color}0e 0%, rgba(13,21,38,0.9) 100%);'
        f'border:1px solid {interp_color}44;border-left:4px solid {interp_color};border-radius:14px;'
        f'padding:1.1rem 1.5rem;margin-top:1.2rem;backdrop-filter:blur(14px);">'
        f'<div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.5rem">'
        f'<span style="font-size:1.5rem">{interp_icon}</span>'
        f'<div style="color:#f8fafc;font-weight:700;font-size:1rem">{interp_title}</div>'
        f'</div>'
        f'<div style="color:#a8c8e0;font-size:0.85rem;line-height:1.65">{interp_body}</div>'
        f'<div style="display:flex;gap:2rem;margin-top:0.85rem;padding-top:0.75rem;'
        f'border-top:1px solid rgba(255,255,255,0.07);font-family:\'JetBrains Mono\', monospace;font-size:0.75rem;">'
        f'<span style="color:#64748b">Pathologies flagged: <b style="color:{interp_color}">{n_pos} / 4</b></span>'
        f'<span style="color:#64748b">Mean confidence: <b style="color:#38bdf8">{mean_conf:.1%}</b></span>'
        f'<span style="color:#64748b">Dominant: <b style="color:#f8fafc">{dom_icon} {dom_name} ({dom_prob:.1%})</b></span>'
        f'<span style="color:#64748b">Threshold: <b style="color:#f59e0b">{threshold:.0%}</b></span>'
        f'</div></div>',
        unsafe_allow_html=True
    )
