"""
src/ui/pages/compare.py
=======================
Longitudinal comparative study, interval pixel subtraction, morph blend, and attention shift analysis.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import streamlit as st

from src.gradcam import overlay_heatmap
from src.image_utils import load_image_from_bytes
from src.ui.common import (
    ROOT,
    LABEL_NAMES,
    LABEL_ICONS,
    section_header,
    glowing_divider,
)
from src.ui.model_service import (
    get_model_and_cam,
    get_image_transforms,
)
from src.ui.components.interval_engine import (
    compute_interval_subtraction,
    compute_cross_dissolve,
    compute_attention_shift_map,
    compute_or_get_longitudinal_analysis,
)


def page_compare(cfg: Dict[str, Any], metrics: Dict[str, Any]):
    """Renders longitudinal comparative study workstation and interval subtraction engine."""
    section_header("🔄", "Longitudinal Comparative Study & Interval Subtraction Engine",
                   "Automated pixel subtraction, trajectory velocity analysis & neural attention shift tracking")

    st.markdown("""
<div style="background:linear-gradient(135deg,rgba(0,242,254,0.06) 0%,rgba(99,102,241,0.05) 100%);border:1px solid rgba(0,242,254,0.22);border-radius:14px;padding:0.95rem 1.4rem;font-size:0.86rem;color:#cbd5e1;margin-bottom:1.4rem;line-height:1.6;">
    <span style="color:#00f2fe;font-weight:700;">Clinical Longitudinal Workflow:</span> Radiologists assess therapeutic efficacy by comparing follow-up radiographs against baseline studies.
    This module performs <b>automated computer-vision pixel subtraction</b> to highlight new or resolving parenchymal opacities, tracks <b>multi-pathology trajectory velocity (Δ)</b>, and computes <b>neural attention shifts</b>.
</div>
""", unsafe_allow_html=True)

    model, cam_gen, device, fitted_temp = get_model_and_cam()
    tf = get_image_transforms()

    longitudinal_presets = {
        "🌊 CHF Decompensation ➔ Rapid Diuresis (Day 1 vs Day 4)": {
            "prior_file": "sample_pulmonary_edema.jpg",
            "curr_file": "sample_normal.jpg",
            "patient": "PT-78214 (68yo F)",
            "interval": "4 Days",
            "therapy": "IV Furosemide diuresis & fluid restriction",
            "indication": "Decompensated heart failure with severe bilateral alveolar edema",
        },
        "💧 Right Pleural Effusion ➔ Post-Thoracentesis Resolution": {
            "prior_file": "sample_pleural_effusion.jpg",
            "curr_file": "sample_normal.jpg",
            "patient": "PT-93041 (54yo M)",
            "interval": "2 Days",
            "therapy": "Ultrasound-guided right thoracentesis (1,200 mL drained)",
            "indication": "Blunted right costophrenic angle and restrictive dyspnea",
        },
        "🫁 Baseline Normal ➔ Post-Operative Atelectasis Evolution": {
            "prior_file": "sample_normal.jpg",
            "curr_file": "sample_atelectasis.jpg",
            "patient": "PT-44109 (71yo M)",
            "interval": "3 Days",
            "therapy": "Exploratory laparotomy; incentive spirometry initiated",
            "indication": "Pre-op baseline vs Day 2 post-op hypoxemia evaluation",
        },
        "🫀 Progressive Cardiomegaly & Ventricular Dilatation": {
            "prior_file": "sample_normal.jpg",
            "curr_file": "sample_cardiomegaly.jpg",
            "patient": "PT-61920 (62yo M)",
            "interval": "6 Months",
            "therapy": "Outpatient cardiomyopathy monitoring; titrating ACE-I/Beta-blocker",
            "indication": "Longitudinal surveillance of transverse cardiac diameter (CTR)",
        },
        "📤 Custom Patient Scans (Upload Prior & Current Files)": {
            "prior_file": None,
            "curr_file": None,
            "patient": "Custom Patient",
            "interval": "Interval Study",
            "therapy": "Clinical Management",
            "indication": "Comparative Radiographic Evaluation",
        },
    }

    preset_names = list(longitudinal_presets.keys())
    selected_preset = st.selectbox("📂 Select Longitudinal Case Scenario or Upload:", preset_names, index=0)
    preset_data = longitudinal_presets[selected_preset]

    img_prior = None
    img_curr = None
    patient_label = preset_data["patient"]
    interval_label = preset_data["interval"]
    therapy_label = preset_data["therapy"]
    indication_label = preset_data["indication"]

    if preset_data["prior_file"] is not None:
        p_path = ROOT / "data" / "samples" / preset_data["prior_file"]
        c_path = ROOT / "data" / "samples" / preset_data["curr_file"]
        if p_path.exists() and c_path.exists():
            img_prior = Image.open(p_path).convert("RGB")
            img_curr = Image.open(c_path).convert("RGB")
    else:
        u_col1, u_col2 = st.columns(2)
        with u_col1:
            st.markdown("<div style='font-weight:700;color:#00d4ff;margin-bottom:0.4rem'>📅 Prior Study (Baseline)</div>", unsafe_allow_html=True)
            up_p = st.file_uploader("Upload prior baseline scan", type=["jpg", "jpeg", "png"], key="up_p_custom")
            if up_p:
                img_prior = load_image_from_bytes(up_p.getvalue())
        with u_col2:
            st.markdown("<div style='font-weight:700;color:#00ff88;margin-bottom:0.4rem'>📅 Current Study (Follow-up)</div>", unsafe_allow_html=True)
            up_c = st.file_uploader("Upload current follow-up scan", type=["jpg", "jpeg", "png"], key="up_c_custom")
            if up_c:
                img_curr = load_image_from_bytes(up_c.getvalue())

    if img_prior is None or img_curr is None:
        st.info("👈 Please ensure both Prior and Current radiographs are selected above to run comparative analysis.")
        return

    # Metadata Strip
    meta_html = (
        f'<div style="background:rgba(15,23,42,0.85);border:1px solid rgba(0,242,254,0.18);border-radius:12px;padding:0.75rem 1.2rem;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:1.4rem;">'
        f'<div><span style="color:#64748b;font-size:0.72rem;font-family:\'JetBrains Mono\',monospace;">PATIENT</span><br><b style="color:#f8fafc;font-size:0.86rem;">{patient_label}</b></div>'
        f'<div><span style="color:#64748b;font-size:0.72rem;font-family:\'JetBrains Mono\',monospace;">INTERVAL</span><br><b style="color:#00f2fe;font-size:0.86rem;">⏱️ {interval_label}</b></div>'
        f'<div><span style="color:#64748b;font-size:0.72rem;font-family:\'JetBrains Mono\',monospace;">INTERVENING THERAPY</span><br><b style="color:#cbd5e1;font-size:0.84rem;">💊 {therapy_label}</b></div>'
        f'<div><span style="color:#64748b;font-size:0.72rem;font-family:\'JetBrains Mono\',monospace;">CLINICAL INDICATION</span><br><b style="color:#a8c8e0;font-size:0.84rem;">🎯 {indication_label}</b></div>'
        f'</div>'
    )
    st.markdown(meta_html, unsafe_allow_html=True)

    # Run dual inference with cache
    with st.spinner("🔄 Running dual deep-learning comparative analysis & Grad-CAM extraction..."):
        analysis = compute_or_get_longitudinal_analysis(
            model, cam_gen, device, img_prior, img_curr, tf, fitted_temp
        )

    p_prior = analysis["p_prior"]
    np_prior = analysis["np_prior"]
    cams_prior = analysis["cams_prior"]
    p_curr = analysis["p_curr"]
    np_curr = analysis["np_curr"]
    cams_curr = analysis["cams_curr"]

    deltas = p_curr - p_prior
    net_delta = float(np.sum(deltas))
    max_worsen_idx = int(np.argmax(deltas))
    max_improve_idx = int(np.argmin(deltas))
    max_worsen_val = float(deltas[max_worsen_idx])
    max_improve_val = float(deltas[max_improve_idx])

    # Trajectory Classification
    if net_delta <= -0.25 or max_improve_val <= -0.30:
        traj_status = "MARKED THERAPEUTIC RESOLUTION"
        traj_badge = "THERAPEUTIC CLEARANCE"
        traj_color = "#10b981"
        traj_icon = "✅"
        traj_summary = (
            f"Marked therapeutic response observed with notable clearance of <b>{LABEL_ICONS[max_improve_idx]} {LABEL_NAMES[max_improve_idx]}</b> "
            f"({max_improve_val:.1%}). Net disease burden across all 4 findings reduced by <b>{abs(net_delta):.1%}</b>. "
            "Parenchymal aerated volumes have substantially improved following intervening therapy."
        )
    elif net_delta >= 0.25 or max_worsen_val >= 0.30:
        traj_status = "ACUTE INTERVAL DETERIORATION"
        traj_badge = "PROGRESSIVE INFILTRATE"
        traj_color = "#f43f5e"
        traj_icon = "🚨"
        traj_summary = (
            f"Significant interval disease progression detected (+{max_worsen_val:.1%} in <b>{LABEL_ICONS[max_worsen_idx]} {LABEL_NAMES[max_worsen_idx]}</b>). "
            f"Net pathology expansion: <b>{net_delta:+.1%}</b>. "
            "Suggests acute clinical decompensation or expanding consolidation. Immediate specialist clinical evaluation recommended."
        )
    elif max_worsen_val > 0.12 and max_improve_val < -0.12:
        traj_status = "MIXED INTERVAL RESPONSE"
        traj_badge = "DIVERGENT EVOLUTION"
        traj_color = "#f59e0b"
        traj_icon = "⚠️"
        traj_summary = (
            f"Divergent interval findings: improvement in <b>{LABEL_ICONS[max_improve_idx]} {LABEL_NAMES[max_improve_idx]}</b> ({max_improve_val:.1%}) "
            f"accompanied by concurrent progression in <b>{LABEL_ICONS[max_worsen_idx]} {LABEL_NAMES[max_worsen_idx]}</b> (+{max_worsen_val:.1%}). "
            "Correlate with clinical trajectory and laboratory parameters."
        )
    else:
        traj_status = "CLINICALLY STABLE STUDY"
        traj_badge = "STATIONARY INTERVAL"
        traj_color = "#38bdf8"
        traj_icon = "↔"
        traj_summary = (
            "Pathology markers remain within stability tolerance (±5%). No acute new consolidations or significant interval "
            "ventricular caliber shifts detected. Findings consistent with chronic baseline stability."
        )

    # ── Executive Assessment Banner ──
    banner_html = (
        f'<div style="background:linear-gradient(135deg, {traj_color}18 0%, rgba(15,23,42,0.9) 100%);border:1px solid {traj_color}55;border-top:3px solid {traj_color};border-radius:16px;padding:1.25rem 1.6rem;backdrop-filter:blur(16px);margin-bottom:1.5rem;box-shadow:0 12px 30px -10px {traj_color}22;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;margin-bottom:0.6rem;">'
        f'<div style="display:flex;align-items:center;gap:12px;">'
        f'<div style="width:46px;height:46px;border-radius:12px;background:{traj_color}22;border:1px solid {traj_color}55;display:flex;align-items:center;justify-content:center;font-size:1.6rem;box-shadow:0 0 15px {traj_color}25;">{traj_icon}</div>'
        f'<div>'
        f'<div style="color:#ffffff;font-weight:800;font-size:1.15rem;letter-spacing:-0.01em;">{traj_status}</div>'
        f'<div style="color:{traj_color};font-size:0.78rem;font-family:\'JetBrains Mono\',monospace;font-weight:700;">{traj_badge} · NET VELOCITY: {net_delta:+.1%}</div>'
        f'</div></div>'
        f'<div style="text-align:right;">'
        f'<span style="background:{traj_color}22;border:1px solid {traj_color}66;color:{traj_color};padding:0.4rem 1.1rem;border-radius:999px;font-family:\'JetBrains Mono\',monospace;font-size:0.76rem;font-weight:700;">INTERVAL: {interval_label}</span>'
        f'</div></div>'
        f'<div style="color:#cbd5e1;font-size:0.86rem;line-height:1.6;">{traj_summary}</div>'
        f'</div>'
    )
    st.markdown(banner_html, unsafe_allow_html=True)

    # ── 4 Comparative Pathology Delta Cards ──
    comp_cols = st.columns(4)
    for col, icon, name, pp, pc in zip(comp_cols, LABEL_ICONS, LABEL_NAMES, p_prior, p_curr):
        delta = pc - pp
        if abs(delta) < 0.05:
            traj_text = "Stable"
            traj_color_c = "#94a3b8"
            arrow = "↔"
        elif delta < 0:
            traj_text = f"Resolving ({abs(delta):.1%} ↓)"
            traj_color_c = "#10b981"
            arrow = "↓"
        else:
            traj_text = f"Worsening ({delta:.1%} ↑)"
            traj_color_c = "#f43f5e"
            arrow = "↑"

        with col:
            card_html = (
                f'<div style="background:linear-gradient(135deg, {traj_color_c}10 0%, rgba(15,23,42,0.82) 100%);border:1px solid {traj_color_c}44;border-top:2px solid {traj_color_c}88;border-radius:16px;padding:1.1rem 0.9rem;text-align:center;backdrop-filter:blur(14px);">'
                f'<div style="font-size:1.9rem;margin-bottom:3px;">{icon}</div>'
                f'<div style="color:#f8fafc;font-weight:700;font-size:0.88rem;margin-bottom:0.4rem;">{name}</div>'
                f'<div style="display:flex;justify-content:space-around;align-items:center;margin:0.5rem 0;font-family:\'JetBrains Mono\',monospace;">'
                f'<div><span style="font-size:0.65rem;color:#64748b;">PRIOR</span><br><b style="color:#38bdf8;font-size:1.05rem;">{pp:.0%}</b></div>'
                f'<div><span style="font-size:0.95rem;color:#64748b;">➔</span></div>'
                f'<div><span style="font-size:0.65rem;color:#64748b;">CURRENT</span><br><b style="color:#10b981;font-size:1.05rem;">{pc:.0%}</b></div>'
                f'</div>'
                f'<div style="background:{traj_color_c}18;border:1px solid {traj_color_c}55;color:{traj_color_c};padding:3px 10px;border-radius:999px;font-size:0.72rem;font-family:\'JetBrains Mono\',monospace;font-weight:700;margin-top:6px;">'
                f'{arrow} {traj_text}</div>'
                f'</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)

    # ── SUBTRACTION STUDIO & VIEWPORT ──
    glowing_divider()
    section_header("🔬", "Interval Subtraction Studio & Longitudinal Viewport",
                   "Computer-vision pixel subtraction, interactive morph blend & deep attention shift")

    sub_mode = st.radio(
        "Display Mode",
        [
            "⚡ Computer-Vision Pixel Subtraction (Red: Worsened · Green: Resolved)",
            "🔬 Synchronized Dual Split View",
            "🎚️ Interactive Alpha Cross-Dissolve / Morph Slider",
            "🔥 Deep Attention Shift (Grad-CAM Delta Matrix)",
        ],
        index=0,
        horizontal=True,
        key="compare_view_mode"
    )

    if "Pixel Subtraction" in sub_mode:
        ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([1.5, 1.2, 1.2])
        with ctrl_c1:
            sub_thresh = st.slider("Subtle Noise Gate Threshold", 8.0, 36.0, 18.0, 1.0, help="Higher values filter out minor patient positioning or respiratory volume variations.")
        with ctrl_c2:
            auto_norm = st.toggle("Exposure Auto-Normalization", value=True, help="Equalizes overall exposure and photon variance between the two studies.")
        with ctrl_c3:
            st.markdown("<div style='margin-top:0.4rem;'></div>", unsafe_allow_html=True)
            st.caption("🔴 Red = New Infiltration/Effusion/Heart Size<br>🟢 Green = Cleared Edema/Drained Fluid", unsafe_allow_html=True)

        sub_rgb, pos_pct, neg_pct = compute_interval_subtraction(
            img_prior, img_curr, threshold_val=sub_thresh, normalize_exposure=auto_norm
        )

        sub_col1, sub_col2 = st.columns([1.6, 1.0])
        with sub_col1:
            st.image(sub_rgb, caption=f"Computer-Vision Interval Subtraction (Noise Gate: {sub_thresh:.0f} · Exposure Norm: {auto_norm})", use_container_width=True)
        with sub_col2:
            net_parenchymal = pos_pct - neg_pct
            net_p_color = "#f43f5e" if net_parenchymal > 0.02 else ("#10b981" if net_parenchymal < -0.02 else "#38bdf8")
            st.markdown(f"""
<div style="background:rgba(15,23,42,0.85);border:1px solid rgba(0,242,254,0.22);border-radius:14px;padding:1.2rem;backdrop-filter:blur(14px);">
    <div style="color:#00f2fe;font-size:0.78rem;font-weight:700;font-family:'JetBrains Mono',monospace;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">
        Quantitative Parenchymal Delta
    </div>
    <div style="background:rgba(244,63,94,0.12);border:1px solid rgba(244,63,94,0.35);border-radius:10px;padding:0.75rem 1rem;margin-bottom:0.7rem;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="color:#fda4af;font-size:0.8rem;font-weight:600;">🔴 New / Worsened Opacity Area:</span>
            <span style="color:#f43f5e;font-size:1.15rem;font-weight:800;font-family:'JetBrains Mono',monospace;">{pos_pct:.1%}</span>
        </div>
        <div style="color:#fda4af;font-size:0.7rem;margin-top:2px;">Fluid accumulation, new consolidation, or cardiomegaly expansion</div>
    </div>
    <div style="background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.35);border-radius:10px;padding:0.75rem 1rem;margin-bottom:0.7rem;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="color:#6ee7b7;font-size:0.8rem;font-weight:600;">🟢 Resolved / Cleared Opacity Area:</span>
            <span style="color:#10b981;font-size:1.15rem;font-weight:800;font-family:'JetBrains Mono',monospace;">{neg_pct:.1%}</span>
        </div>
        <div style="color:#6ee7b7;font-size:0.7rem;margin-top:2px;">Alveolar clearing, diuresis response, or re-expanded lung</div>
    </div>
    <div style="border-top:1px solid rgba(255,255,255,0.08);padding-top:0.75rem;display:flex;justify-content:space-between;align-items:center;">
        <span style="color:#94a3b8;font-size:0.78rem;">Net Parenchymal Trajectory:</span>
        <span style="color:{net_p_color};font-size:1.05rem;font-weight:800;font-family:'JetBrains Mono',monospace;">{net_parenchymal:+.1%}</span>
    </div>
</div>
""", unsafe_allow_html=True)

    elif "Synchronized Dual Split" in sub_mode:
        s_c1, s_c2 = st.columns(2)
        with s_c1:
            st.image(np_prior, caption=f"Prior Radiograph — Baseline ({patient_label})", use_container_width=True)
            st.markdown("<div style='text-align:center;font-size:0.78rem;color:#38bdf8;font-family:\"JetBrains Mono\",monospace;'>Study Date: Baseline Study</div>", unsafe_allow_html=True)
        with s_c2:
            st.image(np_curr, caption=f"Current Radiograph — Follow-up (+{interval_label})", use_container_width=True)
            st.markdown("<div style='text-align:center;font-size:0.78rem;color:#10b981;font-family:\"JetBrains Mono\",monospace;'>Study Date: Follow-up Study</div>", unsafe_allow_html=True)

    elif "Cross-Dissolve" in sub_mode:
        st.markdown("<div style='color:#94a3b8;font-size:0.84rem;margin-bottom:0.6rem;'>Drag the slider below to smoothly morph from Prior Baseline to Follow-up study:</div>", unsafe_allow_html=True)
        morph_val = st.slider("Morph Progression: Prior (0%) ➔ Follow-up (100%)", 0.0, 1.0, 0.5, 0.02)
        blended_img = compute_cross_dissolve(img_prior, img_curr, alpha_curr=morph_val)

        m_col1, m_col2, m_col3 = st.columns([1, 2, 1])
        with m_col2:
            st.image(blended_img, caption=f"Dynamic Morph Viewport ({morph_val:.0%} Follow-up / {1.0 - morph_val:.0%} Prior)", use_container_width=True)

    elif "Attention Shift" in sub_mode:
        selected_cam_name = st.selectbox(
            "Select Pathology to Inspect Neural Attention Shift:",
            [f"{ic} {nm}" for ic, nm in zip(LABEL_ICONS, LABEL_NAMES)],
            index=0,
            key="cam_shift_select"
        )
        cam_idx = [f"{ic} {nm}" for ic, nm in zip(LABEL_ICONS, LABEL_NAMES)].index(selected_cam_name)

        shift_col1, shift_col2, shift_col3 = st.columns(3)
        with shift_col1:
            st.image(overlay_heatmap(np_prior, cams_prior[cam_idx], alpha=0.45), caption=f"Prior Grad-CAM Focus ({p_prior[cam_idx]:.1%})", use_container_width=True)
        with shift_col2:
            shift_overlay = compute_attention_shift_map(cams_prior[cam_idx], cams_curr[cam_idx], img_curr)
            st.image(shift_overlay, caption="Neural Attention Shift (Red: Focus Expanded · Green: Focus Relaxed)", use_container_width=True)
        with shift_col3:
            st.image(overlay_heatmap(np_curr, cams_curr[cam_idx], alpha=0.45), caption=f"Current Grad-CAM Focus ({p_curr[cam_idx]:.1%})", use_container_width=True)

    # ── POLAR RADAR & BAR CHARTS ──
    glowing_divider()
    section_header("📊", "Longitudinal Pathology Shift Analysis", "Polar landscape overlay & comparative confidence spectrum")

    rad_c1, rad_c2 = st.columns(2)
    with rad_c1:
        st.markdown("<div style='text-align:center;color:#94a3b8;font-size:0.82rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;'>🕸 Longitudinal Probability Radar</div>", unsafe_allow_html=True)
        N = len(LABEL_NAMES)
        angles = [n * 2 * np.pi / N for n in range(N)]
        angles += angles[:1]

        val_p = list(p_prior) + [p_prior[0]]
        val_c = list(p_curr) + [p_curr[0]]

        fig_r, ax_r = plt.subplots(figsize=(4.4, 4.4), subplot_kw=dict(polar=True))
        fig_r.patch.set_facecolor("#070d1b")
        ax_r.set_facecolor("#070d1b")

        for ring in [0.25, 0.50, 0.75, 1.0]:
            ax_r.plot(angles, [ring] * (N + 1), color=(1, 1, 1, 0.08), lw=0.7, ls="--")

        ax_r.plot(angles, val_p, color="#6366f1", lw=1.8, ls="--", label="Prior (Baseline)")
        ax_r.fill(angles, val_p, color="#6366f1", alpha=0.10)

        ax_r.plot(angles, val_c, color="#00f2fe", lw=2.4, label="Current (Follow-up)")
        ax_r.fill(angles, val_c, color="#00f2fe", alpha=0.20)

        ax_r.set_xticks(angles[:-1])
        ax_r.set_xticklabels(LABEL_NAMES, fontsize=8.5, color="#cbd5e1")
        ax_r.set_yticks([0.25, 0.50, 0.75, 1.0])
        ax_r.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=6.5, color="#475569")
        ax_r.set_ylim(0, 1)
        ax_r.spines["polar"].set_color("#1e3a5f")
        ax_r.grid(False)
        ax_r.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=7.5, framealpha=0.0, labelcolor="#94a3b8")
        plt.tight_layout()
        st.pyplot(fig_r, use_container_width=True)
        plt.close(fig_r)

    with rad_c2:
        st.markdown("<div style='text-align:center;color:#94a3b8;font-size:0.82rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;'>📊 Comparative Confidence Spectrum</div>", unsafe_allow_html=True)
        fig_b, ax_b = plt.subplots(figsize=(4.8, 3.8))
        fig_b.patch.set_facecolor("#070d1b")
        ax_b.set_facecolor("#0a1020")

        y_pos = np.arange(len(LABEL_NAMES))
        bar_h = 0.35

        ax_b.barh(y_pos - bar_h / 2, p_prior, height=bar_h, color="#6366f1", alpha=0.85, label="Prior (Baseline)")
        ax_b.barh(y_pos + bar_h / 2, p_curr, height=bar_h, color="#00f2fe", alpha=0.90, label="Current (Follow-up)")

        for i, (pp, pc) in enumerate(zip(p_prior, p_curr)):
            ax_b.text(pp + 0.02, i - bar_h / 2, f"{pp:.0%}", va="center", color="#a5b4fc", fontsize=7.5, fontweight="bold")
            ax_b.text(pc + 0.02, i + bar_h / 2, f"{pc:.0%}", va="center", color="#67e8f9", fontsize=7.5, fontweight="bold")

        ax_b.set_yticks(y_pos)
        ax_b.set_yticklabels(LABEL_NAMES, fontsize=9, color="#cbd5e1")
        ax_b.set_xlim(0, 1.15)
        ax_b.set_xlabel("Probability", fontsize=8, color="#64748b")
        ax_b.spines[["top", "right"]].set_visible(False)
        ax_b.spines[["left", "bottom"]].set_color("#1e2d45")
        ax_b.tick_params(axis="x", colors="#475569", labelsize=7.5)
        ax_b.tick_params(axis="y", colors="#94a3b8")
        ax_b.grid(axis="x", color=(1, 1, 0.05), lw=0.7)
        ax_b.legend(fontsize=7.5, framealpha=0.0, loc="lower right")
        plt.tight_layout()
        st.pyplot(fig_b, use_container_width=True)
        plt.close(fig_b)

    # ── STRUCTURED LONGITUDINAL IMPRESSION ADDENDUM ──
    glowing_divider()
    section_header("📄", "Structured RSNA Longitudinal Report Addendum", "Automated comparative study documentation")

    findings_bullets = []
    for nm, pp, pc in zip(LABEL_NAMES, p_prior, p_curr):
        d = pc - pp
        if abs(d) < 0.05:
            change_desc = f"Stable at {pc:.1%} (Prior: {pp:.1%})"
        elif d < 0:
            change_desc = f"Interval improvement / clearing to {pc:.1%} (Prior: {pp:.1%}, Δ: {d:.1%})"
        else:
            change_desc = f"Interval progression / worsening to {pc:.1%} (Prior: {pp:.1%}, Δ: +{d:.1%})"
        findings_bullets.append(f"• {nm}: {change_desc}")

    addendum_text = f"""================================================================================
CHESTAI RADIOLOGY INTELLIGENCE — COMPARATIVE LONGITUDINAL ADDENDUM
================================================================================
PATIENT IDENTIFIER : {patient_label}
COMPARATIVE INTERVAL: {interval_label}
INTERVENING THERAPY : {therapy_label}
CLINICAL INDICATION : {indication_label}
ASSESSMENT DATE     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

COMPARATIVE FINDINGS:
{chr(10).join(findings_bullets)}

OVERALL LONGITUDINAL TRAJECTORY:
{traj_status} (Net Pathology Velocity: {net_delta:+.1%})

CLINICAL IMPRESSION:
{traj_summary}

================================================================================
DISCLAIMER: Investigational AI Pre-Read Workstation. Findings require validation by a
board-certified radiologist.
================================================================================
"""

    rep_c1, rep_c2 = st.columns([3, 1])
    with rep_c1:
        st.text_area("Official Longitudinal Addendum Text", addendum_text, height=250)
    with rep_c2:
        st.markdown("<div style='margin-top:1.6rem;'></div>", unsafe_allow_html=True)
        st.download_button(
            label="📥 Download Longitudinal Report Addendum",
            data=addendum_text,
            file_name=f"Longitudinal_Addendum_{patient_label.replace(' ', '_').replace('(', '').replace(')', '')}_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True,
        )
        st.info("💡 **Clinical Tip:** This addendum can be directly appended into the patient's EHR or PACS clinical report.")
