"""
src/ui/components/zoom_lens.py
==============================
Interactive anatomical quadrant zoom lens, precision magnification viewport,
localized CLAHE enhancement, and landmark survey checklist.
"""

from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np
from PIL import Image
import streamlit as st

from src.ui.common import LABEL_NAMES, LABEL_ICONS, _numpy_to_b64

# ─────────────────────────────────────────────────────────────────────────────
# Anatomical Inspection Zones & Landmark Reference
# ─────────────────────────────────────────────────────────────────────────────
ANATOMICAL_ZONES: Dict[str, Dict[str, Any]] = {
    "🌐 Whole Radiograph (Overview)": {
        "box": (0.0, 0.0, 1.0, 1.0),
        "desc": "Global thorax survey assessing bilateral symmetry, lung expansion, and osseous / soft tissue frame.",
        "landmarks": [
            "Thoracic cage symmetric without displaced rib fractures",
            "Trachea midline with normal caliber",
            "9–10 posterior ribs visible above diaphragm (adequate inspiration)",
            "Both costophrenic angles sharp and clear",
        ],
        "pathologies": ["Systemic survey", "Bilateral lung fields"],
        "primary_label": None,
        "pearl": "Systematic reading: check Airway (trachea midline), Breathing (lung volumes & lucency), Circulation (CTR & borders), Diaphragm (sulci & domes), and Extras.",
    },
    "🫀 Cardiac Silhouette & Mediastinum": {
        "box": (0.18, 0.28, 0.82, 0.85),
        "desc": "Focused view of cardiac borders, aortic knob, hilar pedicle, and transverse cardiothoracic diameter.",
        "landmarks": [
            "Trachea & carina bifurcation midline",
            "Aortic knob contour smooth (<35mm width)",
            "Right heart border (right atrium) distinct without silhouette sign",
            "Left heart border apex within mid-clavicular line",
            "Subcarinal angle normal (<90°)",
        ],
        "pathologies": ["Cardiomegaly", "Pericardial effusion", "Mediastinal widening"],
        "primary_label": "Cardiomegaly",
        "pearl": "Cardiomegaly threshold: CTR > 0.50 on PA projection. Blurring of right border indicates middle lobe or pericardial pathology.",
    },
    "💧 Right Costophrenic Sulcus & Base": {
        "box": (0.0, 0.52, 0.50, 1.0),
        "desc": "Inferolateral right hemithorax focusing on costophrenic angle sharpness and hemidiaphragmatic contour.",
        "landmarks": [
            "Lateral right costophrenic angle acute and sharp",
            "Cardiophphrenic angle clear of fluid meniscus",
            "Right hemidiaphragm dome above 10th posterior rib",
            "Pulmonary vascular markings taper smoothly towards periphery",
        ],
        "pathologies": ["Pleural Effusion", "Atelectasis", "Basilar consolidation"],
        "primary_label": "Pleural Effusion",
        "pearl": "Blunting of the lateral costophrenic sulcus requires ~175–200 mL of pleural fluid on erect PA views. Look for the fluid meniscus sign.",
    },
    "💧 Left Costophrenic Sulcus & Base": {
        "box": (0.50, 0.52, 1.0, 1.0),
        "desc": "Inferolateral left hemithorax focusing on left costophrenic angle, gastric air bubble, and retrocardiac lung field.",
        "landmarks": [
            "Lateral left costophrenic angle sharply defined",
            "Gastric fundus air bubble clearly visualized beneath diaphragm",
            "Retrocardiac lung parenchyma lucent without consolidation",
            "Left hemidiaphragm contour continuous into spine",
        ],
        "pathologies": ["Pleural Effusion", "Atelectasis", "Retrocardiac opacity"],
        "primary_label": "Pleural Effusion",
        "pearl": "Left lower lobe collapse produces a retrocardiac triangular 'sail sign' with loss of the left hemidiaphragm silhouette.",
    },
    "🌊 Central Perihilar 'Bat-Wing' Region": {
        "box": (0.16, 0.28, 0.84, 0.72),
        "desc": "Bilateral hilar reflections, pulmonary vascular branching, and peribronchial cuffing zones.",
        "landmarks": [
            "Right interlobar pulmonary artery caliber normal (<16mm)",
            "Left pulmonary artery concavity preserved",
            "No peribronchial cuffing or 'doughnut' sign on end-on bronchi",
            "Absence of bilateral perihilar butterfly/bat-wing alveolar haze",
        ],
        "pathologies": ["Edema", "Cardiomegaly", "Pulmonary venous hypertension"],
        "primary_label": "Edema",
        "pearl": "Hydrostatic pulmonary edema progresses from cephalization (PCWP 13–18) to interstitial Kerley B lines (18–25) to alveolar bat-wing edema (>25 mmHg).",
    },
    "🫁 Bilateral Apices & Upper Lobes": {
        "box": (0.05, 0.0, 0.95, 0.44),
        "desc": "Supraclavicular and apical parenchymal zones evaluating apical lucency, pleural lines, and rib margins.",
        "landmarks": [
            "Apical pleura closely adherent to thoracic cage (no pneumothorax line)",
            "Symmetric apical lucency without fibro-cavitary lesion",
            "First and second ribs intact without osteolytic lesions",
            "Clavicular heads equidistant from midline spinous process",
        ],
        "pathologies": ["Atelectasis", "Apical capping", "Pneumothorax", "Cavitary process"],
        "primary_label": "Atelectasis",
        "pearl": "Look carefully above the clavicles for visceral pleural reflection lines indicating subtle apical pneumothorax or apical pleural thickening.",
    },
}


def apply_anatomical_lens(
    img_rgb: np.ndarray,
    box: Tuple[float, float, float, float],
    zoom: float = 1.0,
    use_clahe: bool = False,
    clahe_clip: float = 2.5,
) -> Tuple[np.ndarray, Tuple[int, int, int, int], Tuple[float, float, float, float]]:
    """
    Crops an ROI bounding box with a zoom factor and optional local CLAHE micro-enhancement.

    Returns:
        cropped_img: (H_crop, W_crop, 3) uint8 image.
        pixel_box: (px1, py1, px2, py2) in source image pixels.
        norm_box: (cx1, cy1, cx2, cy2) normalized to [0, 1].
    """
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    half_w = (x2 - x1) / (2.0 * max(zoom, 1.0))
    half_h = (y2 - y1) / (2.0 * max(zoom, 1.0))

    cx1 = max(0.0, cx - half_w)
    cx2 = min(1.0, cx + half_w)
    cy1 = max(0.0, cy - half_h)
    cy2 = min(1.0, cy + half_h)

    H, W = img_rgb.shape[:2]
    py1 = max(0, min(H - 1, int(round(cy1 * H))))
    py2 = max(py1 + 10, min(H, int(round(cy2 * H))))
    px1 = max(0, min(W - 1, int(round(cx1 * W))))
    px2 = max(px1 + 10, min(W, int(round(cx2 * W))))

    crop = img_rgb[py1:py2, px1:px2].copy()

    if use_clahe:
        lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(8, 8))
        cl = clahe.apply(l)
        crop = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)

    return crop, (px1, py1, px2, py2), (cx1, cy1, cx2, cy2)


def render_anatomical_zoom_lens(
    image_raw: Image.Image,
    img_np: np.ndarray,
    all_overlays: List[np.ndarray],
    all_heatmaps: List[np.ndarray],
    active_probs: np.ndarray,
    threshold: float,
    heatmap_alpha: float = 0.45,
):
    """
    Renders the Anatomical Quadrant Zoom Lens & Clinical Inspection Viewport.
    Provides targeted ROI inspection, magnification levels, local CLAHE enhancement,
    Grad-CAM heatmap overlays, and an interactive anatomical landmark verification checklist.
    """
    st.markdown(
        '<div style="color: #94a3b8; font-size: 0.86rem; margin-bottom: 1rem; line-height: 1.5;">'
        'Select an anatomical target quadrant below to inspect high-resolution sub-structures with '
        'localized CLAHE parenchymal micro-enhancement, Grad-CAM attention alignment, and clinical landmark verification.'
        '</div>',
        unsafe_allow_html=True
    )

    zone_keys = list(ANATOMICAL_ZONES.keys())

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.6, 1.2, 1.2])
    with ctrl_col1:
        selected_zone = st.selectbox(
            "🎯 Anatomical Target Quadrant",
            zone_keys,
            index=1,  # Default to Cardiac Silhouette
            key="lens_zone_select"
        )
    with ctrl_col2:
        zoom_label = st.radio(
            "🔍 Magnification",
            ["1.0x (Zone)", "1.5x (Target)", "2.0x (Macro)"],
            index=0,
            horizontal=True,
            key="lens_zoom_radio"
        )
        zoom_val = 1.0 if "1.0x" in zoom_label else (1.5 if "1.5x" in zoom_label else 2.0)
    with ctrl_col3:
        layer_mode = st.radio(
            "👁️ Visual Layer",
            ["📷 PACS Radiograph", "🔥 Grad-CAM Overlay"],
            index=0,
            horizontal=True,
            key="lens_layer_radio"
        )

    # Micro-enhancement & secondary controls
    sub_c1, sub_c2, sub_c3 = st.columns([1.2, 1.2, 1.6])
    with sub_c1:
        use_clahe = st.toggle("🔬 Local CLAHE Filter", value=True, help="Applies local adaptive histogram equalization specifically inside this quadrant.")
    with sub_c2:
        use_reticle = st.toggle("🎯 PACS Calibration Reticle", value=False, help="Draws crosshair and caliper reticle on the zoomed inspection frame.")
    with sub_c3:
        if "Grad-CAM" in layer_mode:
            zone_info = ANATOMICAL_ZONES[selected_zone]
            default_path_idx = 0
            if zone_info["primary_label"] in LABEL_NAMES:
                default_path_idx = LABEL_NAMES.index(zone_info["primary_label"])
            overlay_path_choice = st.selectbox(
                "Overlay Pathology Activation:",
                [f"{ic} {nm} ({active_probs[i]:.1%})" for i, (ic, nm) in enumerate(zip(LABEL_ICONS, LABEL_NAMES))],
                index=default_path_idx,
                key="lens_overlay_pathology_select"
            )
            selected_path_idx = [f"{ic} {nm} ({active_probs[i]:.1%})" for i, (ic, nm) in enumerate(zip(LABEL_ICONS, LABEL_NAMES))].index(overlay_path_choice)
        else:
            selected_path_idx = 0

    zone_info = ANATOMICAL_ZONES[selected_zone]
    raw_np = np.array(image_raw.convert("RGB"))
    H_raw, W_raw = raw_np.shape[:2]

    # Prepare base image for cropping (either raw radiograph or high-res Grad-CAM overlay)
    if "Grad-CAM" in layer_mode:
        hm = all_heatmaps[selected_path_idx]
        hm_res = cv2.resize(hm, (W_raw, H_raw), interpolation=cv2.INTER_LINEAR)
        hm_u8 = np.uint8(255 * hm_res)
        hm_col = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
        hm_col = cv2.cvtColor(hm_col, cv2.COLOR_BGR2RGB)
        source_for_crop = np.clip(raw_np.astype(np.float32) * (1.0 - heatmap_alpha) + hm_col.astype(np.float32) * heatmap_alpha, 0, 255).astype(np.uint8)
    else:
        source_for_crop = raw_np

    # Apply crop, zoom, and local CLAHE
    cropped_img, (px1, py1, px2, py2), (cx1, cy1, cx2, cy2) = apply_anatomical_lens(
        source_for_crop,
        zone_info["box"],
        zoom=zoom_val,
        use_clahe=use_clahe,
        clahe_clip=2.5,
    )

    # If reticle requested, draw precision crosshairs
    display_crop = cropped_img.copy()
    if use_reticle:
        ch, cw = display_crop.shape[:2]
        mid_x, mid_y = cw // 2, ch // 2
        reticle_color = (0, 242, 254)
        cv2.line(display_crop, (mid_x - 36, mid_y), (mid_x - 8, mid_y), reticle_color, 1)
        cv2.line(display_crop, (mid_x + 8, mid_y), (mid_x + 36, mid_y), reticle_color, 1)
        cv2.line(display_crop, (mid_x, mid_y - 36), (mid_x, mid_y - 8), reticle_color, 1)
        cv2.line(display_crop, (mid_x, mid_y + 8), (mid_x, mid_y + 36), reticle_color, 1)
        cv2.circle(display_crop, (mid_x, mid_y), 4, reticle_color, 1)
        b_len = 18
        cv2.line(display_crop, (8, 8), (8 + b_len, 8), reticle_color, 1)
        cv2.line(display_crop, (8, 8), (8, 8 + b_len), reticle_color, 1)
        cv2.line(display_crop, (cw - 8, 8), (cw - 8 - b_len, 8), reticle_color, 1)
        cv2.line(display_crop, (cw - 8, 8), (cw - 8, 8 + b_len), reticle_color, 1)
        cv2.line(display_crop, (8, ch - 8), (8 + b_len, ch - 8), reticle_color, 1)
        cv2.line(display_crop, (8, ch - 8), (8, ch - 8 - b_len), reticle_color, 1)
        cv2.line(display_crop, (cw - 8, ch - 8), (cw - 8 - b_len, ch - 8), reticle_color, 1)
        cv2.line(display_crop, (cw - 8, ch - 8), (cw - 8, ch - 8 - b_len), reticle_color, 1)

    # Build Miniature Scout Image
    scout_h, scout_w = 210, 210
    scout_base = cv2.resize(raw_np, (scout_w, scout_h), interpolation=cv2.INTER_AREA)
    bx1 = int(round(cx1 * scout_w))
    by1 = int(round(cy1 * scout_h))
    bx2 = int(round(cx2 * scout_w))
    by2 = int(round(cy2 * scout_h))
    cv2.rectangle(scout_base, (bx1, by1), (bx2, by2), (0, 242, 254), 2)
    for pt in [(bx1, by1), (bx2, by1), (bx1, by2), (bx2, by2)]:
        cv2.circle(scout_base, pt, 3, (255, 71, 87), -1)

    # ── TWO-COLUMN CLINICAL LAYOUT ──
    view_col, info_col = st.columns([1.45, 1.0])

    with view_col:
        crop_h, crop_w = display_crop.shape[:2]
        clahe_badge = '<span style="color:#00f2fe; background:rgba(0,242,254,0.12); border:1px solid rgba(0,242,254,0.3); padding:2px 8px; border-radius:99px; font-size:0.7rem; font-family:\'JetBrains Mono\', monospace;">CLAHE ON</span>' if use_clahe else '<span style="color:#64748b; background:rgba(100,116,139,0.12); padding:2px 8px; border-radius:99px; font-size:0.7rem; font-family:\'JetBrains Mono\', monospace;">RAW PACS</span>'
        reticle_badge = '<span style="color:#38bdf8; background:rgba(56,189,248,0.12); border:1px solid rgba(56,189,248,0.3); padding:2px 8px; border-radius:99px; font-size:0.7rem; font-family:\'JetBrains Mono\', monospace;">RETICLE</span>' if use_reticle else ''
        zone_parts = selected_zone.split(' ')
        zone_short = zone_parts[0] + ' ' + (zone_parts[1] if len(zone_parts) > 1 else "")

        viewport_hdr_html = (
            f'<div style="background:rgba(15,23,42,0.95);border:1px solid rgba(0,242,254,0.25);border-radius:12px 12px 0 0;padding:0.65rem 1rem;display:flex;justify-content:space-between;align-items:center;border-bottom:none;">'
            f'<div style="display:flex;align-items:center;gap:8px;">'
            f'<span style="color:#00f2fe;font-size:0.85rem;font-weight:700;font-family:\'JetBrains Mono\',monospace;">VIEWPORT</span>'
            f'<span style="color:#cbd5e1;font-size:0.82rem;font-weight:600;">{zone_short}</span>'
            f'</div>'
            f'<div style="display:flex;align-items:center;gap:6px;">'
            f'{clahe_badge}{reticle_badge}'
            f'<span style="color:#f59e0b;background:rgba(245,158,11,0.12);border:1px solid rgba(245,158,11,0.3);padding:2px 8px;border-radius:99px;font-size:0.7rem;font-family:\'JetBrains Mono\',monospace;font-weight:700;">{zoom_val:.1f}× ZOOM</span>'
            f'<span style="color:#64748b;font-size:0.72rem;font-family:\'JetBrains Mono\',monospace;">{crop_w}×{crop_h} px</span>'
            f'</div></div>'
        )
        st.markdown(viewport_hdr_html, unsafe_allow_html=True)

        st.image(
            display_crop,
            use_container_width=True,
            caption=f"Magnified ROI Viewport — {selected_zone} ({zoom_val:.1f}x magnification)"
        )

        scout_bar_html = (
            f'<div style="background:rgba(13,20,36,0.9);border:1px solid rgba(0,242,254,0.15);border-radius:0 0 12px 12px;padding:0.8rem 1rem;display:flex;align-items:center;gap:1.2rem;margin-top:-0.6rem;margin-bottom:1rem;">'
            f'<div><img src="data:image/png;base64,{_numpy_to_b64(scout_base)}" style="width:78px;height:78px;border-radius:8px;border:1px solid rgba(0,242,254,0.35);box-shadow:0 0 12px rgba(0,242,254,0.2);" /></div>'
            f'<div style="flex:1;font-family:\'JetBrains Mono\',monospace;font-size:0.75rem;">'
            f'<div style="color:#38bdf8;font-weight:700;margin-bottom:2px;">📍 Thorax Scout Locator Map</div>'
            f'<div style="color:#94a3b8;font-size:0.72rem;">ROI Box: X:[{cx1:.1%}-{cx2:.1%}] · Y:[{cy1:.1%}-{cy2:.1%}]</div>'
            f'<div style="color:#64748b;font-size:0.7rem;margin-top:3px;">Source Res: {W_raw}×{H_raw} px · Sliced Window: ({px1}, {py1}) to ({px2}, {py2})</div>'
            f'</div></div>'
        )
        st.markdown(scout_bar_html, unsafe_allow_html=True)

    with info_col:
        primary_lbl = zone_info["primary_label"]
        if primary_lbl and primary_lbl in LABEL_NAMES:
            p_idx = LABEL_NAMES.index(primary_lbl)
            primary_prob = active_probs[p_idx]
            is_pos = (primary_prob >= threshold)
            prob_color = "#f43f5e" if is_pos else "#10b981"
            prob_status = "POSITIVE / FLAGGED" if is_pos else "NEGATIVE / CLEAR"
            prob_html = (
                f'<div style="margin-top:0.6rem;padding:0.7rem 0.9rem;background:linear-gradient(135deg, {prob_color}14 0%, rgba(15,23,42,0.8) 100%);border:1px solid {prob_color}44;border-radius:10px;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<span style="color:#cbd5e1;font-size:0.75rem;font-weight:600;">{LABEL_ICONS[p_idx]} {primary_lbl} Signal:</span>'
                f'<span style="color:{prob_color};font-size:1.1rem;font-weight:800;font-family:\'JetBrains Mono\',monospace;">{primary_prob:.1%}</span>'
                f'</div>'
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-top:3px;">'
                f'<span style="color:#64748b;font-size:0.68rem;font-family:\'JetBrains Mono\',monospace;">Threshold: {threshold:.0%}</span>'
                f'<span style="color:{prob_color};font-size:0.68rem;font-weight:700;font-family:\'JetBrains Mono\',monospace;">{prob_status}</span>'
                f'</div></div>'
            )
        else:
            prob_html = ""

        profile_html = (
            f'<div style="background:linear-gradient(135deg, rgba(15,23,42,0.85) 0%, rgba(13,20,36,0.95) 100%);border:1px solid rgba(0,242,254,0.2);border-radius:14px;padding:1.1rem;margin-bottom:1rem;backdrop-filter:blur(14px);">'
            f'<div style="color:#00f2fe;font-size:0.78rem;font-weight:700;font-family:\'JetBrains Mono\',monospace;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px;">Quadrant Anatomical Profile</div>'
            f'<div style="color:#f8fafc;font-size:1.05rem;font-weight:700;margin-bottom:0.4rem;">{selected_zone}</div>'
            f'<div style="color:#94a3b8;font-size:0.82rem;line-height:1.55;">{zone_info["desc"]}</div>'
            f'{prob_html}</div>'
        )
        st.markdown(profile_html, unsafe_allow_html=True)

        st.markdown(
            '<div style="color:#e2e8f0;font-size:0.86rem;font-weight:700;display:flex;align-items:center;gap:6px;margin-bottom:0.6rem;">'
            '<span>📋</span> Landmark Survey Checklist'
            '</div>',
            unsafe_allow_html=True
        )

        landmarks = zone_info["landmarks"]
        zone_slug = selected_zone.split(" ")[1].lower() if len(selected_zone.split(" ")) > 1 else "zone"
        checked_count = 0
        for i, lm in enumerate(landmarks):
            chk_key = f"chk_lm_{zone_slug}_{i}"
            is_checked = st.checkbox(lm, value=False, key=chk_key)
            if is_checked:
                checked_count += 1

        all_done = (checked_count == len(landmarks))
        survey_color = "#10b981" if all_done else "#38bdf8"
        progress_html = (
            f'<div style="margin-top:0.6rem;margin-bottom:1rem;display:inline-flex;align-items:center;gap:8px;background:{survey_color}14;border:1px solid {survey_color}44;padding:4px 12px;border-radius:99px;font-size:0.74rem;font-family:\'JetBrains Mono\',monospace;color:{survey_color};font-weight:600;">'
            f'<span>{"✓" if all_done else "○"}</span>'
            f'<span>Survey Progress: {checked_count}/{len(landmarks)} landmarks verified</span>'
            f'</div>'
        )
        st.markdown(progress_html, unsafe_allow_html=True)

        pearl_html = (
            f'<div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.28);border-left:3px solid #818cf8;border-radius:10px;padding:0.85rem 1rem;font-size:0.8rem;line-height:1.55;color:#cbd5e1;">'
            f'<div style="color:#a5b4fc;font-weight:700;font-size:0.78rem;margin-bottom:3px;display:flex;align-items:center;gap:6px;">'
            f'<span>💡</span> Clinical Diagnostic Pearl'
            f'</div>{zone_info["pearl"]}</div>'
        )
        st.markdown(pearl_html, unsafe_allow_html=True)
