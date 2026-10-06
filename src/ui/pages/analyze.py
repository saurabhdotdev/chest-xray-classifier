"""
src/ui/pages/analyze.py
=======================
Primary radiology intelligence workstation: multi-modal acquisition,
PACS windowing studio, CTR calipers, Grad-CAM attention matrix,
anatomical zoom lens, and downloadable RSNA PDF report export.
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
from src.clinical_engine import estimate_ctr, diagnose_syndrome
from src.image_utils import (
    apply_windowing,
    compute_image_stats,
    load_image_from_bytes,
    load_image_from_url,
    SAMPLE_WEB_XRAYS,
)
from src.report_generator import (
    generate_pdf_report,
    generate_structured_report_text,
    compute_triage_tier,
)
from src.ui.common import (
    ROOT,
    LABEL_NAMES,
    LABEL_ICONS,
    LABEL_COLORS,
    PATHOLOGY_GUIDES,
    section_header,
    glowing_divider,
)
from src.ui.model_service import (
    get_model_and_cam,
    get_image_transforms,
    compute_or_get_analysis,
)
from src.ui.components.findings_summary import render_ai_findings_summary
from src.ui.components.zoom_lens import render_anatomical_zoom_lens


def page_analyze(cfg: Dict[str, Any], metrics: Dict[str, Any]):
    """Renders the comprehensive radiology intelligence workstation."""
    section_header("🔬", "Radiology Intelligence Workstation",
                   "Multi-modal X-ray acquisition, PACS windowing, multi-pathology Grad-CAM & report generation")

    # Safety Notice
    st.markdown("""
    <div style="background:rgba(255,183,0,0.08);border:1px solid rgba(255,183,0,0.3);border-radius:12px;padding:0.8rem 1.2rem;color:#ffb300;font-size:0.85rem;margin-bottom:1.2rem;">
        ⚠️ <strong>Investigational AI Pre-Read Assistant:</strong>
        This model supports multi-pathology screening and visual attention mapping. All outputs require clinical validation by a board-certified radiologist.
    </div>
    """, unsafe_allow_html=True)

    # ── WORKFLOW PROGRESS STEPPER BANNER ──
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;background:linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(13, 20, 36, 0.9) 100%);border:1px solid rgba(0, 242, 254, 0.18);border-radius:14px;padding:0.85rem 1.4rem;margin-bottom:1.2rem;flex-wrap:wrap;">
        <div style="display:flex;align-items:center;gap:10px;">
            <span style="background:rgba(0, 242, 254, 0.18);color:#00f2fe;border:1px solid rgba(0, 242, 254, 0.4);border-radius:50%;width:26px;height:26px;display:inline-flex;align-items:center;justify-content:center;font-size:0.8rem;font-weight:800;font-family:'JetBrains Mono', monospace;">1</span>
            <div>
                <div style="color:#f8fafc;font-size:0.84rem;font-weight:700;">Acquire Scan</div>
                <div style="color:#64748b;font-size:0.72rem;">Upload, Demo, or URL</div>
            </div>
        </div>
        <span style="color:#334155;font-size:1rem;">➔</span>
        <div style="display:flex;align-items:center;gap:10px;">
            <span style="background:rgba(99, 102, 241, 0.18);color:#818cf8;border:1px solid rgba(99, 102, 241, 0.4);border-radius:50%;width:26px;height:26px;display:inline-flex;align-items:center;justify-content:center;font-size:0.8rem;font-weight:800;font-family:'JetBrains Mono', monospace;">2</span>
            <div>
                <div style="color:#cbd5e1;font-size:0.84rem;font-weight:600;">AI Inference & Heatmaps</div>
                <div style="color:#64748b;font-size:0.72rem;">DenseNet-121 + Grad-CAM</div>
            </div>
        </div>
        <span style="color:#334155;font-size:1rem;">➔</span>
        <div style="display:flex;align-items:center;gap:10px;">
            <span style="background:rgba(16, 185, 129, 0.18);color:#34d399;border:1px solid rgba(16, 185, 129, 0.4);border-radius:50%;width:26px;height:26px;display:inline-flex;align-items:center;justify-content:center;font-size:0.8rem;font-weight:800;font-family:'JetBrains Mono', monospace;">3</span>
            <div>
                <div style="color:#cbd5e1;font-size:0.84rem;font-weight:600;">Summary & Report</div>
                <div style="color:#64748b;font-size:0.72rem;">Radar Chart & PDF Export</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("💡 Quick Guide: How to Use & Interpret the Workstation", expanded=False):
        g_c1, g_c2, g_c3 = st.columns(3)
        with g_c1:
            st.markdown("""
            <div style="background:rgba(15,23,42,0.6);border:1px solid rgba(0,242,254,0.15);border-radius:10px;padding:0.8rem;height:100%;">
                <div style="color:#00f2fe;font-weight:700;font-size:0.84rem;margin-bottom:4px">🫁 4 Monitored Pathologies</div>
                <div style="color:#94a3b8;font-size:0.77rem;line-height:1.5">
                    • <b>Cardiomegaly:</b> Heart width > 50% chest cavity<br>
                    • <b>Pleural Effusion:</b> Fluid pooling at lung bases<br>
                    • <b>Atelectasis:</b> Partial collapse of lung segments<br>
                    • <b>Edema:</b> Fluid congestion in air sacs
                </div>
            </div>
            """, unsafe_allow_html=True)
        with g_c2:
            st.markdown("""
            <div style="background:rgba(15,23,42,0.6);border:1px solid rgba(245,158,11,0.2);border-radius:10px;padding:0.8rem;height:100%;">
                <div style="color:#fbbf24;font-weight:700;font-size:0.84rem;margin-bottom:4px">🔥 Reading Grad-CAM Heatmaps</div>
                <div style="color:#94a3b8;font-size:0.77rem;line-height:1.5">
                    • <b>Warm/Red zones:</b> Strongest neural network attention (clinical finding trigger)<br>
                    • <b>Cool/Blue zones:</b> Normal or ignored tissue<br>
                    • Use opacity slider to blend with raw scan
                </div>
            </div>
            """, unsafe_allow_html=True)
        with g_c3:
            st.markdown("""
            <div style="background:rgba(15,23,42,0.6);border:1px solid rgba(16,185,129,0.2);border-radius:10px;padding:0.8rem;height:100%;">
                <div style="color:#34d399;font-weight:700;font-size:0.84rem;margin-bottom:4px">📸 Best Image Guidelines</div>
                <div style="color:#94a3b8;font-size:0.77rem;line-height:1.5">
                    • Use upright frontal PA or AP views<br>
                    • Clear contrast between bone and parenchyma<br>
                    • Built-in PACS studio allows brightness, contrast & CLAHE fine-tuning
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Model readiness check
    ckpt_path = ROOT / "results" / "checkpoints" / "best_model.pth"
    if not ckpt_path.exists():
        st.error("Model checkpoint not found. Run `python -m src.train` to generate weights.")
        return

    model, cam_gen, device, default_temp = get_model_and_cam()
    tf = get_image_transforms()

    # ── 1. PATIENT DEMOGRAPHICS & CLINICAL INDICATION (Expandable) ───────────
    with st.expander("👤  Patient Demographics & Clinical Indication", expanded=False):
        c_p1, c_p2, c_p3, c_p4 = st.columns(4)
        with c_p1:
            pt_id = st.text_input("Patient MRN / ID", value="PT-2026-4891", key="pt_id")
        with c_p2:
            pt_age = st.number_input("Age", min_value=1, max_value=110, value=64, key="pt_age")
        with c_p3:
            pt_sex = st.selectbox("Biological Sex", ["Male", "Female", "Other / Unspecified"], key="pt_sex")
        with c_p4:
            pt_view = st.selectbox("Radiographic Projection", ["PA (Posteroanterior)", "AP (Anteroposterior)", "Lateral"], key="pt_view")
        
        pt_indication = st.text_area(
            "Presenting Symptoms & Clinical Indication",
            value="Acute dyspnea on exertion, bilateral lower extremity pitting edema, orthopnea. Suspected congestive heart failure decompensation.",
            key="pt_indication",
            height=68
        )

    patient_info = {
        "patient_id": pt_id,
        "age": pt_age,
        "sex": pt_sex,
        "view": pt_view,
        "indication": pt_indication,
    }

    # ── 2. MULTI-MODAL X-RAY INPUT HUB ──────────────────────────────────────
    st.markdown("<div style='font-size:1.05rem;font-weight:700;color:#e8f4fd;margin:0.8rem 0 0.5rem 0'>📥  Step 1: Enter / Select Chest Radiograph</div>", unsafe_allow_html=True)
    
    source_choice = st.radio(
        "Select Radiograph Acquisition Source:",
        [
            "🗂️  Preloaded Clinical Cases (Instant Demo)",
            "📤  Upload Local Radiograph File",
            "🌐  Web URL / PACS Cloud Link",
            "📷  Live Camera Capture",
        ],
        horizontal=True,
        key="xray_source_radio"
    )

    image_raw = None
    ground_truth_dict = None
    case_notes = None

    if "Preloaded" in source_choice:
        sample_meta = {
            "🫀 Sample 1: Cardiomegaly (Enlarged Cardiac Silhouette)": {
                "path": ROOT / "data" / "samples" / "sample_cardiomegaly.jpg",
                "gt": {"Cardiomegaly": 1, "Pleural Effusion": 0, "Atelectasis": 0, "Edema": 0},
                "notes": "Elderly male with dyspnea. Significant cardiomegaly with CTR > 0.65.",
            },
            "💧 Sample 2: Pleural Effusion (Fluid Meniscus at Base)": {
                "path": ROOT / "data" / "samples" / "sample_pleural_effusion.jpg",
                "gt": {"Cardiomegaly": 0, "Pleural Effusion": 1, "Atelectasis": 0, "Edema": 0},
                "notes": "Costophrenic angle blunting with lateral meniscus formation.",
            },
            "🌊 Sample 3: Pulmonary Edema (Perihilar Bat-wing Infiltrates)": {
                "path": ROOT / "data" / "samples" / "sample_pulmonary_edema.jpg",
                "gt": {"Cardiomegaly": 1, "Pleural Effusion": 0, "Atelectasis": 0, "Edema": 1},
                "notes": "Acute CHF decompensation with diffuse perihilar alveolar haziness.",
            },
            "🫁 Sample 4: Atelectasis (Lower Lobe Discoid Collapse Band)": {
                "path": ROOT / "data" / "samples" / "sample_atelectasis.jpg",
                "gt": {"Cardiomegaly": 0, "Pleural Effusion": 0, "Atelectasis": 1, "Edema": 0},
                "notes": "Horizontal linear discoid collapse opacities in bilateral lower zones.",
            },
            "✨ Sample 5: Normal Chest Radiograph (Clear Lungs & Normal Heart)": {
                "path": ROOT / "data" / "samples" / "sample_normal.jpg",
                "gt": {"Cardiomegaly": 0, "Pleural Effusion": 0, "Atelectasis": 0, "Edema": 0},
                "notes": "Unremarkable study: sharp costophrenic angles, normal mediastinal width, clear lung fields.",
            },
        }
        chosen_sample = st.selectbox("Select demonstration radiograph:", list(sample_meta.keys()), key="demo_select_main")
        s_info = sample_meta[chosen_sample]
        if s_info["path"].exists():
            image_raw = Image.open(s_info["path"]).convert("RGB")
            ground_truth_dict = s_info["gt"]
            case_notes = s_info["notes"]

    elif "Upload" in source_choice:
        st.markdown("""
        <div style="background:linear-gradient(135deg, rgba(0,242,254,0.06) 0%, rgba(99,102,241,0.04) 100%);border:1px solid rgba(0,242,254,0.22);border-radius:14px;padding:1.1rem 1.4rem;margin-bottom:0.9rem;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;flex-wrap:wrap;gap:8px;">
                <div style="font-weight:700;color:#f8fafc;font-size:1.02rem;">
                    📤 Upload Frontal Chest Radiograph
                </div>
                <div style="display:flex;gap:6px;flex-wrap:wrap;">
                    <span style="background:rgba(0,242,254,0.12);color:#00f2fe;border:1px solid rgba(0,242,254,0.3);border-radius:6px;font-size:0.7rem;font-family:'JetBrains Mono',monospace;padding:2px 8px;">PA / AP View</span>
                    <span style="background:rgba(16,185,129,0.12);color:#34d399;border:1px solid rgba(16,185,129,0.3);border-radius:6px;font-size:0.7rem;font-family:'JetBrains Mono',monospace;padding:2px 8px;">JPG · PNG · WEBP · BMP</span>
                    <span style="background:rgba(148,163,184,0.12);color:#cbd5e1;border:1px solid rgba(148,163,184,0.2);border-radius:6px;font-size:0.7rem;font-family:'JetBrains Mono',monospace;padding:2px 8px;">Auto PACS Preprocessing</span>
                </div>
            </div>
            <div style="color:#94a3b8;font-size:0.83rem;line-height:1.5;">
                Select your own patient scan or research radiograph. ChestAI runs automated normalization, DenseNet-121 multi-label inference, Grad-CAM attention localization, and printable PDF report generation.
            </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload file",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="upload_file_main",
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            try:
                image_raw = load_image_from_bytes(uploaded_file.getvalue())
                st.session_state["_uploaded_image"] = image_raw
                st.session_state["_uploaded_filename"] = uploaded_file.name
            except Exception as e:
                st.error(f"Failed to read image: {e}")
        elif "_uploaded_image" in st.session_state:
            image_raw = st.session_state["_uploaded_image"]

        # ⚡ 1-Click Test Samples row
        st.markdown("<div style='margin-top:0.7rem;font-size:0.78rem;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:0.06em'>⚡ Don't have an X-ray on hand? Test 1-click sample:</div>", unsafe_allow_html=True)
        demo_cols = st.columns(4)
        sample_map = [
            ("🫀 Cardiomegaly", ROOT / "data" / "samples" / "sample_cardiomegaly.jpg", "sample_cardiomegaly.jpg"),
            ("💧 Pleural Effusion", ROOT / "data" / "samples" / "sample_pleural_effusion.jpg", "sample_pleural_effusion.jpg"),
            ("🌊 Pulmonary Edema", ROOT / "data" / "samples" / "sample_pulmonary_edema.jpg", "sample_pulmonary_edema.jpg"),
            ("✨ Normal Chest", ROOT / "data" / "samples" / "sample_normal.jpg", "sample_normal.jpg"),
        ]
        for col, (label, path, fname) in zip(demo_cols, sample_map):
            with col:
                if st.button(label, key=f"quick_upload_{fname}", use_container_width=True):
                    if path.exists():
                        img_s = Image.open(path).convert("RGB")
                        st.session_state["_uploaded_image"] = img_s
                        st.session_state["_uploaded_filename"] = fname
                        st.rerun()

        # If an image is loaded, show a crisp status card with clear button
        if image_raw is not None:
            fname = st.session_state.get("_uploaded_filename", "Custom Radiograph")
            stat_c1, stat_c2 = st.columns([3, 1])
            with stat_c1:
                st.markdown(f"""
                <div style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.3);border-radius:12px;padding:0.6rem 1rem;display:flex;align-items:center;gap:10px;margin-top:0.6rem;">
                    <span style="font-size:1.2rem">✅</span>
                    <div>
                        <div style="color:#10b981;font-weight:700;font-size:0.86rem">Active Radiograph: <code>{fname}</code></div>
                        <div style="color:#94a3b8;font-size:0.75rem">Dimensions: {image_raw.width} × {image_raw.height} px · Format: RGB Ready</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with stat_c2:
                st.markdown("<div style='margin-top:0.6rem'></div>", unsafe_allow_html=True)
                if st.button("🗑️ Clear / Reset", key="clear_uploaded_btn", use_container_width=True):
                    st.session_state.pop("_uploaded_image", None)
                    st.session_state.pop("_uploaded_filename", None)
                    st.session_state.pop("_xray_analysis_cache", None)
                    st.rerun()

    elif "Web URL" in source_choice:
        st.markdown("<div style='font-size:0.86rem;color:#7a9ab8;margin-bottom:0.4rem'>Paste image link or choose an open-access sample link:</div>", unsafe_allow_html=True)
        url_col1, url_col2 = st.columns([3, 1])
        with url_col1:
            url_input = st.text_input(
                "Image URL",
                placeholder="https://upload.wikimedia.org/.../Chest_Xray.png",
                key="web_url_input",
                label_visibility="collapsed"
            )
        with url_col2:
            fetch_clicked = st.button("Fetch & Load", key="fetch_url_btn", use_container_width=True)

        quick_pick = st.selectbox("Or quick-load verified public sample:", ["-- Select sample URL --"] + list(SAMPLE_WEB_XRAYS.keys()), key="quick_url_pick")
        target_url = SAMPLE_WEB_XRAYS[quick_pick] if quick_pick != "-- Select sample URL --" else url_input

        if fetch_clicked and target_url:
            with st.spinner("🌐 Fetching radiograph from URL..."):
                try:
                    img_fetched = load_image_from_url(target_url, timeout=10)
                    st.session_state["_web_image"] = img_fetched
                    st.success(f"✓ Successfully fetched radiograph ({img_fetched.width}×{img_fetched.height} px)")
                except Exception as e:
                    st.error(f"Could not load image from URL: {e}")

        if "_web_image" in st.session_state:
            image_raw = st.session_state["_web_image"]

    elif "Live Camera" in source_choice:
        st.markdown("<div style='font-size:0.86rem;color:#7a9ab8;margin-bottom:0.4rem'>Snap a radiograph directly from a viewbox or screen:</div>", unsafe_allow_html=True)
        cam_photo = st.camera_input("Capture radiograph", key="camera_input_main")
        if cam_photo is not None:
            try:
                image_raw = load_image_from_bytes(cam_photo.getvalue())
                st.session_state["_camera_image"] = image_raw
                st.success(f"✓ Camera radiograph captured ({image_raw.width}×{image_raw.height} px)")
            except Exception as e:
                st.error(f"Error capturing image: {e}")
        elif "_camera_image" in st.session_state:
            image_raw = st.session_state["_camera_image"]

    # If no radiograph selected
    if image_raw is None:
        st.markdown("""
        <div style="background:linear-gradient(135deg, rgba(0,242,254,0.04) 0%, rgba(99,102,241,0.03) 100%);border:2px dashed rgba(0,242,254,0.28);border-radius:20px;padding:2.8rem 2rem;text-align:center;margin-top:1.2rem;">
            <div style="font-size:3.2rem;margin-bottom:0.6rem">🫁</div>
            <div style="font-size:1.25rem;font-weight:700;color:#f8fafc;margin-bottom:0.4rem">
                Select or Upload an X-ray Scan Above to Begin
            </div>
            <div style="font-size:0.88rem;color:#94a3b8;max-width:540px;margin:0 auto 1.4rem auto;line-height:1.6">
                Choose an input method above: drag & drop your own X-ray image, try one of our curated demonstration cases, or test with a sample below.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='text-align:center;color:#64748b;font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;margin:1.2rem 0 0.6rem 0'>⚡ Or Instant 1-Click Launch:</div>", unsafe_allow_html=True)
        es_c1, es_c2, es_c3 = st.columns(3)
        with es_c1:
            if st.button("🫀 Try Cardiomegaly Case", key="es_btn_cardio", use_container_width=True):
                s_p = ROOT / "data" / "samples" / "sample_cardiomegaly.jpg"
                if s_p.exists():
                    st.session_state["_uploaded_image"] = Image.open(s_p).convert("RGB")
                    st.session_state["_uploaded_filename"] = "sample_cardiomegaly.jpg"
                    st.session_state["xray_source_radio"] = "📤  Upload Local Radiograph File"
                    st.rerun()
        with es_c2:
            if st.button("💧 Try Pleural Effusion Case", key="es_btn_effusion", use_container_width=True):
                s_p = ROOT / "data" / "samples" / "sample_pleural_effusion.jpg"
                if s_p.exists():
                    st.session_state["_uploaded_image"] = Image.open(s_p).convert("RGB")
                    st.session_state["_uploaded_filename"] = "sample_pleural_effusion.jpg"
                    st.session_state["xray_source_radio"] = "📤  Upload Local Radiograph File"
                    st.rerun()
        with es_c3:
            if st.button("✨ Try Normal Chest Case", key="es_btn_normal", use_container_width=True):
                s_p = ROOT / "data" / "samples" / "sample_normal.jpg"
                if s_p.exists():
                    st.session_state["_uploaded_image"] = Image.open(s_p).convert("RGB")
                    st.session_state["_uploaded_filename"] = "sample_normal.jpg"
                    st.session_state["xray_source_radio"] = "📤  Upload Local Radiograph File"
                    st.rerun()
        return

    # ── 3. RADIOLOGIST PACS WINDOWING & IMAGE ADJUSTMENT STUDIO ──────────────
    glowing_divider()
    with st.expander("🛠️  Radiologist PACS Viewing & Window/Level Studio", expanded=False):
        st.markdown("""
        <div style="color:#7a9ab8;font-size:0.83rem;margin-bottom:0.8rem">
            Adjust radiographic window width (contrast), window level (brightness), invert grayscale (photonegative), or apply CLAHE parenchymal enhancement:
        </div>
        """, unsafe_allow_html=True)

        w_c1, w_c2, w_c3, w_c4, w_c5 = st.columns(5)
        with w_c1:
            invert_toggle = st.toggle("🔄 Invert Grayscale", value=False, help="Inverts intensity (bones white on dark vs dark on white)")
        with w_c2:
            clahe_toggle = st.toggle("🔬 CLAHE Enhance", value=False, help="Contrast-Limited Adaptive Histogram Equalization for lung vessels")
        with w_c3:
            sharpen_toggle = st.toggle("⚡ Sharpen Edges", value=False, help="Enhances fine pulmonary interstitial markings")
        with w_c4:
            contrast_val = st.slider("Window Width (Contrast)", min_value=0.5, max_value=2.5, value=1.0, step=0.1)
        with w_c5:
            brightness_val = st.slider("Window Level (Brightness)", min_value=-50, max_value=50, value=0, step=5)

        apply_to_ai = st.checkbox("🔬 Re-run AI model on enhanced PACS image (default: AI reads original raw radiograph)", value=False)

        stats = compute_image_stats(image_raw)
        st.caption(f"📐 **Radiometric Info:** {stats['width']}×{stats['height']} px · Aspect: {stats['aspect_ratio']} · Mean: {stats['mean_intensity']} · Range: {stats['dynamic_range']}")

        st.markdown("<hr style='border-color:rgba(0,212,255,0.15);margin:0.8rem 0'>", unsafe_allow_html=True)
        show_calipers = st.checkbox("📏 Show Cardiothoracic Ratio (CTR) Digital Calipers", value=False, help="Anatomical calipers measuring maximum cardiac vs thoracic diameter for cardiomegaly grading")
        if show_calipers:
            cal_c1, cal_c2 = st.columns(2)
            with cal_c1:
                c_adj = st.slider("Fine-tune Cardiac Calipers (px)", -30, 30, 0, 1, key="ctr_c_adj")
            with cal_c2:
                t_adj = st.slider("Fine-tune Thoracic Calipers (px)", -30, 30, 0, 1, key="ctr_t_adj")
            ctr_info = estimate_ctr(image_raw, manual_c_offset=c_adj, manual_t_offset=t_adj)

            cal_col1, cal_col2 = st.columns([1, 1.8])
            with cal_col1:
                st.image(ctr_info["annotated_pil"], caption="Digital Calipers (Cyan: Thorax · Coral: Heart)", use_container_width=True)
            with cal_col2:
                st.markdown(f"""
                <div style="background:rgba(13,21,38,0.9);border:1px solid {ctr_info['color']}55;border-radius:12px;padding:1rem;">
                    <div style="color:#7a9ab8;font-size:0.76rem;text-transform:uppercase;letter-spacing:0.05em">Cardiothoracic Ratio (CTR)</div>
                    <div style="color:{ctr_info['color']};font-size:2.2rem;font-weight:800;font-family:'Segoe UI',Consolas,monospace">{ctr_info['ctr']:.1%}</div>
                    <div style="color:{ctr_info['color']};font-weight:700;font-size:0.95rem;margin-top:2px">{ctr_info['classification']}</div>
                    <div style="color:#a8c8e0;font-size:0.84rem;margin-top:6px;line-height:1.5">{ctr_info['clinical_note']}</div>
                    <div style="color:#7a9ab8;font-size:0.76rem;margin-top:10px;border-top:1px solid rgba(255,255,255,0.08);padding-top:6px">
                        Transverse Cardiac: <b style="color:#ff4757">{ctr_info['cardiac_px']} px</b> · Inner Thoracic: <b style="color:#00d4ff">{ctr_info['thorax_px']} px</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Apply adjustments
    image_processed = apply_windowing(
        image_raw,
        brightness=brightness_val,
        contrast=contrast_val,
        invert=invert_toggle,
        use_clahe=clahe_toggle,
        sharpen=sharpen_toggle,
    )
    image_for_model = image_processed if apply_to_ai else image_raw

    # ── 4. SENSITIVITY & CALIBRATION CONTROLS ────────────────────────────────
    col_opt1, col_opt2, col_opt3 = st.columns([1.5, 1.5, 1])
    with col_opt1:
        threshold = st.slider(
            "🎚️  Operating Sensitivity Threshold",
            min_value=0.1, max_value=0.9, value=0.5, step=0.05,
            help="Default: 0.50. Lower threshold increases sensitivity (flags subtle signs); higher threshold increases specificity (flags only confident findings)."
        )
        st.caption("Standard: 50% · Lower = High Sensitivity · Higher = High Specificity")
    with col_opt2:
        calib_mode = st.radio(
            "🌡️  Probability Calibration",
            ["Calibrated (Temperature Scaling T=0.05)", "Raw Model Probabilities"],
            index=0,
            horizontal=True,
            help="Calibrated probabilities adjust overconfident logits to match empirical disease incidence."
        )
        use_calibrated = ("Calibrated" in calib_mode)
    with col_opt3:
        heatmap_alpha = st.slider("🔆 Heatmap Alpha", 0.1, 0.9, 0.45, 0.05)

    # ── 5. RUN INFERENCE & GRAD-CAM (ULTRA-FAST CACHED) ─────────────────────
    with st.spinner("🧠 Performing Deep Neural Radiographic Analysis & Grad-CAM..."):
        analysis = compute_or_get_analysis(
            model, cam_gen, device, image_for_model, tf, default_temp
        )

    raw_probs = analysis["raw_probs"]
    cal_probs = analysis["cal_probs"]
    img_np = analysis["img_np"]
    all_heatmaps = analysis["heatmaps"]

    active_probs = cal_probs if use_calibrated else raw_probs

    # Fast OpenCV heatmap blending with current heatmap_alpha slider (<1ms)
    all_overlays = [
        overlay_heatmap(img_np, hm, alpha=heatmap_alpha)
        for hm in all_heatmaps
    ]

    # ── 6. RESULTS BANNER ───────────────────────────────────────────────────
    glowing_divider()
    triage = compute_triage_tier(active_probs.tolist(), LABEL_NAMES, threshold)
    n_positive = sum(1 for p in active_probs if p >= threshold)

    banner_icon = "🚨" if triage["tier"] == 3 else ("⚠️" if triage["tier"] == 2 else "✅")
    st.markdown(f"""
    <div style="background:linear-gradient(135deg, {triage['color']}18 0%, rgba(15, 23, 42, 0.85) 100%);border:1px solid {triage['color']}55;border-top:2px solid {triage['color']};border-radius:18px;padding:1.25rem 1.7rem;backdrop-filter:blur(16px);box-shadow:0 12px 30px -10px {triage['color']}22;display:flex;align-items:center;justify-content:space-between;margin-bottom:1.5rem;">
        <div style="display:flex;align-items:center;gap:1.3rem">
            <div style="width:52px;height:52px;border-radius:14px;background:{triage['color']}22;border:1px solid {triage['color']}55;display:flex;align-items:center;justify-content:center;font-size:1.8rem;box-shadow:0 0 20px {triage['color']}25;">{banner_icon}</div>
            <div>
                <div style="color:#ffffff;font-weight:800;font-size:1.18rem;letter-spacing:-0.01em;">
                    {triage['badge']} <span style="color:{triage['color']}">· {n_positive} Finding{'s' if n_positive != 1 else ''} Flagged</span>
                </div>
                <div style="color:#94a3b8;font-size:0.85rem;margin-top:3px;font-weight:400">
                    {triage['desc']}
                </div>
            </div>
        </div>
        <div style="text-align:right">
            <span style="background:{triage['color']}22;border:1px solid {triage['color']}66;color:{triage['color']};padding:0.4rem 1.1rem;border-radius:999px;font-family:'JetBrains Mono', monospace;font-size:0.78rem;font-weight:700;letter-spacing:0.05em;box-shadow:0 0 15px {triage['color']}25;">TIER {triage['tier']} RISK</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 7. PER-FINDING PROBABILITY CARDS (Skipper Design) ────────────────────
    pathology_subtitles = {
        "Cardiomegaly": "Enlarged heart (CTR > 50%)",
        "Pleural Effusion": "Fluid pooling in pleural base",
        "Atelectasis": "Partial lung segment collapse",
        "Edema": "Alveolar congestion / bat-wing haze",
    }
    prob_cols = st.columns(4)
    for col, icon, name, prob, raw_p in zip(prob_cols, LABEL_ICONS, LABEL_NAMES, active_probs, raw_probs):
        positive = (prob >= threshold)
        color = "#f43f5e" if positive else "#10b981"
        bar_pct = int(prob * 100)
        status_label = "DETECTED" if positive else "CLEAR / NEGATIVE"
        beacon_color = "#f43f5e" if positive else "#10b981"
        subtitle = pathology_subtitles.get(name, "")

        with col:
            st.markdown(
                f'<div style="background: linear-gradient(135deg, {color}12 0%, rgba(15, 23, 42, 0.78) 100%);'
                f'border: 1px solid {color}44;'
                f'border-top: 2px solid {color}88;'
                f'border-radius: 18px; padding: 1.25rem 1rem; text-align: center;'
                f'backdrop-filter: blur(16px);'
                f'box-shadow: 0 10px 25px -10px {color}25;'
                f'transition: transform 0.25s ease;">'
                f'<div style="font-size: 2.1rem; margin-bottom: 0.4rem; filter: drop-shadow(0 2px 8px {color}44)">{icon}</div>'
                f'<div style="color: #f8fafc; font-weight: 700; font-size: 0.92rem; margin-bottom: 0.2rem; letter-spacing: -0.01em;">{name}</div>'
                f'<div style="color: #94a3b8; font-size: 0.72rem; margin-bottom: 0.6rem; font-weight: 500;">{subtitle}</div>'
                f'<div style="background: rgba(6, 10, 20, 0.8); border-radius: 999px; height: 6px; overflow: hidden; margin-bottom: 0.75rem; border: 1px solid rgba(255,255,255,0.06);">'
                f'<div style="width: {bar_pct}%; height: 100%; background: linear-gradient(90deg, {color}88, {color}); border-radius: 999px; box-shadow: 0 0 10px {color}88;"></div>'
                f'</div>'
                f'<div style="color: {color}; font-size: 1.65rem; font-weight: 800; font-family: \'JetBrains Mono\', Consolas, monospace; letter-spacing: -0.02em; text-shadow: 0 0 18px {color}44;">{prob:.1%}</div>'
                f'<div style="margin-top: 0.45rem;">'
                f'<span style="background: {color}1a; border: 1px solid {color}55; color: {color}; padding: 0.25rem 0.75rem; border-radius: 999px; font-size: 0.7rem; font-family: \'JetBrains Mono\', monospace; font-weight: 700; letter-spacing: 0.04em;">'
                f'<span style="display:inline-block; width:6px; height:6px; border-radius:50%; background:{beacon_color}; box-shadow:0 0 6px {beacon_color}; margin-right:4px;"></span>'
                f'{status_label}</span>'
                f'</div>'
                f'<div style="color: #64748b; font-size: 0.68rem; font-family: \'JetBrains Mono\', monospace; margin-top: 0.5rem;">Raw: {raw_p:.1%}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── 7.5. AI FINDINGS SUMMARY — RADAR CHART & BAR CHART ──────────────────
    glowing_divider()
    section_header(
        "🎯", "AI Findings Summary",
        "Multi-pathology probability landscape — radar overview & comparative confidence analysis"
    )

    render_ai_findings_summary(active_probs, raw_probs, threshold)

    # ── 8. GROUND TRUTH VERIFICATION (If available) ─────────────────────────
    if ground_truth_dict is not None:
        glowing_divider()
        section_header("📋", "Clinical Ground Truth Correlation", "Validation against radiologist's confirmed gold standard")
        if case_notes:
            st.info(f"🏥 **Clinical Case Context:** {case_notes}")

        gt_cols = st.columns(4)
        for col, icon, name, prob in zip(gt_cols, LABEL_ICONS, LABEL_NAMES, active_probs):
            pred_pos = (prob >= threshold)
            true_pos = (ground_truth_dict.get(name, 0) == 1)
            is_match = (pred_pos == true_pos)
            status_text = "Accurate ✓" if is_match else "Discrepancy ✗"
            status_color = "#10b981" if is_match else "#f43f5e"

            with col:
                st.markdown(
                    f'<div style="background: linear-gradient(135deg, {status_color}10 0%, rgba(15, 23, 42, 0.7) 100%);'
                    f'border: 1px solid {status_color}45; border-radius: 14px; padding: 0.9rem; text-align: center; backdrop-filter: blur(12px);">'
                    f'<div style="color: #f8fafc; font-weight: 700; font-size: 0.85rem">{icon} {name}</div>'
                    f'<div style="font-size: 0.75rem; color: #94a3b8; margin-top: 5px">Gold Standard: <b style="color: #f8fafc">{"POS" if true_pos else "NEG"}</b></div>'
                    f'<div style="font-size: 0.75rem; color: #94a3b8">AI Prediction: <b style="color: {status_color}">{"POS" if pred_pos else "NEG"}</b></div>'
                    f'<div style="background: {status_color}22; border: 1px solid {status_color}66; color: {status_color}; padding: 3px 10px; border-radius: 999px; font-size: 0.72rem; font-family: \'JetBrains Mono\', monospace; font-weight: 700; margin-top: 7px; display: inline-block;">{status_text}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── 8B. CLINICAL DIAGNOSTIC SYNDROME SYNTHESIS & MANAGEMENT ─────────────
    glowing_divider()
    section_header("🏥", "Clinical Diagnostic Syndrome Synthesis", "Multi-pathology pattern synthesis & actionable patient management pathways")
    syndrome = diagnose_syndrome(active_probs.tolist(), LABEL_NAMES, threshold)

    st.markdown(f"""
    <div style="background:rgba(13,21,38,0.95);border:1px solid {syndrome['color']}55;border-radius:16px;padding:1.3rem 1.6rem;box-shadow:0 0 25px {syndrome['color']}18;margin-bottom:1.2rem;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.8rem">
            <div style="display:flex;align-items:center;gap:0.8rem">
                <span style="font-size:2rem">{syndrome['icon']}</span>
                <div>
                    <div style="color:#e8f4fd;font-size:1.2rem;font-weight:800">{syndrome['syndrome']}</div>
                    <div style="color:#7a9ab8;font-size:0.78rem">ICD-10 Correlation: <b style="color:#00d4ff">{syndrome['icd10']}</b></div>
                </div>
            </div>
            <div>
                <span style="background:{syndrome['color']}22;border:1px solid {syndrome['color']}66;color:{syndrome['color']};padding:0.4rem 1rem;border-radius:99px;font-size:0.8rem;font-weight:700;letter-spacing:0.05em;">{syndrome['acuity']}</span>
            </div>
        </div>
        <div style="color:#a8c8e0;font-size:0.88rem;line-height:1.6;margin-bottom:1rem">
            {syndrome['summary']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    syn_col1, syn_col2 = st.columns(2)
    with syn_col1:
        differentials_html = ''.join(f'<li>{d}</li>' for d in syndrome['differentials'])
        st.markdown(f"""
        <div style="background:rgba(13,21,38,0.8);border:1px solid rgba(0,212,255,0.2);border-radius:12px;padding:1rem 1.2rem;height:100%;">
            <div style="color:#00d4ff;font-weight:700;font-size:0.9rem;margin-bottom:0.6rem">🎯 Top Differential Diagnoses</div>
            <ul style="color:#a8c8e0;font-size:0.82rem;margin:0;padding-left:1.2rem;line-height:1.7">
                {differentials_html}
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with syn_col2:
        recommendations_html = ''.join(f'<li>{r}</li>' for r in syndrome['recommendations'])
        st.markdown(f"""
        <div style="background:rgba(13,21,38,0.8);border:1px solid rgba(0,255,136,0.25);border-radius:12px;padding:1rem 1.2rem;height:100%;">
            <div style="color:#00ff88;font-weight:700;font-size:0.9rem;margin-bottom:0.6rem">📋 Actionable Clinical Recommendations</div>
            <ul style="color:#a8c8e0;font-size:0.82rem;margin:0;padding-left:1.2rem;line-height:1.7">
                {recommendations_html}
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # ── 9. GRAD-CAM EXPLAINABILITY STUDIO (Single vs 4-Pathology Matrix) ─────
    glowing_divider()
    section_header("🔥", "Grad-CAM Attention & Anatomical Localization", "Explainable visual heatmaps revealing deep neural activations")

    grad_view_mode = st.radio(
        "Display Mode",
        ["Simultaneous 4-Pathology Matrix (All Findings)", "Single Focused Pathology Deep-Dive"],
        index=0,
        horizontal=True,
        key="grad_view_mode"
    )

    if "Simultaneous" in grad_view_mode:
        st.markdown("<div style='font-size:0.85rem;color:#7a9ab8;margin-bottom:0.8rem'>Visualizing model focus for all 4 pathologies simultaneously with anatomical guidance:</div>", unsafe_allow_html=True)
        g_c1, g_c2, g_c3, g_c4 = st.columns(4)
        for col, icon, name, ov, hm, p in zip([g_c1, g_c2, g_c3, g_c4], LABEL_ICONS, LABEL_NAMES, all_overlays, all_heatmaps, active_probs):
            with col:
                st.image(ov, caption=f"{icon} {name} ({p:.1%})", use_container_width=True)
                st.markdown(f"""
                <div style="background:rgba(13,21,38,0.7);border:1px solid rgba(0,212,255,0.15);border-radius:8px;padding:0.6rem;font-size:0.74rem;color:#a8c8e0;line-height:1.4;">
                    <b>Anatomical Target:</b><br>{PATHOLOGY_GUIDES[name]}
                </div>
                """, unsafe_allow_html=True)
    else:
        focused_name = st.selectbox(
            "Choose pathology to inspect:",
            [f"{ic} {nm}" for ic, nm in zip(LABEL_ICONS, LABEL_NAMES)],
            key="focused_grad_select"
        )
        f_idx = [f"{ic} {nm}" for ic, nm in zip(LABEL_ICONS, LABEL_NAMES)].index(focused_name)

        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            st.image(img_np, caption="📷 Radiograph (Model Input)", use_container_width=True)
        with f_col2:
            fig, ax = plt.subplots()
            im = ax.imshow(all_heatmaps[f_idx], cmap="jet")
            ax.axis("off")
            ax.set_title(f"Grad-CAM Heatmap: {LABEL_NAMES[f_idx]}", color="#e8f4fd", fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)
            plt.close()
        with f_col3:
            st.image(all_overlays[f_idx], caption=f"🔥 Radiograph + Heatmap Overlay ({active_probs[f_idx]:.1%})", use_container_width=True)

        st.info(f"💡 **Radiological Focus for {LABEL_NAMES[f_idx]}:** {PATHOLOGY_GUIDES[LABEL_NAMES[f_idx]]}")

    # ── 9.5. ANATOMICAL QUADRANT ZOOM LENS (CLINICAL INSPECTION MODE) ────────
    glowing_divider()
    section_header(
        "🔍", "Anatomical Quadrant Zoom Lens & Clinical Inspection Viewport",
        "Targeted region-of-interest (ROI) examination with local CLAHE parenchymal micro-enhancement & landmark verification"
    )
    render_anatomical_zoom_lens(
        image_raw,
        img_np,
        all_overlays,
        all_heatmaps,
        active_probs,
        threshold,
        heatmap_alpha=heatmap_alpha,
    )

    # ── 10. STRUCTURED RSNA REPORT & DOWNLOADABLE PDF EXPORT ─────────────────
    glowing_divider()
    section_header("📄", "Structured RSNA Radiology Report & Official Export", "Automated clinical pre-read documentation")

    # Generate text report
    report_text = generate_structured_report_text(
        patient_info,
        active_probs.tolist(),
        threshold,
        use_calibrated,
        LABEL_NAMES
    )

    # Cache PDF report build so ReportLab doesn't re-run on every slider change
    pdf_cache_key = f"{analysis['hash']}_{threshold}_{use_calibrated}_{patient_info['patient_id']}"
    if st.session_state.get("_pdf_cache_key") != pdf_cache_key or "_cached_pdf_bytes" not in st.session_state:
        primary_idx = int(np.argmax(active_probs))
        st.session_state["_cached_pdf_bytes"] = generate_pdf_report(
            patient_info,
            active_probs.tolist(),
            threshold,
            use_calibrated,
            LABEL_NAMES,
            image_raw,
            Image.fromarray(all_overlays[primary_idx]),
        )
        st.session_state["_pdf_cache_key"] = pdf_cache_key

    pdf_bytes = st.session_state["_cached_pdf_bytes"]

    rep_col1, rep_col2 = st.columns([3, 1])
    with rep_col1:
        st.markdown("""
        <div style="background:rgba(13,21,38,0.85);border:1px solid rgba(0,212,255,0.22);border-radius:14px;padding:1.2rem;font-family:'Segoe UI', system-ui;">
        """, unsafe_allow_html=True)
        st.markdown(report_text)
        st.markdown("</div>", unsafe_allow_html=True)

    with rep_col2:
        st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)
        st.download_button(
            label="📥 Download Official Radiology Report (PDF)",
            data=pdf_bytes,
            file_name=f"Radiology_Report_{pt_id}_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            key="dl_pdf_btn",
            use_container_width=True,
        )
        st.markdown("""
        <div style="background:rgba(0,255,136,0.06);border:1px solid rgba(0,255,136,0.25);border-radius:12px;padding:0.9rem;font-size:0.8rem;color:#a8c8e0;margin-top:1rem;">
            <b>📄 PDF Contents:</b><br>
            • Hospital-grade letterhead & report ID<br>
            • Patient demographics & indication<br>
            • Quantitative pathology confidence table<br>
            • Embedded radiograph + Grad-CAM figure<br>
            • Structured impression & action plan<br>
            • Official medical research disclaimer
        </div>
        """, unsafe_allow_html=True)
