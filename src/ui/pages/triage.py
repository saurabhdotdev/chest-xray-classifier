"""
src/ui/pages/triage.py
======================
Emergency department batch triage worklist and acuity prioritization engine.
"""

from datetime import datetime
from pathlib import Path
import time
from typing import Dict, Any, List, Tuple
import numpy as np
from PIL import Image
import streamlit as st

from src.image_utils import load_image_from_bytes
from src.report_generator import compute_triage_tier
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
    run_inference,
)
from src.ui.components.triage_components import (
    extract_images_from_zip,
    build_triage_csv,
)


def page_triage(cfg: Dict[str, Any], metrics: Dict[str, Any]):
    """Renders emergency department batch triage worklist with dynamic acuity ranking."""
    section_header("📋", "Emergency Department Batch Triage Worklist",
                   "AI-assisted clinical worklist re-ordering: Prioritize STAT critical emergencies before routine studies")

    st.markdown("""
    <div style="background:linear-gradient(135deg, rgba(0, 242, 254, 0.07) 0%, rgba(99, 102, 241, 0.04) 100%);border:1px solid rgba(0, 242, 254, 0.22);border-radius:14px;padding:1rem 1.4rem;font-size:0.86rem;color:#a8c8e0;margin-bottom:1.3rem;line-height:1.55;">
        <strong>⚡ Emergency Worklist Re-ordering:</strong> In acute hospital and ER settings, high patient volumes create diagnostic backlogs.
        ChestAI screens batches of radiographs in milliseconds, automatically escalating <b>STAT Critical emergencies</b> (acute pulmonary edema, massive effusions, decompensated CHF) to the top of the queue before routine studies.
    </div>
    """, unsafe_allow_html=True)

    model, cam_gen, device, fitted_temp = get_model_and_cam()
    tf = get_image_transforms()

    triage_mode = st.radio(
        "Select Batch Acquisition Mode:",
        [
            "🗂️  Preloaded Emergency Hospital Queue (5 Diverse Patients)",
            "📤  Upload Custom Radiographs Batch / ZIP Archive",
        ],
        horizontal=True,
        key="triage_source_mode"
    )

    cases_to_process = []
    samples_dir = ROOT / "data" / "samples"
    demo_queue = [
        ("PT-901 (Acute CHF Decompensation)", samples_dir / "sample_pulmonary_edema.jpg"),
        ("PT-902 (Suspected Pleural Meniscus)", samples_dir / "sample_pleural_effusion.jpg"),
        ("PT-903 (Enlarged Cardiac Silhouette)", samples_dir / "sample_cardiomegaly.jpg"),
        ("PT-904 (Post-Op Basal Atelectasis)", samples_dir / "sample_atelectasis.jpg"),
        ("PT-905 (Pre-op Normal Screening)", samples_dir / "sample_normal.jpg"),
    ]

    if "Preloaded" in triage_mode:
        for name, p in demo_queue:
            if p.exists():
                cases_to_process.append((name, Image.open(p).convert("RGB")))
    else:
        st.markdown("""
        <div style="background:rgba(15,23,42,0.6);border:1px dashed rgba(0,242,254,0.3);border-radius:12px;padding:0.9rem 1.2rem;margin-bottom:0.8rem;">
            <div style="color:#f8fafc;font-size:0.9rem;font-weight:700;">Upload Individual Radiographs or ZIP Folder</div>
            <div style="color:#94a3b8;font-size:0.78rem;">Drop multiple image files (JPG, PNG, WEBP, BMP) or a single .ZIP archive containing patient radiographs.</div>
        </div>
        """, unsafe_allow_html=True)

        up_col1, up_col2 = st.columns([3, 1])
        with up_col1:
            uploaded_files = st.file_uploader(
                "Upload multiple radiographs or ZIP",
                type=["jpg", "jpeg", "png", "webp", "bmp", "zip"],
                accept_multiple_files=True,
                key="batch_triage_uploader",
                label_visibility="collapsed"
            )
        with up_col2:
            if st.button("⚡ Test 5-Patient Batch", key="load_er_test_batch", use_container_width=True):
                cases_to_process = []
                for name, p in demo_queue:
                    if p.exists():
                        cases_to_process.append((name, Image.open(p).convert("RGB")))
                st.session_state["_custom_triage_batch"] = cases_to_process
                st.rerun()

        if uploaded_files:
            for uf in uploaded_files:
                fname = uf.name
                if fname.lower().endswith(".zip"):
                    extracted = extract_images_from_zip(uf.getvalue())
                    cases_to_process.extend(extracted)
                else:
                    try:
                        img = load_image_from_bytes(uf.getvalue())
                        cases_to_process.append((fname, img))
                    except Exception:
                        pass
        elif "_custom_triage_batch" in st.session_state:
            cases_to_process = st.session_state["_custom_triage_batch"]

    if not cases_to_process:
        st.markdown("""
        <div style="background:rgba(0,242,254,0.03);border:2px dashed rgba(0,242,254,0.22);border-radius:18px;padding:2.5rem;text-align:center;margin-top:1rem;">
            <div style="font-size:2.8rem;margin-bottom:0.5rem">📋</div>
            <div style="font-size:1.15rem;font-weight:700;color:#f8fafc;margin-bottom:0.3rem">Emergency Triage Queue Empty</div>
            <div style="color:#94a3b8;font-size:0.84rem;max-width:500px;margin:0 auto 1.2rem auto;">
                Select the preloaded hospital queue or upload multiple patient radiographs to run automated batch acuity classification.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Fast batch inference with cache
    batch_sig = f"_triage_{len(cases_to_process)}_{cases_to_process[0][0]}_{fitted_temp}"
    if st.session_state.get("_triage_sig") != batch_sig or "_triage_results" not in st.session_state:
        results = []
        t0 = time.time()
        with st.spinner(f"⚡ AI Batch Engine Screening {len(cases_to_process)} Radiographs..."):
            for name, img in cases_to_process:
                _, _, cal_probs, _, _ = run_inference(model, device, img, tf, temperature=fitted_temp)
                triage = compute_triage_tier(cal_probs.tolist(), LABEL_NAMES, threshold=0.5)
                max_idx = int(np.argmax(cal_probs))
                flagged = [
                    (LABEL_ICONS[i], LABEL_NAMES[i], cal_probs[i])
                    for i in range(len(LABEL_NAMES)) if cal_probs[i] >= 0.5
                ]
                results.append({
                    "patient": name,
                    "tier": triage["tier"],
                    "badge": triage["badge"],
                    "color": triage["color"],
                    "desc": triage["desc"],
                    "primary_finding": f"{LABEL_ICONS[max_idx]} {LABEL_NAMES[max_idx]} ({cal_probs[max_idx]:.1%})",
                    "probs": cal_probs,
                    "img": img,
                    "flagged": flagged,
                })
        t_elapsed = time.time() - t0
        # Sort critical tier 3 first, then tier 2, then tier 1
        results.sort(key=lambda r: (r["tier"], float(np.max(r["probs"]))), reverse=True)
        st.session_state["_triage_results"] = results
        st.session_state["_triage_sig"] = batch_sig
        st.session_state["_triage_speed"] = (t_elapsed / max(1, len(cases_to_process)))

    results = st.session_state["_triage_results"]
    avg_speed = st.session_state.get("_triage_speed", 0.04)

    # Count tiers
    n_tier3 = sum(1 for r in results if r["tier"] == 3)
    n_tier2 = sum(1 for r in results if r["tier"] == 2)
    n_tier1 = sum(1 for r in results if r["tier"] == 1)
    total_pts = len(results)

    glowing_divider()
    section_header("📊", "Triage Cohort Analytics", f"Automated risk stratification across {total_pts} studies")

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.markdown(f"""
        <div style="background:rgba(15,23,42,0.8);border:1px solid rgba(0,242,254,0.3);border-top:2px solid #00f2fe;border-radius:14px;padding:1rem;text-align:center;">
            <div style="font-size:1.6rem;color:#00f2fe;font-weight:800;font-family:'JetBrains Mono',monospace;">{total_pts}</div>
            <div style="color:#94a3b8;font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;margin-top:2px;">Total Studies</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div style="background:rgba(244,63,94,0.08);border:1px solid rgba(244,63,94,0.35);border-top:2px solid #f43f5e;border-radius:14px;padding:1rem;text-align:center;">
            <div style="font-size:1.6rem;color:#f43f5e;font-weight:800;font-family:'JetBrains Mono',monospace;">{n_tier3}</div>
            <div style="color:#fda4af;font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;margin-top:2px;">🚨 STAT Critical</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.35);border-top:2px solid #f59e0b;border-radius:14px;padding:1rem;text-align:center;">
            <div style="font-size:1.6rem;color:#f59e0b;font-weight:800;font-family:'JetBrains Mono',monospace;">{n_tier2}</div>
            <div style="color:#fde68a;font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;margin-top:2px;">⚠️ Moderate Risk</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.35);border-top:2px solid #10b981;border-radius:14px;padding:1rem;text-align:center;">
            <div style="font-size:1.6rem;color:#10b981;font-weight:800;font-family:'JetBrains Mono',monospace;">{n_tier1}</div>
            <div style="color:#a7f3d0;font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;margin-top:2px;">🟢 Routine / Clear</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi5:
        st.markdown(f"""
        <div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.35);border-top:2px solid #818cf8;border-radius:14px;padding:1rem;text-align:center;">
            <div style="font-size:1.6rem;color:#818cf8;font-weight:800;font-family:'JetBrains Mono',monospace;">{avg_speed*1000:.0f}ms</div>
            <div style="color:#c7d2fe;font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;margin-top:2px;">⚡ Screen Speed</div>
        </div>
        """, unsafe_allow_html=True)

    glowing_divider()
    section_header("🚨", "Prioritized Emergency Worklist Queue", "High-acuity studies dynamically ranked to the top of the clinical queue")

    f_col1, f_col2, f_col3 = st.columns([2, 1.5, 1.2])
    with f_col1:
        tier_filter = st.selectbox(
            "Filter by Urgency Tier:",
            [
                f"All Cases ({total_pts})",
                f"🚨 STAT Critical Only ({n_tier3})",
                f"⚠️ Moderate Risk Only ({n_tier2})",
                f"🟢 Routine / Clear Only ({n_tier1})",
            ],
            key="triage_filter_select"
        )
    with f_col2:
        sort_by = st.selectbox(
            "Sort Order:",
            ["Acuity Priority (Highest First)", "Max Confidence %", "Patient ID / Name"],
            key="triage_sort_select"
        )
    with f_col3:
        st.markdown("<div style='margin-top:1.7rem'></div>", unsafe_allow_html=True)
        csv_data = build_triage_csv(results)
        st.download_button(
            label="📥 Export Worklist (CSV)",
            data=csv_data,
            file_name=f"ER_Triage_Worklist_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key="dl_triage_csv_btn",
            use_container_width=True
        )

    # Filter and sort
    filtered_results = list(results)
    if "STAT" in tier_filter:
        filtered_results = [r for r in filtered_results if r["tier"] == 3]
    elif "Moderate" in tier_filter:
        filtered_results = [r for r in filtered_results if r["tier"] == 2]
    elif "Routine" in tier_filter:
        filtered_results = [r for r in filtered_results if r["tier"] == 1]

    if "Max Confidence" in sort_by:
        filtered_results.sort(key=lambda r: float(np.max(r["probs"])), reverse=True)
    elif "Patient ID" in sort_by:
        filtered_results.sort(key=lambda r: r["patient"])
    else:
        filtered_results.sort(key=lambda r: (r["tier"], float(np.max(r["probs"]))), reverse=True)

    for idx, r in enumerate(filtered_results, 1):
        tier_color = r["color"]
        beacon_style = f"box-shadow: 0 0 15px {tier_color};" if r["tier"] == 3 else ""

        with st.container():
            card_c1, card_c2, card_c3, card_c4 = st.columns([0.6, 1.2, 4.2, 1.8])
            with card_c1:
                st.markdown(f"""
                <div style="font-size:1.15rem;font-weight:800;color:{tier_color};background:{tier_color}22;width:44px;height:44px;border-radius:50%;border:1px solid {tier_color}66;font-family:'JetBrains Mono',monospace;display:flex;align-items:center;justify-content:center;margin-top:4px;{beacon_style}">
                    #{idx}
                </div>
                """, unsafe_allow_html=True)

            with card_c2:
                thumb = r["img"].copy()
                thumb.thumbnail((90, 90))
                st.image(thumb, use_container_width=True)

            with card_c3:
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:4px;">
                    <span style="color:#f8fafc;font-weight:700;font-size:1.02rem;">{r['patient']}</span>
                    <span style="background:{tier_color}22;border:1px solid {tier_color}66;color:{tier_color};padding:0.2rem 0.75rem;border-radius:999px;font-family:'JetBrains Mono',monospace;font-size:0.72rem;font-weight:700;">{r['badge']}</span>
                </div>
                <div style="color:#94a3b8;font-size:0.83rem;margin-bottom:0.5rem;">
                    Primary Finding: <b style="color:#f8fafc">{r['primary_finding']}</b> · <span style="color:{tier_color}">{r['desc']}</span>
                </div>
                """, unsafe_allow_html=True)

                # Mini Sparkline / Multi-pathology meter
                m_cols = st.columns(4)
                for col_m, ic, nm, p in zip(m_cols, LABEL_ICONS, LABEL_NAMES, r["probs"]):
                    is_pos = (p >= 0.5)
                    bar_c = "#f43f5e" if is_pos else "#38bdf8" if p >= 0.25 else "#64748b"
                    with col_m:
                        st.markdown(f"""
                        <div style="background:rgba(13,20,36,0.6);border:1px solid rgba(255,255,255,0.06);border-radius:6px;padding:3px 6px;font-family:'JetBrains Mono',monospace;font-size:0.68rem;">
                            <span style="color:#94a3b8">{ic} {nm[:5]}:</span> <b style="color:{bar_c}">{p:.0%}</b>
                        </div>
                        """, unsafe_allow_html=True)

            with card_c4:
                st.markdown("<div style='margin-top:0.4rem'></div>", unsafe_allow_html=True)
                if st.button("🔬 Open in Workstation", key=f"triage_inspect_{idx}_{r['patient']}", use_container_width=True):
                    st.session_state["_uploaded_image"] = r["img"]
                    st.session_state["_uploaded_filename"] = r["patient"]
                    st.session_state["nav_choice"] = "🔬  Analyze X-ray"
                    st.session_state["xray_source_radio"] = "📤  Upload Local Radiograph File"
                    st.session_state.pop("_xray_analysis_cache", None)
                    st.rerun()

            st.markdown("<hr style='border-color:rgba(0,242,254,0.12);margin:0.6rem 0 1rem 0'>", unsafe_allow_html=True)
