"""
app.py — ChestAI Radiology Intelligence & Clinical Diagnostic Workstation
========================================================================
Modular, ultra-responsive workstation orchestrator for Chest X-ray classification.

Architecture & Folder Structure:
--------------------------------
src/ui/
├── styles.py               # Skipper dark medical theme CSS & style injection
├── common.py               # Brand headers, glowing dividers, status indicators, and constants
├── model_service.py        # Model checkpoint loading, inference, and Grad-CAM caching
├── components/
│   ├── navigation.py       # Sidebar navigation, clinical governance, and page router
│   ├── findings_summary.py # Polar radar chart & horizontal calibrated confidence spectrum
│   ├── zoom_lens.py        # Anatomical Quadrant Zoom Lens & Clinical Viewport
│   ├── interval_engine.py  # Computer-vision interval subtraction & morph blend
│   └── triage_components.py# Batch ZIP extraction & triage worklist serialization
└── pages/
    ├── home.py             # Executive dashboard & workstation launchpad
    ├── analyze.py          # Radiology intelligence workstation (Analyze)
    ├── compare.py          # Comparative longitudinal study & interval subtraction
    ├── triage.py           # Emergency department batch triage worklist
    ├── performance.py      # Model validation performance & AUROC
    ├── calibration.py      # Temperature scaling & reliability curves
    ├── shift.py            # Dataset shift & NIH zero-shot evaluation
    └── about.py            # Model card & clinical governance
"""

import streamlit as st

# ── 1. Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChestAI — Radiology Intelligence",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 2. Style Injection ────────────────────────────────────────────────────────
from src.ui.styles import inject_custom_css
inject_custom_css()

# ── 3. Imports ────────────────────────────────────────────────────────────────
from src.ui.model_service import load_config, load_metrics
from src.ui.components.navigation import render_sidebar
from src.ui.pages import (
    page_home,
    page_analyze,
    page_compare,
    page_triage,
    page_performance,
    page_calibration,
    page_shift,
    page_about,
)


def main():
    """Main application lifecycle and page routing orchestrator."""
    cfg = load_config()
    metrics = load_metrics()

    # Render sidebar navigation and retrieve active page route
    page = render_sidebar()

    # Route to modular workstation pages
    if page == "home":
        page_home(cfg, metrics)
    elif page == "analyze":
        page_analyze(cfg, metrics)
    elif page == "compare":
        page_compare(cfg, metrics)
    elif page == "triage":
        page_triage(cfg, metrics)
    elif page == "performance":
        page_performance(cfg, metrics)
    elif page == "calibration":
        page_calibration(cfg, metrics)
    elif page == "shift":
        page_shift(cfg, metrics)
    elif page == "about":
        page_about(cfg)

    # Workstation Footer
    st.markdown("""
    <div style="
        margin-top: 3.5rem; padding: 1.5rem 1rem 1rem 1rem;
        border-top: 1px solid rgba(0, 242, 254, 0.14);
        text-align: center; color: #64748b; font-size: 0.76rem;
        font-family: 'JetBrains Mono', monospace; line-height: 1.6;
    ">
        <div>🫁 <b>ChestAI Radiology Intelligence Workstation</b> · DenseNet-121 Multi-Label Foundation Engine</div>
        <div style="color: #475569; margin-top: 4px;">
            Research Pre-Read Prototype · Stanford CheXpert & NIH ChestX-ray14 Benchmark Cohorts · Not an FDA-Approved Medical Device
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
