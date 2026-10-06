"""
src/ui/components/navigation.py
===============================
Workstation sidebar navigation, active page router, and clinical governance disclaimers.
"""

import streamlit as st


def render_sidebar() -> str:
    """Renders the executive sidebar navigation and clinical governance status badge."""
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1.6rem 0.5rem 1.2rem 0.5rem;border-bottom:1px solid rgba(0, 242, 254, 0.16);margin-bottom:1.3rem;">
            <div style="display:inline-flex;align-items:center;justify-content:center;width:58px;height:58px;border-radius:18px;background:linear-gradient(135deg, rgba(0, 242, 254, 0.22) 0%, rgba(99, 102, 241, 0.16) 100%);border:1px solid rgba(0, 242, 254, 0.4);box-shadow:0 0 22px rgba(0, 242, 254, 0.22);font-size:2rem;margin-bottom:0.6rem;">🫁</div>
            <div style="font-size:1.45rem;font-weight:800;background:linear-gradient(135deg, #ffffff 0%, #38bdf8 50%, #00f2fe 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-0.02em;line-height:1.1;">ChestAI</div>
            <div style="display:flex;align-items:center;justify-content:center;gap:6px;margin-top:5px;">
                <span class="skipper-beacon"></span>
                <span style="color:#38bdf8;font-size:0.68rem;font-family:'JetBrains Mono', monospace;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;">Skipper Workstation</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        pages = {
            "🏠  Home":                 "home",
            "🔬  Analyze X-ray":        "analyze",
            "🔄  Comparative Study":    "compare",
            "📋  Emergency Triage":     "triage",
            "📊  Performance":          "performance",
            "📐  Calibration":          "calibration",
            "🌍  Dataset Shift":        "shift",
            "📖  Model Card":           "about",
        }
        
        # Support programmatic navigation from quick launch buttons
        default_index = 0
        target_page = st.session_state.get("_nav_target")
        if target_page:
            for idx, p_key in enumerate(pages.values()):
                if p_key == target_page:
                    default_index = idx
                    break
            st.session_state["_nav_target"] = None

        choice = st.radio("Navigation", list(pages.keys()), index=default_index, label_visibility="collapsed", key="nav_choice")

        st.markdown("""
        <div style="margin-top:2rem;padding:0.9rem 1rem;background:linear-gradient(135deg, rgba(244, 63, 94, 0.08) 0%, rgba(15, 23, 42, 0.7) 100%);border:1px solid rgba(244, 63, 94, 0.3);border-radius:14px;font-size:0.74rem;color:#fda4af;line-height:1.5;backdrop-filter:blur(12px);">
            <div style="display:flex;align-items:center;gap:6px;font-weight:700;color:#f43f5e;margin-bottom:4px;font-family:'JetBrains Mono', monospace;">
                <span>⚠️</span> CLINICAL GOVERNANCE
            </div>
            Research & pre-read demonstration workstation. All findings require board-certified clinical review.
        </div>
        """, unsafe_allow_html=True)

        return pages[choice]
