"""
src/ui/styles.py
================
High-performance, modern medical AI dark theme stylesheet for ChestAI.
Features:
  - Deep space navy background (#060913) with ambient radial lighting & dot-matrix grid
  - Precision medical telemetry typography (Plus Jakarta Sans, JetBrains Mono)
  - Ambient glassmorphic card surfaces with cyan, emerald, and indigo glow borders
  - Fluid micro-animations (pulsing beacons, smooth hover lifts, gradient buttons)
"""

import streamlit as st

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
  --skipper-bg:            #060913;
  --skipper-surface:       rgba(13, 20, 36, 0.72);
  --skipper-surface-card:  rgba(15, 23, 42, 0.82);
  --skipper-cyan:          #00f2fe;
  --skipper-sky:           #38bdf8;
  --skipper-emerald:       #10b981;
  --skipper-amber:         #f59e0b;
  --skipper-rose:          #f43f5e;
  --skipper-indigo:        #6366f1;
  --skipper-text-primary:  #f8fafc;
  --skipper-text-muted:    #94a3b8;
  --skipper-border:        rgba(0, 242, 254, 0.16);
  --skipper-border-glow:   rgba(0, 242, 254, 0.45);
  --skipper-glow:          0 0 25px rgba(0, 242, 254, 0.18);
}

/* ── Ambient Radial Lighting & Dot Matrix Grid ── */
html, body, [data-testid="stAppViewContainer"] {
  background-color: var(--skipper-bg) !important;
  background-image: 
    radial-gradient(at 0% 0%, rgba(14, 165, 233, 0.12) 0px, transparent 50%),
    radial-gradient(at 100% 0%, rgba(99, 102, 241, 0.10) 0px, transparent 50%),
    radial-gradient(at 50% 100%, rgba(16, 185, 129, 0.08) 0px, transparent 50%),
    radial-gradient(rgba(255, 255, 255, 0.04) 1px, transparent 1px) !important;
  background-size: 100% 100%, 100% 100%, 100% 100%, 28px 28px !important;
  font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif !important;
  color: var(--skipper-text-primary) !important;
}

/* ── Modern Custom Scrollbars ── */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}
::-webkit-scrollbar-track {
  background: #060913;
}
::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, rgba(0, 242, 254, 0.4), rgba(99, 102, 241, 0.3));
  border-radius: 99px;
}
::-webkit-scrollbar-thumb:hover {
  background: rgba(0, 242, 254, 0.7);
}

/* ── Skipper Sidebar ── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(6, 10, 20, 0.96) 0%, rgba(10, 16, 32, 0.98) 100%) !important;
  backdrop-filter: blur(24px) !important;
  border-right: 1px solid var(--skipper-border) !important;
  box-shadow: 4px 0 28px rgba(0, 0, 0, 0.6) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.2rem !important; }

/* ── Skipper Pulse Beacon Animation ── */
.skipper-beacon {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #00f2fe;
  box-shadow: 0 0 0 0 rgba(0, 242, 254, 0.7);
  animation: skipper-pulse 2s infinite;
}

@keyframes skipper-pulse {
  0% {
    transform: scale(0.95);
    box-shadow: 0 0 0 0 rgba(0, 242, 254, 0.7);
  }
  70% {
    transform: scale(1);
    box-shadow: 0 0 0 10px rgba(0, 242, 254, 0);
  }
  100% {
    transform: scale(0.95);
    box-shadow: 0 0 0 0 rgba(0, 242, 254, 0);
  }
}

/* ── Skipper Metric Card ── */
div[data-testid="stMetric"] {
  background: linear-gradient(135deg, rgba(15, 23, 42, 0.75) 0%, rgba(15, 23, 42, 0.45) 100%) !important;
  backdrop-filter: blur(16px) !important;
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  border-top: 2px solid rgba(0, 242, 254, 0.4) !important;
  border-radius: 16px !important;
  padding: 1.15rem 1.4rem !important;
  box-shadow: 0 10px 25px -10px rgba(0, 0, 0, 0.5) !important;
  transition: all 0.25s ease !important;
}
div[data-testid="stMetric"]:hover {
  transform: translateY(-2px);
  border-color: rgba(0, 242, 254, 0.5) !important;
  box-shadow: 0 14px 30px -10px rgba(0, 242, 254, 0.2) !important;
}
div[data-testid="stMetric"] label {
  color: var(--skipper-text-muted) !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-size: 0.82rem !important;
  font-weight: 600 !important;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
  color: var(--skipper-cyan) !important;
  font-family: 'JetBrains Mono', Consolas, monospace !important;
  font-size: 1.85rem !important;
  font-weight: 700 !important;
  text-shadow: 0 0 20px rgba(0, 242, 254, 0.3);
}

/* ── Skipper Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  gap: 8px;
  background: rgba(13, 20, 36, 0.75) !important;
  backdrop-filter: blur(12px) !important;
  padding: 6px;
  border-radius: 14px;
  border: 1px solid rgba(0, 242, 254, 0.18) !important;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 10px !important;
  color: var(--skipper-text-muted) !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-weight: 500 !important;
  font-size: 0.88rem !important;
  padding: 8px 18px !important;
  transition: all 0.2s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
  color: var(--skipper-text-primary) !important;
  background: rgba(255, 255, 255, 0.04) !important;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, rgba(0, 242, 254, 0.2) 0%, rgba(99, 102, 241, 0.15) 100%) !important;
  border: 1px solid rgba(0, 242, 254, 0.4) !important;
  color: #ffffff !important;
  font-weight: 700 !important;
  box-shadow: 0 0 18px rgba(0, 242, 254, 0.25) !important;
}

/* ── Sliders ── */
div[data-baseweb="slider"] {
  accent-color: var(--skipper-cyan) !important;
}

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, rgba(0, 242, 254, 0.2) 0%, rgba(99, 102, 241, 0.18) 100%) !important;
  border: 1px solid rgba(0, 242, 254, 0.45) !important;
  color: #ffffff !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-weight: 600 !important;
  border-radius: 12px !important;
  padding: 0.6rem 1.3rem !important;
  transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
.stButton > button:hover {
  background: linear-gradient(135deg, rgba(0, 242, 254, 0.38) 0%, rgba(99, 102, 241, 0.32) 100%) !important;
  box-shadow: 0 0 20px rgba(0, 242, 254, 0.45) !important;
  border-color: #00f2fe !important;
  transform: translateY(-2px);
}

/* ── Download Button ── */
.stDownloadButton > button {
  background: linear-gradient(135deg, rgba(16, 185, 129, 0.25) 0%, rgba(0, 242, 254, 0.18) 100%) !important;
  border: 1px solid rgba(16, 185, 129, 0.6) !important;
  color: #10b981 !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-weight: 700 !important;
  border-radius: 12px !important;
  padding: 0.75rem 1.4rem !important;
  box-shadow: 0 4px 18px rgba(16, 185, 129, 0.2) !important;
  transition: all 0.25s ease !important;
}
.stDownloadButton > button:hover {
  background: linear-gradient(135deg, rgba(16, 185, 129, 0.4) 0%, rgba(0, 242, 254, 0.28) 100%) !important;
  box-shadow: 0 0 25px rgba(16, 185, 129, 0.5) !important;
  transform: translateY(-2px);
}

/* ── Radio Pills in Sidebar & Main ── */
[data-testid="stSidebar"] div[role="radiogroup"] label {
  background: rgba(13, 20, 36, 0.6) !important;
  border: 1px solid rgba(0, 242, 254, 0.12) !important;
  border-radius: 12px;
  padding: 0.65rem 0.95rem;
  margin-bottom: 0.4rem;
  transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
  border-color: rgba(0, 242, 254, 0.45) !important;
  background: rgba(0, 242, 254, 0.08) !important;
  transform: translateX(3px);
}

/* ── Expanders ── */
[data-testid="stExpander"] {
  background: linear-gradient(135deg, rgba(15, 23, 42, 0.7) 0%, rgba(15, 23, 42, 0.4) 100%) !important;
  backdrop-filter: blur(14px) !important;
  border: 1px solid rgba(0, 242, 254, 0.2) !important;
  border-radius: 14px !important;
}

/* ── File Uploader Styling ── */
[data-testid="stFileUploader"] {
  background: rgba(13, 20, 36, 0.6) !important;
  border: 1px dashed rgba(0, 242, 254, 0.35) !important;
  border-radius: 16px !important;
  padding: 1.2rem !important;
  transition: all 0.25s ease !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: rgba(0, 242, 254, 0.7) !important;
  background: rgba(0, 242, 254, 0.04) !important;
  box-shadow: 0 0 25px rgba(0, 242, 254, 0.15) !important;
}
[data-testid="stFileUploader"] section {
  padding: 0.8rem 0 !important;
}
[data-testid="stFileUploader"] section button {
  background: linear-gradient(135deg, rgba(0, 242, 254, 0.2) 0%, rgba(99, 102, 241, 0.2) 100%) !important;
  border: 1px solid rgba(0, 242, 254, 0.5) !important;
  color: #38bdf8 !important;
  font-weight: 700 !important;
  border-radius: 10px !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
  background: rgba(0, 242, 254, 0.06) !important;
  backdrop-filter: blur(12px) !important;
  border-left: 3px solid var(--skipper-cyan) !important;
  border-radius: 0 12px 12px 0 !important;
  color: var(--skipper-text-primary) !important;
}
</style>
"""

def inject_custom_css():
    """Injects high-DPI Medical AI Dark Workstation styles into Streamlit page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
