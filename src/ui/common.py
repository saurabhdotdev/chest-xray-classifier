"""
src/ui/common.py
================
Common UI components, brand headers, status indicators, and label configuration constants.
"""

import base64
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ── Project Root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent

# ── Label Configuration Constants ─────────────────────────────────────────────
LABEL_NAMES  = ["Cardiomegaly", "Pleural Effusion", "Atelectasis", "Edema"]
LABEL_ICONS  = ["🫀", "💧", "🫁", "🌊"]
LABEL_COLORS = ["#ff4757", "#00d4ff", "#ffb300", "#a29bfe"]

PATHOLOGY_GUIDES = {
    "Cardiomegaly": "Attention concentrates on the cardiac silhouette, aortic arch, and transverse cardiac diameter (CTR > 0.50).",
    "Pleural Effusion": "Attention localizes to the bilateral costophrenic angles and hemidiaphragmatic contours for meniscus blunting.",
    "Atelectasis": "Attention identifies linear, platelike, or lobar opacification and volume loss patterns in lung bases.",
    "Edema": "Attention highlights perihilar bilateral 'bat-wing' alveolar haziness and vascular engorgement.",
}

PATHOLOGY_SUBTITLES = {
    "Cardiomegaly": "Enlarged heart (CTR > 50%)",
    "Pleural Effusion": "Fluid pooling in pleural base",
    "Atelectasis": "Partial lung segment collapse",
    "Edema": "Alveolar congestion / bat-wing haze",
}


def hero_header():
    """Renders the executive hero brand header with animated beacon and gradient text."""
    st.markdown("""
<div style="position:relative;text-align:center;padding:2.8rem 1.5rem 2rem 1.5rem;background:linear-gradient(180deg, rgba(14, 165, 233, 0.12) 0%, rgba(99, 102, 241, 0.05) 50%, transparent 100%);border:1px solid rgba(0, 242, 254, 0.2);border-top:2px solid rgba(0, 242, 254, 0.6);border-radius:26px;margin-bottom:2rem;backdrop-filter:blur(20px);box-shadow:0 20px 40px -15px rgba(0, 242, 254, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.1);overflow:hidden;">
    <div style="display:inline-flex;align-items:center;gap:8px;background:rgba(0, 242, 254, 0.08);border:1px solid rgba(0, 242, 254, 0.3);padding:0.35rem 1rem;border-radius:999px;font-family:'JetBrains Mono', monospace;font-size:0.74rem;font-weight:600;color:#38bdf8;letter-spacing:0.12em;margin-bottom:1.2rem;">
        <span class="skipper-beacon"></span>
        NEXT-GEN RADIOLOGY WORKSTATION // DENSENET-121 CO-PILOT
    </div>
    <div style="font-size:3.6rem;font-weight:800;background:linear-gradient(135deg, #ffffff 0%, #38bdf8 45%, #00f2fe 75%, #10b981 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-0.03em;line-height:1.08;margin-bottom:0.6rem;text-shadow:0 0 35px rgba(0, 242, 254, 0.25);">
        🫁 ChestAI
    </div>
    <div style="color:#94a3b8;font-size:1.05rem;letter-spacing:0.16em;text-transform:uppercase;font-weight:500;font-family:'Plus Jakarta Sans', sans-serif;max-width:650px;margin:0 auto;">
        Clinical Diagnostic Workstation & PACS Visual Explainability Engine
    </div>
    <div style="margin-top:1.4rem;display:inline-flex;gap:12px;align-items:center;background:rgba(245, 158, 11, 0.1);border:1px solid rgba(245, 158, 11, 0.35);color:#fbbf24;padding:0.4rem 1.2rem;border-radius:99px;font-size:0.76rem;font-weight:600;font-family:'JetBrains Mono', monospace;letter-spacing:0.05em;">
        <span>⚡ RESEARCH & EDUCATIONAL WORKSTATION · NOT AN APPROVED MEDICAL DEVICE</span>
    </div>
</div>
""", unsafe_allow_html=True)


def section_header(icon: str, title: str, subtitle: str = ""):
    """Renders a standard section header with icon badge and glowing underline."""
    sub_html = f'<div style="color:#94a3b8;font-size:0.84rem;margin-top:2px;font-weight:400">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:0.9rem;border-bottom:1px solid rgba(0,242,254,0.18);padding-bottom:0.8rem;margin-bottom:1.4rem;margin-top:0.5rem;">'
        f'<div style="width:42px;height:42px;border-radius:12px;background:linear-gradient(135deg,rgba(0,242,254,0.15) 0%,rgba(99,102,241,0.1) 100%);border:1px solid rgba(0,242,254,0.3);display:flex;align-items:center;justify-content:center;font-size:1.4rem;box-shadow:0 0 15px rgba(0,242,254,0.15);">{icon}</div>'
        f'<div><div style="font-size:1.35rem;font-weight:700;color:#f8fafc;letter-spacing:-0.01em;">{title}</div>{sub_html}</div>'
        f'</div>',
        unsafe_allow_html=True
    )


def glowing_divider():
    """Renders a sleek glowing horizontal divider."""
    st.markdown("""
<div style="height:1px;background:linear-gradient(90deg, transparent, rgba(0, 242, 254, 0.4), rgba(16, 185, 129, 0.3), transparent);margin:1.8rem 0;"></div>
""", unsafe_allow_html=True)


def status_dot(ok: bool):
    """Returns an inline HTML status dot badge."""
    color = "#10b981" if ok else "#f43f5e"
    label = "ACTIVE" if ok else "PENDING"
    return f"""<span style="display:inline-flex;align-items:center;gap:6px;color:{color};font-size:0.75rem;font-weight:700;font-family:'JetBrains Mono', monospace;"><span style="width:7px;height:7px;border-radius:50%;background:{color};box-shadow:0 0 10px {color};display:inline-block;"></span>{label}</span>"""


def _numpy_to_b64(arr: np.ndarray, ext: str = ".png") -> str:
    """Encodes a numpy RGB image to a base64 string for inline HTML rendering."""
    success, encoded = cv2.imencode(ext, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    if success:
        return base64.b64encode(encoded.tobytes()).decode("utf-8")
    return ""


def fig_exists(name: str) -> bool:
    """Checks whether an evaluation figure exists in results/figures."""
    return (ROOT / "results" / "figures" / name).exists()


def load_fig(name: str) -> Image.Image:
    """Loads an evaluation figure from results/figures."""
    return Image.open(ROOT / "results" / "figures" / name)
