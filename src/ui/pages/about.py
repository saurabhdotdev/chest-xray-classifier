"""
src/ui/pages/about.py
=====================
Model card, architectural specifications, clinical governance, and limitations.
"""

from typing import Dict, Any
import streamlit as st

from src.ui.common import section_header


def page_about(cfg: Dict[str, Any]):
    """Renders algorithmic model card and regulatory governance guidelines."""
    section_header("📖", "Model Card & Clinical Governance", "Regulatory specifications, algorithmic architecture, and intended use guidelines")

    st.markdown(r"""
    ### 🔬 Algorithmic Specifications
    - **Architecture:** DenseNet-121 (Huang et al., CVPR 2017)
    - **Pretraining:** ImageNet-1K initialization, fine-tuned on frontal chest radiographs
    - **Input Dimensions:** 224 × 224 × 3 (RGB standard normalization)
    - **Loss Function:** Multi-label Binary Cross-Entropy with Logits
    - **Optimization:** AdamW ($\eta = 3 \times 10^{-4}$, weight decay $10^{-5}$) with Cosine Annealing
    - **Calibration Method:** Post-hoc Temperature Scaling (Guo et al., ICML 2017)
    - **Visual Explainability:** Gradient-weighted Class Activation Mapping (Grad-CAM)

    ---

    ### ⚠️ Limitations & Governance
    1. **Not a Diagnostic Device:** ChestAI is an investigational software prototype designed for research and pre-read assistance. It is not approved by the US FDA, CE Mark, or CDSCO.
    2. **Single-View Frontal Images:** Current architecture is trained on PA/AP views; lateral chest radiographs are not formally benchmarked.
    3. **Pediatric Generalization:** Evaluated primarily on adult cohorts (CheXpert & NIH); performance on neonatal or pediatric cohorts is unvalidated.
    4. **Resolution Downsampling:** High-resolution digital radiography (3000×3000) is downsampled to 224×224; subtle pneumothorax or micro-nodules may fall below detection resolution.
    """, unsafe_allow_html=True)
