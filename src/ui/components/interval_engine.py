"""
src/ui/components/interval_engine.py
====================================
Longitudinal comparative study utilities: computer-vision interval subtraction,
cross-dissolve blend morphing, neural attention delta tracking, and dual-scan caching.
"""

import hashlib
from typing import Tuple, Dict, Any, List
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import torch
from torchvision import transforms

from src.gradcam import GradCAM
from src.ui.model_service import run_inference


def compute_interval_subtraction(
    img_prior: Image.Image,
    img_curr: Image.Image,
    threshold_val: float = 18.0,
    normalize_exposure: bool = True
) -> Tuple[np.ndarray, float, float]:
    """
    Computes registered computer-vision interval subtraction image between baseline & follow-up.
    Red/Coral: New opacities, worsening infiltration/effusion/cardiomegaly.
    Emerald Green: Interval clearing, resolved opacities, improved aeration.
    """
    g1 = cv2.cvtColor(np.array(img_prior.convert("RGB")), cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(np.array(img_curr.convert("RGB")), cv2.COLOR_RGB2GRAY)

    H, W = g2.shape[:2]
    g1_r = cv2.resize(g1, (W, H))

    if normalize_exposure:
        m_p, s_p = float(np.mean(g1_r)), float(np.std(g1_r))
        m_c, s_c = float(np.mean(g2)), float(np.std(g2))
        g1_norm = np.clip((g1_r - m_p) * (s_c / (s_p + 1e-5)) + m_c, 0, 255).astype(np.float32)
    else:
        g1_norm = g1_r.astype(np.float32)

    g2_f = g2.astype(np.float32)
    delta = g2_f - g1_norm

    base_gray = (g2_f * 0.38).astype(np.uint8)
    sub_rgb = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2RGB)

    pos_mask = delta > threshold_val
    pos_intensity = np.clip((delta - threshold_val) / 60.0, 0, 1)

    neg_mask = delta < -threshold_val
    neg_intensity = np.clip((-delta - threshold_val) / 60.0, 0, 1)

    for c, val in enumerate([244, 63, 94]):  # Red (Worsening opacity)
        sub_rgb[..., c] = np.where(pos_mask, np.clip(sub_rgb[..., c] * (1 - pos_intensity) + val * pos_intensity, 0, 255), sub_rgb[..., c])

    for c, val in enumerate([16, 185, 129]):  # Emerald (Resolving opacity)
        sub_rgb[..., c] = np.where(neg_mask, np.clip(sub_rgb[..., c] * (1 - neg_intensity) + val * neg_intensity, 0, 255), sub_rgb[..., c])

    pos_pct = float(np.mean(pos_mask))
    neg_pct = float(np.mean(neg_mask))
    return sub_rgb, pos_pct, neg_pct


def compute_cross_dissolve(img_prior: Image.Image, img_curr: Image.Image, alpha_curr: float = 0.5) -> np.ndarray:
    """Computes smooth cross-dissolve blend between Prior (0.0) and Current (1.0)."""
    arr1 = np.array(img_prior.convert("RGB"))
    arr2 = np.array(img_curr.convert("RGB"))
    H, W = arr2.shape[:2]
    arr1_r = cv2.resize(arr1, (W, H))
    blended = np.clip(arr1_r.astype(np.float32) * (1.0 - alpha_curr) + arr2.astype(np.float32) * alpha_curr, 0, 255).astype(np.uint8)
    return blended


def compute_attention_shift_map(hm_prior: np.ndarray, hm_curr: np.ndarray, img_curr: Image.Image) -> np.ndarray:
    """Computes neural attention delta (Current CAM - Prior CAM) overlaid on Current scan."""
    arr = np.array(img_curr.convert("RGB"))
    H, W = arr.shape[:2]
    hm1_r = cv2.resize(hm_prior, (W, H), interpolation=cv2.INTER_LINEAR)
    hm2_r = cv2.resize(hm_curr, (W, H), interpolation=cv2.INTER_LINEAR)
    diff = hm2_r - hm1_r

    overlay = arr.copy()
    pos_mask = diff > 0.15
    neg_mask = diff < -0.15

    overlay[pos_mask] = [244, 63, 94]   # Model focus expanded
    overlay[neg_mask] = [16, 185, 129]  # Model focus cleared
    return cv2.addWeighted(arr, 0.65, overlay, 0.35, 0)


def compute_or_get_longitudinal_analysis(
    model: torch.nn.Module,
    cam_gen: GradCAM,
    device: torch.device,
    img_prior: Image.Image,
    img_curr: Image.Image,
    tf: transforms.Compose,
    fitted_temp: float
) -> Dict[str, Any]:
    """Caches dual forward passes and all 8 Grad-CAM heatmaps."""
    h_prior = hashlib.md5(img_prior.tobytes()[::128]).hexdigest()
    h_curr = hashlib.md5(img_curr.tobytes()[::128]).hexdigest()
    pair_key = f"_longitudinal_{h_prior}_{h_curr}_{fitted_temp}"

    if pair_key in st.session_state:
        return st.session_state[pair_key]

    _, _, p_prior, t_prior, np_prior = run_inference(model, device, img_prior, tf, temperature=fitted_temp)
    _, _, p_curr, t_curr, np_curr = run_inference(model, device, img_curr, tf, temperature=fitted_temp)

    cams_prior = [cam_gen(t_prior, i, device) for i in range(4)]
    cams_curr = [cam_gen(t_curr, i, device) for i in range(4)]

    res = {
        "p_prior": p_prior,
        "np_prior": np_prior,
        "cams_prior": cams_prior,
        "p_curr": p_curr,
        "np_curr": np_curr,
        "cams_curr": cams_curr,
    }
    st.session_state[pair_key] = res
    return res
