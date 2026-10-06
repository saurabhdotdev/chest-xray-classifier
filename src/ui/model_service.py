"""
src/ui/model_service.py
=======================
Service layer for configuration loading, evaluation metrics retrieval,
neural network checkpoint weights, PyTorch inference, and Grad-CAM caching.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import yaml

from src.train import build_model
from src.gradcam import GradCAM, denormalize

ROOT = Path(__file__).resolve().parent.parent.parent


@st.cache_resource
def load_config() -> Dict[str, Any]:
    """Loads config.yaml from project root."""
    cfg_path = ROOT / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


@st.cache_data
def load_metrics() -> Dict[str, Any]:
    """Loads evaluation metrics and training history from results/metrics."""
    metrics = {}
    metrics_dir = ROOT / "results" / "metrics"
    for fname in ["train_history.json", "external_eval.json", "calibration.json"]:
        fp = metrics_dir / fname
        if fp.exists():
            with open(fp, "r", encoding="utf-8") as f:
                metrics[fname.replace(".json", "")] = json.load(f)
    return metrics


@st.cache_resource
def get_model_and_cam() -> Tuple[torch.nn.Module, GradCAM, torch.device, float]:
    """
    Loads checkpoint, temperature scaling parameters, and sets up Grad-CAM.
    CRITICAL OPTIMIZATION: Always uses pretrained=False because best_model.pth holds
    all trained weights. This guarantees ZERO network calls and instant startup.
    """
    ckpt_path = ROOT / "results" / "checkpoints" / "best_model.pth"
    cal_ckpt_path = ROOT / "results" / "checkpoints" / "best_model_calibrated.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_path = ROOT / "config.yaml"
    backbone = "densenet121"
    if cfg_path.exists():
        import yaml
        with open(cfg_path) as f:
            backbone = yaml.safe_load(f).get("model", {}).get("backbone", "densenet121")

    fast_cfg = {"model": {"backbone": backbone, "pretrained": False}}
    model = build_model(fast_cfg, num_classes=4)
    if ckpt_path.exists():
        model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
    model.eval().to(device)

    fitted_temp = 0.05
    if cal_ckpt_path.exists():
        try:
            cal_data = torch.load(str(cal_ckpt_path), map_location=device)
            if isinstance(cal_data, dict) and "temperature" in cal_data:
                fitted_temp = float(cal_data["temperature"])
        except Exception:
            pass

    cam_gen = GradCAM(model, backbone=backbone)
    return model, cam_gen, device, fitted_temp


def get_image_transforms() -> transforms.Compose:
    """Returns standard ImageNet evaluation transforms for 224x224 input."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def run_inference(
    model: torch.nn.Module,
    device: torch.device,
    img_pil: Image.Image,
    tf: transforms.Compose,
    temperature: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor, np.ndarray]:
    """Runs model forward pass returning raw logits, raw probs, calibrated probs, and denormalized image."""
    img_t = tf(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_t)
        raw_probs = torch.sigmoid(logits).cpu().numpy()[0]
        if temperature and temperature > 0:
            cal_probs = torch.sigmoid(logits / float(temperature)).cpu().numpy()[0]
        else:
            cal_probs = raw_probs
    img_np = denormalize(img_t.squeeze(0))
    return logits.cpu().numpy()[0], raw_probs, cal_probs, img_t, img_np


def compute_or_get_analysis(
    model: torch.nn.Module,
    cam_gen: GradCAM,
    device: torch.device,
    img_pil: Image.Image,
    tf: transforms.Compose,
    default_temp: float
) -> Dict[str, Any]:
    """
    Computes DenseNet-121 forward pass and 4 Grad-CAM heatmaps ONCE per radiograph.
    Caches in st.session_state using an image hash.
    Tweaking sliders (threshold, alpha, calibration, patient info) responds in 0.01 seconds.
    """
    img_rgb = img_pil.convert("RGB")
    raw_sample = img_rgb.tobytes()[::128]
    h = hashlib.md5(f"{img_rgb.size}_{len(raw_sample)}_{raw_sample[:256]}".encode("latin1", errors="ignore")).hexdigest()

    cache = st.session_state.get("_xray_analysis_cache")
    if cache is not None and cache.get("hash") == h:
        return cache

    logits, raw_probs, cal_probs, img_t, img_np = run_inference(
        model, device, img_rgb, tf, temperature=default_temp
    )

    all_heatmaps = []
    for i in range(4):
        hm = cam_gen(img_t, i, device)
        all_heatmaps.append(hm)

    result = {
        "hash": h,
        "logits": logits,
        "raw_probs": raw_probs,
        "cal_probs": cal_probs,
        "img_t": img_t,
        "img_np": img_np,
        "heatmaps": all_heatmaps,
    }
    st.session_state["_xray_analysis_cache"] = result
    return result
